import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from utils import setup_seed, cos_sim
from model.moe_adapter import SFMoECLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS, DATA_PATH, TEST_BRIGHTNESS_FACTOR, TEST_GAMMA
from dataset.constants import ASPECT_PRESERVING_DATASETS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    group_indices_by_variant,
    metrics_eval,
    calcuate_metric_pixel,
    letterbox_resize_bgr,
    apply_photometric_bgr,
)
from sklearn.metrics import precision_recall_curve
import warnings
import re
import sys
import cv2
import time

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_epoch_number(filename):
    match = re.search(r'moe_epoch_(\d+)\.pth', filename)
    return int(match.group(1)) if match else -1

def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader: 
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features

def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
    eval_fp16: bool = False,
):
    # When eval_fp16=True, per-image anomaly score tensors are cast to fp16
    # before being appended to the CPU accumulators. metrics_eval casts back
    # to fp32 before normalization + AUC, so the only precision loss is the
    # storage step. Off by default; fp32 path is bit-identical to original.
    store_dtype = np.float16 if eval_fp16 else None

    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "path": []}
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]              
        class_name = input_data["class_name"]            
        assert len(set(class_name)) == 1, "mixed class not supported"

        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        epoch_text_feature = class_text_embeddings

        patch_features, det_feature, _, _ = model(image)

        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())

        patch_preds = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        if store_dtype is not None:
            patch_preds = patch_preds.astype(store_dtype)
        preds.append(patch_preds)

    
        anomaly_maps_b = []
        for f in patch_features:
            patch_anomaly_scores = 100.0 * torch.matmul(f, epoch_text_feature) 
            B, L, C = patch_anomaly_scores.shape
            H = int(np.sqrt(L))
            patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)  
            patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2          
            temp = F.interpolate(
                patch_pred.unsqueeze(1), size=img_size, mode="bilinear", align_corners=True
            )                                                                    
            anomaly_maps_b.append(temp)
        anomaly_maps_b = torch.cat(anomaly_maps_b, dim=1).sum(1).cpu().numpy()  

        
        base_path = DATA_PATH[dataset]
        gt_b = input_data["mask"].cpu().numpy()                                   
        for b in range(len(file_name)):
            gt = gt_b[b]
            if gt.ndim == 3 and gt.shape[0] == 1:
                gt = gt[0]
            gt = np.nan_to_num(gt)
            gt = (gt > 0.5).astype(np.uint8)
            gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            am = anomaly_maps_b[b]
            if store_dtype is not None:
                am = am.astype(store_dtype)

            results["cls_names"].append(class_name[b])
            results["imgs_masks"].append(gt)
            results["anomaly_maps"].append(am)
            results["path"].append(os.path.join(base_path, file_name[b]))
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names, results


def _normalize01(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _heatmap_overlay(bgr, score_map, alpha=0.5):
    norm = _normalize01(score_map)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, alpha, heat, 1 - alpha, 0)


def compute_image_scores(preds, preds_image, domain):
    """Combine pixel-max and image-level scores in raw space.

    Same formula `metrics_eval` uses (industrial: 0.5/0.5, medical: pmax only),
    minus the per-batch min-max so the result is comparable across labeled
    and unlabeled splits of the same dataset.
    """
    pmax = np.asarray(preds, dtype=np.float64).max(axis=(1, 2))
    image_preds = np.asarray(preds_image, dtype=np.float64)
    if domain != "Medical":
        return 0.5 * pmax + 0.5 * image_preds
    return pmax


def derive_image_threshold(labels, scores):
    """Pick the F1-optimal image-score threshold from the labeled set.

    Mirrors the PR-curve approach `calcuate_metric_pixel` uses for pixel masks,
    but applied to image-level scores. Returns (threshold, best_f1).
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.max() == labels.min():
        return float(np.median(scores)), 0.0
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    f1_per_th = f1[:-1]
    if f1_per_th.size == 0:
        return float(np.median(scores)), 0.0
    idx = int(np.argmax(f1_per_th))
    return float(thresholds[idx]), float(f1_per_th[idx])


def _save_score_histogram(scores, threshold, out_path, title, bins=40, w=900, h=420):
    """Render a histogram of raw image scores to a PNG using cv2 only.

    Auto-ranges to [min, max] of the data (raw scores aren't pinned to [0,1]).
    Marks the threshold so the user can see where it falls in the distribution.
    """
    scores = np.asarray(scores, dtype=np.float64)
    lo = float(min(scores.min(), threshold))
    hi = float(max(scores.max(), threshold))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    counts, edges = np.histogram(scores, bins=bins, range=(lo, hi))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    pad_l, pad_r, pad_t, pad_b = 60, 20, 50, 60
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    cv2.rectangle(canvas, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (0, 0, 0), 1)

    cmax = max(int(counts.max()), 1)
    bar_w = plot_w / bins
    for i, c in enumerate(counts):
        x0 = int(round(pad_l + i * bar_w))
        x1 = int(round(pad_l + (i + 1) * bar_w)) - 1
        bh = int(round((c / cmax) * plot_h))
        y0 = pad_t + plot_h - bh
        bin_center = 0.5 * (edges[i] + edges[i + 1])
        color = (60, 60, 200) if bin_center >= threshold else (180, 120, 60)
        cv2.rectangle(canvas, (x0, y0), (x1, pad_t + plot_h), color, -1)

    tx = int(round(pad_l + (threshold - lo) / (hi - lo) * plot_w))
    cv2.line(canvas, (tx, pad_t), (tx, pad_t + plot_h), (0, 0, 0), 1)
    cv2.putText(
        canvas, f"thr={threshold:.3f}", (tx + 4, pad_t + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
    )

    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(pad_l + f * plot_w))
        v = lo + f * (hi - lo)
        cv2.line(canvas, (x, pad_t + plot_h), (x, pad_t + plot_h + 4), (0, 0, 0), 1)
        cv2.putText(
            canvas, f"{v:.3f}", (x - 18, pad_t + plot_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA,
        )
    cv2.putText(
        canvas, f"max count: {cmax}", (pad_l, pad_t - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, title, (pad_l, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "image score (raw, pmax+image_preds combined)",
        (pad_l, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.imwrite(out_path, canvas)


def unlabeled_summary(
    scores,
    threshold,
    file_names,
    results,
    class_name,
    args,
    logger,
):
    """Apply a precomputed image-score threshold to unlabeled samples.

    Scores are raw (pmax + image_preds combined per `compute_image_scores`,
    no min-max), so the threshold passed in must be in that same raw space —
    typically the F1-optimal value from `derive_image_threshold` on the
    labeled split. This makes `% defected` reflect model quality rather than
    just the shape of the score distribution within the unlabeled batch.
    """
    out_root = os.path.join(args.save_path, "unlabeled", class_name)
    os.makedirs(out_root, exist_ok=True)

    scores = np.asarray(scores, dtype=np.float64)
    pred_labels = (scores >= threshold).astype(np.int64)
    n_total = scores.shape[0]
    n_def = int(pred_labels.sum())
    n_norm = n_total - n_def
    logger.info("-----------------------------------------------")
    logger.info(
        "[unlabeled %s] threshold=%.4f -> normal=%d, defected=%d, total=%d (%.1f%% defected)",
        class_name, threshold, n_norm, n_def, n_total,
        100.0 * n_def / max(n_total, 1),
    )
    logger.info(
        "[unlabeled %s] score stats: min=%.4f mean=%.4f median=%.4f max=%.4f",
        class_name, float(scores.min()), float(scores.mean()),
        float(np.median(scores)), float(scores.max()),
    )

    csv_path = os.path.join(out_root, "predictions.csv")
    df = DataFrame({
        "file_name": file_names,
        "score": scores,
        "predicted_label": pred_labels,
    })
    df.sort_values("score", ascending=False).to_csv(csv_path, index=False)
    logger.info("[unlabeled %s] wrote %s", class_name, csv_path)

    hist_path = os.path.join(out_root, "score_histogram.png")
    _save_score_histogram(
        scores, threshold, hist_path,
        title=f"{args.dataset} / {class_name} unlabeled image-score distribution",
    )
    logger.info("[unlabeled %s] wrote %s", class_name, hist_path)

    k = max(1, min(args.num_viz, n_total))
    order = np.argsort(scores)
    bottom_idx = order[:k]
    top_idx = order[-k:][::-1]

    aspect_preserving = args.dataset in ASPECT_PRESERVING_DATASETS
    brightness_factor = TEST_BRIGHTNESS_FACTOR.get(args.dataset, 1.0)
    gamma = TEST_GAMMA.get(args.dataset, 1.0)
    paths = results["path"]
    anomaly_maps = results["anomaly_maps"]

    for tag, idx_list in (("defected_topK", top_idx), ("normal_bottomK", bottom_idx)):
        sub_dir = os.path.join(out_root, tag)
        os.makedirs(sub_dir, exist_ok=True)
        for rank, i in enumerate(idx_list):
            raw = cv2.imread(paths[i])
            if raw is None:
                logger.warning("[unlabeled %s] could not read %s", class_name, paths[i])
                continue
            raw = apply_photometric_bgr(raw, brightness_factor, gamma)
            if aspect_preserving:
                img = letterbox_resize_bgr(raw, args.img_size)
            else:
                img = cv2.resize(raw, (args.img_size, args.img_size))
            overlay = _heatmap_overlay(img, anomaly_maps[i])
            panel = np.concatenate([img, overlay], axis=1)
            label = f"score={scores[i]:.3f}  pred={'defected' if pred_labels[i] else 'normal'}"
            cv2.putText(
                panel, label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            stem = os.path.splitext(os.path.basename(paths[i]))[0]
            out = os.path.join(sub_dir, f"{rank:02d}_{stem}.png")
            cv2.imwrite(out, panel)
    logger.info(
        "[unlabeled %s] saved %d top + %d bottom visualizations under %s",
        class_name, k, k, out_root,
    )

    return {
        "class name": class_name,
        "total": int(n_total),
        "normal": int(n_norm),
        "defected": int(n_def),
        "% defected": round(100.0 * n_def / max(n_total, 1), 2),
    }


def save_alpha_logs(alphas_per_layer, file_names, dataset, class_name, save_path, logger):
    """Save per-image alpha values as CSV and a per-layer histogram PNG.

    alphas_per_layer: {layer_idx: np.ndarray of shape (N,)} from
        SFMoECLIP.pop_alpha_collection(). file_names is the matching per-image
        list returned by get_predictions().
    """
    layers = sorted(k for k, v in alphas_per_layer.items() if v is not None)
    if not layers:
        logger.info("[alpha %s] no alpha values collected (alpha gate off?)", class_name)
        return

    out_dir = os.path.join(save_path, "alpha_logs", f"{dataset}_{class_name}")
    os.makedirs(out_dir, exist_ok=True)

    cols = {"file_name": list(file_names)}
    n_files = len(file_names)
    for li in layers:
        a = alphas_per_layer[li]
        if a.shape[0] != n_files:
            logger.warning(
                "[alpha %s] layer %d alpha length %d != file count %d; truncating",
                class_name, li, a.shape[0], n_files,
            )
            a = a[:n_files]
        cols[f"alpha_layer_{li}"] = a
    df = DataFrame(cols)
    csv_path = os.path.join(out_dir, "alpha_values.csv")
    df.to_csv(csv_path, index=False)

    stats = " | ".join(
        f"L{li}: mean={alphas_per_layer[li].mean():.3f} std={alphas_per_layer[li].std():.3f} "
        f"min={alphas_per_layer[li].min():.3f} max={alphas_per_layer[li].max():.3f}"
        for li in layers
    )
    logger.info("[alpha %s] %s", class_name, stats)
    logger.info("[alpha %s] wrote %s", class_name, csv_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("[alpha %s] matplotlib not available (%s); skipping plot", class_name, e)
        return

    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, li in zip(axes, layers):
        a = alphas_per_layer[li]
        ax.hist(a, bins=30, range=(0.0, 1.0), color="#3b6ea5", alpha=0.85)
        ax.axvline(float(a.mean()), color="red", linestyle="--", linewidth=1,
                   label=f"mean={a.mean():.3f}")
        ax.set_xlim(0, 1)
        ax.set_title(f"layer {li}")
        ax.set_xlabel(r"$\alpha(x)$")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("# images")
    fig.suptitle(f"{dataset} / {class_name}: per-image alpha distribution")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "alpha_histogram.png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    logger.info("[alpha %s] wrote %s", class_name, png_path)


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument(
        "--no_use_paa",
        action="store_false",
        help="Whether to use patch average aggregation (true/false)",
    )
    parser.add_argument(
        "--seg_proj_sharing_strategy",
        type=str,
        choices=["separate", "shared"],
        default="shared",
        help="Projector sharing strategy for patch average aggregation",
    )
    # MoE controls
    parser.add_argument("--moe_r", type=int, default=8)
    parser.add_argument("--moe_lora_alpha", type=int, default=16)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument(
        "--moe_layers",
        type=str,
        default="5,11,17,23",
        help="Comma-separated layer indices for MoE application",
    )
    parser.add_argument("--no_use_fofs", action="store_false", help="Use fixed-A LoRA partitioning (default: True)")
    parser.add_argument(
        "--fofs_mode",
        type=str,
        choices=["block", "pca"],
        default="block",
        help="FOFS partitioning strategy. Must match the value used at training time.",
    )
    parser.add_argument(
        "--fofs_basis_path",
        type=str,
        default=None,
        help="Path to precomputed PCA basis (.pt) for fofs_mode='pca'. "
             "Only used during model construction; the trained checkpoint overrides A weights.",
    )
    parser.add_argument(
        "--use_wavelet_gate",
        action="store_true",
        help="Match the training-time setting of the Haar gate descriptor. ",
    )
    parser.add_argument(
        "--wavelet_levels",
        type=int,
        default=1,
        help="Number of dyadic dilations for the Haar gate descriptor. Must "
             "match the training-time value when --use_wavelet_gate is set.",
    )
    parser.add_argument(
        "--use_alpha_gate",
        action="store_true",
        help="Match the training-time setting of the conditional alpha head. "
             "Off by default; must mirror the training value to load the "
             "checkpoint.",
    )
    parser.add_argument(
        "--alpha_hidden_dim",
        type=int,
        default=64,
        help="Hidden width of the alpha MLP. Must match the training value "
             "when --use_alpha_gate is set.",
    )
    parser.add_argument(
        "--log_alpha",
        action="store_true",
        help="Record per-image alpha(x) from each MoE layer during testing, "
             "save as CSV, and render a per-layer histogram. Off by default "
             "and only effective when --use_alpha_gate is set.",
    )
    parser.add_argument(
        "--use_freq_experts",
        action="store_true",
        help="Match the training-time setting of the parallel frequency-domain "
             "MoE stream. Off by default; must mirror training to load the "
             "checkpoint.",
    )
    parser.add_argument(
        "--freq_levels",
        type=int,
        default=1,
        help="Number of dyadic dilations summed inside the Haar high-pass "
             "feeding the freq experts. Must match the training value when "
             "--use_freq_experts is set.",
    )
    parser.add_argument(
        "--freq_beta_init",
        type=float,
        default=-4.0,
        help="Initial freq_beta logit at model construction; the trained "
             "checkpoint overrides it. Only used with --use_freq_experts.",
    )
    parser.add_argument(
        "--log_freq_beta",
        action="store_true",
        help="Log sigmoid(freq_beta) for each MoE layer at test-time. "
             "Off by default; only effective with --use_freq_experts.",
    )
    parser.add_argument(
        "--freq_num_experts",
        type=int,
        default=-1,
        help="Number of experts in the frequency stream. Must match the "
             "training value when --use_freq_experts is set; -1 (default) "
             "mirrors --moe_num_experts.",
    )
    parser.add_argument(
        "--freq_top_k",
        type=int,
        default=-1,
        help="Top-k for the frequency stream. Must match the training value; "
             "-1 (default) mirrors --moe_top_k.",
    )
    parser.add_argument(
        "--freq_moe_r",
        type=int,
        default=-1,
        help="LoRA rank for the frequency experts. Must match the training "
             "value; -1 (default) mirrors --moe_r.",
    )
    parser.add_argument(
        "--freq_lora_alpha",
        type=int,
        default=-1,
        help="LoRA alpha for the frequency experts. Must match the training "
             "value; -1 (default) mirrors --moe_lora_alpha.",
    )
    parser.add_argument(
        "--freq_expert_type",
        type=str,
        default="lora",
        choices=["lora", "convlora"],
        help="Expert architecture for the frequency stream. Must match the "
             "training value to load the checkpoint.",
    )
    parser.add_argument(
        "--freq_transform",
        type=str,
        default="haar",
        choices=["haar", "laplacian"],
        help="High-pass transform feeding the frequency stream. Must match "
             "the training value.",
    )
    parser.add_argument(
        "--freq_beta_frozen",
        action="store_true",
        help="Must match the training value (frozen fusion scalar ablation).",
    )
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--visual_path",
        type=str,
        default=None,
        help="Where to write visualizations. Defaults to --save_path so they "
        "land in the checkpoint folder.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--save_path", default="ckpt/baseline")
    parser.add_argument(
        "--tiled",
        action="store_true",
        help=(
            "Use the pre-tiled metadata (full-shot-tiled.jsonl) produced by "
            "preprocess_pvf1.py --mode tile. Disables letterbox padding since "
            "tiles are already square."
        ),
    )
    parser.add_argument(
        "--use_topk_pool",
        action="store_true",
        help=(
            "Replace per-image max(pixel_preds) with the mean of the top-k "
            "pixel scores when computing the image-level prediction. Off by "
            "default; the original pmax behaviour is preserved bit-identically."
        ),
    )
    parser.add_argument(
        "--topk_frac",
        type=float,
        default=0.01,
        help=(
            "Fraction of pixels to average for --use_topk_pool "
            "(k = round(topk_frac * H*W), clamped to >=1)."
        ),
    )
    parser.add_argument(
        "--eval_fp16",
        action="store_true",
        help=(
            "Store per-image anomaly score arrays as float16 in the CPU "
            "accumulator (halves accumulator memory). metrics_eval casts back "
            "to float32 before normalization + AUC. Off by default; fp32 "
            "behaviour reproduces bit-identically when off."
        ),
    )
    parser.add_argument(
        "--eval_pixel_subsample",
        type=int,
        default=0,
        help=(
            "If > 0, uniformly subsample this many (pixel_label, pixel_pred) "
            "pairs before computing pixel AUC / AP. Used to cap roc_auc_score "
            "memory on large medical splits. 0 = off (use all pixels, exact "
            "match to original). 50000000 (50M) gives <0.1pp metric drift."
        ),
    )
    parser.add_argument(
        "--eval_subsample_seed",
        type=int,
        default=42,
        help=(
            "Random seed for --eval_pixel_subsample so the sampled metric is "
            "reproducible across reruns of the same configuration."
        ),
    )
    # unlabeled inference
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help=(
            "Run on an unlabeled JSONL (label=0, no mask_path). Skips metric "
            "computation; instead writes per-image scores, normal/defected "
            "counts, a score histogram and a few sample visualizations."
        ),
    )
    parser.add_argument(
        "--unlabeled_meta",
        type=str,
        default="unlabeled.jsonl",
        help=(
            "Filename under dataset/metadata/<dataset>/ or absolute path to "
            "the unlabeled JSONL produced by make_unlabeled_metadata.py."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Image-score threshold (after min-max norm); above => defected.",
    )
    parser.add_argument(
        "--num_viz",
        type=int,
        default=8,
        help="Top-K and bottom-K scoring images to visualize for sanity check.",
    )
    args = parser.parse_args()
    if args.visual_path is None:
        args.visual_path = args.save_path
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_path, "test.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    # parse moe_layers string to list[int]
    try:
        moe_layers = [int(x.strip()) for x in args.moe_layers.split(',') if x.strip() != ""]
    except Exception:
        moe_layers = [5, 11, 17, 23]

    model = SFMoECLIP(
        clip_model=clip_model,
        use_paa=args.no_use_paa,
        seg_proj_sharing_strategy=args.seg_proj_sharing_strategy,
        image_adapt_weight=args.image_adapt_weight,
        moe_r=args.moe_r,
        moe_lora_alpha=args.moe_lora_alpha,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_layers=moe_layers,
        use_fofs=args.no_use_fofs,
        fofs_mode=args.fofs_mode,
        fofs_basis_path=args.fofs_basis_path,
        relu=args.relu,
        use_wavelet_gate=args.use_wavelet_gate,
        wavelet_levels=args.wavelet_levels,
        use_alpha_gate=args.use_alpha_gate,
        alpha_hidden_dim=args.alpha_hidden_dim,
        use_freq_experts=args.use_freq_experts,
        freq_levels=args.freq_levels,
        freq_beta_init=args.freq_beta_init,
        freq_num_experts=(args.freq_num_experts if args.freq_num_experts > 0 else None),
        freq_top_k=(args.freq_top_k if args.freq_top_k > 0 else None),
        freq_moe_r=(args.freq_moe_r if args.freq_moe_r > 0 else None),
        freq_lora_alpha=(args.freq_lora_alpha if args.freq_lora_alpha > 0 else None),
        freq_expert_type=args.freq_expert_type,
        freq_transform=args.freq_transform,
        freq_beta_frozen=args.freq_beta_frozen,
    ).to(device)
    model.eval()
    
    #load checkpoints if exists
    files = sorted(
        glob(args.save_path + "/moe_epoch_*.pth"),
        key=extract_epoch_number
    )
    
    # Load epochs 20-25 if available.
    # files = [f for f in files if 20 <= extract_epoch_number(f) <= 20]
    
    # # Load last checkpoint
    # files = glob(args.save_path + "/moe_last*.pth")

    logger.info("-----------------------------------------------")
    logger.info(f"dataset: {args.dataset}")
    logger.info("-----------------------------------------------")
    for checkpoint_path in files:
        logger.info(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "text_adapter" in checkpoint: 
            model.text_adapter.load_state_dict(checkpoint["text_adapter"])
            adapt_text = True
        else:
            adapt_text = False

        if "image_adapter" in checkpoint:
            model.image_adapter.load_state_dict(checkpoint["image_adapter"])

        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info(f"Successfully loaded model from epoch {test_epoch} from {checkpoint_path}")
        logger.info("-----------------------------------------------")

        if args.log_freq_beta and args.use_freq_experts:
            betas = model.get_freq_betas()
            fmt = ", ".join(f"L{l}={v:.4f}" if v is not None else f"L{l}=off"
                            for l, v in betas.items())
            logger.info(f"freq_beta per MoE layer: {fmt}")
        
        # load dataset
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
            tiled=args.tiled,
        )
        unlabeled_datasets = None
        if args.unlabeled:
            unlabeled_datasets = get_dataset(
                args.dataset,
                args.img_size,
                None,
                args.shot,
                "test",
                logger=logger,
                tiled=args.tiled,
                meta_file=args.unlabeled_meta,
            )
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        unlabeled_df = DataFrame(
            columns=[
                "class name", "total", "normal", "defected", "% defected",
                "threshold", "F1@thr (labeled)",
            ]
        )

        combined_results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "path": []}
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # ========================================================
            # testing
            collect_alpha = args.log_alpha and args.use_alpha_gate
            if collect_alpha:
                model.start_alpha_collection()

            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names, results = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                    eval_fp16=args.eval_fp16,
                )

            if collect_alpha:
                model.stop_alpha_collection()
                alphas_per_layer = model.pop_alpha_collection()
                save_alpha_logs(
                    alphas_per_layer=alphas_per_layer,
                    file_names=file_names,
                    dataset=args.dataset,
                    class_name=class_name,
                    save_path=args.save_path,
                    logger=logger,
                )
            # ========================================================
            if args.unlabeled:
                domain = DOMAINS[args.dataset]
                labeled_scores = compute_image_scores(preds, preds_image, domain)
                threshold, f1 = derive_image_threshold(labels, labeled_scores)
                logger.info(
                    "[unlabeled %s] derived F1-optimal threshold=%.4f (F1=%.4f) "
                    "from %d labeled samples (%d defected)",
                    class_name, threshold, f1,
                    int(labels.shape[0]), int(np.sum(labels)),
                )

                u_loader = torch.utils.data.DataLoader(
                    unlabeled_datasets[class_name], batch_size=args.batch_size,
                    shuffle=False, **kwargs,
                )
                with torch.no_grad():
                    _, _, u_preds, u_preds_image, u_file_names, u_results = get_predictions(
                        model=model,
                        class_text_embeddings=class_text_embeddings,
                        test_loader=u_loader,
                        device=device,
                        img_size=args.img_size,
                        dataset=args.dataset,
                        eval_fp16=args.eval_fp16,
                    )
                u_scores = compute_image_scores(u_preds, u_preds_image, domain)

                row = unlabeled_summary(
                    scores=u_scores,
                    threshold=threshold,
                    file_names=u_file_names,
                    results=u_results,
                    class_name=class_name,
                    args=args,
                    logger=logger,
                )
                row["threshold"] = round(threshold, 4)
                row["F1@thr (labeled)"] = round(f1, 4) if f1 == f1 else "n/a"
                unlabeled_df.loc[len(unlabeled_df)] = Series(row)
                continue
            # ========================================================
            variant_indices = group_indices_by_variant(file_names)
            if args.visualize:
                if args.dataset == "DTD-Synthetic" and len(variant_indices) > 1:
                    for variant_name, idx_list in variant_indices.items():
                        v_results = {
                            "cls_names": [results["cls_names"][i] for i in idx_list],
                            "imgs_masks": [results["imgs_masks"][i] for i in idx_list],
                            "anomaly_maps": [results["anomaly_maps"][i] for i in idx_list],
                            "path": [results["path"][i] for i in idx_list],
                        }
                        calcuate_metric_pixel(v_results, [variant_name], args)
                else:
                    calcuate_metric_pixel(results, [class_name], args)


            if args.dataset == "DTD-Synthetic" and len(variant_indices) > 1:
                for variant_name, idx_list in variant_indices.items():
                    idx_arr = np.array(idx_list)
                    v_masks = masks[idx_arr]
                    v_labels = labels[idx_arr]
                    v_preds = preds[idx_arr]
                    v_preds_image = preds_image[idx_arr]
                    v_result = metrics_eval(
                        v_masks,
                        v_labels,
                        v_preds,
                        v_preds_image,
                        variant_name,
                        domain=DOMAINS[args.dataset],
                        use_topk_pool=args.use_topk_pool,
                        topk_frac=args.topk_frac,
                        eval_pixel_subsample=args.eval_pixel_subsample,
                        eval_subsample_seed=args.eval_subsample_seed,
                    )
                    df.loc[len(df)] = Series(v_result)
            else:
                class_result_dict = metrics_eval(
                    masks,
                    labels,
                    preds,
                    preds_image,
                    class_name,
                    domain=DOMAINS[args.dataset],
                    use_topk_pool=args.use_topk_pool,
                    topk_frac=args.topk_frac,
                    eval_pixel_subsample=args.eval_pixel_subsample,
                    eval_subsample_seed=args.eval_subsample_seed,
                )
                df.loc[len(df)] = Series(class_result_dict)
        if args.unlabeled:
            if len(unlabeled_df) > 1:
                totals = unlabeled_df[["total", "normal", "defected"]].sum()
                pct = round(100.0 * totals["defected"] / max(int(totals["total"]), 1), 2)
                unlabeled_df.loc[len(unlabeled_df)] = Series({
                    "class name": "Total",
                    "total": int(totals["total"]),
                    "normal": int(totals["normal"]),
                    "defected": int(totals["defected"]),
                    "% defected": pct,
                    "threshold": "",
                    "F1@thr (labeled)": "",
                })
            logger.info(
                "unlabeled results:\n%s",
                unlabeled_df.to_string(index=False, justify="center"),
            )
            continue
        numeric_cols = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
        df.loc[len(df)] = df[numeric_cols].mean()
        df.loc[len(df) - 1, "class name"] = "Average"
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()
    