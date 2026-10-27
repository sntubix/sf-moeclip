import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj, SimpleLoRA, ConvAdapterProj
from .config import LoraConfig, MixLoraConfig
from typing import Dict, Optional, Tuple, List
import math


_HAAR_KERNEL_SIGNS = torch.tensor(
    [
        [[1.0, 1.0], [1.0, 1.0]],    # LL (low-low)
        [[1.0, -1.0], [1.0, -1.0]],  # LH (vertical detail)
        [[1.0, 1.0], [-1.0, -1.0]],  # HL (horizontal detail)
        [[1.0, -1.0], [-1.0, 1.0]],  # HH (diagonal detail)
    ]
)


def _haar_kernels(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return (_HAAR_KERNEL_SIGNS.to(dtype=dtype, device=device) * 0.5).unsqueeze(1)


def haar_token_descriptor(hidden_states: torch.Tensor, levels: int) -> torch.Tensor:
    """Per-token dense Haar wavelet descriptor (LL/LH/HL/HH) for MoE routing.

    Input layout matches the rest of the MoE: (L, B, d) with CLS at index 0,
    L-1 a perfect square (the ViT patch grid). Returns (L, B, 4*levels); the
    CLS row is zero so its routing is unaffected. Each level is an undecimated
    (stride-1) 2x2 Haar at dilation 2**level — every token gets a multi-scale
    descriptor at the original patch-grid resolution.
    """
    L, B, _ = hidden_states.shape
    n_patch = L - 1
    H = int(round(math.sqrt(n_patch)))
    if H * H != n_patch:
        raise ValueError(
            f"haar_token_descriptor expects a square patch grid; got {n_patch} tokens."
        )

    energy = hidden_states[1:].norm(dim=-1)                         # (n_patch, B)
    energy = energy.reshape(H, H, B).permute(2, 0, 1).unsqueeze(1)  # (B, 1, H, H)

    kernels = _haar_kernels(energy.dtype, energy.device)            # (4, 1, 2, 2)

    levels_out = []
    for level in range(levels):
        dilation = 2 ** level
        out = F.conv2d(energy, kernels, stride=1, padding=0, dilation=dilation)
        out = F.pad(out, (0, dilation, 0, dilation), mode="replicate")  # (B, 4, H, H)
        levels_out.append(out)
    bands = torch.cat(levels_out, dim=1)                            # (B, 4*levels, H, H)

    bands = bands.permute(2, 3, 0, 1).reshape(n_patch, B, -1)       # (n_patch, B, 4*levels)
    cls_pad = bands.new_zeros(1, B, bands.shape[-1])
    return torch.cat([cls_pad, bands], dim=0)                       # (L, B, 4*levels)


def haar_token_features(hidden_states: torch.Tensor, levels: int) -> torch.Tensor:
    """Per-channel Haar high-pass of patch-token features for the freq stream.

    Same (L, B, d) layout as :func:`haar_token_descriptor`, but instead of
    returning a 4-d energy descriptor it band-passes the *full d-dim feature
    map* per channel. Summed LH+HL+HH across `levels` dyadic dilations; LL is
    deliberately excluded so the freq experts cannot see the low-frequency
    content the spatial stream already handles. CLS row is zeroed.
    """
    L, B, d = hidden_states.shape
    n_patch = L - 1
    H = int(round(math.sqrt(n_patch)))
    if H * H != n_patch:
        raise ValueError(
            f"haar_token_features expects a square patch grid; got {n_patch} tokens."
        )

    patch = hidden_states[1:].reshape(H, H, B, d).permute(2, 3, 0, 1)   # (B, d, H, H)

    kernels = _haar_kernels(patch.dtype, patch.device)                  # (4, 1, 2, 2)
    high_pass = kernels[1:].sum(dim=0, keepdim=True)                    # (1, 1, 2, 2)
    weight = high_pass.expand(d, 1, 2, 2).contiguous()                  # depthwise

    acc = torch.zeros_like(patch)
    for level in range(levels):
        dilation = 2 ** level
        out = F.conv2d(patch, weight, stride=1, padding=0, dilation=dilation, groups=d)
        out = F.pad(out, (0, dilation, 0, dilation), mode="replicate")  # back to H×H
        acc = acc + out

    acc = acc.permute(2, 3, 0, 1).reshape(n_patch, B, d)
    cls_pad = acc.new_zeros(1, B, d)
    return torch.cat([cls_pad, acc], dim=0)                             # (L, B, d)


def laplacian_token_features(hidden_states: torch.Tensor, levels: int) -> torch.Tensor:
    """Per-channel Laplacian high-pass of patch-token features (ablation).

    Drop-in alternative to :func:`haar_token_features` for the freq stream:
    a fixed zero-DC 3x3 Laplacian kernel applied depthwise at `levels` dyadic
    dilations. Zero kernel sum plays the role of the excluded LL band — the
    output carries no low-frequency content. CLS row is zeroed.
    """
    L, B, d = hidden_states.shape
    n_patch = L - 1
    H = int(round(math.sqrt(n_patch)))
    if H * H != n_patch:
        raise ValueError(
            f"laplacian_token_features expects a square patch grid; got {n_patch} tokens."
        )

    patch = hidden_states[1:].reshape(H, H, B, d).permute(2, 3, 0, 1)   # (B, d, H, H)

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=patch.dtype, device=patch.device,
    ) * 0.25
    weight = kernel.expand(d, 1, 3, 3).contiguous()

    acc = torch.zeros_like(patch)
    for level in range(levels):
        dilation = 2 ** level
        out = F.conv2d(
            patch, weight, stride=1, padding=dilation, dilation=dilation, groups=d,
        )
        acc = acc + out

    acc = acc.permute(2, 3, 0, 1).reshape(n_patch, B, d)
    cls_pad = acc.new_zeros(1, B, d)
    return torch.cat([cls_pad, acc], dim=0)                             # (L, B, d)


def etf_loss(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    num_tokens, num_experts, _ = expert_outputs.shape

    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_outputs.device)


    norm_outputs = F.normalize(expert_outputs, p=2, dim=-1, eps=eps)
    
    gram_matrix = torch.bmm(norm_outputs, norm_outputs.transpose(1, 2))

    target_val = -1.0 / (num_experts - 1)
    target_matrix = torch.full(
        (num_experts, num_experts), 
        target_val, 
        device=expert_outputs.device, 
        dtype=expert_outputs.dtype
    )

    target_matrix.fill_diagonal_(1.0)

    loss = F.mse_loss(gram_matrix, target_matrix.unsqueeze(0).expand_as(gram_matrix))
    
    return loss

class SimpleLoraExpert(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        config: LoraConfig,
        weight: Tuple[torch.Tensor, torch.Tensor] = (None, None),
        device: str = None,
    ):
        super().__init__()

        self.config_ = config
        self.initializer_ = config.lora_init_

        self.dtype_ = config.dtype_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_
        self.device_ = torch.device("cuda:0")
        
        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        if config.lora_dropout_ > 0.0:
            self.dropout_ = nn.Dropout(p=config.lora_dropout_)
        else:
            self.dropout_ = nn.Identity()

        self.lora_A = nn.Linear(in_features, self.r_, bias=False, dtype=self.dtype_, device=self.device_)
        self.lora_B = nn.Linear(self.r_, out_features, bias=False, dtype=self.dtype_, device=self.device_)
        
        self.use_dora_: bool = config.use_dora_
        self.magnitude_vector_: nn.Parameter = None
        self.reset_parameters(weight)

    def reset_parameters(
        self, weight: Tuple[torch.Tensor, torch.Tensor] = (None, None)
    ) -> None:
        assert isinstance(weight, Tuple)
        assert len(weight) == 2
        assert ((weight[0] is None) and (weight[1] is None)) or (
            isinstance(weight[0], torch.Tensor)
        )

        if weight[0] is None and weight[1] is None:
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_A.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_B.weight)
        else:
            with torch.no_grad():
                if weight[0] is not None:
                    self.lora_A.weight.copy_(weight[0])
                    self.lora_A.weight.requires_grad = False
                
                if weight[1] is not None:
                    self.lora_B.weight.copy_(weight[1])
                else:
                    nn.init.zeros_(self.lora_B.weight)
    
    def get_lora_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout_(hidden_states))) * self.scaling_

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lora_output = self.get_lora_output(hidden_states)
        return lora_output


class SimpleConvLoraExpert(nn.Module):
    """Spatial-aware LoRA expert for the frequency stream.

    Replaces the linear `lora_A` (d -> r) with a 3x3 Conv2d so the expert can
    see a small spatial neighborhood of the band-passed token grid; `lora_B`
    (r -> d) is unchanged. Forward signature takes the structured (L, B, d)
    tensor (CLS at index 0) rather than flat tokens — the spatial conv needs
    the patch grid. The CLS row in the output is zeroed (no neighborhood).

    FOFS A-init is not used here (it's defined for linear layers); the conv
    weight gets PyTorch's default Kaiming init and lora_B is zero-init so the
    expert starts contributing exactly zero, like the linear LoRA expert.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoraConfig,
        weight: Tuple[torch.Tensor, torch.Tensor] = (None, None),
        device: str = None,
    ):
        super().__init__()
        self.config_ = config
        self.dtype_ = config.dtype_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_
        self.device_ = torch.device("cuda:0")

        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        if config.lora_dropout_ > 0.0:
            self.dropout_ = nn.Dropout(p=config.lora_dropout_)
        else:
            self.dropout_ = nn.Identity()

        # Depthwise-broadcastable 3x3 conv that projects d -> r per token,
        # mixing each token with its 8 spatial neighbours. groups=1 (full
        # cross-channel mixing) keeps capacity comparable to the linear LoRA's
        # d x r mixing matrix.
        self.lora_A_conv = nn.Conv2d(
            in_features, self.r_, kernel_size=3, padding=1, bias=False,
            dtype=self.dtype_, device=self.device_,
        )
        self.lora_B = nn.Linear(
            self.r_, out_features, bias=False,
            dtype=self.dtype_, device=self.device_,
        )

        # weight[] is accepted only for interface symmetry with SimpleLoraExpert;
        # FOFS chunking is undefined for conv kernels and is ignored here.
        nn.init.kaiming_uniform_(self.lora_A_conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (L, B, d) with CLS at index 0. Returns (L, B, d)."""
        L, B, d = hidden_states.shape
        n_patch = L - 1
        H = int(round(math.sqrt(n_patch)))
        if H * H != n_patch:
            raise ValueError(
                f"SimpleConvLoraExpert expects a square patch grid; got {n_patch} tokens."
            )

        patch = hidden_states[1:].reshape(H, H, B, d).permute(2, 3, 0, 1)   # (B, d, H, H)
        patch = self.dropout_(patch)
        a = self.lora_A_conv(patch)                                         # (B, r, H, H)
        a = a.permute(2, 3, 0, 1).reshape(n_patch, B, self.r_)              # (n_patch, B, r)
        out = self.lora_B(a) * self.scaling_                                # (n_patch, B, d)

        cls_pad = out.new_zeros(1, B, d)
        return torch.cat([cls_pad, out], dim=0)                             # (L, B, d)


class BaseIndependentMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        config,
        use_fofs: bool = True,
        fofs_mode: str = "block",
        fofs_basis: Optional[torch.Tensor] = None,
        use_wavelet_gate: bool = False,
        wavelet_levels: int = 1,
        use_alpha_gate: bool = False,
        alpha_hidden_dim: int = 64,
        use_freq_experts: bool = False,
        freq_levels: int = 1,
        freq_beta_init: float = -4.0,
        freq_config: Optional["MixLoraConfig"] = None,
        freq_expert_type: str = "lora",
        freq_transform: str = "haar",
        freq_beta_frozen: bool = False,
    ):
        super().__init__()
        if freq_expert_type not in ("lora", "convlora"):
            raise ValueError(
                f"freq_expert_type must be 'lora' or 'convlora', got {freq_expert_type!r}"
            )
        if freq_transform not in ("haar", "laplacian"):
            raise ValueError(
                f"freq_transform must be 'haar' or 'laplacian', got {freq_transform!r}"
            )
        self.freq_transform = freq_transform
        self.freq_expert_type = freq_expert_type if use_freq_experts else "lora"
        self.config = config
        self.d_model = d_model
        self.fofs_mode = fofs_mode

        self.use_wavelet_gate = use_wavelet_gate
        self.wavelet_levels = wavelet_levels if use_wavelet_gate else 0
        self.wavelet_dim = 4 * self.wavelet_levels
        self.gate = nn.Linear(d_model + self.wavelet_dim, config.num_experts_, bias=False)

        # Conditional wavelet via a per-image scalar alpha(x) in [0, 1].
        # Only created when both the wavelet gate and alpha are enabled;
        # otherwise no extra parameters appear in the state_dict and the
        # forward path is identical to the wavelet-only / no-wavelet baseline.
        self.use_alpha_gate = use_alpha_gate and use_wavelet_gate
        if self.use_alpha_gate:
            self.alpha_fc1 = nn.Linear(d_model, alpha_hidden_dim, bias=True)
            self.alpha_fc2 = nn.Linear(alpha_hidden_dim, 1, bias=True)

        # Side channel for collecting per-image alpha values during eval.
        # Off by default; enable via start_alpha_collection()/stop_alpha_collection().
        # When off (default), the forward path does not touch the buffer.
        self._alpha_collect = False
        self._alpha_buffer: list = []

        # Parallel frequency-domain experts. Independent gate + expert pool that
        # consumes a per-channel Haar high-pass of the patch tokens. Outputs are
        # combined with the spatial stream via sigmoid(freq_beta), initialised
        # near zero so training starts from the spatial-only baseline. None of
        # these parameters are constructed when the flag is off, so the default
        # state_dict matches the original model bit-for-bit.
        self.use_freq_experts = use_freq_experts
        self.freq_levels = freq_levels if use_freq_experts else 0
        # Freq stream uses its own MixLoraConfig so num_experts / top_k / rank
        # / lora_alpha can differ from the spatial stream. When the caller
        # passes None we mirror the spatial config, which keeps the freq path
        # identical to the symmetric version added earlier.
        if self.use_freq_experts:
            self.freq_config = freq_config if freq_config is not None else config
            self.freq_gate = nn.Linear(d_model, self.freq_config.num_experts_, bias=False)
            # Frozen-beta ablation: a buffer keeps the same state_dict key but
            # never receives gradient, fixing the fusion weight at
            # sigmoid(freq_beta_init) for the whole run.
            if freq_beta_frozen:
                self.register_buffer("freq_beta", torch.tensor(float(freq_beta_init)))
            else:
                self.freq_beta = nn.Parameter(torch.tensor(float(freq_beta_init)))
        else:
            self.freq_config = None

        fixed_A_weights = None
        if use_fofs:
            if fofs_mode == "block":
                fixed_A_weights = self._create_fofs_A_matrices()
            elif fofs_mode == "pca":
                if fofs_basis is None:
                    raise ValueError(
                        "fofs_mode='pca' requires a precomputed basis. "
                        "Run compute_fofs_pca.py to generate one."
                    )
                fixed_A_weights = self._create_pca_fofs_A_matrices(fofs_basis)
            else:
                raise ValueError(
                    f"Unknown fofs_mode={fofs_mode!r}. Use 'block' or 'pca'."
                )

        self.experts = nn.ModuleList()
        for i in range(config.num_experts_):
            lora_A_weight = fixed_A_weights[i] if use_fofs else None
            expert = SimpleLoraExpert(d_model, d_model, config, weight=(lora_A_weight, None))
            self.experts.append(expert)

        # Freq experts get their own FOFS chunking based on the freq config's
        # (num_experts, lora_r). When the freq config mirrors the spatial one
        # the chunking is structurally identical to the spatial chunking (the
        # weights differ only by RNG advance, which is irrelevant here since
        # this branch only runs when use_freq_experts is on — and that
        # configuration already diverges from the baseline). Joint ETF over
        # the concatenated expert outputs handles asymmetric pool sizes
        # natively (cat along dim=1).
        #
        # ConvLoRA experts replace the linear lora_A with a 3x3 Conv2d so
        # they can mix neighbouring band-passed tokens. FOFS chunking is
        # undefined for conv kernels, so we skip the A-init prep for that
        # path — each ConvLoRA expert gets a fresh Kaiming-initialised conv
        # and a zero-initialised lora_B.
        if self.use_freq_experts:
            if self.freq_expert_type == "lora" and use_fofs:
                if fofs_mode == "block":
                    freq_fixed_A_weights = self._create_fofs_A_matrices(
                        num_experts=self.freq_config.num_experts_,
                        rank=self.freq_config.lora_r_,
                    )
                else:  # 'pca'
                    freq_fixed_A_weights = self._create_pca_fofs_A_matrices(
                        fofs_basis,
                        num_experts=self.freq_config.num_experts_,
                        rank=self.freq_config.lora_r_,
                    )
            else:
                freq_fixed_A_weights = None

            expert_cls = (
                SimpleConvLoraExpert if self.freq_expert_type == "convlora"
                else SimpleLoraExpert
            )
            self.freq_experts = nn.ModuleList()
            for i in range(self.freq_config.num_experts_):
                lora_A_weight = freq_fixed_A_weights[i] if freq_fixed_A_weights is not None else None
                expert = expert_cls(
                    d_model, d_model, self.freq_config, weight=(lora_A_weight, None),
                )
                self.freq_experts.append(expert)

        self.jitter_noise = getattr(config, "jitter_noise_", 0.0)
        self.init_custom_weights()
        
    def _create_fofs_A_matrices(self, num_experts: Optional[int] = None, rank: Optional[int] = None) -> list:
        num_experts = num_experts if num_experts is not None else self.config.num_experts_
        rank = rank if rank is not None else self.config.lora_r_
        in_features = self.d_model
        
        base_chunk_size = in_features // num_experts
        remainder = in_features % num_experts
        
        fixed_A_matrices = []
        current_start_idx = 0
        
        print(f"Creating fofs LoRA A matrices for d_model={in_features}, num_experts={num_experts}")

        for i in range(num_experts):
            chunk_size = base_chunk_size + 1 if i < remainder else base_chunk_size
            
            start_idx = current_start_idx
            end_idx = start_idx + chunk_size
            
            print(f"  - Expert {i+1}: features from {start_idx} to {end_idx-1} (size: {chunk_size})")

            temp_matrix = torch.randn(chunk_size, rank, device=self.gate.weight.device, dtype=self.config.dtype_)
            q, _ = torch.linalg.qr(temp_matrix)
            
            q_ortho = q.T

            A_matrix = torch.zeros(rank, in_features, device=self.gate.weight.device, dtype=self.config.dtype_)
            A_matrix[:, start_idx:end_idx] = q_ortho
            
            fixed_A_matrices.append(A_matrix)
            
            current_start_idx = end_idx

        return fixed_A_matrices

    def _create_pca_fofs_A_matrices(
        self, basis: torch.Tensor,
        num_experts: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> list:
        num_experts = num_experts if num_experts is not None else self.config.num_experts_
        rank = rank if rank is not None else self.config.lora_r_
        in_features = self.d_model

        if basis.shape != (in_features, in_features):
            raise ValueError(
                f"FOFS basis shape mismatch: expected ({in_features}, {in_features}), "
                f"got {tuple(basis.shape)}"
            )

        base_chunk_size = in_features // num_experts
        remainder = in_features % num_experts

        device = self.gate.weight.device
        dtype = self.config.dtype_
        basis = basis.to(device=device, dtype=dtype)

        fixed_A_matrices = []
        current_start_idx = 0

        print(
            f"Creating PCA FOFS A matrices for d_model={in_features}, "
            f"num_experts={num_experts} (PCs sorted by descending variance)"
        )

        for i in range(num_experts):
            chunk_size = base_chunk_size + 1 if i < remainder else base_chunk_size
            start_idx = current_start_idx
            end_idx = start_idx + chunk_size

            print(f"  - Expert {i+1}: PCA components {start_idx} to {end_idx-1} (size: {chunk_size})")

            temp_matrix = torch.randn(chunk_size, rank, device=device, dtype=dtype)
            q, _ = torch.linalg.qr(temp_matrix)

            A_matrix = (basis[:, start_idx:end_idx] @ q).T.contiguous()

            fixed_A_matrices.append(A_matrix)
            current_start_idx = end_idx

        return fixed_A_matrices

    def init_custom_weights(self):
        nn.init.zeros_(self.gate.weight)
        if self.use_freq_experts:
            nn.init.zeros_(self.freq_gate.weight)
        if self.use_alpha_gate:
            # Initialise alpha(x) ≈ 1 at training start so the conditional
            # model begins from the existing wavelet-on behaviour. fc2.weight
            # is zeroed and fc2.bias is large positive: sigmoid(4.0) ≈ 0.982.
            # fc1 receives gradient through fc2 once fc2.weight becomes
            # non-zero, so the head still trains end-to-end.
            nn.init.kaiming_uniform_(self.alpha_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.alpha_fc1.bias)
            nn.init.zeros_(self.alpha_fc2.weight)
            nn.init.constant_(self.alpha_fc2.bias, 4.0)

    def start_alpha_collection(self) -> None:
        """Begin recording per-image alpha values produced by forward()."""
        self._alpha_collect = True
        self._alpha_buffer = []

    def stop_alpha_collection(self) -> None:
        self._alpha_collect = False

    def pop_alpha_buffer(self) -> Optional["np.ndarray"]:
        """Return the concatenated alpha values collected so far, or None."""
        import numpy as np
        if not self._alpha_buffer:
            return None
        out = np.concatenate(self._alpha_buffer)
        self._alpha_buffer = []
        return out

    def _vit_forward(self, expert_mask: torch.Tensor, hidden_states: torch.Tensor) -> Dict[int, torch.Tensor]:
        return self._vit_forward_with_experts(
            self.experts, expert_mask, hidden_states, self.config.num_experts_,
        )

    def _vit_forward_with_experts(
        self,
        experts: nn.ModuleList,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        num_experts: int,
    ) -> Dict[int, torch.Tensor]:
        expert_outputs_dict = {}
        for expert_idx in range(num_experts):
            _, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            current_hidden_states = hidden_states[top_x]
            expert_output = experts[expert_idx](current_hidden_states)
            expert_outputs_dict[expert_idx] = expert_output

        return expert_outputs_dict

    def _route_and_combine(
        self,
        gate: nn.Linear,
        experts: nn.ModuleList,
        gate_input_flat: torch.Tensor,
        expert_input_flat: torch.Tensor,
        input_dtype: torch.dtype,
        dense_expert_forward: bool,
        num_experts: int,
        top_k: int,
        expert_input_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Single-stream MoE routing + combine. Returns the same tuple shape as
        the original forward for either stream (spatial or freq). The caller
        supplies `num_experts` and `top_k` so the spatial and freq streams can
        run with independent values.

        When `expert_input_grid` is provided (shape `(L, B, d)`), each expert
        is called on the full structured tensor — needed for spatial-aware
        experts (ConvLoRA) that mix neighbouring tokens. In that case the
        dense path is used unconditionally because the sparse per-token
        subset path is incompatible with a spatial convolution kernel.
        """
        device = expert_input_flat.device
        hidden_dim = expert_input_flat.shape[1]
        use_structured = expert_input_grid is not None

        router_logits = gate(gate_input_flat)
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(router_probs, top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(input_dtype)

        load_balance_loss = torch.tensor(0.0, device=device)
        all_expert_outputs = None
        out_selected_experts = None

        use_dense = self.training or dense_expert_forward or use_structured

        if use_dense:
            aux = getattr(self.config, "router_aux_loss_coef", 0.0)
            if aux > 0:
                gate_sum = router_probs.sum(dim=0)
                mean, std = gate_sum.mean(), gate_sum.std()
                load_balance_loss = (std / (mean + 1e-6)).pow(2)

            if use_structured:
                # Each expert returns (L, B, d). Stack along a new expert axis
                # and reshape to (N, E, d) — row-major flatten of (L, B)
                # matches the gate's (l*B + b) token ordering.
                outputs = [expert(expert_input_grid) for expert in experts]
                stacked = torch.stack(outputs, dim=2)                       # (L, B, E, d)
                L_, B_, E_, D_ = stacked.shape
                all_expert_outputs = stacked.reshape(L_ * B_, E_, D_)
            else:
                all_expert_outputs = torch.stack(
                    [expert(expert_input_flat) for expert in experts],
                    dim=1,
                )

            token_indices = torch.arange(expert_input_flat.shape[0], device=device).unsqueeze(1)
            activated_outputs = all_expert_outputs[token_indices, selected_experts]
            weighted_outputs = activated_outputs * routing_weights.unsqueeze(-1)
            final_hidden_states = weighted_outputs.sum(dim=1)
            out_selected_experts = selected_experts
        else:
            expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
            expert_outputs_dict = self._vit_forward_with_experts(
                experts, expert_mask, expert_input_flat, num_experts,
            )
            per_token_expert_outputs = torch.zeros(
                (expert_input_flat.shape[0], top_k, hidden_dim),
                dtype=input_dtype, device=device,
            )
            for expert_idx, output_tensor in expert_outputs_dict.items():
                top_k_idx, top_x = torch.where(expert_mask[expert_idx])
                per_token_expert_outputs.index_put_((top_x, top_k_idx), output_tensor)
            weighted_outputs = per_token_expert_outputs * routing_weights.unsqueeze(-1)
            final_hidden_states = weighted_outputs.sum(dim=1)

        return (
            final_hidden_states,
            load_balance_loss,
            all_expert_outputs,
            out_selected_experts,
            router_probs,
            routing_weights,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        dense_expert_forward: bool = False,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.jitter_noise > 0 and self.training:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        input_dtype = hidden_states.dtype
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

        # ---- spatial stream gate input (unchanged) ----
        if self.use_wavelet_gate:
            descriptor = haar_token_descriptor(hidden_states, self.wavelet_levels)
            if self.use_alpha_gate:
                # Per-image scalar in [0, 1] derived from the CLS token at
                # this layer. Broadcasts (B, 1) -> (L, B, W) so the same
                # alpha scales every token's descriptor for that image.
                cls_feat = hidden_states[0]
                alpha = torch.sigmoid(
                    self.alpha_fc2(F.gelu(self.alpha_fc1(cls_feat)))
                )
                if self._alpha_collect:
                    self._alpha_buffer.append(
                        alpha.detach().to(torch.float32).cpu().numpy().reshape(-1)
                    )
                descriptor = descriptor * alpha.to(descriptor.dtype).unsqueeze(0)
            descriptor_flat = descriptor.reshape(-1, descriptor.shape[-1]).to(input_dtype)
            spatial_gate_input = torch.cat([hidden_states_flat, descriptor_flat], dim=-1)
        else:
            spatial_gate_input = hidden_states_flat

        # ---- spatial stream routing + combine ----
        (
            spatial_out_flat,
            load_balance_loss,
            spatial_expert_outputs,
            selected_experts,
            router_probs,
            routing_weights,
        ) = self._route_and_combine(
            self.gate, self.experts, spatial_gate_input, hidden_states_flat,
            input_dtype, dense_expert_forward,
            num_experts=self.config.num_experts_,
            top_k=self.config.top_k_,
        )

        final_hidden_states = spatial_out_flat
        all_expert_outputs = spatial_expert_outputs
        freq_debug = None

        # ---- frequency stream (parallel) ----
        self.last_pool_expert_outputs = None
        if self.use_freq_experts:
            transform_fn = (
                laplacian_token_features
                if self.freq_transform == "laplacian"
                else haar_token_features
            )
            freq_in = transform_fn(hidden_states, self.freq_levels)
            freq_in_flat = freq_in.reshape(-1, hidden_dim).to(input_dtype)
            # ConvLoRA experts need the (L, B, d) structure so a 3x3 conv can
            # see neighbouring tokens; LoRA experts are token-independent and
            # use the existing flat path (which can additionally take the
            # sparse eval shortcut).
            freq_grid = (
                freq_in.to(input_dtype)
                if self.freq_expert_type == "convlora"
                else None
            )

            (
                freq_out_flat,
                freq_lb_loss,
                freq_expert_outputs,
                freq_selected_experts,
                freq_router_probs,
                freq_routing_weights,
            ) = self._route_and_combine(
                self.freq_gate, self.freq_experts, freq_in_flat, freq_in_flat,
                input_dtype, dense_expert_forward,
                num_experts=self.freq_config.num_experts_,
                top_k=self.freq_config.top_k_,
                expert_input_grid=freq_grid,
            )

            # sigmoid(freq_beta) initialised near 0 → freq path contributes
            # nothing at step 0; training opens it only where it helps.
            beta = torch.sigmoid(self.freq_beta).to(input_dtype)
            final_hidden_states = spatial_out_flat + beta * freq_out_flat
            load_balance_loss = load_balance_loss + freq_lb_loss

            # Joint 8-expert ETF: stacking [spatial, freq] along the expert axis
            # makes the etf_loss in MoECLIP push the two pools into mutually
            # complementary directions. Only available in the dense path
            # (training / dense_expert_forward); in the sparse eval path both
            # streams return None and the loss is skipped, matching baseline.
            if spatial_expert_outputs is not None and freq_expert_outputs is not None:
                all_expert_outputs = torch.cat(
                    [spatial_expert_outputs, freq_expert_outputs], dim=1,
                )
                # Kept separately so the caller can compute per-pool ETF
                # (ablation of the joint regularization) without re-splitting.
                self.last_pool_expert_outputs = (
                    spatial_expert_outputs, freq_expert_outputs,
                )
            else:
                all_expert_outputs = None

            freq_debug = {
                "freq_router_probs": freq_router_probs,
                "freq_routing_weights": freq_routing_weights,
                "freq_selected_experts": freq_selected_experts,
                "freq_expert_outputs": freq_expert_outputs,
                "freq_beta": beta.detach(),
            }

        debug_payload = None
        if return_debug:
            if all_expert_outputs is not None:
                all_expert_outputs.retain_grad()
            debug_payload = {
                "router_probs": router_probs,
                "routing_weights": routing_weights,
                "selected_experts": selected_experts,
                "all_expert_outputs": all_expert_outputs,
            }
            if freq_debug is not None:
                debug_payload.update(freq_debug)
        self.last_debug = debug_payload

        return (
            final_hidden_states.reshape(batch_size, sequence_length, hidden_dim),
            load_balance_loss,
            all_expert_outputs,
            selected_experts,
        )

    def get_freq_beta(self) -> Optional[float]:
        """Return current sigmoid(freq_beta) as a python float, or None if
        the freq path is disabled. Cheap interpretability hook for logging."""
        if not self.use_freq_experts:
            return None
        return torch.sigmoid(self.freq_beta).detach().to(torch.float32).item()

class SFMoECLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        use_paa: bool = True,
        seg_proj_sharing_strategy: str = "shared",
        image_adapt_weight: float = 0.1,
        levels: list = [6, 12, 18, 24],
        moe_r: int = 8,
        moe_lora_alpha: int = 16,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        use_fofs: bool = True,
        fofs_mode: str = "block",
        fofs_basis_path: Optional[str] = None,
        moe_layers: Optional[List[int]] = None,
        relu: bool = True,
        use_wavelet_gate: bool = False,
        wavelet_levels: int = 1,
        use_alpha_gate: bool = False,
        alpha_hidden_dim: int = 64,
        use_freq_experts: bool = False,
        freq_levels: int = 1,
        freq_beta_init: float = -4.0,
        freq_num_experts: Optional[int] = None,
        freq_top_k: Optional[int] = None,
        freq_moe_r: Optional[int] = None,
        freq_lora_alpha: Optional[int] = None,
        freq_expert_type: str = "lora",
        freq_transform: str = "haar",
        freq_beta_frozen: bool = False,
        etf_per_pool: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.etf_per_pool = etf_per_pool
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.i_w = image_adapt_weight
        self.levels = levels
        self.moe_layers = moe_layers if moe_layers is not None else [5, 11, 17, 23]
        
        self.use_paa = use_paa
        
        self.seg_proj_sharing_strategy = seg_proj_sharing_strategy
        if self.use_paa:
            assert seg_proj_sharing_strategy in ["separate", "shared"], \
                "seg_proj_sharing_strategy must be 'separate' or 'shared' when using paa"
            
            if self.seg_proj_sharing_strategy == "separate":
                num_seg_projs = len(levels) * 3
            else:
                num_seg_projs = len(levels)
            
            self._create_gaussian_kernels()
        else:
            num_seg_projs = len(levels)
        
        d_model = 1024
        
        moe_config = MixLoraConfig.from_config(
            {
                "bias": "none",
                "peft_type": "MIXLORA",
                "r": moe_r,
                "lora_alpha": moe_lora_alpha,
                "lora_dropout": 0.05,
                "target_modules": ["c_fc", "c_proj"],
                "routing_strategy": "mixlora",
                "num_experts": moe_num_experts,
                "num_lora_experts": moe_num_experts,
                "top_k": moe_top_k,
                "act_fn": "silu",
                "base_model_name_or_path": "CLIP_VIT",
                "task_type": "VISION",
            }
        )

        # When any freq-specific override is provided, build a separate
        # MixLoraConfig for the freq stream. Otherwise reuse the spatial
        # config (so the freq path is symmetric and matches the previous
        # behaviour). Resolves None overrides to the spatial values.
        freq_overrides = (freq_num_experts, freq_top_k, freq_moe_r, freq_lora_alpha)
        if use_freq_experts and any(v is not None for v in freq_overrides):
            freq_config = MixLoraConfig.from_config(
                {
                    "bias": "none",
                    "peft_type": "MIXLORA",
                    "r": freq_moe_r if freq_moe_r is not None else moe_r,
                    "lora_alpha": (
                        freq_lora_alpha if freq_lora_alpha is not None else moe_lora_alpha
                    ),
                    "lora_dropout": 0.05,
                    "target_modules": ["c_fc", "c_proj"],
                    "routing_strategy": "mixlora",
                    "num_experts": (
                        freq_num_experts if freq_num_experts is not None else moe_num_experts
                    ),
                    "num_lora_experts": (
                        freq_num_experts if freq_num_experts is not None else moe_num_experts
                    ),
                    "top_k": freq_top_k if freq_top_k is not None else moe_top_k,
                    "act_fn": "silu",
                    "base_model_name_or_path": "CLIP_VIT",
                    "task_type": "VISION",
                }
            )
        else:
            freq_config = None


        fofs_bases = None
        if use_fofs and fofs_mode == "pca":
            if fofs_basis_path is None:
                raise ValueError(
                    "fofs_mode='pca' requires --fofs_basis_path. "
                    "Run compute_fofs_pca.py to generate a basis file."
                )
            fofs_bases = torch.load(fofs_basis_path, map_location="cpu")
            for layer_idx in self.moe_layers:
                if layer_idx not in fofs_bases:
                    raise KeyError(
                        f"FOFS basis missing for layer {layer_idx} in {fofs_basis_path}. "
                        f"Available layers: {sorted(fofs_bases.keys())}"
                    )

        moe_adapters = nn.ModuleList([
            BaseIndependentMoE(
                d_model=d_model,
                config=moe_config,
                use_fofs=use_fofs,
                fofs_mode=fofs_mode,
                fofs_basis=(fofs_bases[layer_idx] if fofs_bases is not None else None),
                use_wavelet_gate=use_wavelet_gate,
                wavelet_levels=wavelet_levels,
                use_alpha_gate=use_alpha_gate,
                alpha_hidden_dim=alpha_hidden_dim,
                use_freq_experts=use_freq_experts,
                freq_levels=freq_levels,
                freq_beta_init=freq_beta_init,
                freq_config=freq_config,
                freq_expert_type=freq_expert_type,
                freq_transform=freq_transform,
                freq_beta_frozen=freq_beta_frozen,
            )
            for layer_idx in self.moe_layers
        ])

        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(num_seg_projs)]
        )
        
        
        det_proj = ConvAdapterProj(1024, 768)
        self.image_adapter = nn.ModuleDict(
            {
                "seg_proj": seg_proj,
                "det_proj": det_proj,
                "moe_adapters": moe_adapters,
            }
        )
        self.text_adapter = nn.ModuleList(
            [SimpleProj(768, 768, relu=True)]
        )

    def start_alpha_collection(self) -> None:
        for moe in self.image_adapter["moe_adapters"]:
            moe.start_alpha_collection()

    def stop_alpha_collection(self) -> None:
        for moe in self.image_adapter["moe_adapters"]:
            moe.stop_alpha_collection()

    def pop_alpha_collection(self) -> Dict[int, "np.ndarray"]:
        """Return {layer_idx: alpha_values} for all MoE layers and reset buffers."""
        return {
            layer_idx: moe.pop_alpha_buffer()
            for layer_idx, moe in zip(self.moe_layers, self.image_adapter["moe_adapters"])
        }

    def get_freq_betas(self) -> Dict[int, Optional[float]]:
        """Return {layer_idx: sigmoid(freq_beta)} for each MoE layer. Values
        are None where the freq stream is disabled. Useful for tracking how
        much each layer ends up relying on the frequency path."""
        return {
            layer_idx: moe.get_freq_beta()
            for layer_idx, moe in zip(self.moe_layers, self.image_adapter["moe_adapters"])
        }

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float = 2.0) -> torch.Tensor:
        x_coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        y_coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _create_gaussian_kernels(self):
        kernel_3x3 = self._gaussian_kernel(3)
        kernel_5x5 = self._gaussian_kernel(5)
        self.register_buffer('gaussian_kernel_3', kernel_3x3)
        self.register_buffer('gaussian_kernel_5', kernel_5x5)

    def _aggregate_neighbor(self, x: torch.Tensor, r: int) -> torch.Tensor:
        if r == 1:
            return x
        
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        b, l, c = patch_tokens.shape
        h = w = int(math.sqrt(l))
        
        patch_tokens = patch_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        padding = r // 2
        unfolded_patches = F.unfold(patch_tokens, kernel_size=r, padding=padding, stride=1)
        
        unfolded_patches = unfolded_patches.permute(0, 2, 1)
        unfolded_patches = unfolded_patches.reshape(b * l, c, r * r).permute(0, 2, 1)


        aggregated_features = torch.mean(unfolded_patches, dim=1)
            
        aggregated_patches = aggregated_features.reshape(b, l, c)
        
        return torch.cat([cls_token, aggregated_patches], dim=1)

    def _aggregate_neighbors(self, tokens_from_layers: list) -> list:
        aggregated_token_list = []
        for token_map in tokens_from_layers:
            for r in [1, 3, 5]:
                permuted_token_map = token_map.permute(1, 0, 2)
                aggregated_token = self._aggregate_neighbor(permuted_token_map, r)
                aggregated_token_list.append(aggregated_token.permute(1, 0, 2))
        return aggregated_token_list

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x, return_moe_debug: bool = False, dense_moe: bool = False):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)
        
        tokens = []
        total_load_balance_loss = torch.tensor(0.0, device=x.device)
        total_etf_loss = torch.tensor(0.0, device=x.device)

        moe_debug_infos = []
        for i in range(24):
            x, attn, _ = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            
            if i in self.moe_layers:
                moe_idx = self.moe_layers.index(i)
                
                moe_output, moe_lb_loss, all_expert_outputs, selected_experts = \
                    self.image_adapter["moe_adapters"][moe_idx](
                        x,
                        dense_expert_forward=dense_moe,
                        return_debug=return_moe_debug,
                    )
                if return_moe_debug:
                    layer_debug = self.image_adapter["moe_adapters"][moe_idx].last_debug
                    if layer_debug is not None:
                        layer_debug = dict(layer_debug)
                        layer_debug["block_index"] = i
                        layer_debug["moe_index"] = moe_idx
                        moe_debug_infos.append(layer_debug)
                
                if self.training and all_expert_outputs is not None:
                    pool_outputs = self.image_adapter["moe_adapters"][moe_idx].last_pool_expert_outputs
                    if self.etf_per_pool and pool_outputs is not None:
                        # Ablation: regularize each pool independently instead
                        # of the joint equiangular set over both pools.
                        moe_etf_l = etf_loss(pool_outputs[0]) + etf_loss(pool_outputs[1])
                    else:
                        moe_etf_l = etf_loss(all_expert_outputs)
                    total_etf_loss += moe_etf_l
                moe_output_normalized = (
                    moe_output * x.norm(dim=-1, keepdim=True)
                    / (moe_output.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.i_w * moe_output_normalized + (1 - self.i_w) * x
                
                total_load_balance_loss += moe_lb_loss
                
            if i + 1 in self.levels:
                if self.use_paa:
                    tokens.append(x)
                else:
                    tokens.append(x)
                
        if self.use_paa:
            tokens = self._aggregate_neighbors(tokens)

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        tokens = [t[:, 1:, :] for t in tokens]
        if self.use_paa and self.seg_proj_sharing_strategy == "shared":
            seg_tokens = [
                self.image_adapter["seg_proj"][i // 3](t)
                for i, t in enumerate(tokens)
            ]
        else:
            seg_tokens = [
                self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
            ]

        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]

        det_token = self.image_adapter["det_proj"](tokens[-3])
        det_token = F.normalize(det_token, dim=-1).mean(1)

        total_aux_loss = total_load_balance_loss
        special_loss = total_etf_loss
        
        if return_moe_debug:
            return seg_tokens, det_token, total_aux_loss, special_loss, moe_debug_infos
        return seg_tokens, det_token, total_aux_loss, special_loss

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn, _ = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x
