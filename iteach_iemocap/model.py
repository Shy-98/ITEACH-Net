"""The fixed IEMOCAP4 ITS-NAS model used by this review copy."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

MODALITIES = ("text", "audio", "video")


class _SoftmaxKernel(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.softmax(dim=-1)


class _SharedTemporalConv1d(nn.Module):
    """One normalized 7-tap temporal kernel shared over hidden channels."""

    def __init__(self):
        super().__init__()
        # Keep the initialization stream identical to the dense-Conv construction
        # used by the reference model before this operator is installed.
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
        parametrize.register_parametrization(self.conv, "weight", _SoftmaxKernel())
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = x.shape
        y = self.conv(x.reshape(batch * channels, 1, steps))
        return y.reshape(batch, channels, steps)


def split_modalities(x: torch.Tensor, audio_dim: int, text_dim: int) -> dict[str, torch.Tensor]:
    return {
        "audio": x[..., :audio_dim],
        "text": x[..., audio_dim:audio_dim + text_dim],
        "video": x[..., audio_dim + text_dim:],
    }


def _attention_mask(valid: torch.Tensor, heads: int, dtype: torch.dtype,
                    ecce: torch.Tensor | None = None) -> torch.Tensor:
    batch, steps = valid.shape
    mask = torch.zeros((batch, steps, steps), dtype=dtype, device=valid.device)
    mask.masked_fill_(~valid[:, None, :], float("-inf"))
    if ecce is not None:
        mask = mask + ecce.to(dtype=dtype)
    return mask.repeat_interleave(heads, dim=0)


class EmotionContextChangingEncoder(nn.Module):
    """Centered local convolution, inclusive interval mean, scalar logit bias."""

    def __init__(self, hidden: int):
        super().__init__()
        self.local_convolution = True
        self.local_window = "centered"
        self.local_dropout_rate = 0.5
        self.local_alpha = 1.0
        self.local_norm_type = "none"
        self.local_norm_affine = False
        self.local_norm_eps = 1e-5
        self.local_padding_left = 3
        self.local_padding_right = 3
        # This dense biased module is deliberately created first to preserve the
        # source model's parameter initialization order and RNG consumption.
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.local_norm = nn.Identity()
        self.max_distance = 30
        self.to_scalar = nn.Linear(hidden, 1)
        self.local_dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z = self.local_norm(z)
        z = self.local_dropout(z)
        positions = torch.arange(x.shape[1], device=x.device)
        left = torch.minimum(positions[:, None], positions[None, :])
        right = torch.maximum(positions[:, None], positions[None, :])
        prefix = F.pad(z.cumsum(dim=1), (0, 0, 1, 0))
        means = prefix[:, right + 1, :] - prefix[:, left, :]
        means = means / (right - left + 1).to(dtype=x.dtype)[None, :, :, None]
        means = means * ((right - left) <= self.max_distance)[None, :, :, None]
        return self.to_scalar(means).squeeze(-1)


class SequenceMLPMixer(nn.Module):
    """Two-layer MLP acting on the utterance/time axis."""

    def __init__(self, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens
        self.mlp = nn.Sequential(
            nn.Linear(max_tokens, max_tokens), nn.GELU(),
            nn.Linear(max_tokens, max_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        if length > self.max_tokens:
            raise ValueError(f"conversation length {length} exceeds max_tokens={self.max_tokens}")
        y = F.pad(x.transpose(1, 2), (0, self.max_tokens - length))
        return self.mlp(y)[..., :length].transpose(1, 2)


class PoolMixer(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.kernel_size = 3
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        pad = self.kernel_size // 2
        if self.mode == "avg":
            y = F.avg_pool1d(y, 3, stride=1, padding=pad, count_include_pad=False)
        else:
            y = F.max_pool1d(y, 3, stride=1, padding=pad)
        return y.transpose(1, 2)


class TokenAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 8, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, ecce: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        mask = _attention_mask(valid, 8, x.dtype, ecce)
        return self.attention(x, x, x, attn_mask=mask, need_weights=False)[0]


class RouterMixer(nn.Module):
    def __init__(self, max_tokens: int, dropout: float):
        super().__init__()
        self.router = nn.Linear(128, 4)
        self.operations = nn.ModuleList([
            TokenAttention(dropout), SequenceMLPMixer(max_tokens),
            PoolMixer("avg"), PoolMixer("max"),
        ])

    def forward(self, x: torch.Tensor, ecce: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        weights = self.router(x).softmax(dim=-1)
        outputs = [
            self.operations[0](x, ecce, valid),
            self.operations[1](x),
            self.operations[2](x),
            self.operations[3](x),
        ]
        return (weights.unsqueeze(-1) * torch.stack(outputs, dim=2)).sum(dim=2)


class FeedForward(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, max_tokens: int, dropout: float, routed: bool):
        super().__init__()
        self.norm_mixer = nn.LayerNorm(128)
        self.norm_ffn = nn.LayerNorm(128)
        self.mixer = RouterMixer(max_tokens, dropout) if routed else TokenAttention(dropout)
        self.ffn = FeedForward(dropout)

    def forward(self, x: torch.Tensor, ecce: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm_mixer(x), ecce, valid)
        return x + self.ffn(self.norm_ffn(x))


class CrossModalFusion(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.norm_t = nn.LayerNorm(128)
        self.norm_a = nn.LayerNorm(128)
        self.norm_v = nn.LayerNorm(128)
        self.cross_audio = nn.MultiheadAttention(128, 8, dropout=dropout, batch_first=True)
        self.cross_video = nn.MultiheadAttention(128, 8, dropout=dropout, batch_first=True)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        query = self.norm_t(text)
        a, v = self.norm_a(audio), self.norm_v(video)
        mask_a = _attention_mask(valid, 8, query.dtype)
        mask_v = _attention_mask(valid, 8, query.dtype)
        add_a = self.cross_audio(query, a, a, attn_mask=mask_a, need_weights=False)[0]
        add_v = self.cross_video(query, v, v, attn_mask=mask_v, need_weights=False)[0]
        return text + add_a + add_v


class ITEACHModel(nn.Module):
    """One fixed Teacher or Student encoder with source-compatible state keys."""

    def __init__(self, input_dims: dict[str, int], max_tokens: int,
                 layers: int, dropout: float, routed: bool):
        super().__init__()
        self.hidden_size = 128
        self.max_tokens = max_tokens
        self.routed = routed
        self.input_projection = nn.ModuleDict({
            name: nn.Linear(input_dims[name], 128) for name in MODALITIES
        })
        self.ecce = nn.ModuleDict({name: EmotionContextChangingEncoder(128)
                                   for name in MODALITIES})
        self.ecce_input_dropout = nn.Dropout(0.5)
        self.ecce_input_activation = nn.Identity()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                name: EncoderBlock(max_tokens, dropout, routed)
                for name in MODALITIES
            }) for _ in range(layers)
        ])
        self.fusions = nn.ModuleList([CrossModalFusion(dropout) for _ in range(layers)])
        self.classifier = nn.Linear(128, 4)

        # Keep the reference initialization sequence: construct every dense ECCE
        # Conv and classifier, remove only ECCE Conv biases, then install the
        # parametrized shared 7-tap operator without advancing the parent RNG.
        for modality in MODALITIES:
            self.ecce[modality].conv.bias = None
        for modality in MODALITIES:
            self.ecce[modality].conv = _SharedTemporalConv1d()

    def forward(self, features: dict[str, torch.Tensor],
                valid_mask: torch.Tensor) -> dict[str, Any]:
        if valid_mask.ndim != 2 or features["text"].shape[:2] != valid_mask.shape:
            raise ValueError("features must be [batch,time,dim] with matching valid_mask")
        valid_mask = valid_mask.to(device=features["text"].device, dtype=torch.bool)
        x = {name: self.input_projection[name](features[name]) for name in MODALITIES}
        ecce_x = {name: self.ecce_input_activation(self.ecce_input_dropout(x[name]))
                  for name in MODALITIES}
        biases = {name: self.ecce[name](ecce_x[name]) for name in MODALITIES}
        text, audio, video = x["text"], x["audio"], x["video"]
        hidden_text = []
        for layer, fusion in zip(self.blocks, self.fusions):
            text = layer["text"](text, biases["text"], valid_mask)
            audio = layer["audio"](audio, biases["audio"], valid_mask)
            video = layer["video"](video, biases["video"], valid_mask)
            text = fusion(text, audio, video, valid_mask)
            hidden_text.append(text)
        return {"logits": self.classifier(text), "hidden": hidden_text}

    def router_parameters(self):
        if not self.routed:
            return iter(())
        params = []
        for layer in self.blocks:
            for name in MODALITIES:
                mixer = layer[name].mixer
                if isinstance(mixer, RouterMixer):
                    params.extend(mixer.router.parameters())
        return iter(params)


class ITEACHPair(nn.Module):
    """Three-layer complete-input Teacher plus four-layer masked NAS Student."""

    def __init__(self, audio_dim: int, text_dim: int, video_dim: int,
                 max_tokens: int, teacher_dropout: float = 0.5,
                 student_dropout: float = 0.0):
        super().__init__()
        dims = {"audio": audio_dim, "text": text_dim, "video": video_dim}
        self.teacher = ITEACHModel(dims, max_tokens, 3, teacher_dropout, routed=False)
        self.student = ITEACHModel(dims, max_tokens, 4, student_dropout, routed=True)
        self.architecture = {
            "framework": "its-nas", "teacher_layers": 3, "student_layers": 4,
            "teacher_routed": False, "student_routed": True,
            "hidden_size": 128, "heads": 8, "ffn_ratio": 4,
            "max_tokens": int(max_tokens), "teacher_dropout": teacher_dropout,
            "student_dropout": student_dropout,
            "ecce": "per-modality projected-128 shared-softmax 7-tap Conv1d; centered pad3; inclusive intervals; distance<=30",
            "ecce_input_dropout": 0.5, "ecce_local_dropout": 0.5,
            "ecce_conv_bias": False, "ecce_local_norm": "none",
            "ecce_input_activation": "identity", "ecce_local_alpha": 1.0,
            "ecce_mask_padding": False,
            "padding_policy": "attention keys masked; padded positions otherwise processed",
            "parameter_values_teacher": sum(p.numel() for p in self.teacher.parameters()),
            "parameter_values_student": sum(p.numel() for p in self.student.parameters()),
            "parameter_values": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, complete: torch.Tensor, incomplete: torch.Tensor,
                utterance_mask: torch.Tensor, audio_dim: int, text_dim: int,
                video_dim: int, train: bool):
        time_steps, batch_size, feature_size = complete.shape
        if feature_size != audio_dim + text_dim + video_dim or incomplete.shape != complete.shape:
            raise ValueError("complete/incomplete tensors must have [T,B,A+T+V] shape")
        if utterance_mask.shape != (batch_size, time_steps):
            raise ValueError("utterance_mask must be [B,T]")
        valid = utterance_mask.to(device=complete.device, dtype=torch.bool)
        full_b = complete.transpose(0, 1).contiguous()
        incomplete_b = incomplete.transpose(0, 1).contiguous()
        teacher_features = split_modalities(full_b, audio_dim, text_dim)
        student_features = split_modalities(incomplete_b, audio_dim, text_dim)
        with torch.set_grad_enabled(train):
            teacher_out = self.teacher(teacher_features, valid)
            student_out = self.student(student_features, valid)
        return {
            "teacher_logits": teacher_out["logits"].transpose(0, 1).contiguous(),
            "student_logits": student_out["logits"].transpose(0, 1).contiguous(),
            "teacher_hidden": [h.transpose(0, 1).contiguous() for h in teacher_out["hidden"]],
            "student_hidden": [h.transpose(0, 1).contiguous() for h in student_out["hidden"]],
        }

    def router_parameters(self):
        return self.student.router_parameters()
