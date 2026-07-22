"""Project-side adapter for the pinned official SwinJSCC SA architecture.

The third-party source is kept unmodified.  This adapter preserves its Swin
blocks and Channel ModNet topology while making the project comparison
contract explicit: per-image SNR, per-image power normalization, paired-real
AWGN, and optional externally supplied standard-normal noise.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SOURCE = ROOT / "third_party" / "SwinJSCC"


def _load_official_factories():
    if not OFFICIAL_SOURCE.is_dir():
        raise FileNotFoundError(
            f"Pinned SwinJSCC source is missing: {OFFICIAL_SOURCE}. "
            "See third_party/README.md for the frozen upstream commit."
        )
    source = str(OFFICIAL_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    from net.decoder import create_decoder  # type: ignore[import-not-found]
    from net.encoder import create_encoder  # type: ignore[import-not-found]

    return create_encoder, create_decoder


@dataclass(frozen=True)
class SwinJSCCObservation:
    latent: torch.Tensor
    transmitted: torch.Tensor
    received: torch.Tensor
    normalized_power: torch.Tensor
    snr_db: torch.Tensor


class OfficialSwinJSCCSA(nn.Module):
    """Official-source fixed-rate SwinJSCC with vectorized SNR adaptation."""

    def __init__(
        self,
        *,
        image_size: int = 256,
        latent_channels: int = 64,
        encoder_depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        decoder_depths: tuple[int, int, int, int] = (2, 6, 2, 2),
    ) -> None:
        super().__init__()
        if image_size != 256:
            raise ValueError("the S34A comparison is frozen to 256x256 inputs")
        if len(encoder_depths) != 4 or len(decoder_depths) != 4:
            raise ValueError("S34A requires the official four-stage HR topology")

        create_encoder, create_decoder = _load_official_factories()
        model_name = "SwinJSCC_w/_SA"
        encoder_dims = [128, 192, 256, 320]
        decoder_dims = [320, 256, 192, 128]
        encoder_heads = [4, 6, 8, 10]
        decoder_heads = [10, 8, 6, 4]
        common = dict(
            model=model_name,
            img_size=(image_size, image_size),
            C=int(latent_channels),
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        self.encoder = create_encoder(
            patch_size=2,
            in_chans=3,
            embed_dims=encoder_dims,
            depths=list(encoder_depths),
            num_heads=encoder_heads,
            **common,
        )
        self.decoder = create_decoder(
            embed_dims=decoder_dims,
            depths=list(decoder_depths),
            num_heads=decoder_heads,
            **common,
        )
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.encoder_depths = tuple(int(value) for value in encoder_depths)
        self.decoder_depths = tuple(int(value) for value in decoder_depths)
        self.real_symbols = self.latent_channels * (self.image_size // 16) ** 2

    @staticmethod
    def _snr_tensor(
        snr_db: torch.Tensor | float, batch: int, device: torch.device
    ) -> torch.Tensor:
        value = torch.as_tensor(snr_db, device=device, dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(f"expected one SNR per image, got {value.numel()} for {batch}")
        return value

    @staticmethod
    def _channel_modulate(module: nn.Module, value: torch.Tensor, snr: torch.Tensor):
        """Vectorized equivalent of the official ``w/_SA`` Channel ModNet."""

        condition = snr.reshape(-1, 1)
        hidden: torch.Tensor | None = None
        for index in range(module.layer_num):
            if index == 0:
                hidden = module.sm_list[index](value.detach())
            else:
                assert hidden is not None
                hidden = module.sm_list[index](hidden)
            gain = module.bm_list[index](condition).unsqueeze(1)
            hidden = hidden * gain
        assert hidden is not None
        modulation = module.sigmoid(module.sm_list[-1](hidden))
        return value * modulation

    def encode(self, image: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
        if tuple(image.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(
                f"expected Bx3x{self.image_size}x{self.image_size}, got {tuple(image.shape)}"
            )
        snr = self._snr_tensor(snr_db, image.shape[0], image.device)
        value = self.encoder.patch_embed(image)
        for layer in self.encoder.layers:
            value = layer(value)
        value = self.encoder.norm(value)
        value = self._channel_modulate(self.encoder, value, snr)
        return self.encoder.head_list(value)

    @staticmethod
    def normalize_channel_input(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = latent.flatten(start_dim=1)
        power = flat.float().square().mean(dim=1, keepdim=True).clamp_min(1e-12)
        normalized = flat / torch.sqrt(power).to(flat.dtype)
        return normalized.reshape_as(latent), power.reshape(-1)

    def transmit(
        self,
        transmitted: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        snr = self._snr_tensor(snr_db, transmitted.shape[0], transmitted.device)
        if standard_normal is None:
            noise = torch.randn_like(transmitted)
        else:
            noise = standard_normal.to(device=transmitted.device, dtype=transmitted.dtype)
            if noise.shape != transmitted.shape:
                if noise.numel() != transmitted.numel():
                    raise ValueError("standard-normal tensor does not match channel tensor")
                noise = noise.reshape_as(transmitted)
        power = transmitted.float().flatten(start_dim=1).square().mean(dim=1)
        gamma = torch.pow(10.0, snr / 10.0)
        sigma = torch.sqrt(power / (2.0 * gamma)).to(transmitted.dtype)
        return transmitted + noise * sigma[:, None, None]

    def decode(self, received: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
        if received.ndim != 3:
            raise ValueError(f"expected Bx256xC latent tokens, got {tuple(received.shape)}")
        snr = self._snr_tensor(snr_db, received.shape[0], received.device)
        value = self.decoder.head_list(received)
        value = self._channel_modulate(self.decoder, value, snr)
        for layer in self.decoder.layers:
            value = layer(value)
        batch, tokens, channels = value.shape
        expected_tokens = self.image_size * self.image_size
        if tokens != expected_tokens or channels != 3:
            raise RuntimeError(f"unexpected decoder output tokens: {tuple(value.shape)}")
        return value.reshape(batch, self.image_size, self.image_size, 3).permute(0, 3, 1, 2)

    def forward_with_observation(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SwinJSCCObservation]:
        snr = self._snr_tensor(snr_db, image.shape[0], image.device)
        latent = self.encode(image, snr)
        transmitted, _ = self.normalize_channel_input(latent)
        received = self.transmit(transmitted, snr, standard_normal)
        reconstruction = self.decode(received, snr)
        normalized_power = transmitted.float().flatten(start_dim=1).square().mean(dim=1)
        return reconstruction, SwinJSCCObservation(
            latent=latent,
            transmitted=transmitted,
            received=received,
            normalized_power=normalized_power,
            snr_db=snr,
        )

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_observation(image, snr_db, standard_normal)[0]


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
