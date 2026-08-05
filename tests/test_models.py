from __future__ import annotations

import torch
from torch import nn

from vision_action_tokenizer.losses import LossConfig, TokenizerLoss
from vision_action_tokenizer.models.decoder import TrajectoryDecoder
from vision_action_tokenizer.models.pe import PEFeatureExtractor
from vision_action_tokenizer.models.tokenizer import VisionActionTokenizer


class FakePE(nn.Module):
    def forward_features(self, images, **kwargs):
        del kwargs
        return images.new_ones((images.shape[0], 16, 8))


def test_pe_wrapper_preserves_frame_dimension() -> None:
    extractor = PEFeatureExtractor(model=FakePE(), pool_size=2)
    output = extractor(torch.randn(2, 6, 3, 16, 16))
    assert output.shape == (2, 6, 4, 8)


def test_context_free_decoders() -> None:
    latent = torch.randn(2, 2, 8)
    times = torch.arange(1, 13).float().unsqueeze(0).repeat(2, 1) / 12
    for decoder_type in ("direct", "kinematic"):
        decoder = TrajectoryDecoder(
            latent_dim=8,
            model_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0,
            decoder_type=decoder_type,
        )
        trajectory = decoder(latent, times)
        assert trajectory.shape == (2, 12, 3)
        assert torch.isfinite(trajectory).all()


def test_full_tokenizer_loss_backward() -> None:
    tokenizer = VisionActionTokenizer(
        pe_feature_dim=16,
        model_dim=32,
        latent_dim=8,
        num_action_tokens=2,
        resampled_tokens_per_frame=4,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0,
        decoder_type="direct",
    )
    visual = torch.randn(3, 6, 4, 16)
    trajectory = torch.randn(3, 12, 3)
    frame_times = torch.tensor([[0, 1, 2, 3, 4, 5]]).repeat(3, 1).float()
    future_times = torch.arange(1, 13).float().unsqueeze(0).repeat(3, 1) / 12
    mask = torch.ones(3, 12, dtype=torch.bool)
    output = tokenizer(visual, trajectory, frame_times, future_times, mask)
    loss, terms = TokenizerLoss(LossConfig(kl_warmup_steps=1))(
        output, trajectory, future_times, mask, global_step=1
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss/alignment" in terms
    assert tokenizer.visual_encoder.to_mean.weight.grad is not None

