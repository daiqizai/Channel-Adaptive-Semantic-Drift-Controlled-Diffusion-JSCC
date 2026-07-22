import unittest

import torch
from torch import nn

from cadsd_jscc.b1_feature_injection import (
    FrozenB1FeatureInjection,
    envelope_tensor,
    trainable_parameter_count,
)


class TinyB1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.SiLU())
        self.body = nn.Identity()
        self.tail = nn.Conv2d(4, 3, 3, padding=1)

    def structural_conditions(self, image: torch.Tensor):
        return []

    def forward(self, b0, snr_norm, gate):
        snr = snr_norm[:, None, None, None].expand(-1, 1, b0.shape[2], b0.shape[3])
        residual = torch.tanh(self.tail(self.body(self.head(torch.cat([b0, snr], 1)))))
        return (b0 + gate[:, None, None, None] * residual).clamp(0, 1)


class B1FeatureInjectionTests(unittest.TestCase):
    def test_zero_projection_and_zero_difference_are_exact_b1(self) -> None:
        b1 = TinyB1()
        model = FrozenB1FeatureInjection(b1, feature_channels=4)
        b0 = torch.rand(2, 3, 12, 12)
        auxiliary = torch.rand_like(b0)
        snr = torch.tensor([0.05, 0.35])
        gate = torch.tensor([0.12, 0.08])
        expected = b1(b0, snr, gate)
        self.assertTrue(torch.equal(model(b0, auxiliary, snr, gate, torch.ones(2)), expected))
        with torch.no_grad():
            model.aux_projection.weight.normal_()
        self.assertTrue(torch.equal(model(b0, b0, snr, gate, torch.ones(2)), expected))

    def test_zero_envelope_is_exact_b1_after_training(self) -> None:
        b1 = TinyB1()
        model = FrozenB1FeatureInjection(b1, feature_channels=4)
        with torch.no_grad():
            model.aux_projection.weight.normal_()
        b0 = torch.rand(1, 3, 12, 12)
        auxiliary = torch.rand_like(b0)
        snr = torch.tensor([0.95])
        gate = torch.tensor([0.04])
        self.assertTrue(
            torch.equal(model(b0, auxiliary, snr, gate, torch.zeros(1)), b1(b0, snr, gate))
        )

    def test_trainable_parameter_count_and_envelope(self) -> None:
        model = FrozenB1FeatureInjection(TinyB1(), feature_channels=4)
        self.assertEqual(trainable_parameter_count(model), 3 * 4 * 3 * 3)
        values = envelope_tensor(
            torch.tensor([1.0, 13.0]), {"1": 1.0, "13": 0.0}, torch.device("cpu")
        )
        self.assertTrue(torch.equal(values, torch.tensor([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
