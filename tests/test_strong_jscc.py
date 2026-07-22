from __future__ import annotations

import torch

from cadsd_jscc.strong_jscc import StrongJSCC, trainable_parameter_count


def tiny_model() -> StrongJSCC:
    return StrongJSCC(
        image_size=32,
        latent_channels=2,
        stage_channels=(8, 16, 24, 32),
        stage_blocks=(1, 1, 1, 1),
        condition_dim=16,
    )


def test_default_model_is_native_exact_rate_and_strong_capacity() -> None:
    model = StrongJSCC()
    assert model.real_symbols == 19712
    parameters = trainable_parameter_count(model)
    assert 25_000_000 <= parameters <= 45_000_000


def test_forward_observation_has_exact_power_and_gradients() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    image = torch.rand(2, 3, 32, 32)
    snr = torch.tensor([1.0, 13.0])
    standard_normal = torch.zeros(2, 2, 2, 2)
    output, observation = model.forward_with_observation(image, snr, standard_normal)
    assert output.shape == image.shape
    assert observation.transmitted[0].numel() == model.real_symbols == 8
    assert torch.allclose(
        observation.normalized_power, torch.ones(2), atol=1e-6, rtol=0.0
    )
    loss = (output - image).square().mean()
    loss.backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient_sum > 0.0


def test_awgn_uses_project_half_variance_convention() -> None:
    model = tiny_model()
    transmitted = torch.ones(1, 2, 2, 2)
    standard_normal = torch.ones_like(transmitted)
    received = model.transmit(transmitted, 0.0, standard_normal)
    expected = transmitted + 2.0**-0.5
    assert torch.allclose(received, expected, atol=1e-6, rtol=0.0)


def test_snr_and_shape_contracts_fail_closed() -> None:
    model = tiny_model()
    bad_image = torch.rand(1, 3, 31, 32)
    try:
        model(bad_image, 7.0)
    except ValueError as error:
        assert "expected 32x32" in str(error)
    else:
        raise AssertionError("wrong image shape was accepted")

    image = torch.rand(2, 3, 32, 32)
    try:
        model(image, torch.tensor([1.0, 4.0, 7.0]))
    except ValueError as error:
        assert "one SNR per image" in str(error)
    else:
        raise AssertionError("wrong SNR batch was accepted")
