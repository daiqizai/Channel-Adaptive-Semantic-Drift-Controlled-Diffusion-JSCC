import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.semantic_sketch import (
    embed_repeated_sketch,
    fixed_rademacher_projection,
    bits_to_integer_codes,
    integer_codes_to_bits,
    probabilities_to_simplex_sketch,
    probabilities_to_sketch,
    quantize_probabilities_uniform,
    recover_repeated_sketch_and_erase,
    reserved_symbol_indices,
    semantic_payload_accounting,
    simplex_sketch_to_probabilities,
)


class SemanticSketchTests(unittest.TestCase):
    def test_exact_payload_accounting(self) -> None:
        result = semantic_payload_accounting(2, 256, 32, 16)
        self.assertEqual(result["total_real_symbols"], 16384)
        self.assertEqual(result["payload_real_symbols"], 512)
        self.assertAlmostEqual(result["payload_fraction_of_structure"], 0.03125)

    def test_projection_is_deterministic_and_normalized(self) -> None:
        probabilities = torch.softmax(torch.randn(4, 1000), dim=1)
        first = fixed_rademacher_projection(1000, 32, 17)
        second = fixed_rademacher_projection(1000, 32, 17)
        self.assertTrue(torch.equal(first, second))
        sketch = probabilities_to_sketch(probabilities, first)
        self.assertTrue(torch.allclose(sketch.square().mean(dim=1), torch.ones(4), atol=1e-5))

    def test_noiseless_payload_roundtrip_and_erasure(self) -> None:
        latent = torch.randn(3, 4, 64, 64)
        sketch = torch.randn(3, 32)
        sketch = sketch * (32**0.5) / sketch.norm(dim=1, keepdim=True)
        embedded, indices = embed_repeated_sketch(latent, sketch, 16)
        recovered, erased = recover_repeated_sketch_and_erase(embedded, 32, 16, indices)
        cosine = F.cosine_similarity(sketch, recovered, dim=1)
        self.assertTrue(torch.all(cosine > 0.99999))
        self.assertTrue(torch.all(erased.flatten(start_dim=1)[:, indices] == 0))

    def test_noiseless_simplex_sketch_roundtrip(self) -> None:
        probabilities = torch.softmax(torch.randn(4, 10), dim=1)
        sketch = probabilities_to_simplex_sketch(probabilities)
        recovered = simplex_sketch_to_probabilities(sketch)
        self.assertTrue(torch.allclose(probabilities, recovered, atol=1e-6))
        self.assertTrue(torch.allclose(sketch.square().mean(dim=1), torch.ones(4)))

    def test_all_negative_simplex_sketch_falls_back_to_uniform(self) -> None:
        recovered = simplex_sketch_to_probabilities(-torch.ones(2, 10))
        self.assertTrue(torch.allclose(recovered, torch.full((2, 10), 0.1)))

    def test_uint4_probability_bit_roundtrip(self) -> None:
        probabilities = torch.softmax(torch.randn(5, 10), dim=1)
        codes, decoded = quantize_probabilities_uniform(probabilities, 4)
        bits = integer_codes_to_bits(codes, 4)
        recovered_codes = bits_to_integer_codes(bits, 10, 4)
        self.assertTrue(torch.equal(codes, recovered_codes))
        self.assertTrue(torch.allclose(decoded.sum(dim=1), torch.ones(5)))
        self.assertEqual(bits.shape, (5, 40))

    def test_reserved_indices_are_even_and_unique(self) -> None:
        indices = reserved_symbol_indices(16384, 512)
        self.assertEqual(torch.unique(indices).numel(), 512)
        self.assertEqual(int(indices[0]), 0)
        self.assertLess(int(indices[-1]), 16384)

    def test_payload_budget_rejects_exhaustion(self) -> None:
        with self.assertRaises(ValueError):
            semantic_payload_accounting(2, 256, 1024, 16)


if __name__ == "__main__":
    unittest.main()
