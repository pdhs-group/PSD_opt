"""Tests for the active wmcpbe fixed packet-weight interface."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_path in (
    PROJECT_ROOT / "pbe-core" / "src",
    PROJECT_ROOT / "lmc" / "src",
    PROJECT_ROOT / "breakage-rate-model" / "src",
    PROJECT_ROOT / "mcpbe" / "src",
):
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from wmcpbe import MCPBESolver  # noqa: E402


def _make_solver(
    process_type: str,
    weights: tuple[float, ...],
    *,
    break_dW_const: float = 1.0,
    agg_dW_const: float = 1.0,
) -> MCPBESolver:
    """Build a minimal initialized 1D solver without LMC dependencies."""
    count = len(weights)
    solver = MCPBESolver(
        dim=1,
        t_vec=np.array([0.0, 1.0]),
        load_attr=False,
        init=False,
        seed=42,
    )
    solver.process_type = process_type
    solver.Vc = 1.0
    solver.break_dW_const = break_dW_const
    solver.agg_dW_const = agg_dW_const
    solver.lmc_use_breakage_model = False

    volumes = np.ones((2, count), dtype=float)
    volumes[-1, :] = volumes[0, :]
    solver._initialize_particles(
        init_Vc=False,
        V_flat=volumes,
        W_init=np.asarray(weights, dtype=float),
    )
    return solver


class TestPacketWeightConstants(unittest.TestCase):
    def test_breakage_delta_uses_direct_constant(self) -> None:
        solver = _make_solver("breakage", (0.25, 2.0), break_dW_const=1.0)
        solver._initialize_samplers()
        np.testing.assert_allclose(solver._delta_break[:2], (0.25, 1.0))

    def test_agglomeration_delta_and_packet_are_weight_limited(self) -> None:
        solver = _make_solver("agglomeration", (0.25, 2.0), agg_dW_const=1.0)
        solver._initialize_samplers()
        np.testing.assert_allclose(solver._delta_agg[:2], (0.25, 1.0))
        self.assertEqual(solver._compute_agg_dW(0, 1), 0.25)

    def test_invalid_packet_constants_fail_before_sampler_construction(self) -> None:
        invalid_breakage = _make_solver("breakage", (1.0,), break_dW_const=0.0)
        with self.assertRaisesRegex(ValueError, "break_dW_const"):
            invalid_breakage._initialize_samplers()

        invalid_agglomeration = _make_solver(
            "agglomeration", (1.0, 1.0), agg_dW_const=float("inf")
        )
        with self.assertRaisesRegex(ValueError, "agg_dW_const"):
            invalid_agglomeration._initialize_samplers()

    def test_legacy_packet_attribute_fails_loudly(self) -> None:
        solver = _make_solver("breakage", (1.0,))
        solver.break_dW_max = 1.0
        with self.assertRaisesRegex(AttributeError, "no longer supported"):
            solver._initialize_samplers()


if __name__ == "__main__":
    unittest.main()
