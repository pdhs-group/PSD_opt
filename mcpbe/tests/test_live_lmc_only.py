"""Tests for the active live-LMC-only fragmentation interface."""

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
from wmcpbe import lmc_adapter  # noqa: E402
from wmcpbe.lmc_adapter import LMCLiveAdapter, LMCLiveUnbreakable  # noqa: E402


def _minimal_live_adapter() -> LMCLiveAdapter:
    """Return an adapter whose undersized-parent check needs no pool asset."""
    adapter = LMCLiveAdapter()
    adapter._sim = object()  # The test exits before invoking the simulator.
    adapter.pool_dir = "unused-by-small-parent-test"
    adapter.Df = 1.8
    adapter.MAS = 0.5
    adapter.NO_FRAG = 2
    adapter.A0_run = 1.0
    adapter.delta_cells = 0.1
    return adapter


class TestLiveLMCOnly(unittest.TestCase):
    def test_module_exposes_only_live_adapter(self) -> None:
        self.assertTrue(hasattr(lmc_adapter, "LMCLiveAdapter"))
        self.assertTrue(hasattr(lmc_adapter, "LMCLiveUnbreakable"))
        for removed_name in (
            "LMCBaseAdapter",
            "LMCTableAdapter",
            "LMCRankAdapter",
            "LMCCopulaAdapter",
            "LMCFlowAdapter",
        ):
            self.assertFalse(hasattr(lmc_adapter, removed_name), removed_name)

    def test_removed_offline_configuration_fails_before_initialization(self) -> None:
        solver = MCPBESolver(
            dim=1,
            t_vec=np.array([0.0, 1.0]),
            load_attr=False,
            init=False,
            seed=42,
        )
        solver.use_lmc_pre_model = False
        with self.assertRaisesRegex(AttributeError, "removed offline LMC adapter"):
            solver._init_lmc()

    def test_small_live_parent_is_unbreakable_without_uniform_fallback(self) -> None:
        adapter = _minimal_live_adapter()
        with self.assertRaises(LMCLiveUnbreakable):
            adapter.sample_one_shot(np.array([1.0]), np.random.default_rng(3))

        solver = MCPBESolver(
            dim=1,
            t_vec=np.array([0.0, 1.0]),
            load_attr=False,
            init=False,
            seed=42,
        )
        solver.use_lmc_live = True
        solver.lmc_live = adapter
        status, fragments = solver._break_build_fragments(np.array([1.0]))
        self.assertEqual(status, "disable")
        self.assertEqual(fragments, [])

        solver.V_flat = np.array([[1.0], [1.0]])
        solver.W = np.array([1.0])
        solver.a_tot = 1
        solver._cap = 1
        solver._break_rate = np.array([1.0])
        solver._delta_break = np.array([1.0])
        solver._break_sampler = None
        solver._do_one_break()
        self.assertEqual(float(solver._break_rate[0]), 0.0)
        self.assertEqual(float(solver._delta_break[0]), 0.0)
        self.assertEqual(solver._last_break_dW, 0.0)

    def test_analytical_path_remains_available_when_live_lmc_is_disabled(self) -> None:
        solver = MCPBESolver(
            dim=1,
            t_vec=np.array([0.0, 1.0]),
            load_attr=False,
            init=False,
            seed=17,
        )
        solver.use_lmc_live = False
        solver.BREAKFVAL = 2
        solver.pl_v = 2.0
        solver.pl_q = 1.0
        solver._compute_frag_num()
        solver._prepare_break_config()

        status, fragments = solver._break_build_fragments(np.array([8.0]))
        self.assertEqual(status, "ok")
        self.assertGreaterEqual(len(fragments), 2)
        self.assertTrue(all(float(np.sum(fragment)) > 0.0 for fragment in fragments))
        self.assertAlmostEqual(sum(float(np.sum(fragment)) for fragment in fragments), 8.0)


if __name__ == "__main__":
    unittest.main()
