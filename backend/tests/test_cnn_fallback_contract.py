import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import detector  # noqa: E402
import main  # noqa: E402


class TestCnnFallbackGate(unittest.TestCase):
    def test_activate_when_auc_below_gate(self):
        active, reason = main._resolve_cnn_fallback_state(handcrafted_auc=0.64)
        self.assertTrue(active)
        self.assertEqual(reason, "gated_auc")

    def test_deactivate_when_auc_meets_gate(self):
        active, reason = main._resolve_cnn_fallback_state(handcrafted_auc=0.65)
        self.assertFalse(active)
        self.assertEqual(reason, "off")

    def test_force_override_activates_fallback(self):
        active, reason = main._resolve_cnn_fallback_state(
            handcrafted_auc=0.95,
            force_cnn_fallback=True,
        )
        self.assertTrue(active)
        self.assertEqual(reason, "forced")


class TestModelUsedStates(unittest.TestCase):
    def test_cnn_fallback_state_when_cnn_available(self):
        model_used, cnn_active = detector._resolve_model_used(
            model=None,
            scaler=None,
            cnn_fallback_active=True,
            cnn_infer_available=True,
        )
        self.assertEqual(model_used, "cnn_fallback")
        self.assertTrue(cnn_active)

    def test_cnn_fallback_degraded_when_cnn_unavailable(self):
        model_used, cnn_active = detector._resolve_model_used(
            model=None,
            scaler=None,
            cnn_fallback_active=True,
            cnn_infer_available=False,
        )
        self.assertEqual(model_used, "cnn_fallback_degraded")
        self.assertTrue(cnn_active)

    def test_handcrafted_model_states_when_fallback_off(self):
        # Backend uses equal_weights scoring as primary (LR model retained for
        # diagnostics only — it underperforms equal-weights on FF++ C23).
        model_used, cnn_active = detector._resolve_model_used(
            model=object(),
            scaler=object(),
            cnn_fallback_active=False,
            cnn_infer_available=False,
        )
        self.assertEqual(model_used, "equal_weights")
        self.assertFalse(cnn_active)


if __name__ == "__main__":
    unittest.main()
