import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from lbm_utils import classify_lbm_regime


class TestLBMCompressibilityRegime(unittest.TestCase):
    def test_low_mach_is_claim_grade_only_for_current_weakly_compressible_model(self):
        regime = classify_lbm_regime(0.3)
        self.assertEqual(regime["validity_regime"], "validated_low_mach_envelope")
        self.assertEqual(regime["claim_grade"], "low_mach_sanity_only")
        self.assertIsNone(regime["high_mach_warning"])
        self.assertEqual(regime["compressibility_model"], "weakly_compressible_isothermal_lbm")
        self.assertEqual(regime["thermal_model"], "none_isothermal")

    def test_high_mach_internal_lbm_is_experimental_and_not_claim_grade(self):
        regime = classify_lbm_regime(0.31)
        self.assertEqual(regime["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(regime["claim_grade"], "no_claim_experimental")
        self.assertIn("not validated compressible CFD", regime["high_mach_warning"])
        self.assertFalse(regime["claim_grade"].startswith("low_mach"))

    def test_external_validation_can_be_recorded_without_upgrading_internal_physics(self):
        regime = classify_lbm_regime(0.8, external_validation="openfoam_compressible_converged")
        self.assertEqual(regime["validity_regime"], "external_compressible_reference_available")
        self.assertEqual(regime["claim_grade"], "external_reference_only")
        self.assertIn("Internal D3Q27", regime["high_mach_warning"])


if __name__ == "__main__":
    unittest.main()
