import os
import sys
import unittest
from pathlib import Path


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import build_nasa_crm_whole_aircraft_context as crm_builder


class TestNasaCrmSourceCatalog(unittest.TestCase):
    def test_default_catalog_loads_ready_assets(self):
        assets = crm_builder.load_source_catalog(crm_builder.DEFAULT_SOURCE_CATALOG_PATH)

        self.assertGreaterEqual(len(assets), 15)
        self.assertEqual(len({asset.source_id for asset in assets}), len(assets))
        self.assertTrue(all(asset.candidate_status == "ready" for asset in assets))
        self.assertTrue(all(asset.source_url.startswith("https://") for asset in assets))

    def test_source_id_filter_selects_subset(self):
        assets = crm_builder.load_source_catalog(
            crm_builder.DEFAULT_SOURCE_CATALOG_PATH,
            source_ids={"crm_hl_reference_ldg", "crm_hs_dpw6_cf"},
        )

        self.assertEqual({asset.source_id for asset in assets}, {"crm_hl_reference_ldg", "crm_hs_dpw6_cf"})


if __name__ == "__main__":
    unittest.main()
