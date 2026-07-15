import json
import os
import sys


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from build_openvsp_airshow_corpus import AIRSHOW_COLLECTION, select_exact_vsp_records


def test_select_exact_vsp_records_uses_only_catalogued_airshow_vsp3_sources(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "records": [
                    {"source_id": "good-a", "source_collection": AIRSHOW_COLLECTION, "file_format": "vsp3", "exact_cad_url": "https://example/a"},
                    {"source_id": "good-b", "source_collection": AIRSHOW_COLLECTION, "file_format": "vsp3", "exact_cad_url": "https://example/b"},
                    {"source_id": "preview-only", "source_collection": AIRSHOW_COLLECTION, "file_format": "obj", "exact_cad_url": "https://example/c"},
                    {"source_id": "other", "source_collection": "other", "file_format": "vsp3", "exact_cad_url": "https://example/d"},
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = select_exact_vsp_records(catalog, target_count=10, seed=7)

    assert {record["source_id"] for record in selected} == {"good-a", "good-b"}
    assert select_exact_vsp_records(catalog, target_count=1, seed=7) == select_exact_vsp_records(catalog, target_count=1, seed=7)
    assert select_exact_vsp_records(catalog, target_count=1, seed=7, selection_offset=1) != select_exact_vsp_records(catalog, target_count=1, seed=7)
