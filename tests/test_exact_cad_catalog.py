import io
import subprocess
import sys
import zipfile

from CLI.build_exact_cad_catalog import (
    build_airshow_exact_cad_records,
    build_hiliftaeroml_records,
    build_hiliftaeroml_surface_records,
    build_nasa_uam_records,
    collect_nasa_uam_archive_metadata,
    render_markdown_report,
)


def test_airshow_catalog_keeps_exact_vsp_under_allowed_licenses():
    models = [
        {
            "id": "model-a",
            "name": "Allowed Model",
            "manufacturer": "Example",
            "license": "1",
            "newVspUrl": "https://storage.example/model-a.vsp3",
            "newX3dUrl": "https://storage.example/model-a.x3d",
            "document_name": "projects/demo/databases/(default)/documents/models/model-a",
        },
        {
            "id": "model-b",
            "name": "Non Commercial",
            "license": "4",
            "newVspUrl": "https://storage.example/model-b.vsp3",
        },
        {
            "id": "model-c",
            "name": "Preview Only",
            "license": "2",
            "newX3dUrl": "https://storage.example/model-c.x3d",
        },
    ]

    records = build_airshow_exact_cad_records(models)

    assert [record["source_id"] for record in records] == ["airshow_model-a"]
    record = records[0]
    assert record["exact_cad_url"] == "https://storage.example/model-a.vsp3"
    assert record["file_format"] == "vsp3"
    assert record["source_license"] == "No Rights Reserved (CC0)"
    assert record["preview_geometry_url"] == "https://storage.example/model-a.x3d"
    assert record["claim_boundary"].startswith("Exact OpenVSP source file URL")


def test_nasa_uam_catalog_reads_vsp3_members_from_zip_bytes():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as handle:
        handle.writestr("vehicle-a.vsp3", "<vsp>")
        handle.writestr("notes/readme.txt", "metadata")
        handle.writestr("nested/vehicle-b.VSP3", "<vsp>")

    records = build_nasa_uam_records(
        [
            {
                "vehicle": "Tiltduct",
                "zip_url": "https://www.nasa.gov/example/tiltduct.zip",
                "zip_sha256": "abc123",
                "zip_size_bytes": len(archive.getvalue()),
                "members": zipfile.ZipFile(io.BytesIO(archive.getvalue())).namelist(),
            }
        ]
    )

    assert [record["source_id"] for record in records] == [
        "nasa_uam_tiltduct_nested_vehicle-b",
        "nasa_uam_tiltduct_vehicle-a",
    ]
    assert all(record["file_format"] == "vsp3" for record in records)
    assert all(record["source_license"] == "NASA public reference vehicle release" for record in records)
    assert records[0]["archive_url"] == "https://www.nasa.gov/example/tiltduct.zip"


def test_hiliftaeroml_catalog_generates_one_canonical_stp_per_geometry_variant():
    records = build_hiliftaeroml_records(geometry_count=3, canonical_aoa=4)

    assert len(records) == 3
    assert records[0]["source_id"] == "hiliftaeroml_geo_LHC001"
    assert records[0]["file_format"] == "stp"
    assert records[0]["exact_cad_url"].endswith(
        "/geo_LHC001_AoA_4/geo_LHC001_AoA_4.stp"
    )
    assert records[2]["geometry_variant_id"] == "geo_LHC003"
    assert records[2]["available_flow_solution_count"] == 10


def test_hiliftaeroml_surface_catalog_generates_per_aoa_stl_records_with_duplicate_boundary():
    records = build_hiliftaeroml_surface_records(
        geometry_count=2,
        aoa_degrees=[4, 8],
    )

    assert len(records) == 4
    assert records[0]["source_id"] == "hiliftaeroml_surface_geo_LHC001_AoA_4"
    assert records[0]["file_format"] == "stl"
    assert records[0]["exact_cad_url"].endswith(
        "/geo_LHC001_AoA_4/geo_LHC001_AoA_4.stl"
    )
    assert records[1]["geometry_variant_id"] == "geo_LHC001"
    assert records[1]["angle_of_attack_deg"] == 8
    assert records[1]["geometry_uniqueness"] == "repeated_geometry_variant_across_aoa"


def test_catalog_script_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "CLI/build_exact_cad_catalog.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "exact aircraft CAD sources" in result.stdout


def test_nasa_uam_archive_collection_reports_bad_zip_without_aborting():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as handle:
        handle.writestr("vehicle.vsp3", "<vsp>")

    class FakeResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, timeout, verify):
            if "bad" in url:
                return FakeResponse(b"<html>login</html>")
            return FakeResponse(archive.getvalue())

    metadata = collect_nasa_uam_archive_metadata(
        FakeSession(),
        [
            {"vehicle": "Good", "zip_url": "https://example.test/good.zip"},
            {"vehicle": "Bad", "zip_url": "https://example.test/bad.zip"},
        ],
        verify_tls=False,
    )

    assert len(metadata["archives"]) == 1
    assert metadata["archives"][0]["vehicle"] == "Good"
    assert len(metadata["failures"]) == 1
    assert metadata["failures"][0]["vehicle"] == "Bad"
    assert metadata["failures"][0]["error_type"] == "BadZipFile"


def test_markdown_report_surfaces_uam_failures_and_source_links():
    markdown = render_markdown_report(
        {
            "generated_at": "2026-06-24T00:00:00+00:00",
            "catalog_path": "catalog.json",
            "machine_report_path": "report.json",
            "summary": {
                "record_count": 0,
                "source_collection_counts": {},
                "file_format_counts": {},
            },
            "source_metadata": {
                "airshow": {
                    "all_public_model_documents": 0,
                    "exact_vsp_url_documents": 0,
                    "license_qualified_exact_vsp_records": 0,
                },
                "nasa_uam": {
                    "archive_count": 7,
                    "archive_failure_count": 1,
                    "vsp3_member_count": 9,
                },
                "hiliftaeroml": {
                    "geometry_variant_count": 180,
                    "canonical_aoa_deg": 4,
                },
                "local_nasa_crm": {"record_count": 15},
                "nasa_crm_candidate_sweep": {
                    "candidate_group_count": 41,
                    "ready_group_count": 31,
                    "apparent_record_count": 1873,
                },
            },
        },
        [],
    )

    assert "Archive fetch failures recorded: `1`" in markdown
    assert "https://www.nasa.gov/reference/uam-refs/" in markdown
    assert "https://huggingface.co/datasets/nvidia/HiLiftAeroML" in markdown
