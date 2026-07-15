from run_with_resource_monitor import _sample_processes, _summarize


def test_summary_ignores_invalid_power_outlier():
    samples = [
        {"gpu": [{"power_draw_w": None}], "process": {}, "system": {}},
        {"gpu": [{"power_draw_w": 74.0}], "process": {}, "system": {}},
    ]

    summary = _summarize(samples, return_code=0, elapsed_s=1.0)

    assert summary["gpu_power_draw_w"] == {"min": 74.0, "max": 74.0, "mean": 74.0}


def test_process_gpu_memory_is_unknown_when_driver_reports_no_process_rows(monkeypatch):
    class Process:
        pid = 7

        def is_running(self):
            return True

        def children(self, recursive=True):
            return []

        def memory_info(self):
            return type("Memory", (), {"rss": 1024, "vms": 2048})()

        def cpu_percent(self, interval=None):
            return 1.0

        def name(self):
            return "python"

        def status(self):
            return "running"

    monkeypatch.setattr("run_with_resource_monitor._query_gpu_process_memory", lambda: {})

    sample = _sample_processes(Process())

    assert sample["gpu_process_memory_mb"] is None
