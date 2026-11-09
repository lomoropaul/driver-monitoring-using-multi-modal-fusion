import driver_monitoring


def test_package_imports():
    assert driver_monitoring.__version__ == "0.1.0"


def test_config_loads():
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())
    assert cfg["data"]["window_seconds"] == 4
    assert cfg["labels"]["kss_decline_min"] == 7
