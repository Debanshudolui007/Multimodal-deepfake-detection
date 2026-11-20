"""
Tests for utils/config.py — YAML loading, validation, and path resolution.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from deepfake_detector.utils.config import (
    _validate_required_keys,
    load_yaml,
    resolve_paths,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def valid_yaml_file(tmp_path: Path) -> Path:
    """Write a minimal valid YAML file and return its path."""
    content = textwrap.dedent("""\
        name: test_config
        value: 42
        nested:
          key: hello
    """)
    p = tmp_path / "test.yaml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def invalid_yaml_file(tmp_path: Path) -> Path:
    """Write a syntactically broken YAML file."""
    p = tmp_path / "bad.yaml"
    p.write_text(": : broken: yaml: [[[", encoding="utf-8")
    return p


# ── load_yaml tests ───────────────────────────────────────────────────────────


class TestLoadYaml:
    def test_loads_valid_file(self, valid_yaml_file: Path) -> None:
        cfg = load_yaml(valid_yaml_file)
        assert cfg["name"] == "test_config"
        assert cfg["value"] == 42
        assert cfg["nested"]["key"] == "hello"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_raises_on_invalid_yaml(self, invalid_yaml_file: Path) -> None:
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_yaml(invalid_yaml_file)

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        cfg = load_yaml(p)
        assert cfg == {}


# ── _validate_required_keys tests ────────────────────────────────────────────


class TestValidateRequiredKeys:
    def test_passes_when_all_keys_present(self) -> None:
        cfg = {"raw": {}, "processed": {}, "split": {}, "random_seed": 42}
        # Should not raise
        _validate_required_keys(cfg, "dataset")

    def test_raises_on_missing_key(self) -> None:
        cfg = {"raw": {}}  # missing "processed", "split", "random_seed"
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_required_keys(cfg, "dataset")

    def test_unknown_config_name_skips_validation(self) -> None:
        # No required keys defined for "custom" — should not raise
        _validate_required_keys({}, "custom")


# ── resolve_paths tests ───────────────────────────────────────────────────────


class TestResolvePaths:
    def test_resolves_dataset_prefix(self, tmp_path: Path) -> None:
        cfg = {"root": "dataset/Celeb DF", "value": 42}
        resolved = resolve_paths(cfg, tmp_path)
        assert resolved["root"] == str(tmp_path / "dataset/Celeb DF")

    def test_leaves_non_path_strings_unchanged(self, tmp_path: Path) -> None:
        cfg = {"name": "CelebDF", "version": "v2"}
        resolved = resolve_paths(cfg, tmp_path)
        assert resolved["name"] == "CelebDF"
        assert resolved["version"] == "v2"

    def test_resolves_nested_paths(self, tmp_path: Path) -> None:
        cfg = {
            "processed": {
                "frames": "deepfake_detector/processed/frames",
                "count": 100,
            }
        }
        resolved = resolve_paths(cfg, tmp_path)
        assert resolved["processed"]["frames"] == str(
            tmp_path / "deepfake_detector/processed/frames"
        )
        assert resolved["processed"]["count"] == 100

    def test_resolves_list_paths(self, tmp_path: Path) -> None:
        cfg = {"dirs": ["dataset/a", "dataset/b", "not_a_path"]}
        resolved = resolve_paths(cfg, tmp_path)
        assert resolved["dirs"][0] == str(tmp_path / "dataset/a")
        assert resolved["dirs"][1] == str(tmp_path / "dataset/b")
        assert resolved["dirs"][2] == "not_a_path"


# ── Integration: load real config files ───────────────────────────────────────


class TestRealConfigs:
    """Integration test: load the actual config files from the project."""

    @pytest.fixture()
    def config_dir(self) -> Path:
        here = Path(__file__).resolve().parent.parent
        return here / "deepfake_detector" / "configs"

    def test_all_config_files_exist(self, config_dir: Path) -> None:
        stems = ["dataset", "train", "model", "frequency", "audio", "fusion"]
        for stem in stems:
            assert (config_dir / f"{stem}.yaml").exists(), f"Missing: {stem}.yaml"

    def test_dataset_yaml_has_required_structure(self, config_dir: Path) -> None:
        cfg = load_yaml(config_dir / "dataset.yaml")
        assert "raw" in cfg
        assert "celebdf" in cfg["raw"]
        assert "ff_plus" in cfg["raw"]
        assert "fakeavceleb" in cfg["raw"]
        assert "split" in cfg
        assert cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"] == pytest.approx(1.0)

    def test_model_yaml_backbone_field(self, config_dir: Path) -> None:
        cfg = load_yaml(config_dir / "model.yaml")
        assert "backbone" in cfg["spatial"]
        assert cfg["spatial"]["backbone"] == "resnext50_32x4d"
