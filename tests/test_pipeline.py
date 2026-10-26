"""
Tests unitaires d'OptoBind-AI — mode dry_run (sans GPU)
=========================================================
Lancez avec : pytest tests/ -v
"""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import sys

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    """Retourne un ensemble de répertoires temporaires pour les tests."""
    (tmp_path / "pdbs").mkdir()
    (tmp_path / "fasta").mkdir()
    (tmp_path / "reports").mkdir()
    return {
        "root"    : tmp_path,
        "pdbs"    : tmp_path / "pdbs",
        "fasta"   : tmp_path / "fasta",
        "reports" : tmp_path / "reports",
    }


@pytest.fixture
def dummy_target(tmp_path) -> Path:
    """Crée un fichier PDB fictif pour la cible Zn²⁺."""
    pdb = tmp_path / "zn_site.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    2  ZN  ZN  A   2       3.000   0.000   0.000  1.00  0.00          ZN  \n"
        "END\n",
        encoding="utf-8",
    )
    return pdb


@pytest.fixture
def dummy_ligand(tmp_path) -> Path:
    """Crée un fichier SDF minimal pour DFHBI-1T."""
    sdf = tmp_path / "DFHBI-1T.sdf"
    sdf.write_text(
        "\n  DFHBI-1T\n\n"
        "  0  0  0  0  0  0  0  0  0  0999 V2000\n"
        "M  END\n$$$$\n",
        encoding="utf-8",
    )
    return sdf


@pytest.fixture
def default_cfg(tmp_path) -> Path:
    """Crée un fichier YAML de configuration minimal."""
    import yaml
    cfg = {
        "design": {"n_variants": 2, "min_length": 70, "max_length": 90, "pH": 7.4},
        "rfdiffusion": {},
        "proteinmpnn": {"num_seq_per_target": 2},
        "alphafold": {"backend": "chai-1"},
        "validation": {"rmsd_threshold": 2.0, "plddt_threshold": 90.0},
    }
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Configuration
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_default_config(self, default_cfg):
        from optobind_ai.utils.config import load_config
        cfg = load_config(default_cfg)
        assert cfg["design"]["n_variants"] == 2
        assert cfg["design"]["pH"] == 7.4

    def test_validate_config_ok(self, default_cfg):
        from optobind_ai.utils.config import load_config, validate_config
        cfg = load_config(default_cfg)
        validate_config(cfg)  # ne doit pas lever d'exception

    def test_validate_config_bad_n_variants(self, default_cfg):
        from optobind_ai.utils.config import load_config, validate_config
        cfg = load_config(default_cfg)
        cfg["design"]["n_variants"] = 0
        with pytest.raises(ValueError, match="n_variants"):
            validate_config(cfg)

    def test_validate_config_bad_length(self, default_cfg):
        from optobind_ai.utils.config import load_config, validate_config
        cfg = load_config(default_cfg)
        cfg["design"]["min_length"] = 100
        cfg["design"]["max_length"] = 70
        with pytest.raises(ValueError, match="min_length"):
            validate_config(cfg)

    def test_missing_config_file(self, tmp_path):
        from optobind_ai.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Tests — IO Helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestIOHelpers:
    def test_ensure_output_dirs(self, tmp_path):
        from optobind_ai.utils.io_helpers import ensure_output_dirs
        dirs = ensure_output_dirs(tmp_path, "run_test")
        for key in ["root", "pdbs", "fasta", "reports", "logs", "inputs_snapshot"]:
            assert dirs[key].exists(), f"Répertoire manquant : {key}"

    def test_count_pdb_residues(self, dummy_target):
        from optobind_ai.utils.io_helpers import count_pdb_residues
        n = count_pdb_residues(dummy_target)
        assert n == 1  # seul l'ATOM record compte (pas HETATM)

    def test_read_fasta(self, tmp_path):
        from optobind_ai.utils.io_helpers import read_fasta
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nACDEFGHI\n>seq2\nKLMNPQRS\n", encoding="utf-8")
        entries = list(read_fasta(fasta))
        assert len(entries) == 2
        assert entries[0] == ("seq1", "ACDEFGHI")
        assert entries[1] == ("seq2", "KLMNPQRS")

    def test_scaffold_name(self):
        from optobind_ai.utils.io_helpers import Scaffold
        s = Scaffold(index=3, pdb_path=Path("dummy.pdb"), run_id="run_001", n_residues=80)
        assert s.name == "run_001_scaffold_003"

    def test_designed_sequence_fasta(self):
        from optobind_ai.utils.io_helpers import Scaffold, DesignedSequence
        sc = Scaffold(index=0, pdb_path=Path("dummy.pdb"), run_id="run_001", n_residues=75)
        ds = DesignedSequence(
            index=1, scaffold=sc, sequence="ACDEFGHIKLM",
            fasta_path=Path("dummy.fasta"), mpnn_score=0.45,
        )
        fasta_str = ds.to_fasta()
        assert fasta_str.startswith(">run_001_scaffold_000_seq_01")
        assert "ACDEFGHIKLM" in fasta_str


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Module Scaffolding (dry_run)
# ──────────────────────────────────────────────────────────────────────────────

class TestScaffolding:
    def test_dry_run_scaffolding(self, dummy_target, dummy_ligand, tmp_dirs, default_cfg):
        import logging
        from optobind_ai.utils.config import load_config
        from optobind_ai.modules.scaffolding import run_rfdiffusion

        cfg = load_config(default_cfg)
        logger = logging.getLogger("test")

        scaffolds = run_rfdiffusion(
            target_pdb   = dummy_target,
            ligand_file  = dummy_ligand,
            output_dir   = tmp_dirs["pdbs"],
            n_variants   = 2,
            backbone_len = (70, 90),
            ph           = 7.4,
            dry_run      = True,
            cfg          = cfg.get("rfdiffusion", {}),
            logger       = logger,
        )
        assert len(scaffolds) == 2
        for s in scaffolds:
            assert s.pdb_path.exists()
            assert 70 <= s.n_residues <= 90


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Module Séquençage (dry_run)
# ──────────────────────────────────────────────────────────────────────────────

class TestSequencing:
    def _make_scaffold(self, tmp_path):
        from optobind_ai.utils.io_helpers import Scaffold
        pdb = tmp_path / "scaffold_000.pdb"
        pdb.write_text("REMARK test\nEND\n", encoding="utf-8")
        return Scaffold(index=0, pdb_path=pdb, run_id="run_test", n_residues=75)

    def test_dry_run_sequencing(self, tmp_dirs, tmp_path, default_cfg):
        import logging
        from optobind_ai.utils.config import load_config
        from optobind_ai.modules.sequencing import run_proteinmpnn

        cfg = load_config(default_cfg)
        logger = logging.getLogger("test")
        scaffold = self._make_scaffold(tmp_path)

        sequences = run_proteinmpnn(
            scaffolds  = [scaffold],
            output_dir = tmp_dirs["fasta"],
            ph         = 7.4,
            dry_run    = True,
            cfg        = cfg.get("proteinmpnn", {}),
            logger     = logger,
        )
        assert len(sequences) == 2  # num_seq_per_target = 2
        for s in sequences:
            assert len(s.sequence) == 75
            assert s.fasta_path.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Module Validation (dry_run)
# ──────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def _make_sequence(self, tmp_path):
        from optobind_ai.utils.io_helpers import Scaffold, DesignedSequence
        sc = Scaffold(index=0, pdb_path=tmp_path / "scaffold.pdb",
                      run_id="run_test", n_residues=75)
        return DesignedSequence(
            index=0, scaffold=sc,
            sequence="A" * 75,
            fasta_path=tmp_path / "seq.fasta",
            mpnn_score=0.5,
        )

    def test_dry_run_validation(self, dummy_target, dummy_ligand, tmp_dirs, tmp_path, default_cfg):
        import logging
        from optobind_ai.utils.config import load_config
        from optobind_ai.modules.validation import run_alphafold

        cfg = load_config(default_cfg)
        logger = logging.getLogger("test")
        seq = self._make_sequence(tmp_path)

        results = run_alphafold(
            sequences   = [seq],
            ligand_file = dummy_ligand,
            target_pdb  = dummy_target,
            output_dir  = tmp_dirs["pdbs"],
            dry_run     = True,
            cfg         = cfg.get("alphafold", {}),
            logger      = logger,
        )
        assert len(results) == 1
        r = results[0]
        assert 0 <= r.plddt_mean <= 100
        assert r.rmsd >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Module Reporting
# ──────────────────────────────────────────────────────────────────────────────

class TestReporting:
    def test_build_report_no_validation(self, tmp_dirs, tmp_path, default_cfg):
        import logging
        import csv
        from optobind_ai.utils.config import load_config
        from optobind_ai.utils.io_helpers import Scaffold, DesignedSequence
        from optobind_ai.modules.reporting import build_report

        cfg = load_config(default_cfg)
        logger = logging.getLogger("test")

        sc = Scaffold(index=0, pdb_path=tmp_path / "s.pdb", run_id="run_test", n_residues=75)
        seq = DesignedSequence(
            index=0, scaffold=sc, sequence="A"*75,
            fasta_path=tmp_path / "seq.fasta", mpnn_score=0.5
        )

        report = build_report(
            scaffolds=[], sequences=[seq], validation_results=[],
            output_path=tmp_dirs["reports"] / "report.csv",
            run_id="run_test", cfg=cfg, logger=logger,
        )
        assert report.exists()
        with report.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["passed"] == "N/A"
