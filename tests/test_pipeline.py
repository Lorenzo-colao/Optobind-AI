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

    def test_auto_model_selection(self, dummy_target, dummy_ligand, tmp_path):
        from optobind_ai.modules.sequencing import detect_optimal_mpnn_model

        # 1. Cible avec ion métallique (dummy_target contient ZN)
        model, reason = detect_optimal_mpnn_model(target_pdb=dummy_target)
        assert model == "ligandmpnn"
        assert "métallique" in reason or "hétéroatomes" in reason

        # 2. Cible avec fichier ligand
        pure_protein = tmp_path / "pure.pdb"
        pure_protein.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n")
        model, _ = detect_optimal_mpnn_model(target_pdb=pure_protein, ligand_file=dummy_ligand)
        assert model == "ligandmpnn"

        # 3. Cible purement protéique (sans ligand ni métal)
        model, reason = detect_optimal_mpnn_model(target_pdb=pure_protein)
        assert model == "proteinmpnn"
        assert "protéique pure" in reason

        # 4. Forçage utilisateur via config
        model, _ = detect_optimal_mpnn_model(target_pdb=dummy_target, configured_model="proteinmpnn")
        assert model == "proteinmpnn"
        model, _ = detect_optimal_mpnn_model(target_pdb=pure_protein, configured_model="ligandmpnn")
        assert model == "ligandmpnn"


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


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Constructeur Automatisé PDB, Contigs 3D & Optimisation ADN
# ──────────────────────────────────────────────────────────────────────────────

class TestPdbBuilderAndOptimizations:
    def test_automated_pdb_builder_on_zn_site(self, tmp_path):
        from optobind_ai.modules.pdb_builder import inspect_structure, build_target_pdb

        ref_pdb = Path(__file__).parent.parent / "inputs" / "targets" / "zn2+_site.pdb"
        ref_sdf = Path(__file__).parent.parent / "inputs" / "ligands" / "DFHBI-1T.sdf"

        info = inspect_structure(ref_pdb)
        assert len(info["metals"]) >= 1
        assert info["metals"][0]["element"] == "ZN"
        assert info["metals"][0]["n_coordinating"] == 4

        out_pdb = tmp_path / "built_zn_target.pdb"
        res = build_target_pdb(
            source=ref_pdb,
            output_pdb=out_pdb,
            mode="metal_site",
            metal_id="ZN",
            reporter_ligand=ref_sdf,
            min_length=70,
            max_length=100,
        )
        assert out_pdb.is_file()
        assert "A1-1," in res.recommended_contigs
        assert "A4-4," in res.recommended_contigs
        assert len(res.gap_analysis) == 3
        content = out_pdb.read_text(encoding="utf-8")
        assert "OPTOBIND CONTIGS:" in content
        assert "CONECT" in content

    def test_multi_segment_3d_motif_resolver(self, tmp_path):
        from optobind_ai.modules.sequencing import motif_positions_from_contigs

        tgt = tmp_path / "tgt.pdb"
        tgt.write_text(
            "ATOM      1  CA  CYS A   1       0.000   0.000   0.000  1.00 20.00           C\n"
            "ATOM      2  CA  HIS A   2       8.000   0.000   0.000  1.00 20.00           C\n"
            "END\n",
            encoding="utf-8",
        )
        # Squelette de 12 résidus où CYS1 est en pos 4 et HIS2 en pos 9 (écart 4 résidus libres)
        lines = []
        for idx in range(1, 13):
            if idx == 4:
                rname, x, y, z = "CYS", 0.0, 0.0, 0.0
            elif idx == 9:
                rname, x, y, z = "HIS", 8.0, 0.0, 0.0
            else:
                rname, x, y, z = "ALA", float(idx) * 2.0, 5.0, 0.0
            lines.append(
                f"ATOM  {idx:5d}  CA  {rname} A{idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            )
        lines.append("END\n")
        scaf = tmp_path / "scaf.pdb"
        scaf.write_text("".join(lines), encoding="utf-8")

        pos = motif_positions_from_contigs(
            "2-5,A1,3-6,A2,2-5",
            n_residues=12,
            scaffold_pdb=scaf,
            target_pdb=tgt,
        )
        assert pos == [4, 9]

    def test_codon_optimization_restriction_free(self):
        from optobind_ai.modules.biophysics import optimize_dna_sequence

        res = optimize_dna_sequence("MCHHCADEFGKLMNPQRSTVWY", host="ecoli", add_his_tev=True)
        assert res["restriction_sites_free"] is True
        assert res["dna_length_bp"] % 3 == 0
        assert 35.0 <= res["gc_percent"] <= 70.0

    def test_rfdiffusion_contig_normalization_and_sdf_conect(self, tmp_path):
        from optobind_ai.modules.scaffolding import _normalize_contigs_for_rfdiffusion
        from optobind_ai.modules.pdb_builder import (
            parse_sdf_heavy_atoms,
            _build_ligand_conect_lines,
        )

        assert (
            _normalize_contigs_for_rfdiffusion("20-25,A1,3-7,A2,4-10,A3,36-46")
            == "20-25,A1-1,3-7,A2-2,4-10,A3-3,36-46"
        )
        assert (
            _normalize_contigs_for_rfdiffusion("10-15,A1-3,5-8,A4-4,20-30")
            == "10-15,A1-3,5-8,A4-4,20-30"
        )

        sdf = Path("inputs/ligands/DFHBI-1T.sdf")
        if sdf.is_file():
            atoms = parse_sdf_heavy_atoms(sdf)
            assert len(atoms) == 22
            # 2D SDF rescalé : C-F / C-C autour de 1.42 Å
            serials = [(35 + i, a) for i, a in enumerate(atoms)]
            conect = _build_ligand_conect_lines(serials)
            # Chaque fluor (atomes 35, 36, 37) ne doit avoir qu'une seule liaison covalente (au carbone CF3)
            f_conects = [l for l in conect if l.startswith("CONECT   35 ") or l.startswith("CONECT   36 ") or l.startswith("CONECT   37 ")]
            assert len(f_conects) == 3
            for line in f_conects:
                parts = line.split()
                assert len(parts) == 3  # CONECT <atom> <1 partenaire>

    def test_chai_restraints_use_mapped_scaffold_motif_and_best_model_idx(self, tmp_path):
        import numpy as np
        import logging
        from optobind_ai.modules.validation import (
            _write_chai_restraints,
            _select_best_chai_model_idx,
            _parse_chai_tm_scores,
        )

        ref_pdb = Path("inputs/targets/zn2+_site.pdb")
        if ref_pdb.is_file():
            seq_07 = "APLLPYPPLRDIRYPRGPAPRCLQEGEPKCTVASPTRPFLHLTIHIESGSAKRVSAVSGGRTVSITEDDERVEGPPGEVAYRRRP"
            out_csv = tmp_path / "restraints_sc1.csv"
            ok = _write_chai_restraints(
                out_csv,
                ref_pdb,
                motif_positions=[22, 30, 41, 45],
                protein_seq=seq_07,
            )
            assert ok is True
            txt = out_csv.read_text(encoding="utf-8")
            assert ",A,C22," in txt
            assert ",A,C30," in txt
            assert ",A,H41," in txt
            assert ",A,H45," in txt
            assert ",A,C1," not in txt

        out_dir = tmp_path / "chai_out"
        out_dir.mkdir()
        np.savez(out_dir / "scores.model_idx_0.npz", aggregate_score=np.array([0.27]), ptm=np.array([0.57]), iptm=np.array([0.19]))
        np.savez(out_dir / "scores.model_idx_1.npz", aggregate_score=np.array([0.65]), ptm=np.array([0.72]), iptm=np.array([0.62]))
        assert _select_best_chai_model_idx(out_dir) == 1
        ptm, iptm = _parse_chai_tm_scores(out_dir, logging.getLogger("test"))
        assert round(ptm, 2) == 0.72 and round(iptm, 2) == 0.62

    def test_ligandmpnn_preserves_last_id_and_ranks_confidence_descending(self, tmp_path):
        from optobind_ai.utils.io_helpers import Scaffold, DesignedSequence
        from optobind_ai.modules.sequencing import _collect_sequences
        from optobind_ai.modules.biophysics import smart_pre_filter_sequences

        sc_pdb = tmp_path / "sc.pdb"
        sc_pdb.write_text("END\n", encoding="utf-8")
        sc = Scaffold(index=0, pdb_path=sc_pdb, run_id="run_t", n_residues=10)

        seqs_dir = tmp_path / "seqs"
        seqs_dir.mkdir()
        (seqs_dir / "sc.fa").write_text(
            ">sc, T=0.1, seed=42\n"
            "AAAAAAAAAA\n"
            ">sc, id=1, T=0.1, seed=42, overall_confidence=0.3100, ligand_confidence=0.2500\n"
            "MKEEEEEEEE\n"
            ">sc, id=2, T=0.1, seed=42, overall_confidence=0.4900, ligand_confidence=0.4100\n"
            "MKEEEEEEEK\n",
            encoding="utf-8",
        )
        out_fa = tmp_path / "out_fa"
        out_fa.mkdir()
        collected = _collect_sequences(sc, tmp_path, num_seq=2, output_dir=out_fa)
        assert len(collected) == 3  # seq_00 native + id=1 + id=2

        for ds in collected:
            ds.metadata["mpnn_model"] = "ligandmpnn"
        kept, _ = smart_pre_filter_sequences(
            collected,
            target_pdb=sc_pdb,
            cfg={"pre_filter": {"enabled": True, "max_liabilities": 1, "filter_cysteines": False, "min_net_charge": 0.0, "avoid_neutral_pi": False, "top_k_per_scaffold": 1, "filter_native": True}},
        )
        assert len(kept) == 1
        assert kept[0].mpnn_score == pytest.approx(0.49)



