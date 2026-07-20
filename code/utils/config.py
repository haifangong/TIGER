"""Configuration dataclass + JSON loader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # --- data paths ---
    train_csv: str = "metadata/train_val_by_cfu_group_ug_per_mL.csv"
    test_csv: str = "metadata/test_LL37_by_cfu_group_ug_per_mL.csv"
    train_pdb_dir: str = "data/3D_data_train_eva_Rosetta"
    test_pdb_dir: str = "data/3D_data_train_eva_Rosetta"
    feature_path: str = "data/metadata/features.txt"
    shared_cache_dir: str = "artifacts/cache_ll37_holdout_cfu"
    out_dir: str = "outputs/tiger_code_run"
    csv_encoding: str | None = "utf-8"

    # --- protocol ---
    seed: int = 20260714
    folds: int = 5
    min_len: int = 6
    max_len: int = 50
    use_cfu_protocol: bool = True
    use_cfu_feature: bool = True
    primary_cfu_group: str = "1E5 - 1E6"
    eval_external: bool = False

    # --- model ---
    emb_dim: int = 90
    num_layer: int = 2
    dropout_ratio: float = 0.35
    gnn_type: str = "gatv2"
    graph_pooling: str = "attention"
    fusion: str = "attention"
    feature_modalities: str = "gsh"
    structure_features: str = "se"
    pair_interaction: str = "diff"
    # integer | embedding | onehot
    # Default integer + zscore matched the strongest LL37-holdout CV in ablations;
    # AA ids are always sequential 1..20 (see utils.constants). Prefer embedding
    # when you need a categorically valid sequence branch (uses positional encoding).
    seq_encoding: str = "integer"
    # legacy_x10 | zscore  (zscore: per-fold train stats; empirically strongest)
    global_feature_scaling: str = "zscore"
    # Node features: AA descriptors (20) + Rosetta energies (20); coords optional.
    include_node_coords: bool = False
    # none | zscore  (per-fold train stats over all residue nodes)
    node_feature_scaling: str = "zscore"

    # --- optimizer ---
    lr: float = 0.0015
    weight_decay: float = 5e-4
    lr_scheduler: str = "none"  # none | cosine
    graph_epochs: int = 70
    pair_batch_size: int = 2048
    eval_batch_size: int = 4096
    early_stop_patience: int = 16
    min_delta: float = 1e-4
    final_epoch_strategy: str = "cv_median_best"

    # --- pairing ---
    similarity_threshold: float = 0.30
    neighbor_top_k: int = 50
    fallback_top_k: int = 20
    max_candidate_per_anchor: int = 2500
    max_train_pairs: int = 250000
    pair_balance_num: int = 1000
    delta_bin_width: float = 0.50
    use_signed_sampling: bool = False
    use_similarity_strata: bool = False

    # --- runtime ---
    model_kind: str = "pair"  # pair | single
    name: str = "fusion_attention"
    checkpoint: str | None = None
    device: str = "cuda:0"
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        raw = json.loads(Path(path).read_text())
        names = {f.name for f in fields(Config)}
        for k, v in raw.items():
            if k in names:
                setattr(cfg, k, v)
    if overrides:
        names = {f.name for f in fields(Config)}
        for k, v in overrides.items():
            if k in names and v is not None:
                setattr(cfg, k, v)
    if cfg.smoke:
        cfg.graph_epochs = min(cfg.graph_epochs, 3)
        cfg.early_stop_patience = min(cfg.early_stop_patience, 4)
        cfg.max_train_pairs = min(cfg.max_train_pairs, 12000)
        cfg.pair_balance_num = min(cfg.pair_balance_num, 80)
    return cfg
