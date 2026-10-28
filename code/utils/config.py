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
    # MIC column name in train/test CSVs (default E. coli). Wetlab multi-species
    # tables may keep a species-specific name or remap into this column.
    target_col: str = "MIC_Escherichia_coli"

    # --- protocol ---
    seed: int = 1
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
    # Multimodal attention Q/K/V role assignment (used when fusion=attention):
    #   self_gsp | self_sgp | self_pgs  — self-attn over stacked tokens in that order
    #   cross_qg | cross_qs | cross_qp  — Q from one modality, K/V from the other two
    # Letters: g=graph(structure), s=sequence, p=physicochemical(handcrafted)
    fusion_attn_mode: str = "self_gsp"
    feature_modalities: str = "gps"
    # Graph-node amino-acid representation: "prior" = frozen 20-dim handcrafted
    # descriptor (legacy behaviour); "prior+emb" = descriptor concat learnable
    # embedding; "emb" = learnable embedding only.
    node_features_mode: str = "prior"
    node_emb_dim: int = 90
    # Sequence-embedding prior coupling: init from a fixed random orthonormal
    # projection of the 20-dim AA descriptor table; and/or anchor the table to
    # that projection with an L2 penalty (main-loop only, not cls-pretraining).
    seq_emb_prior_init: bool = False
    seq_emb_prior_reg: float = 0.0
    structure_features: str = "se"
    # Graph-edge construction: "dist" = 3D CA-proximity graph (<=5 A, legacy
    # behaviour); "chain" = sequential-adjacency edges only (v2-10 edge-geometry
    # ablation: strips 3D information from the graph channel). Edge weights are
    # discarded at PeptideGraph construction either way (adjacency only).
    edge_mode: str = "dist"
    pair_interaction: str = "diff"
    # integer | embedding | onehot
    # Default integer + zscore matched the strongest LL37-holdout CV in ablations;
    # AA ids are always sequential 1..20 (see utils.constants). Prefer embedding
    # when you need a categorically valid sequence branch (uses positional encoding).
    seq_encoding: str = "integer"
    # legacy_x10 | zscore  (zscore: per-fold train stats; empirically strongest)
    p_feature_scaling: str = "zscore"
    # Append helical(100°)/beta(160°) normalized hydrophobic moments (Eisenberg
    # scale) to p_f: +2 dims. Amphipathic-face descriptors that bulk GRAVY
    # cannot represent. Default off — all existing runs unchanged.
    p_feature_hmom: bool = False
    # Pair-head hydrophobic-moment delta: attach raw [muH_100, muH_160] to each
    # graph and concatenate (hmom_a - hmom_b) directly to the pair head input.
    # Signed per-pair features on the same scale as the regression target
    # (delta_log2), unlike absolute z-scored dims (see ablation A2 vs A2p).
    pair_hmom_delta: bool = False
    # CV fold grouping: "sequence" (default, GroupKFold by exact sequence) or
    # "family" (groups = NW>=0.7 connected clusters; whole families stay in one
    # fold so checkpoint selection measures cross-family generalization, matching
    # the sequence-disjoint external test regime). See utils/family_cluster.py.
    fold_grouping: str = "sequence"
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
    # Explicitly disable only the TRAINING similarity gate. A threshold of
    # zero still excludes negative NW scores and is not a disabled gate.
    train_similarity_gate_enabled: bool = True
    # None retains the historical coupled setting for old configurations.
    # The v15 campaign explicitly fixes validation to 0.7 in training-gate arms.
    validation_similarity_threshold: float | None = None
    neighbor_top_k: int = 50
    fallback_top_k: int = 20
    # If False, val/test neighbor graphs skip the distant fallback (empty
    # neighbor list → that query is omitted from pair-Δ checkpoint selection).
    use_neighbor_fallback: bool = True
    max_candidate_per_anchor: int = 2500
    max_train_pairs: int = 250000
    pair_balance_num: int = 1000
    delta_bin_width: float = 0.50
    use_signed_sampling: bool = False
    use_similarity_strata: bool = False
    # Optional per-sequence exemption for training-pair gating: sequences listed in
    # ``pair_sim_exempt_csv`` (single ``sequence`` column) are paired against the pool
    # at ``pair_sim_exempt_threshold`` instead of ``similarity_threshold``. Length
    # +-5 gating and balanced sampling are unchanged. ``None`` disables (default),
    # keeping the behavior of all existing runs identical.
    pair_sim_exempt_threshold: float | None = None
    pair_sim_exempt_csv: str | None = None
    # Optional directed top-K injection for training pairs: sequences listed in
    # ``pair_direct_topk_csv`` (single ``sequence`` column) are paired, ignoring the
    # similarity gate and the balance quota, with their ``pair_direct_topk`` most
    # similar pool peptides inside the length +-5 window, once as anchor and once
    # as query (2K directed pairs per listed peptide per fold). ``None`` disables.
    pair_direct_topk: int | None = None
    pair_direct_topk_csv: str | None = None
    # Optional pair-pool restriction: training pairs are only built among rows
    # whose sequence appears in ``pair_pool_csv`` (single ``sequence`` column).
    # Rows outside the pool still enter the graph cache, fold z-score stats and
    # encoder pretraining — only the ΔMIC pair-construction universe shrinks.
    # Used by v2-3b (S4v2_pool: increment sequences pretrain but never pair).
    # ``None`` disables (default), keeping the behavior of all existing runs.
    pair_pool_csv: str | None = None
    # Optional "new-row" composition levers (v2-3c). ``pair_new_ref_csv`` (single
    # ``sequence`` column) defines the REFERENCE (old) sequences; rows outside it
    # are "new". ``pair_new_cfu_same_only=true`` bans pairs where exactly one
    # endpoint is new AND the two rows' CFU groups differ (new rows may only pair
    # within their own — unknown — protocol group). ``pair_new_min_sim`` raises
    # the similarity gate for any pair with a new endpoint. Old×old pairing is
    # untouched by both. All default to None/False (existing runs unchanged).
    pair_new_ref_csv: str | None = None
    pair_new_cfu_same_only: bool = False
    pair_new_min_sim: float | None = None
    # v2-3d intra-publication lever: rows listed in ``pair_new_pub_csv`` (columns
    # sequence, pub — one row per sequence×publication, multi-pub sequences get
    # several rows) are "new". With ``pair_new_intra_pub_only`` a new row only
    # pairs with rows sharing >=1 publication (pub-set intersection) and never
    # with unlisted (old) rows. With ``pair_new_quota_separate`` new-involving
    # pairs are booked in a SECOND per-bin ledger (same ``pair_balance_num``),
    # so old pairs keep the reference regime pair-for-pair (two-phase anchor
    # scheduling: old pass reproduces the old-only rng stream exactly). All
    # default off — existing runs unchanged.
    pair_new_intra_pub_only: bool = False
    pair_new_pub_csv: str | None = None
    pair_new_quota_separate: bool = False
    # --- validation-constitution axis (what signal picks each fold's
    # checkpoint: interpolation vs extrapolation) ---
    # Upstream switch for BOTH regimes: fold_grouping (sequence | family, see
    # above) decides whether the validation fold is near-neighbor-interpolative
    # or novel-family-extrapolative by construction.
    # pair regime — val_pair_mode: "neighbor" (cross-fold >=0.7 neighbor pairs
    # + distant fallback; default) | "within_fold_strict" (val-fold pairs at
    # strict >=0.7) | "within_fold_family" (same-family val pairs; requires
    # fold_grouping=family, capped by val_pair_cap).
    # single regime — val_far_only (v2-6 intervention B): when true, checkpoint
    # selection is scored only on validation peptides WITHOUT
    # >=similarity_threshold neighbors in the training fold (extrapolative
    # selection). Falls back to the full fold when fewer than 50 such peptides
    # exist. Label quality of the far subset decides the sign of the effect
    # (v2-6b: curated old set +0.020 p=.014; noisy v2 set -0.023 p=.031).
    # Default False.
    val_far_only: bool = False
    # Training-time antisymmetry constraint for the pair head. "none" (default)
    # keeps the plain forward f(a,b); "symmetric" trains on the antisymmetrized
    # prediction 0.5*(f(a,b) - f(b,a)) so that delta(a,b) = -delta(b,a) holds by
    # construction (doubles the per-batch forward cost). Evaluation protocols
    # are unchanged (single-direction) either way.
    antisym_loss: str = "none"

    # --- non-AMP negatives + two-stage recipe (N1/N2 ablation; defaults off) ---
    # Sampled UniProt non-AMP pool (sequence[,length] CSV). When set, a separate
    # negative graph cache is built (never merged into the main graphs/rows, so
    # folds/validation/OOF/calibrator stay AMP-only by construction).
    neg_csv: str | None = None
    neg_pdb_dir: str | None = None
    # N1: inject negatives into the main objective (directed AMP-neg pairs for the
    # pair head, pseudo-label rows for the single head). N2 only needs neg_csv
    # without this flag (negatives used solely by the classification pretrain).
    neg_train_inject: bool = False
    # log2(MIC ug/mL) pseudo label for negatives = log2(1e4); above 99.93% of
    # training AMP labels (max observed 14.20). DeepAMP/Pandi anchoring convention.
    neg_cap_log2: float = 13.287712379549449
    # Per negative: pair with its ``neg_pair_topk`` most-NW-similar FOLD-TRAIN AMPs
    # (within +-5 length), both directions, capped at ``neg_max_pairs``.
    neg_pair_topk: int = 1
    neg_max_pairs: int = 4000
    # N2: AMP-vs-neg classification pretraining of the shared encoder (BCE via a
    # temporary Linear head) before the normal objective. 0 = off.
    cls_pretrain_epochs: int = 0
    cls_lr: float | None = None  # defaults to cfg.lr when None
    # Anchor multitask (pair arms): weight lambda of the auxiliary absolute-score
    # loss on BOTH endpoints of each training pair (shared-encoder score head).
    # 0 = off. anchor_neg_rows additionally feeds the negative pool as standalone
    # score-only rows (the single+N1 mechanism ported to the pair model; the
    # negatives never enter the pair-delta loss).
    anchor_multitask: float = 0.0
    anchor_neg_rows: bool = False
    # 方案 B: pretrain the shared encoder on the SINGLE objective (per-peptide
    # log2MIC MSE incl. cap-anchored negative rows) before pair fine-tuning.
    single_pretrain_epochs: int = 0
    # 实验一: replicate the single+N1 formal regime during pretraining (cosine
    # schedule + early stopping on fold-val peptide MSE + best-encoder keeping)
    # instead of a fixed-epoch plain loop.
    single_pretrain_full: bool = False
    # Negative-pair construction mode: "topk" (default; NW-most-similar fold-train
    # AMP per negative, bypassing the similarity gate), "joint" (experiment 12:
    # every fold-train AMP whose NATURAL NW similarity passes the gate within
    # +-5 length — the AMP-AMP pairing protocol applied verbatim to negatives;
    # no neg-neg pairs), or "parent" (constructed analog negatives pair with
    # their recorded parent peptide ONLY, both directions; parents sitting in
    # the validation fold are skipped).
    neg_pair_mode: str = "topk"
    # Joint-mode gate override: NW threshold for neg-AMP pairing. None = use
    # cfg.similarity_threshold (0.7 in the A0 regime). Experiment 12 uses 0.5
    # for the dose/similarity intermediate-band arm. AMP-AMP pairing is NEVER
    # affected by this field.
    neg_pair_gate: float | None = None
    # Checkpoint-selection validation pairs (experiment 16/17). "neighbor" = the
    # default cross-fold neighbor protocol (anchor = training-fold peptide,
    # cfg.similarity_threshold gate). "within_fold_family" requires
    # fold_grouping="family": validation pairs are built INSIDE the validation
    # fold between same-cluster (family) peptides, both endpoints held out —
    # matching the external within-family pairing regime. "within_fold_strict"
    # (experiment 17) works with EITHER fold grouping: pairs are built inside
    # the validation fold with the strict +-5 length window + NW >= threshold
    # pairwise filter (no clustering). Only the checkpoint selection changes;
    # OOF/calibrator keep the neighbor construction.
    val_pair_mode: str = "neighbor"
    # Per-fold cap on within-fold validation pairs (balanced across clusters).
    val_pair_cap: int = 4000

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
