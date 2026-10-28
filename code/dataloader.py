"""Data loading: CSV preprocess, graph construction, pair sampling, datasets."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa

try:
    from Bio.PDB.Polypeptide import three_to_one
except Exception:  # pragma: no cover
    from Bio.SeqUtils import seq1 as three_to_one
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx

from .utils.config import Config
from .utils.constants import CFU_COL, CFU_LEVELS, TARGET_COL, VALID_RE, encode_sequence
from .utils.features import calculate_property, hydrophobic_moments, normalize_cfu_group, tabular_features


class PeptideGraph(Data):
    def __init__(self, edge_index_s=None, x_s=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def parse_value(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x).replace(",", ""))
    return float(m.group(0)) if m else np.nan


def _target_col(cfg: Config) -> str:
    return str(getattr(cfg, "target_col", None) or TARGET_COL)


def preprocess_dataset(csv_path: str, pdb_dir: str, cfg: Config, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding=cfg.csv_encoding) if cfg.csv_encoding else pd.read_csv(csv_path)
    pdb_names = {p.stem for p in Path(pdb_dir).glob("*.pdb")}
    tcol = _target_col(cfg)
    rows = []
    counters: Counter[str] = Counter()
    for _, row in df.iterrows():
        raw = "" if pd.isna(row.get("sequence")) else str(row.get("sequence")).strip()
        seq = raw.upper()
        mic = parse_value(row.get(tcol, np.nan))
        if np.isfinite(mic) and mic <= 0:
            mic = np.nan
        if not np.isfinite(mic):
            counters["missing_or_nonpositive_mic"] += 1
            continue
        if not VALID_RE.fullmatch(seq):
            counters["invalid_symbols"] += 1
            continue
        if not (cfg.min_len <= len(seq) <= cfg.max_len):
            counters["length_outside_range"] += 1
            continue
        if seq not in pdb_names:
            counters["missing_pdb_file"] += 1
            continue
        rows.append(
            {
                "sequence": seq,
                "original_all_upper": bool(raw) and raw == raw.upper(),
                "n_terminus": row.get("n_terminus", ""),
                "c_terminus": row.get("c_terminus", ""),
                # Always persist under TARGET_COL so downstream code stays stable;
                # also keep the source column name when it differs.
                TARGET_COL: mic,
                tcol: mic,
                CFU_COL: normalize_cfu_group(row.get(CFU_COL, np.nan)),
                "pdb_path": str(Path(pdb_dir) / f"{seq}.pdb"),
            }
        )
    filt = pd.DataFrame(rows)
    stats: dict[str, Any] = {
        "split": split,
        "raw_rows": int(len(df)),
        "pdb_files": int(len(pdb_names)),
        "target_col": tcol,
        **{f"removed_{k}": int(v) for k, v in counters.items()},
        "rows_before_dedup": int(len(filt)),
    }
    if filt.empty:
        stats["rows_after_dedup"] = 0
        return filt, stats

    def pick_duplicate(g: pd.DataFrame) -> pd.Series:
        upper = g[g["original_all_upper"]]
        cand = upper if not upper.empty else g
        return cand.iloc[int(np.argmin(cand[TARGET_COL].fillna(np.inf).to_numpy()))]

    dedup_keys = ["sequence", CFU_COL] if cfg.use_cfu_protocol else ["sequence"]
    dedup_rows = [pick_duplicate(group) for _, group in filt.groupby(dedup_keys, sort=False, dropna=False)]
    dedup = pd.DataFrame(dedup_rows).reset_index(drop=True)
    stats["duplicate_rows_removed"] = int(len(filt) - len(dedup))
    stats["rows_after_dedup"] = int(len(dedup))
    return dedup, stats


def load_aa_features(feature_path: str) -> dict[str, list[float]]:
    out = {}
    with open(feature_path) as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                out[parts[0]] = [float(v) for v in parts[1:]]
    return out


def read_rosetta_energies(pdb: str, expected_len: int, energy_dim: int = 20) -> list[list[float]]:
    scoring = False
    profile = []
    with open(pdb) as handle:
        for line in handle:
            if line.startswith("#END_POSE_ENERGIES_TABLE"):
                scoring = False
            if scoring:
                label = line.split()[0]
                if label not in {"label", "weights", "pose"}:
                    profile.append([float(v) for v in line.split()[1 : 1 + energy_dim]])
            if line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
                scoring = True
    if len(profile) < expected_len:
        profile.extend([[0.0] * energy_dim for _ in range(expected_len - len(profile))])
    return profile[:expected_len]


def construct_graph(row: Any, aa_features: dict[str, list[float]], cfg: Config) -> PeptideGraph | None:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(row.sequence, row.pdb_path)
    except Exception:
        return None
    if "A" not in structure[0]:
        return None
    graph = nx.Graph()
    node_aa_ids: list[int] = []
    for res in structure[0]["A"]:
        if is_aa(res.get_resname(), standard=True) and "CA" in res:
            aa = three_to_one(res.get_resname())
            if aa not in aa_features:
                continue
            x, y, z = res["CA"].get_coord()
            node_aa_ids.append(1 + "ACDEFGHIKLMNPQRSTVWY".index(aa))
            graph.add_node(graph.number_of_nodes(), aa=aa, position=(float(x), float(y), float(z)))
    if graph.number_of_nodes() == 0:
        return None
    coords = np.asarray([graph.nodes[i]["position"] for i in graph.nodes], dtype=float)
    edge_mode = str(getattr(cfg, "edge_mode", "dist") or "dist").lower()
    if edge_mode == "chain":
        # [v2-10 patch] edge-geometry ablation: sequential-adjacency edges
        # replace the 3D 5 A proximity graph — identical node features and
        # message passing, geometry stripped from the graph channel.
        for i in range(len(coords) - 1):
            graph.add_edge(i, i + 1, weight=1.0)
    else:
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                if dist <= 5.0:
                    graph.add_edge(i, j, weight=5.0 / max(dist, 1e-6))
    if graph.number_of_edges() == 0 and graph.number_of_nodes() > 1:
        for i in range(graph.number_of_nodes() - 1):
            graph.add_edge(i, i + 1, weight=1.0)
    energies = read_rosetta_energies(row.pdb_path, graph.number_of_nodes(), energy_dim=20)
    use_coords = bool(getattr(cfg, "include_node_coords", False))
    for node_id, node_data in graph.nodes(data=True):
        feats = aa_features[node_data["aa"]] + energies[node_id]
        if use_coords:
            feats = feats + list(node_data["position"])
        node_data["x"] = feats
    data_wt = from_networkx(graph)
    data = PeptideGraph(data_wt.edge_index, data_wt.x.float())
    # Sequential AA codes 1..20 (0=pad)
    seq_emb = encode_sequence(row.sequence, cfg.max_len)
    data.seq = torch.tensor([seq_emb], dtype=torch.float32)
    props = calculate_property(
        row.sequence, row.n_terminus, row.c_terminus, scaling=cfg.p_feature_scaling
    )
    if bool(getattr(cfg, "p_feature_hmom", False)):
        props = props + hydrophobic_moments(row.sequence)
    data.p_f = torch.tensor([props], dtype=torch.float32)
    if bool(getattr(cfg, "pair_hmom_delta", False)):
        # Raw per-peptide moments for the pair head's signed delta features.
        data.hmom = torch.tensor([hydrophobic_moments(row.sequence)], dtype=torch.float32)
    data.tab_f = torch.from_numpy(np.asarray(row.tab_f, dtype=np.float32).reshape(1, -1))
    data.node_aa = torch.tensor(node_aa_ids, dtype=torch.long)
    data.cfu_id = torch.tensor([CFU_LEVELS[normalize_cfu_group(getattr(row, CFU_COL, "unknown"))]], dtype=torch.long)
    data.gt = float(row.label)
    data.sequence = row.sequence
    return data


def build_graphs(df: pd.DataFrame, cfg: Config) -> tuple[list[PeptideGraph], pd.DataFrame, dict[str, Any]]:
    target_df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    target_df["label"] = np.log2(target_df[TARGET_COL].astype(float))
    prop_scale = "raw" if str(cfg.p_feature_scaling).lower() == "zscore" else "legacy_x10"
    tabs = tabular_features(target_df, cfg.use_cfu_feature, property_scaling=prop_scale)
    target_df["tab_f"] = [row for row in tabs]
    aa_features = load_aa_features(cfg.feature_path)
    graphs, kept, failed = [], [], 0
    for row in target_df.itertuples():
        graph = construct_graph(row, aa_features, cfg)
        if graph is None:
            failed += 1
            continue
        graphs.append(graph)
        kept.append(row._asdict())
    return graphs, pd.DataFrame(kept), {"input_rows": int(len(target_df)), "graph_rows": int(len(graphs)), "graph_failed_rows": int(failed)}


def make_batch(data_list: list[PeptideGraph]) -> Batch:
    batch = Batch.from_data_list(list(data_list))
    batch.x_s_batch = torch.cat([torch.full((g.x_s.size(0),), i, dtype=torch.long) for i, g in enumerate(data_list)])
    return batch


def pair_collate(batch):
    """Collate (a, b, y, teacher[, y_a, y_b]) items; the endpoint labels are
    optional so legacy 4-tuple callers (evaluation.py, eval scripts) keep working."""
    cols = list(zip(*batch))
    out = [make_batch(list(cols[0])), make_batch(list(cols[1])),
           torch.stack(cols[2]), torch.stack(cols[3])]
    if len(batch[0]) >= 6:
        out += [torch.stack(cols[4]), torch.stack(cols[5])]
    return tuple(out)


def single_collate(items):
    graphs, labels = zip(*items)
    return make_batch(list(graphs)), torch.stack(labels)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels, pairs, teacher_pred=None):
        self.graphs = graphs
        self.labels = labels
        self.pairs = pairs
        self.teacher_pred = teacher_pred

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        y_a, y_b = float(self.labels[a]), float(self.labels[b])
        y = y_a - y_b
        teacher = np.nan if self.teacher_pred is None else float(self.teacher_pred[a] - self.teacher_pred[b])
        # Endpoint labels feed the anchor-multitask score loss (train.py); the
        # pair-delta target y is unchanged.
        return (self.graphs[a], self.graphs[b],
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(teacher, dtype=torch.float32),
                torch.tensor(y_a, dtype=torch.float32),
                torch.tensor(y_b, dtype=torch.float32))


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels, indices):
        self.graphs, self.labels, self.indices = graphs, labels, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        return self.graphs[idx], torch.tensor(float(self.labels[idx]), dtype=torch.float32)


def setup_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    return aligner


def sequence_similarity(aligner: PairwiseAligner, a: str, b: str) -> float:
    """Length-normalized global alignment score (same contract as legacy pipeline)."""
    if not a or not b:
        return 0.0
    return float(aligner.score(a, b) / max(1, len(a)))


def validation_similarity_threshold(cfg: Config) -> float:
    value = getattr(cfg, "validation_similarity_threshold", None)
    return float(cfg.similarity_threshold if value is None else value)


def directed_pair_eligible(aligner, anchor: str, query: str, threshold: float, enabled: bool = True, score=None) -> bool:
    """Shared length-window and direction-specific NW gate, before sampling."""
    if abs(len(anchor)-len(query)) > 5:
        return False
    return not enabled or (sequence_similarity(aligner, anchor, query) if score is None else score) >= threshold


def similarity_stratum(sim: float) -> str:
    if sim >= 0.7:
        return "high"
    if sim >= 0.4:
        return "mid"
    return "low"


def signed_delta_bin(delta: float, width: float) -> int:
    return int(np.floor(delta / width))


def build_training_pairs(rows: pd.DataFrame, labels: np.ndarray, cfg: Config, seed: int) -> list[tuple[int, int]]:
    aligner = setup_aligner()
    rng = np.random.default_rng(seed)
    seqs = rows.sequence.tolist()
    # Optional per-sequence similarity-gate exemption (see Config.pair_sim_exempt_*):
    # listed sequences pair against the pool at the exempt threshold. None → disabled.
    exempt: set[str] = set()
    if getattr(cfg, "pair_sim_exempt_csv", None) and getattr(cfg, "pair_sim_exempt_threshold", None) is not None:
        exempt = set(pd.read_csv(cfg.pair_sim_exempt_csv)["sequence"].astype(str).str.upper())
    # Optional pair-pool restriction (see Config.pair_pool_csv): pairing happens
    # only among rows whose sequence is listed in the pool CSV. Out-of-pool rows
    # keep their indices (they remain valid PairDataset members via pretraining /
    # other channels) but never appear as anchor or query in any training pair.
    pool_seqs: set[str] | None = None
    if getattr(cfg, "pair_pool_csv", None):
        pool_seqs = set(pd.read_csv(cfg.pair_pool_csv)["sequence"].astype(str).str.upper())
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, seq in enumerate(seqs):
        if pool_seqs is not None and seq.upper() not in pool_seqs:
            continue
        by_len[len(seq)].append(i)
    # Optional directed top-K injection (see Config.pair_direct_topk*): listed
    # sequences are paired with their K most-similar pool peptides inside the
    # length +-5 window, once as anchor and once as query, bypassing the
    # similarity gate and the balance quota. Injected first so both return
    # paths below include them; pool sampling cannot duplicate these pairs
    # (pool gate 0.7 keeps WT-pool pairs out; WT-WT pairs share no length window).
    pairs: list[tuple[int, int]] = []
    if getattr(cfg, "pair_direct_topk_csv", None) and getattr(cfg, "pair_direct_topk", None):
        direct = set(pd.read_csv(cfg.pair_direct_topk_csv)["sequence"].astype(str).str.upper())
        for wi, wseq in enumerate(seqs):
            if wseq not in direct:
                continue
            if pool_seqs is not None and wseq.upper() not in pool_seqs:
                continue
            cands = [
                j
                for length in range(max(1, len(wseq) - 5), len(wseq) + 6)
                for j in by_len.get(length, [])
                if j != wi
            ]
            as_anchor = sorted(
                ((sequence_similarity(aligner, wseq, seqs[j]), j) for j in cands), reverse=True
            )[: cfg.pair_direct_topk]
            as_query = sorted(
                ((sequence_similarity(aligner, seqs[j], wseq), j) for j in cands), reverse=True
            )[: cfg.pair_direct_topk]
            pairs.extend((wi, j) for _, j in as_anchor)
            pairs.extend((j, wi) for _, j in as_query)
    counts: Counter = Counter()
    order = list(range(len(seqs)))
    if pool_seqs is not None:
        order = [i for i in order if seqs[i].upper() in pool_seqs]
    # NOTE: no shuffle here — each pass below shuffles its own anchor list once,
    # so the single-pass path consumes exactly one shuffle of `order` (identical
    # rng stream to the pre-v2-3d code; existing runs reproduce pair-for-pair).
    # v2-3c "new-row" levers (see Config.pair_new_ref_csv): flags for rows outside
    # the reference (old) CSV, plus their per-row CFU group for the cross-protocol
    # ban. Both levers leave old×old pairing untouched.
    new_flags: list[bool] | None = None
    if getattr(cfg, "pair_new_ref_csv", None) and (
        bool(getattr(cfg, "pair_new_cfu_same_only", False))
        or getattr(cfg, "pair_new_min_sim", None) is not None
    ):
        ref = set(pd.read_csv(cfg.pair_new_ref_csv)["sequence"].astype(str).str.upper())
        new_flags = [s.upper() not in ref for s in seqs]
    cfu_groups = [normalize_cfu_group(getattr(r, CFU_COL, "unknown")) for r in rows.itertuples()] \
        if new_flags is not None else None
    # v2-3d intra-publication lever (see Config.pair_new_pub_csv): rows listed in
    # the pub CSV are "new"; a new row only pairs with rows sharing >=1 publication
    # (set intersection), never with unlisted (old) rows. Old×old is untouched.
    pub_sets: dict[str, set[str]] | None = None
    if bool(getattr(cfg, "pair_new_intra_pub_only", False)) and getattr(cfg, "pair_new_pub_csv", None):
        _t = pd.read_csv(cfg.pair_new_pub_csv)
        pub_sets = {}
        for _s, _r in zip(_t["sequence"].astype(str).str.upper(), _t["pub"].astype(str)):
            pub_sets.setdefault(_s, set()).add(_r)

    def _pub_ok(i: int, j: int) -> bool:
        if pub_sets is None:
            return True
        ni, nj = seqs[i].upper() in pub_sets, seqs[j].upper() in pub_sets
        if ni != nj:
            return False
        return not (ni and pub_sets[seqs[i].upper()].isdisjoint(pub_sets[seqs[j].upper()]))

    # Two-phase anchor scheduling when the new-pair ledger is separate: phase 1
    # runs old anchors over an old-only bucket map (byte-identical rng stream to
    # the old-only regime → old pairs reproduce the reference run pair-for-pair);
    # phase 2 runs new anchors over their own bucket map. Single-pass otherwise
    # (rng behavior unchanged for all existing configs).
    passes: list[tuple[list[int], dict[int, list[int]]]] = []
    if pub_sets is not None:
        old_idx = [i for i in range(len(seqs)) if seqs[i].upper() not in pub_sets]
        new_idx = [i for i in range(len(seqs)) if seqs[i].upper() in pub_sets]
        if pool_seqs is not None:
            old_idx = [i for i in old_idx if seqs[i].upper() in pool_seqs]
            new_idx = [i for i in new_idx if seqs[i].upper() in pool_seqs]
        by_len_old: dict[int, list[int]] = defaultdict(list)
        by_len_new: dict[int, list[int]] = defaultdict(list)
        for i in old_idx:
            by_len_old[len(seqs[i])].append(i)
        for i in new_idx:
            by_len_new[len(seqs[i])].append(i)
        passes = [(old_idx, by_len_old), (new_idx, by_len_new)]
    else:
        passes = [(order, by_len)]
    counts_new: Counter = Counter()
    sep_ledger = bool(getattr(cfg, "pair_new_quota_separate", False))
    for pass_order, pass_by_len in passes:
        po = list(pass_order)
        rng.shuffle(po)
        for i in po:
            candidates = []
            for length in range(max(1, len(seqs[i]) - 5), len(seqs[i]) + 6):
                candidates.extend(pass_by_len.get(length, []))
            rng.shuffle(candidates)
            for j in candidates[: cfg.max_candidate_per_anchor]:
                if i == j:
                    continue
                if not _pub_ok(i, j):
                    continue
                # Gate = 0.3 when EITHER endpoint is an exempt sequence, else the normal
                # threshold. min() implements "any-endpoint exemption".
                thr_i = cfg.pair_sim_exempt_threshold if seqs[i] in exempt else cfg.similarity_threshold
                thr_j = cfg.pair_sim_exempt_threshold if seqs[j] in exempt else cfg.similarity_threshold
                if new_flags is not None and getattr(cfg, "pair_new_min_sim", None) is not None \
                        and (new_flags[i] or new_flags[j]):
                    nms = float(cfg.pair_new_min_sim)
                    thr_i, thr_j = max(thr_i, nms), max(thr_j, nms)
                gate_enabled = bool(getattr(cfg, "train_similarity_gate_enabled", True))
                sim = sequence_similarity(aligner, seqs[i], seqs[j]) if gate_enabled or cfg.use_similarity_strata else 0.0
                if not directed_pair_eligible(aligner, seqs[i], seqs[j], min(thr_i, thr_j),
                                              gate_enabled, score=sim):
                    continue
                # v2-3c cross-protocol ban: a new row never pairs outside its own
                # (unknown) CFU group; old×old and same-group pairs are unaffected.
                if new_flags is not None and bool(getattr(cfg, "pair_new_cfu_same_only", False)) \
                        and new_flags[i] != new_flags[j] and cfu_groups[i] != cfu_groups[j]:
                    continue
                delta = float(labels[i] - labels[j])
                if cfg.use_signed_sampling:
                    key: Any = signed_delta_bin(delta, cfg.delta_bin_width)
                else:
                    key = int(round(abs(delta) * 100))
                if cfg.use_similarity_strata:
                    key = (key, similarity_stratum(sim))
                ledger = counts
                if sep_ledger and pub_sets is not None \
                        and (seqs[i].upper() in pub_sets or seqs[j].upper() in pub_sets):
                    ledger = counts_new
                if ledger[key] >= cfg.pair_balance_num:
                    continue
                ledger[key] += 1
                pairs.append((i, j))
                if len(pairs) >= cfg.max_train_pairs:
                    rng.shuffle(pairs)
                    return pairs
    rng.shuffle(pairs)
    return pairs


def far_val_mask(tr_rows: pd.DataFrame, va_rows: pd.DataFrame, cfg: Config) -> np.ndarray:
    """v2-6 intervention B: boolean mask of validation rows that have NO
    >= similarity_threshold NW neighbor (within the length +-5 window) among
    the training rows — "extrapolative" validation peptides whose checkpoint-
    selection signal probes family-level generalization rather than near-
    neighbor interpolation. Training split and OOF/calibrator are unchanged."""
    aligner = setup_aligner()
    by_len: dict[int, list[str]] = defaultdict(list)
    for s in tr_rows.sequence:
        by_len[len(s)].append(s)
    out = []
    for s in va_rows.sequence:
        hit = False
        for length in range(max(1, len(s) - 5), len(s) + 6):
            for t in by_len.get(length, []):
                if sequence_similarity(aligner, s, t) >= validation_similarity_threshold(cfg):
                    hit = True
                    break
            if hit:
                break
        out.append(not hit)
    return np.asarray(out)


def precompute_neighbors(anchor_rows: pd.DataFrame, query_rows: pd.DataFrame, cfg: Config) -> list[list[tuple[int, float]]]:
    aligner = setup_aligner()
    anchors = list(anchor_rows.itertuples())
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(anchors):
        if cfg.use_cfu_protocol and normalize_cfu_group(getattr(row, CFU_COL, "unknown")) != cfg.primary_cfu_group:
            continue
        by_len[len(row.sequence)].append(i)
    out = []
    for qr in query_rows.itertuples():
        candidates = []
        for length in range(max(1, len(qr.sequence) - 5), len(qr.sequence) + 6):
            candidates.extend(by_len.get(length, []))
        sims = []
        for i in candidates:
            sim = sequence_similarity(aligner, qr.sequence, anchors[i].sequence)
            if sim >= validation_similarity_threshold(cfg):
                sims.append((i, sim))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[: cfg.neighbor_top_k]
        use_fallback = bool(getattr(cfg, "use_neighbor_fallback", True)) and int(getattr(cfg, "fallback_top_k", 0) or 0) > 0
        if not sims and use_fallback:
            fallback_candidates = candidates[:500] if candidates else list(range(min(len(anchors), 500)))
            fallback = [(i, sequence_similarity(aligner, qr.sequence, anchors[i].sequence)) for i in fallback_candidates]
            sims = sorted(fallback, key=lambda x: x[1], reverse=True)[: cfg.fallback_top_k]
        out.append(sims)
    return out


def ensure_node_aa(graphs: list, feature_path: str) -> None:
    """Attach data.node_aa (residue AA ids 1..20) to graphs built before the
    node_features_mode work, by exact-matching the RAW descriptor channels
    (x_s[:, :20], pre-scaling) against the handcrafted prior table. Idempotent."""
    if not graphs:
        return
    from .utils.features import aa_prior_table

    table = aa_prior_table(feature_path)[1:]  # 20 x 20
    for g in graphs:
        if hasattr(g, "node_aa"):
            continue
        raw = g.x_s[:, :20].float()
        dist = torch.cdist(raw.unsqueeze(0), table.unsqueeze(0)).squeeze(0)
        g.node_aa = dist.argmin(dim=1) + 1


def build_negative_rows(neg_csv: str, neg_pdb_dir: str, cfg: Config, root: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the non-AMP negative row frame (N1/N2 ablation).

    Negatives carry a pseudo MIC of 2**cfg.neg_cap_log2 (label = log2 of it via
    build_graphs), the primary CFU group, and no terminal modifications. They
    live in a SEPARATE cache (see load_or_build_neg_cache) and are never merged
    into the main graphs/rows, so folds/validation/OOF stay AMP-only.
    """
    path = Path(neg_csv)
    if not path.is_absolute() and root is not None:
        path = root / path
    pdb_dir = Path(neg_pdb_dir)
    if not pdb_dir.is_absolute() and root is not None:
        pdb_dir = root / pdb_dir
    df = pd.read_csv(path)
    has_parent = "parent" in df.columns
    pdb_names = {p.stem for p in pdb_dir.glob("*.pdb")}
    mic = float(2 ** float(cfg.neg_cap_log2))
    rows = []
    n_missing = 0
    for _, r in df.iterrows():
        seq = str(r.sequence).upper().strip()
        if not VALID_RE.fullmatch(seq):
            continue
        if not (cfg.min_len <= len(seq) <= cfg.max_len):
            continue
        if seq not in pdb_names:
            n_missing += 1
            continue
        row = {
            "sequence": seq,
            "n_terminus": "",
            "c_terminus": "",
            TARGET_COL: mic,
            CFU_COL: "1E5 - 1E6",
            "pdb_path": str(pdb_dir / f"{seq}.pdb"),
        }
        if has_parent:
            row["parent"] = str(r.parent).upper().strip()
        rows.append(row)
    out = pd.DataFrame(rows)
    stats = {"pool_rows": int(len(df)), "kept": int(len(out)), "missing_pdb": int(n_missing),
             "pseudo_mic": mic, "cap_log2": float(cfg.neg_cap_log2)}
    return out, stats


def build_within_fold_pairs(va_rows: pd.DataFrame, fam_map: dict[str, int],
                            cap: int, seed: int) -> pd.DataFrame:
    """Experiment 16 (val_pair_mode="within_fold_family"): directed validation
    pairs BETWEEN SAME-CLUSTER peptides inside the validation fold — both
    endpoints held out, matching the external within-family pairing regime.

    All bidirectional pairs within each cluster (>=2 rows), shuffled and
    sampled to ``cap`` with a fold-seeded rng (deterministic per fold).
    Returns a DataFrame with anchor_sequence / query_sequence columns, ready
    for evaluation.eval_explicit_pairs.
    """
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str]] = []
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for s in va_rows.sequence:
        by_cluster[fam_map.get(s, -1)].append(s)
    n_clusters = sum(1 for v in by_cluster.values() if len(v) >= 2)
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(len(members)):
                if i != j:
                    pairs.append((members[i], members[j]))
    rng.shuffle(pairs)
    pairs = pairs[:cap]
    print(f"[val-pairs/within_fold_family] clusters_with_pairs={n_clusters} "
          f"directed_pairs={len(pairs)} (cap={cap})", flush=True)
    return pd.DataFrame(pairs, columns=["anchor_sequence", "query_sequence"])


def build_within_fold_strict_pairs(va_rows: pd.DataFrame, cfg: Config,
                                   cap: int, seed: int) -> pd.DataFrame:
    """Experiment 17 (val_pair_mode="within_fold_strict"): directed validation
    pairs INSIDE the validation fold with the strict pairwise filter —
    +-5 length window + NW >= cfg.similarity_threshold — no clustering.
    Works with either fold grouping (sequence or family).
    """
    aligner = setup_aligner()
    rng = np.random.default_rng(seed)
    seqs = va_rows.sequence.tolist()
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(seqs):
        by_len[len(s)].append(i)
    thr = validation_similarity_threshold(cfg)
    pairs: list[tuple[str, str]] = []
    n_cand = 0
    for i, s in enumerate(seqs):
        for length in range(max(1, len(s) - 5), len(s) + 6):
            for j in by_len.get(length, []):
                if j == i:
                    continue
                n_cand += 1
                if sequence_similarity(aligner, s, seqs[j]) >= thr:
                    pairs.append((s, seqs[j]))
    rng.shuffle(pairs)
    pairs = pairs[:cap]
    print(f"[val-pairs/within_fold_strict] candidates={n_cand} "
          f"qualifying>= {thr}: directed_pairs={len(pairs)} (cap={cap})", flush=True)
    return pd.DataFrame(pairs, columns=["anchor_sequence", "query_sequence"])


def build_neg_pairs(tr_rows: pd.DataFrame, neg_rows: pd.DataFrame, cfg: Config) -> list[tuple[int, int]]:
    """Directed AMP<->negative training pairs (N1), indices local to the fold's
    combined graph list [fold-train AMPs ..., negatives ...].

    Each negative is paired with its ``neg_pair_topk`` most NW-similar fold-train
    AMPs inside the +-5 length window (deterministic, ties by index), once as
    anchor and once as query, capped at ``neg_max_pairs``. Targets are computed
    by PairDataset as label[a]-label[b] (AMP: true log2MIC, negative: cap), so
    (amp, neg) -> label_amp - cap and (neg, amp) -> cap - label_amp.
    """
    if len(neg_rows) == 0:
        return []
    offset = len(tr_rows)
    mode = str(getattr(cfg, "neg_pair_mode", "topk")).lower()
    # "parent" mode (constructed-analog negatives): each negative pairs with its
    # recorded parent ONLY, both directions. Parents sitting in this fold's
    # validation split are skipped (never train on val-fold labels).
    if mode == "parent" and "parent" in neg_rows.columns:
        seq_to_j = {s: j for j, s in enumerate(tr_rows.sequence)}
        pairs = []
        for k, parent in enumerate(neg_rows.parent):
            j = seq_to_j.get(parent)
            if j is None:
                continue
            pairs.append((j, offset + k))
            pairs.append((offset + k, j))
        return pairs[: int(cfg.neg_max_pairs)]
    aligner = setup_aligner()
    amp_seqs = tr_rows.sequence.tolist()
    by_len: dict[int, list[int]] = defaultdict(list)
    for j, s in enumerate(amp_seqs):
        by_len[len(s)].append(j)
    offset = len(amp_seqs)
    # Joint mode shares the directed length/similarity eligibility rule with
    # AMP-AMP pairs. It uses its own explicitly capped candidate ledger, not
    # the positive-pair signed-bin quota; no neg-neg pairs are constructed.
    # "topk" mode (default): top-``neg_pair_topk`` AMPs by similarity regardless
    # of the gate (bypassing it, as the N1 mechanism did).
    gate = mode == "joint"
    gate_thr = (float(cfg.neg_pair_gate) if getattr(cfg, "neg_pair_gate", None) is not None
                else float(getattr(cfg, "similarity_threshold", 0.7)))
    n_neg_with_pair = 0
    pairs: list[tuple[int, int]] = []
    for k, neg_seq in enumerate(neg_rows.sequence):
        candidates: list[int] = []
        for length in range(max(1, len(neg_seq) - 5), len(neg_seq) + 6):
            candidates.extend(by_len.get(length, []))
        if not candidates:
            continue
        ranked = sorted(
            ((sequence_similarity(aligner, neg_seq, amp_seqs[j]), j) for j in candidates),
            key=lambda x: (-x[0], x[1]),
        )
        take = [j for _, j in ranked] if gate else [j for _, j in ranked[:int(cfg.neg_pair_topk)]]
        before = len(pairs)
        for j in take:
            enabled = gate and bool(getattr(cfg, "train_similarity_gate_enabled", True))
            if directed_pair_eligible(aligner, amp_seqs[j], neg_seq, gate_thr, enabled):
                pairs.append((j, offset + k))
            if directed_pair_eligible(aligner, neg_seq, amp_seqs[j], gate_thr, enabled):
                pairs.append((offset + k, j))
            if len(pairs) >= int(cfg.neg_max_pairs):
                break
        n_neg_with_pair += int(len(pairs) > before)
        if len(pairs) >= int(cfg.neg_max_pairs):
            break
    if gate:
        print(f"[neg-pairs/joint] gate={gate_thr} negatives_with_amp_pair="
              f"{n_neg_with_pair}/{len(neg_rows)} directed_pairs={len(pairs[: int(cfg.neg_max_pairs)])}",
              flush=True)
    return pairs[: int(cfg.neg_max_pairs)]


def load_or_build_cache(cfg: Config, root: Path | None = None) -> dict[str, Any]:
    """Load/build a content-identified cache; never fall back to legacy caches."""
    from .utils.graph_cache import load_cached_graphs
    return load_cached_graphs(cfg, root, negative=False)


def load_or_build_neg_cache(cfg: Config, root: Path | None = None) -> dict[str, Any] | None:
    """Separate negative cache, with identical content/atomicity guarantees."""
    from .utils.graph_cache import load_cached_graphs
    return load_cached_graphs(cfg, root, negative=True)
