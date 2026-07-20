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
from .utils.features import calculate_property, normalize_cfu_group, tabular_features


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


def preprocess_dataset(csv_path: str, pdb_dir: str, cfg: Config, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding=cfg.csv_encoding) if cfg.csv_encoding else pd.read_csv(csv_path)
    pdb_names = {p.stem for p in Path(pdb_dir).glob("*.pdb")}
    rows = []
    counters: Counter[str] = Counter()
    for _, row in df.iterrows():
        raw = "" if pd.isna(row.get("sequence")) else str(row.get("sequence")).strip()
        seq = raw.upper()
        mic = parse_value(row.get(TARGET_COL, np.nan))
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
                TARGET_COL: mic,
                CFU_COL: normalize_cfu_group(row.get(CFU_COL, np.nan)),
                "pdb_path": str(Path(pdb_dir) / f"{seq}.pdb"),
            }
        )
    filt = pd.DataFrame(rows)
    stats: dict[str, Any] = {
        "split": split,
        "raw_rows": int(len(df)),
        "pdb_files": int(len(pdb_names)),
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
    for res in structure[0]["A"]:
        if is_aa(res.get_resname(), standard=True) and "CA" in res:
            aa = three_to_one(res.get_resname())
            if aa not in aa_features:
                continue
            x, y, z = res["CA"].get_coord()
            graph.add_node(graph.number_of_nodes(), aa=aa, position=(float(x), float(y), float(z)))
    if graph.number_of_nodes() == 0:
        return None
    coords = np.asarray([graph.nodes[i]["position"] for i in graph.nodes], dtype=float)
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
        row.sequence, row.n_terminus, row.c_terminus, scaling=cfg.global_feature_scaling
    )
    data.global_f = torch.tensor([props], dtype=torch.float32)
    data.tab_f = torch.from_numpy(np.asarray(row.tab_f, dtype=np.float32).reshape(1, -1))
    data.cfu_id = torch.tensor([CFU_LEVELS[normalize_cfu_group(getattr(row, CFU_COL, "unknown"))]], dtype=torch.long)
    data.gt = float(row.label)
    data.sequence = row.sequence
    return data


def build_graphs(df: pd.DataFrame, cfg: Config) -> tuple[list[PeptideGraph], pd.DataFrame, dict[str, Any]]:
    target_df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    target_df["label"] = np.log2(target_df[TARGET_COL].astype(float))
    prop_scale = "raw" if str(cfg.global_feature_scaling).lower() == "zscore" else "legacy_x10"
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
    a, b, y, teacher = zip(*batch)
    return make_batch(list(a)), make_batch(list(b)), torch.stack(y), torch.stack(teacher)


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
        y = float(self.labels[a] - self.labels[b])
        teacher = np.nan if self.teacher_pred is None else float(self.teacher_pred[a] - self.teacher_pred[b])
        return self.graphs[a], self.graphs[b], torch.tensor(y, dtype=torch.float32), torch.tensor(teacher, dtype=torch.float32)


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
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, seq in enumerate(seqs):
        by_len[len(seq)].append(i)
    pairs: list[tuple[int, int]] = []
    counts: Counter = Counter()
    order = list(range(len(seqs)))
    rng.shuffle(order)
    for i in order:
        candidates = []
        for length in range(max(1, len(seqs[i]) - 5), len(seqs[i]) + 6):
            candidates.extend(by_len.get(length, []))
        rng.shuffle(candidates)
        for j in candidates[: cfg.max_candidate_per_anchor]:
            if i == j:
                continue
            sim = sequence_similarity(aligner, seqs[i], seqs[j])
            if sim < cfg.similarity_threshold:
                continue
            delta = float(labels[i] - labels[j])
            if cfg.use_signed_sampling:
                key: Any = signed_delta_bin(delta, cfg.delta_bin_width)
            else:
                key = int(round(abs(delta) * 100))
            if cfg.use_similarity_strata:
                key = (key, similarity_stratum(sim))
            if counts[key] >= cfg.pair_balance_num:
                continue
            counts[key] += 1
            pairs.append((i, j))
            if len(pairs) >= cfg.max_train_pairs:
                rng.shuffle(pairs)
                return pairs
    rng.shuffle(pairs)
    return pairs


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
            if sim >= cfg.similarity_threshold:
                sims.append((i, sim))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[: cfg.neighbor_top_k]
        if not sims:
            fallback_candidates = candidates[:500] if candidates else list(range(min(len(anchors), 500)))
            fallback = [(i, sequence_similarity(aligner, qr.sequence, anchors[i].sequence)) for i in fallback_candidates]
            sims = sorted(fallback, key=lambda x: x[1], reverse=True)[: cfg.fallback_top_k]
        out.append(sims)
    return out


def load_or_build_cache(cfg: Config, root: Path | None = None) -> dict[str, Any]:
    """Load peptide graphs from disk cache or build from CSVs."""
    def _p(path: str) -> str:
        p = Path(path)
        if p.is_absolute() or root is None:
            return str(p)
        return str(root / p)

    scale_tag = "rawprops" if str(cfg.global_feature_scaling).lower() == "zscore" else "legacyprops"
    coord_tag = "withcoord" if bool(getattr(cfg, "include_node_coords", False)) else "nocoord"
    # Tag includes aa_v2 so sequential 1..20 encoding does not collide with legacy AMAS caches.
    # Node-feature zscore is applied at train time (not baked into cache).
    cache_tag = (
        ("cfu_v1_trainonly" if not cfg.eval_external else "cfu_v1")
        + f"_{scale_tag}_aa_seq1to20_{coord_tag}"
    )
    cache_dir = Path(_p(cfg.shared_cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mic_graph_cache_len{cfg.max_len}_seed{cfg.seed}_{cache_tag}.pt"

    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    train_df, train_stats = preprocess_dataset(_p(cfg.train_csv), _p(cfg.train_pdb_dir), cfg, "train")
    graphs, rows, gs = build_graphs(train_df, cfg)
    train_stats["graph_stats"] = gs
    if cfg.eval_external:
        test_df, test_stats = preprocess_dataset(_p(cfg.test_csv), _p(cfg.test_pdb_dir), cfg, "test")
        test_graphs, test_rows, tgs = build_graphs(test_df, cfg)
        test_stats["graph_stats"] = tgs
    else:
        test_graphs, test_rows, test_stats = [], pd.DataFrame(), {}
    payload = {
        "graphs": graphs,
        "rows": rows,
        "train_stats": train_stats,
        "test_graphs": test_graphs,
        "test_rows": test_rows,
        "test_stats": test_stats,
        "aa_encoding": "sequential_1_to_20_alphabetical",
    }
    torch.save(payload, cache_path)
    return payload
