"""Family clustering for fold grouping (Config.fold_grouping="family").

Greedy representative clustering: sequences are sorted by length and assigned to
the first representative within the +-5 length window whose symmetric NW
similarity >= threshold; otherwise they become a new representative. This keeps
pair-connected families (everything the pair sampler can link at the training
similarity gate) inside a single CV fold, so checkpoint selection scores the
cross-family regime the external tests actually use.

Clustering is cached per threshold in artifacts/family_clusters/ as JSON
{threshold: {sequence: cluster_id}}; unknown sequences are merged into the
cached map on first use.
"""
from __future__ import annotations

import json
from pathlib import Path

from Bio.Align import PairwiseAligner

_CACHE_DIR = Path("artifacts/family_clusters")


def _setup_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    return aligner


def cluster_families(sequences: list[str], threshold: float = 0.7, cache: bool = True) -> dict[str, int]:
    """Return {sequence: cluster_id}. Deterministic given input order+lengths."""
    seqs = list(dict.fromkeys(sequences))
    cache_file = _CACHE_DIR / f"clusters_thr{str(threshold).replace('.', 'p')}.json"
    cached: dict[str, int] = {}
    if cache and cache_file.exists():
        cached = {k: int(v) for k, v in json.loads(cache_file.read_text()).items()}
    todo = [s for s in seqs if s not in cached]
    if todo:
        aligner = _setup_aligner()
        reps: list[str] = []
        # Deterministic order: shorter first, then lexicographic — representatives
        # stay stable across invocations with overlapping inputs.
        todo.sort(key=lambda s: (len(s), s))
        for s in todo:
            hit = -1
            for ri, r in enumerate(reps):
                if abs(len(r) - len(s)) > 5:
                    continue
                score = aligner.score(s, r)
                if score / max(len(s), len(r)) >= threshold:
                    hit = ri
                    break
            if hit < 0:
                reps.append(s)
                cached[s] = len(reps) - 1
            else:
                cached[s] = cached[reps[hit]]
        if cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached))
    return {s: cached[s] for s in sequences}
