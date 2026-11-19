"""Recompute HC50 external and out-of-fold discrimination from frozen case scores."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
SEEDS = {1, 2, 3, 4, 42}
METRICS = ["auroc", "auprc_ap", "pr_trapezoid"]
SECONDARY_METRICS = ["mcc", "auroc", "ap", "recall", "specificity", "brier", "tp", "tn", "fp", "fn"]


def read(relative):
    return pd.read_csv(ROOT / relative, float_precision="round_trip")


def compare(actual, reference, keys, columns):
    def indexed(frame):
        return frame.set_index(keys, verify_integrity=True).sort_index()[columns]
    pd.testing.assert_frame_equal(indexed(actual), indexed(reference), check_exact=False,
                                  check_dtype=False, rtol=0, atol=1e-12)


def summary(frame, keys, measures):
    groups = frame.groupby(keys)
    if not groups.seed.agg(lambda s: len(s) == 5 and set(s) == SEEDS).all():
        raise ValueError("Expected one result from each of the five seeds")
    table = groups[measures].agg(["mean", "std"])
    table.columns = ["_".join(parts) for parts in table.columns]
    return table.reset_index()


def measure(group, method):
    y = group.y1024.to_numpy()
    scores = group[method].to_numpy()
    precision, recall, _ = precision_recall_curve(y, scores)
    return {"n": len(y), "n_positive": int(y.sum()), "auroc": roc_auc_score(y, scores),
            "auprc_ap": average_precision_score(y, scores), "pr_trapezoid": auc(recall, precision)}


def validate_scores(frame, methods, arms, cohort_counts):
    meta = {"case_id", "y1024", "label_evidence", "arm", "seed"}
    if set(frame) != meta | set(methods) or set(frame.arm) != arms:
        raise ValueError("Unexpected score schema or training arms")
    if frame.duplicated(["case_id", "arm", "seed"]).any():
        raise ValueError("Duplicate model score for a case")
    scores = frame[methods].to_numpy()
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Expected finite probabilities in [0, 1]")
    labels = frame[["case_id", "y1024", "label_evidence"]].drop_duplicates()
    if not labels.case_id.is_unique or labels.groupby(["y1024", "label_evidence"]).size().to_dict() != cohort_counts:
        raise ValueError("Inconsistent case metadata")
    for arm in arms:
        arm_data = frame[frame.arm == arm]
        if set(arm_data.seed) != SEEDS or not arm_data.groupby("seed").size().eq(len(labels)).all():
            raise ValueError(f"Incomplete scores for {arm}")


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        if sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch for {relative}")
    reference = read("benchmark/per_seed.csv")
    methods = sorted(reference.method.unique())
    if len(methods) != 17:
        raise ValueError("Expected 17 internal methods")
    measured, background = "measured_HC50", "uniprot_presumed_negative"
    rows = []
    for file, arms, counts in [
        ("external", {"augmented", "measured_training"}, {(1, measured):159, (0, measured):37, (0, background):122}),
        ("validation", {"augmented"}, {(1, measured):1048, (0, measured):133, (0, background):915}),
    ]:
        frame = read(f"scores/{file}.csv")
        validate_scores(frame, methods, arms, counts)
        for (arm, seed), block in frame.groupby(["arm", "seed"]):
            real = block.label_evidence == measured
            if file == "external":
                populations = {"balanced_318": block, "measured_196": block[real],
                               "measured_positive_vs_uniprot": block[(block.y1024 == 1) | ~real]}
            else:
                populations = {"balanced_oof": block, "measured_oof": block[real]}
            for population, group in populations.items():
                for method in methods:
                    rows.append(dict(arm=arm, seed=seed, population=population, method=method,
                                     **measure(group, method)))
    per_seed = pd.DataFrame(rows)
    keys = ["arm", "population", "method"]
    compare(per_seed, reference, [*keys, "seed"], ["n", "n_positive", *METRICS])
    totals = summary(per_seed, keys, METRICS)
    compare(totals, read("benchmark/summary.csv"), keys, totals.columns.difference(keys).tolist())
    differences = read("benchmark/training_differences.csv")
    deltas = []
    for metric in ["auroc", "auprc_ap"]:
        panel = totals[totals.population.isin(["balanced_318", "measured_196"])]
        paired = panel.pivot(index=["population", "method"], columns="arm", values=f"{metric}_mean")
        delta = (paired.augmented - paired.measured_training).rename("delta_mean").reset_index()
        deltas.append(delta.assign(metric=metric))
    compare(pd.concat(deltas), differences, ["population", "method", "metric"], ["delta_mean"])

    for prefix, measures in [("", SECONDARY_METRICS), ("s15_", ["mcc", "auroc", "ap"])]:
        keys = ["population", "scenario", "method"]
        ablation = summary(read(f"ablation/{prefix}per_seed.csv"), keys, measures)
        compare(ablation, read(f"ablation/{prefix}summary.csv"), keys, ablation.columns.difference(keys).tolist())
    print(f"Verified {len(per_seed)} seed-level rows, {len(totals)} benchmark summaries, "
          f"{len(differences)} paired point differences and both ablation summary tables.")


if __name__ == "__main__":
    main()
