from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score


@dataclass(frozen=True)
class EvalResult:
    auc: float
    ap: float
    acc: float
    threshold: float

    def __str__(self) -> str:
        return f"AUC={self.auc:.4f}  AP={self.ap:.4f}  ACC={self.acc:.4f}  threshold={self.threshold:.2f}"


def _best_threshold(labels: list[int], scores: list[float]) -> float:
    thresholds = np.linspace(0, 1, 201)
    return float(max(thresholds, key=lambda t: accuracy_score(labels, [s >= t for s in scores])))


def evaluate(labels: list[int], scores: list[float]) -> EvalResult:
    if len(set(labels)) < 2:
        raise ValueError("Need both real and fake samples to evaluate.")
    t = _best_threshold(labels, scores)
    return EvalResult(
        auc=roc_auc_score(labels, scores),
        ap=average_precision_score(labels, scores),
        acc=accuracy_score(labels, [s >= t for s in scores]),
        threshold=t,
    )
