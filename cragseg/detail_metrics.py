"""Diagnostics measured on the evaluation mask grid (background=0, smoke=1, fire=2)."""
from collections import deque

import cv2
import numpy as np

from .metrics import NAMES, metrics_from_confusion


_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def _boundary(mask, valid):
    edge = np.zeros_like(valid)
    horizontal = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (mask[1:] != mask[:-1]) & valid[1:] & valid[:-1]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    return edge


def _near(edge):
    # Two dilations by the 4-connected cross match the original tolerance=2.
    return cv2.dilate(edge.astype(np.uint8), _CROSS, iterations=2,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _boundary_counts(pred, truth, valid):
    predicted = _boundary(pred, valid)
    actual = _boundary(truth, valid)
    return np.array([(predicted & _near(actual)).sum(), predicted.sum(),
                     (actual & _near(predicted)).sum(), actual.sum()], dtype=np.int64)


def _maximum_matching(neighbors):
    """Return matched GT components, maximizing pair count, one use per component.

    GT is visited in raster label order, then eligible predictions in descending
    IoU and raster label order. These tie rules also define assignment to the two
    disjoint area bins when several maximum-cardinality matchings exist.
    """
    assigned_pred = {}
    assigned_gt = {}
    for start in sorted(neighbors):
        queue = deque([start])
        parent_gt = {start: None}
        tail = None
        while queue and tail is None:
            gt = queue.popleft()
            for pred in neighbors[gt]:
                owner = assigned_pred.get(pred)
                if owner is None:
                    tail = gt, pred
                    break
                if owner not in parent_gt:
                    parent_gt[owner] = gt
                    queue.append(owner)
        if tail is None:
            continue
        gt, pred = tail
        while gt is not None:
            old_pred = assigned_gt.get(gt)
            assigned_gt[gt] = pred
            assigned_pred[pred] = gt
            pred = old_pred
            gt = parent_gt[gt]
    return list(assigned_gt)


def _components(binary):
    # SAUF assigns contiguous labels in raster order including 0 for background.
    n, labels = cv2.connectedComponentsWithAlgorithm(binary.astype(np.uint8), 8,
                                                      cv2.CV_32S, cv2.CCL_WU)
    sizes = np.bincount(labels.ravel(), minlength=n)
    return labels, sizes


def _small_fire_counts(pred, truth, valid):
    gt, gt_sizes = _components((truth == 2) & valid)
    small = gt_sizes <= 16
    small[0] = False
    small_sizes = gt_sizes[small]
    totals = [np.count_nonzero(small_sizes <= 4), np.count_nonzero(small_sizes > 4)]
    counts = np.array([0, totals[0], 0, totals[1]], dtype=np.int64)
    if not small.any():
        return counts
    pred_labels, pred_sizes = _components((pred == 2) & valid)
    overlapping = small[gt] & (pred_labels != 0)
    if not overlapping.any():
        return counts
    # Only overlapping pairs can meet the threshold. A dense GT x prediction
    # intersection matrix is unnecessarily large for noisy predictions.
    stride = len(pred_sizes)
    pairs = gt[overlapping].astype(np.int64) * stride + pred_labels[overlapping]
    pairs, overlap = np.unique(pairs, return_counts=True)
    gt_ids, pred_ids = np.divmod(pairs, stride)
    union = gt_sizes[gt_ids] + pred_sizes[pred_ids] - overlap
    eligible = overlap * 10 >= union  # IoU >= 0.1, including equality
    ranked = {}
    for g, p, iou in zip(gt_ids[eligible], pred_ids[eligible], overlap[eligible] / union[eligible]):
        ranked.setdefault(int(g), []).append((-float(iou), int(p)))
    neighbors = {g: [p for _, p in sorted(edges)] for g, edges in ranked.items()}
    matched = _maximum_matching(neighbors)
    if matched:
        sizes = gt_sizes[matched]
        counts[0] = np.count_nonzero(sizes <= 4)
        counts[2] = np.count_nonzero(sizes > 4)
    return counts


class DetailCounts:
    """Accumulate confusion, boundary, connected-component and background counts."""
    def __init__(self):
        self.cm = np.zeros((3, 3), dtype=np.int64)
        self.boundary = np.zeros(4, dtype=np.int64)
        self.small_fire = np.zeros(4, dtype=np.int64)
        self.samples = 0
        self.pure_bg_frames = 0
        self.pure_bg_valid_pixels = 0
        self.pure_bg_fp_foreground = 0
        self.pure_bg_fp_fire = 0

    def update(self, prediction, truth):
        pred, truth = np.asarray(prediction), np.asarray(truth)
        if pred.shape != truth.shape or truth.ndim != 2:
            raise ValueError("Diagnostic metrics expect two aligned H x W masks.")
        valid = truth != 255
        if not np.isin(truth[valid], (0, 1, 2)).all() or not np.isin(pred[valid], (0, 1, 2)).all():
            raise ValueError("Unexpected class ID at a valid pixel.")
        self.samples += 1
        if not valid.any():
            return
        actual = truth[valid].astype(np.int64)
        predicted = pred[valid].astype(np.int64)
        self.cm += np.bincount(3 * actual + predicted, minlength=9).reshape(3, 3)
        self.boundary += _boundary_counts(pred, truth, valid)
        self.small_fire += _small_fire_counts(pred, truth, valid)
        if not actual.any():
            self.pure_bg_frames += 1
            self.pure_bg_valid_pixels += int(actual.size)
            self.pure_bg_fp_foreground += int(np.count_nonzero(predicted))
            self.pure_bg_fp_fire += int(np.count_nonzero(predicted == 2))

    def merge(self, other):
        """Combine counts without re-evaluating pixels (e.g. group to split)."""
        self.cm += other.cm
        self.boundary += other.boundary
        self.small_fire += other.small_fire
        self.samples += other.samples
        self.pure_bg_frames += other.pure_bg_frames
        self.pure_bg_valid_pixels += other.pure_bg_valid_pixels
        self.pure_bg_fp_foreground += other.pure_bg_fp_foreground
        self.pure_bg_fp_fire += other.pure_bg_fp_fire

    def summary(self):
        if self.cm.any():
            result = metrics_from_confusion(self.cm)
        else:
            keys = ["macro_recall", "miou", "macro_f1", "pixel_accuracy", "foreground_miou"]
            keys += [f"{name}_{metric}" for name in NAMES for metric in ("iou", "f1", "precision", "recall")]
            result = dict.fromkeys(keys, float("nan"))
        good_pred, n_pred, good_true, n_true = map(int, self.boundary)
        prec, recall = _ratio(good_pred, n_pred), _ratio(good_true, n_true)
        bf1 = float("nan") if not n_true else 0.0
        if n_true and n_pred and prec + recall:
            bf1 = 2 * prec * recall / (prec + recall)
        result.update(samples=self.samples, boundary_f1=bf1, boundary_precision=prec,
                      boundary_recall=recall, boundary_matched_pred=good_pred,
                      boundary_total_pred=n_pred, boundary_matched_gt=good_true,
                      boundary_total_gt=n_true, pure_bg_frames=self.pure_bg_frames,
                      pure_bg_valid_pixels=self.pure_bg_valid_pixels,
                      pure_bg_fp_foreground=self.pure_bg_fp_foreground,
                      pure_bg_fp_fire=self.pure_bg_fp_fire,
                      pure_bg_fg_rate=_ratio(self.pure_bg_fp_foreground, self.pure_bg_valid_pixels),
                      pure_bg_fire_rate=_ratio(self.pure_bg_fp_fire, self.pure_bg_valid_pixels))
        matched4, total4, matched16, total16 = map(int, self.small_fire)
        for suffix, hits, total in (("le4", matched4, total4), ("5to16", matched16, total16),
                                    ("le16", matched4 + matched16, total4 + total16)):
            result["small_fire_matched_" + suffix] = hits
            result["small_fire_total_" + suffix] = total
            result["small_fire_recall_" + suffix] = _ratio(hits, total)
        return result
