"""Conservation-based anomaly detection for PX4 flight data — Optimized.

Pre-clusters the entire flight, then computes conservation on sliding windows
of cluster labels (fast — no k-means per window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import sys
sys.path.insert(0, '/home/phoenix/.openclaw/workspace/conservation-spectral-python/src')
from conservation_spectral import (
    TensionGraph,
    build_laplacian,
    eigendecompose,
    conservation_ratios,
    spectral_gap,
    analyze,
    spectral_fingerprint,
)

from simulator import FlightState


@dataclass
class FlightAnalysis:
    timestamps: np.ndarray
    conservation_scores: np.ndarray
    spectral_entropies: np.ndarray
    fiedler_values: np.ndarray
    state_labels: np.ndarray
    anomalies: list
    baseline_conservation: float
    detection_events: list[tuple[float, float, str]]


def _extract_features(states):
    vectors = np.array([s.vector for s in states])
    return np.hstack([
        vectors[:, :4] * 2.0,       # quat
        vectors[:, 4:7] / 5.0,      # vel
        vectors[:, 7:10] / 100.0,   # pos
        vectors[:, 10:13] / 2.0,    # gyro
        vectors[:, 13:16] / 10.0,   # accel
    ])


def _window_conservation(labels, features, n_total_clusters):
    """Compute conservation metrics from a window of cluster labels."""
    transitions = [(labels[i], labels[i+1]) for i in range(len(labels)-1)]
    graph = TensionGraph.build_from_transitions(transitions, directed=True)
    if graph.vertex_count < 2:
        return None

    # Tension attribute
    mean_vec = features.mean(axis=0)
    tensions = np.zeros(graph.vertex_count)
    for lbl in range(graph.vertex_count):
        mask = labels == lbl
        if mask.any():
            tensions[lbl] = np.mean(np.linalg.norm(features[mask] - mean_vec, axis=1))
    tmax = tensions.max()
    if tmax > 0:
        tensions /= tmax

    try:
        lap = build_laplacian(graph)
        eigen = eigendecompose(lap)

        ratios = [r.ratio for r in conservation_ratios(eigen, tensions, "tension")]
        mean_cr = float(np.mean(ratios))

        # Spectral entropy
        evals = eigen.eigenvalues
        total = float(np.sum(np.abs(evals)))
        if total > 1e-15:
            probs = np.abs(evals) / total
            probs = probs[probs > 1e-15]
            entropy = float(-np.sum(probs * np.log(probs)))
        else:
            entropy = 0.0

        # Fiedler value (λ₂)
        fiedler = float(eigen.eigenvalues[1]) if len(eigen.eigenvalues) >= 2 else 0.0

        return (mean_cr, entropy, fiedler)
    except Exception:
        return None


def analyze_flight(
    states: list[FlightState],
    anomaly_time: Optional[float] = None,
    window_size: int = 150,
    stride: int = 25,
    n_clusters: int = 20,
) -> FlightAnalysis:
    """Full conservation analysis of a flight trajectory.

    Step 1: Pre-cluster entire flight with k-means (done once).
    Step 2: Sliding window over cluster labels → tension graph → conservation.
    """
    print("    Extracting features...", end='', flush=True)
    features = _extract_features(states)
    print(f" {features.shape}")

    print("    Clustering states...", end='', flush=True)
    n_clust = min(n_clusters, len(states) // 5)
    kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=42, n_init=3, batch_size=1024)
    labels = kmeans.fit_predict(features)
    print(f" {n_clust} clusters")

    print("    Computing conservation windows...", end='', flush=True)
    conservation_scores = []
    entropy_scores = []
    fiedler_scores = []
    result_timestamps = []
    detection_events = []

    # Compute baseline from first ~15 seconds (before any anomaly)
    baseline_limit = anomaly_time if anomaly_time else float('inf')
    baseline_crs = []
    for i in range(0, min(len(states) - window_size, int(25.0 / (states[1].timestamp - states[0].timestamp) if len(states) > 1 else 500)), stride):
        if states[i + window_size - 1].timestamp > baseline_limit - 2:
            break
        win_labels = labels[i:i + window_size]
        win_features = features[i:i + window_size]
        metrics = _window_conservation(win_labels, win_features, n_clust)
        if metrics is not None:
            baseline_crs.append(metrics[0])

    if baseline_crs:
        baseline_mean = float(np.mean(baseline_crs))
        baseline_std = float(np.std(baseline_crs))
        if baseline_std < 1e-10:
            baseline_std = abs(baseline_mean) * 0.15 + 1e-6
    else:
        baseline_mean = None
        baseline_std = None

    # Sliding window over entire flight
    for i in range(0, len(states) - window_size, stride):
        win_labels = labels[i:i + window_size]
        win_features = features[i:i + window_size]
        ts = states[i + window_size - 1].timestamp

        metrics = _window_conservation(win_labels, win_features, n_clust)
        if metrics is None:
            continue

        cr, entropy, fiedler = metrics
        conservation_scores.append(cr)
        entropy_scores.append(entropy)
        fiedler_scores.append(fiedler)
        result_timestamps.append(ts)

        # Detection
        if baseline_mean is not None:
            deviation = (cr - baseline_mean) / baseline_std
            if abs(deviation) > 2.0:
                det_type = "DROP" if deviation < 0 else "SPIKE"
                detection_events.append((ts, cr, det_type))

    print(f" {len(conservation_scores)} windows")

    bl = float(np.mean(conservation_scores[:15])) if len(conservation_scores) >= 15 else (
        float(np.mean(conservation_scores)) if conservation_scores else 0.0)

    return FlightAnalysis(
        timestamps=np.array(result_timestamps),
        conservation_scores=np.array(conservation_scores),
        spectral_entropies=np.array(entropy_scores),
        fiedler_values=np.array(fiedler_scores),
        state_labels=labels,
        anomalies=[],
        baseline_conservation=bl,
        detection_events=detection_events,
    )


def simple_threshold_detection(
    states: list[FlightState],
    window_size: int = 50,
    threshold_sigma: float = 4.0,
) -> list[tuple[float, float, str]]:
    """Baseline: simple thresholding on state deviation magnitude.

    Uses a higher threshold to avoid saturating with false positives.
    Deduplicates detections: only report the first detection in each
    consecutive run of threshold crossings.
    """
    vectors = np.array([s.vector for s in states])
    detections = []
    in_alert = False

    for i in range(window_size, len(vectors)):
        window = vectors[i - window_size:i]
        mean_vec = window.mean(axis=0)
        std_vec = window.std(axis=0)
        std_vec[std_vec < 1e-10] = 1.0

        deviation = np.linalg.norm((vectors[i] - mean_vec) / std_vec)
        if deviation > threshold_sigma:
            if not in_alert:
                detections.append((states[i].timestamp, deviation, "THRESHOLD"))
                in_alert = True
        else:
            in_alert = False

    return detections
