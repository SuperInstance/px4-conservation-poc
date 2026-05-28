"""Visualization for PX4 conservation-based anomaly detection.

Generates publication-quality plots showing:
1. 3D flight trajectory
2. Conservation ratio over time with anomaly markers
3. Fiedler vector / algebraic connectivity
4. Spectral fingerprint comparison (healthy vs unhealthy)
5. Detection comparison: conservation vs simple thresholding
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from simulator import FlightState
from detector import FlightAnalysis


# Professional color palette
COLORS = {
    'normal': '#2196F3',      # Blue
    'anomaly': '#F44336',     # Red
    'detect': '#FF9800',      # Orange
    'conservation': '#4CAF50', # Green
    'threshold': '#9C27B0',   # Purple
    'baseline': '#607D8B',    # Gray-blue
    'fiedler': '#00BCD4',     # Cyan
    'entropy': '#FF5722',     # Deep orange
}

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


def plot_trajectory(
    states: list[FlightState],
    anomaly_time: Optional[float] = None,
    title: str = "PX4 Flight Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 3D flight trajectory with anomaly region highlighted."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    positions = np.array([s.position for s in states])
    # NED to display: flip z for altitude
    x, y, z = positions[:, 0], positions[:, 1], -positions[:, 2]

    # Color by time
    times = np.array([s.timestamp for s in states])

    scatter = ax.scatter(x, y, z, c=times, cmap='viridis', s=1, alpha=0.6)

    if anomaly_time is not None:
        # Find anomaly region (±3 seconds)
        mask = (times >= anomaly_time - 1) & (times <= anomaly_time + 5)
        if mask.any():
            ax.scatter(x[mask], y[mask], -positions[mask, 2], c=COLORS['anomaly'],
                      s=8, alpha=0.9, label='Anomaly region', zorder=5)

    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(title)
    ax.legend(loc='upper left')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Time (s)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


def plot_conservation_timeseries(
    analysis: FlightAnalysis,
    anomaly_time: Optional[float] = None,
    anomaly_label: str = "Anomaly",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot conservation ratio over time with anomaly markers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Align timestamps with analysis arrays (offset by window warmup)
    offset = len(analysis.timestamps) - len(analysis.conservation_scores)
    ts = analysis.timestamps[offset:]

    # 1. Conservation ratio
    ax = axes[0]
    if len(ts) == len(analysis.conservation_scores):
        ax.plot(ts, analysis.conservation_scores, color=COLORS['conservation'],
                linewidth=1.0, alpha=0.8, label='Conservation Ratio')

        if analysis.baseline_conservation > 0:
            ax.axhline(y=analysis.baseline_conservation, color=COLORS['baseline'],
                      linestyle='--', alpha=0.7, label=f'Baseline ({analysis.baseline_conservation:.3f})')

        # Mark detection events
        for det_time, det_score, det_type in analysis.detection_events:
            ax.axvline(x=det_time, color=COLORS['detect'], alpha=0.5, linewidth=1.5)
            ax.scatter([det_time], [det_score], color=COLORS['detect'], s=50, zorder=5,
                      marker='v', label=f'Conservation {det_type}')

        if anomaly_time is not None:
            ax.axvline(x=anomaly_time, color=COLORS['anomaly'], linestyle='--',
                      linewidth=2, alpha=0.8, label=f'{anomaly_label} onset')

    ax.set_ylabel('Conservation Ratio')
    ax.set_title('Conservation Spectral Analysis — PX4 Flight Anomaly Detection')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. Spectral entropy
    ax = axes[1]
    if len(ts) == len(analysis.spectral_entropies):
        ax.plot(ts, analysis.spectral_entropies, color=COLORS['entropy'],
                linewidth=1.0, alpha=0.8, label='Spectral Entropy')

        if anomaly_time is not None:
            ax.axvline(x=anomaly_time, color=COLORS['anomaly'], linestyle='--',
                      linewidth=2, alpha=0.8)

    ax.set_ylabel('Spectral Entropy')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Algebraic connectivity (Fiedler value)
    ax = axes[2]
    if len(ts) == len(analysis.fiedler_values):
        ax.plot(ts, analysis.fiedler_values, color=COLORS['fiedler'],
                linewidth=1.0, alpha=0.8, label='Algebraic Connectivity (λ₂)')

        if anomaly_time is not None:
            ax.axvline(x=anomaly_time, color=COLORS['anomaly'], linestyle='--',
                      linewidth=2, alpha=0.8)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('λ₂')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


def plot_spectral_fingerprint(
    normal_analysis: FlightAnalysis,
    anomaly_analysis: FlightAnalysis,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare spectral fingerprints of healthy vs unhealthy flight."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Conservation ratio histograms
    ax = axes[0]
    normal_scores = normal_analysis.conservation_scores
    anomaly_scores = anomaly_analysis.conservation_scores

    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=30, alpha=0.7, color=COLORS['conservation'],
                label='Healthy Flight', density=True)
    if len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=30, alpha=0.7, color=COLORS['anomaly'],
                label='Anomalous Flight', density=True)

    ax.set_xlabel('Conservation Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Conservation Distribution: Healthy vs Anomalous')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # State cluster distribution (Fiedler partition)
    ax = axes[1]
    if len(normal_analysis.fiedler_values) > 0 and len(anomaly_analysis.fiedler_values) > 0:
        ax.plot(normal_analysis.fiedler_values, color=COLORS['conservation'],
                alpha=0.7, label='Healthy λ₂')
        ax.plot(anomaly_analysis.fiedler_values, color=COLORS['anomaly'],
                alpha=0.7, label='Anomalous λ₂')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Algebraic Connectivity')
        ax.set_title('Algebraic Connectivity Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


def plot_detection_comparison(
    conservation_detections: list[tuple[float, float, str]],
    threshold_detections: list[tuple[float, float, str]],
    anomaly_time: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare detection timelines: conservation vs simple thresholding."""
    fig, ax = plt.subplots(figsize=(14, 4))

    # Conservation detections
    if conservation_detections:
        times_c = [d[0] for d in conservation_detections]
        ax.scatter(times_c, [1.0]*len(times_c), color=COLORS['conservation'],
                  s=30, alpha=0.7, label='Conservation Detection', marker='|', linewidths=2)

    # Threshold detections
    if threshold_detections:
        times_t = [d[0] for d in threshold_detections]
        ax.scatter(times_t, [0.0]*len(times_t), color=COLORS['threshold'],
                  s=30, alpha=0.7, label='Threshold Detection', marker='|', linewidths=2)

    if anomaly_time is not None:
        ax.axvline(x=anomaly_time, color=COLORS['anomaly'], linestyle='--',
                  linewidth=2, alpha=0.8, label='Anomaly Onset')

    ax.set_xlabel('Time (s)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Threshold', 'Conservation'])
    ax.set_title('Anomaly Detection Timeline Comparison')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig
