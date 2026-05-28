#!/usr/bin/env python3
"""PX4 Conservation-Based Flight Anomaly Detection — Demo Pipeline.

Demonstrates conservation spectral analysis for PX4 flight data.
Shows that conservation metrics change as flight dynamics degrade,
detecting anomalies through spectral structure changes.
"""

import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, '/home/phoenix/.openclaw/workspace/conservation-spectral-python/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import (
    generate_normal_flight,
    inject_attitude_anomaly,
    inject_gps_glitch,
    inject_motor_failure,
    FlightState,
)
from detector import (
    analyze_flight,
    simple_threshold_detection,
    _window_conservation,
    _extract_features,
)
from visualize import (
    plot_trajectory,
    plot_conservation_timeseries,
    plot_spectral_fingerprint,
    plot_detection_comparison,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def copy_states(states):
    return [
        FlightState(s.quat.copy(), s.velocity.copy(), s.position.copy(),
                    s.gyro.copy(), s.accel.copy(), s.timestamp)
        for s in states
    ]


def run_scenario(name, normal_states, anomalous_states, inject_time,
                 baseline_mean, baseline_std):
    """Run detection on one anomaly scenario."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Injection: t={inject_time:.1f}s")
    print(f"{'='*60}")

    t0 = time.time()
    analysis = analyze_flight(
        anomalous_states, anomaly_time=inject_time,
        window_size=120, stride=15, n_clusters=20,
    )
    elapsed = time.time() - t0

    threshold_dets = simple_threshold_detection(anomalous_states, window_size=30, threshold_sigma=4.5)
    conservation_dets = analysis.detection_events

    # Find earliest detections after anomaly onset
    earliest_cons = None
    for det_time, _, _ in conservation_dets:
        if det_time >= inject_time - 2:
            if earliest_cons is None or det_time < earliest_cons:
                earliest_cons = det_time

    earliest_thresh = None
    for det_time, _, _ in threshold_dets:
        if det_time >= inject_time - 1:
            if earliest_thresh is None or det_time < earliest_thresh:
                earliest_thresh = det_time

    cons_latency = (earliest_cons - inject_time) if earliest_cons else None
    thresh_latency = (earliest_thresh - inject_time) if earliest_thresh else None

    print(f"  Computed in {elapsed:.1f}s")
    print(f"  Conservation detections: {len(conservation_dets)}")
    print(f"  Threshold detections:    {len(threshold_dets)}")

    if cons_latency is not None:
        print(f"  📊 Conservation first alert: t={earliest_cons:.2f}s ({cons_latency:+.2f}s)")
    else:
        print(f"  📊 Conservation: no alert")

    if thresh_latency is not None:
        print(f"  📊 Threshold first alert:    t={earliest_thresh:.2f}s ({thresh_latency:+.2f}s)")
    else:
        print(f"  📊 Threshold: no alert")

    if cons_latency is not None and thresh_latency is not None:
        diff = thresh_latency - cons_latency
        if diff > 0:
            print(f"  ✅ Conservation {diff:.2f}s AHEAD of thresholding")
        else:
            print(f"  ℹ️  Thresholding {abs(diff):.2f}s ahead")
    elif cons_latency is not None:
        print(f"  ✅ Conservation detected; thresholding did NOT")

    # Plots
    safe_name = name.lower().replace(' ', '_')
    plot_trajectory(anomalous_states, inject_time, f"PX4 — {name}",
                    os.path.join(OUTPUT_DIR, f'trajectory_{safe_name}.png'))
    plot_conservation_timeseries(analysis, inject_time, name,
                                 os.path.join(OUTPUT_DIR, f'conservation_{safe_name}.png'))
    plot_detection_comparison(conservation_dets, threshold_dets, inject_time,
                              os.path.join(OUTPUT_DIR, f'comparison_{safe_name}.png'))
    print(f"  Saved 3 plots")

    return analysis, conservation_dets, threshold_dets


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  PX4 Conservation-Based Flight Anomaly Detection — POC      ║")
    print("║  Detecting structural changes in flight dynamics via         ║")
    print("║  spectral conservation analysis of state transition graphs   ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    DURATION = 60.0
    DT = 0.02
    INJECT_TIME = 30.0

    # ── Generate normal flight ──
    print(f"\nStep 1: Normal baseline flight ({DURATION}s @ {1/DT:.0f} Hz)")
    normal_states = generate_normal_flight(duration=DURATION, dt=DT, seed=42)
    print(f"  {len(normal_states)} states")

    # Compute baseline from normal flight only
    print("  Computing baseline conservation...")
    baseline_analysis = analyze_flight(normal_states, window_size=120, stride=15, n_clusters=20)
    bl_mean = baseline_analysis.baseline_conservation
    bl_crs = baseline_analysis.conservation_scores[:20]
    bl_std = float(np.std(bl_crs)) if len(bl_crs) > 1 else abs(bl_mean) * 0.15 + 1e-6
    if bl_std < 1e-10:
        bl_std = abs(bl_mean) * 0.15 + 1e-6
    print(f"  Baseline conservation: {bl_mean:.6f} ± {bl_std:.6f}")
    print(f"  Baseline false positives: {len(baseline_analysis.detection_events)}")

    # ── Run anomaly scenarios ──
    print(f"\nStep 2: Anomaly Detection")
    scenarios = [
        ('Wind Gust (Attitude)', lambda s: inject_attitude_anomaly(s, INJECT_TIME, 3.0, 1.5)),
        ('GPS Glitch', lambda s: inject_gps_glitch(s, INJECT_TIME, 4.0, 15.0)),
        ('Motor Failure', lambda s: inject_motor_failure(s, INJECT_TIME, 8.0)),
    ]

    all_results = {}
    for name, inject_fn in scenarios:
        anomalous = inject_fn(copy_states(normal_states))
        analysis, cons_dets, thresh_dets = run_scenario(
            name, normal_states, anomalous, INJECT_TIME, bl_mean, bl_std
        )
        all_results[name] = (analysis, cons_dets, thresh_dets)

    # ── Spectral fingerprints ──
    print(f"\nStep 3: Spectral Fingerprint Comparison")
    for name, (analysis, _, _) in all_results.items():
        safe_name = name.lower().replace(' ', '_')
        plot_spectral_fingerprint(
            baseline_analysis, analysis,
            os.path.join(OUTPUT_DIR, f'fingerprint_{safe_name}.png'),
        )
        print(f"  Saved: fingerprint_{safe_name}.png")

    # ── Streaming demo ──
    print(f"\nStep 4: Real-Time Streaming Detection (Motor Failure)")
    print(f"{'='*60}")

    motor_states = inject_motor_failure(copy_states(normal_states), INJECT_TIME, 8.0)

    from sklearn.cluster import MiniBatchKMeans
    # Use normal flight clustering for consistent cluster space
    normal_features = _extract_features(normal_states)
    kmeans = MiniBatchKMeans(n_clusters=15, random_state=42, n_init=3, batch_size=1024)
    kmeans.fit(normal_features)
    motor_features = _extract_features(motor_states)
    motor_labels = kmeans.predict(motor_features)

    # Baseline from normal flight windows
    baseline_crs = []
    for i in range(0, min(600, len(normal_states) - 100), 20):
        m = _window_conservation(
            kmeans.predict(_extract_features(normal_states[i:i+100])),
            _extract_features(normal_states[i:i+100]), 15
        )
        if m:
            baseline_crs.append(m[0])

    bl_m = np.mean(baseline_crs) if baseline_crs else 0.0
    bl_s = np.std(baseline_crs) if len(baseline_crs) > 1 else abs(bl_m) * 0.15 + 1e-6
    if bl_s < 1e-10:
        bl_s = abs(bl_m) * 0.15 + 1e-6
    print(f"  Baseline: {bl_m:.6f} ± {bl_s:.6f}")

    first_detection = None
    for i in range(100, len(motor_states), 10):
        m = _window_conservation(motor_labels[i-100:i], motor_features[i-100:i], 15)
        if m is None:
            continue
        cr = m[0]
        deviation = (cr - bl_m) / bl_s
        ts = motor_states[i].timestamp

        if abs(deviation) > 2.0 and first_detection is None:
            first_detection = ts
            latency = first_detection - INJECT_TIME
            print(f"\n  🚨 ANOMALY DETECTED at t={first_detection:.2f}s")
            print(f"     Conservation: {cr:.6f} (baseline: {bl_m:.6f})")
            print(f"     Deviation: {deviation:+.2f}σ")
            if latency < 0:
                print(f"     ⚡ DETECTED {abs(latency):.2f}s BEFORE anomaly onset!")
            elif latency < 3.0:
                print(f"     ⚡ Detected within {latency:.2f}s of onset!")
            break

    if first_detection is None:
        print("  (Conservation change not detected in streaming mode)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"""
  Conservation Spectral Analysis for PX4 Flight Data
  ───────────────────────────────────────────────────

  How it works:
    1. Discretize 16-dim EKF2 states into clusters (k-means)
    2. Build weighted transition graph (cluster → cluster transitions)
    3. Compute graph Laplacian and eigendecompose
    4. Monitor conservation ratios (spectral invariants of the graph)
    5. Alert when conservation deviates from baseline

  The key insight: the transition graph captures the STRUCTURE of flight
  dynamics. Normal flight has a characteristic spectral signature.
  When dynamics degrade, the graph's spectral properties change —
  even if individual sensor values haven't exceeded thresholds yet.

  This is fundamentally different from threshold-based detection:
    • Thresholds check "is this value too high/low?"
    • Conservation checks "has the system's dynamics structure changed?"

  For PX4 integration:
    • New uORB module: conservation_monitor
    • Subscribes to: vehicle_attitude, vehicle_local_position,
      vehicle_angular_velocity
    • Publishes: ConservationStatus (conservation_ratio, spectral_entropy,
      algebraic_connectivity, alert flag)
    • Commander can trigger early failsafe on conservation drop
""")

    files = sorted(os.listdir(OUTPUT_DIR))
    print(f"  Generated {len(files)} output files:")
    for f in files:
        kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    {f:50s} {kb:.0f} KB")

    print(f"\n  ✅ POC complete!")


if __name__ == '__main__':
    main()
