# PX4 Conservation-Based Flight Anomaly Detection — POC

Proof-of-concept demonstrating conservation spectral analysis for PX4 flight data.
Detects anomalies by monitoring changes in the spectral structure of state transition graphs.

## Architecture

```
PX4 EKF2 States (16-dim) ──→ K-Means Clustering ──→ Tension Graph
       ↓                                                    │
  quaternion (4)                                  Laplacian + Eigendecomposition
  velocity (3)                                             │
  position (3)                              Conservation Ratios (spectral invariants)
  gyro (3)                                                 │
  accel (3)                              Anomaly Detection (conservation deviation)
```

## Files

| File | Purpose |
|------|---------|
| `simulator.py` | Synthetic PX4 EKF2 state sequences (normal + 3 anomaly types) |
| `detector.py` | Conservation-based anomaly detection using tension graphs |
| `visualize.py` | Dark-themed matplotlib plots |
| `demo.py` | Full pipeline: baseline → inject → detect → compare → plot |
| `output/` | Generated plots (12 files) |

## Anomaly Types

1. **Wind Gust** — sudden attitude change from external force
2. **GPS Glitch** — position jump with velocity inconsistency
3. **Motor Failure** — asymmetric thrust → yaw spin + altitude drop

## Key Concept

**Threshold detection** asks: "Is this sensor reading too high/low?"
**Conservation detection** asks: "Has the dynamics structure changed?"

The transition graph captures the *shape* of flight dynamics. Normal hover-cruise-hover
produces a characteristic spectral fingerprint. When dynamics degrade, the graph's
eigenstructure shifts — even if individual readings haven't crossed thresholds.

## Run

```bash
cd px4-conservation-poc
pip install numpy scipy scikit-learn matplotlib
python3 demo.py
```

Output plots saved to `output/`.

## Conservation Spectral SDK

Uses the `conservation-spectral-python` SDK:
- `TensionGraph` — weighted directed graph with vertex attributes
- `build_laplacian()` — graph Laplacian construction
- `eigendecompose()` — spectral decomposition
- `conservation_ratios()` — spectral invariants
- `spectral_fingerprint()` — eigenvalue distribution hashing
- `ConservationTracker` — real-time sliding window tracker

## PX4 Integration Path

1. **ConservationStatus.msg** — new uORB message type
2. **conservation_monitor** module — subscribes to `vehicle_attitude`, `vehicle_local_position`, `vehicle_angular_velocity`
3. **Commander** integration — trigger failsafe on conservation drop
4. **Logger** integration — record conservation metrics in ULog flight logs

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
