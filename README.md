# px4-conservation-poc

**PX4 flight anomaly detection via conservation spectral analysis — detect attitude failures, GPS glitches, and motor failures from EKF2 state sequences.**

Proof of concept applying the conservation spectral framework to PX4 autopilot data. Simulates 16-dimensional flight states (quaternion + velocity + position + gyro + accel), injects three types of anomalies, and detects them through conservation ratio drops in sliding-window spectral analysis.

## What This Gives You

- **Flight state simulator** — synthetic PX4 EKF2 16D state sequences with smooth SLERP interpolation
- **Three anomaly types** — sudden attitude change, GPS glitch, motor failure
- **Conservation-based detection** — sliding window Laplacian, track CR drops
- **Spectral fingerprinting** — compare healthy vs unhealthy flight signatures
- **Threshold comparison** — conservation detection vs simple statistical thresholding
- **Publication-quality visualization** — 3D trajectories, CR timeseries, spectral fingerprints

## Quick Start

```bash
pip install numpy scipy matplotlib scikit-learn
python demo.py
```

Outputs go to `output/` with PNG plots.

## Architecture

```
simulator.py   # FlightState + synthetic flight generation + anomaly injection
detector.py    # Conservation-based anomaly detection pipeline
visualize.py   # Professional dark-themed plots
demo.py        # End-to-end demo pipeline
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python)** — Core SDK
- **[anomaly-atlas](https://github.com/SuperInstance/anomaly-atlas)** — Unified anomaly detection across domains
- **px4-conservation-poc** — PX4-specific proof of concept (this repo)

## License

MIT
