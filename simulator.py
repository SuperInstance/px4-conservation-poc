"""PX4 EKF2 flight state simulator for conservation-based anomaly detection.

Generates synthetic 16-dimensional state sequences:
  quaternion (4) + velocity (3) + position (3) + gyro (3) + accel (3) = 16-dim

Three flight profiles:
  - Normal: hover → cruise → hover (smooth transitions, high conservation)
  - Anomaly types: sudden attitude change, GPS glitch, motor failure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class FlightState:
    """Single EKF2 state sample (16-dimensional)."""
    quat: np.ndarray      # [w, x, y, z] quaternion (4)
    velocity: np.ndarray   # [vx, vy, vz] m/s (3)
    position: np.ndarray   # [x, y, z] m (3)
    gyro: np.ndarray       # [wx, wy, wz] rad/s (3)
    accel: np.ndarray      # [ax, ay, az] m/s² (3)
    timestamp: float       # seconds

    @property
    def vector(self) -> np.ndarray:
        """Full 16-dim state vector."""
        return np.concatenate([self.quat, self.velocity, self.position, self.gyro, self.accel])

    @property
    def attitude_euler(self) -> np.ndarray:
        """Roll, pitch, yaw in degrees."""
        r = Rotation.from_quat([self.quat[1], self.quat[2], self.quat[3], self.quat[0]])
        return np.degrees(r.as_euler('xyz'))


def _slerp_quat(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions."""
    key_rots = Rotation.from_quat([
        [q1[1], q1[2], q1[3], q1[0]],
        [q2[1], q2[2], q2[3], q2[0]]
    ])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp = slerp(t)
    q = interp.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # -> [w, x, y, z]


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def generate_normal_flight(
    duration: float = 60.0,
    dt: float = 0.02,  # 50 Hz
    seed: int = 42,
) -> list[FlightState]:
    """Generate a normal flight: hover → cruise → hover.

    Smooth transitions, physically consistent dynamics, high conservation.
    """
    rng = np.random.RandomState(seed)
    states = []
    t = 0.0

    n_steps = int(duration / dt)

    # Flight phases
    hover1_end = int(n_steps * 0.25)
    cruise_start = int(n_steps * 0.30)
    cruise_end = int(n_steps * 0.70)
    hover2_start = int(n_steps * 0.75)

    # Start position
    pos = np.array([0.0, 0.0, -5.0])  # NED: 5m up
    vel = np.array([0.0, 0.0, 0.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # level

    cruise_vel = np.array([5.0, 2.0, 0.0])  # 5 m/s forward, 2 m/s right

    for i in range(n_steps):
        t = i * dt

        # Phase logic
        if i < hover1_end:
            # Hover: small perturbations only
            accel = rng.normal(0, 0.05, 3)
            gyro_noise = rng.normal(0, 0.005, 3)
            target_vel = np.array([0.0, 0.0, 0.0])

        elif i < cruise_start:
            # Transition: hover → cruise
            alpha = (i - hover1_end) / (cruise_start - hover1_end)
            alpha = 0.5 * (1 - np.cos(np.pi * alpha))  # smooth ease
            target_vel = alpha * cruise_vel
            accel = (target_vel - vel) * 2.0 + rng.normal(0, 0.1, 3)
            gyro_noise = rng.normal(0, 0.02, 3)

        elif i < cruise_end:
            # Cruise: steady with small oscillations
            accel = rng.normal(0, 0.08, 3)
            gyro_noise = rng.normal(0, 0.008, 3)
            target_vel = cruise_vel.copy()

        elif i < hover2_start:
            # Transition: cruise → hover
            alpha = (i - cruise_end) / (hover2_start - cruise_end)
            alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            target_vel = (1 - alpha) * cruise_vel
            accel = (target_vel - vel) * 2.0 + rng.normal(0, 0.1, 3)
            gyro_noise = rng.normal(0, 0.02, 3)

        else:
            # Hover again
            accel = rng.normal(0, 0.05, 3)
            gyro_noise = rng.normal(0, 0.005, 3)
            target_vel = np.array([0.0, 0.0, 0.0])

        # Integrate
        vel = vel + accel * dt
        pos = pos + vel * dt

        # Attitude from velocity direction (simplified)
        if np.linalg.norm(vel[:2]) > 0.1:
            pitch_angle = -np.arctan2(accel[0], 9.81) * 0.5
            roll_angle = np.arctan2(accel[1], 9.81) * 0.5
            yaw_angle = np.arctan2(vel[1], vel[0])
        else:
            pitch_angle = rng.normal(0, 0.01)
            roll_angle = rng.normal(0, 0.01)
            yaw_angle = 0.0

        r = Rotation.from_euler('xyz', [roll_angle, pitch_angle, yaw_angle])
        q = r.as_quat()
        quat = np.array([q[3], q[0], q[1], q[2]])
        # Apply gyro noise to quaternion via small rotation
        if np.linalg.norm(gyro_noise) > 1e-10:
            dR = Rotation.from_rotvec(gyro_noise * dt)
            q_orig = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            q_new = dR * q_orig
            q = q_new.as_quat()
            quat = np.array([q[3], q[0], q[1], q[2]])
        quat = _normalize_quat(quat)

        # Gyro = angular rates + noise
        gyro = gyro_noise / dt * 0.1

        # Accelerometer: gravity + motion
        accel_meas = accel + np.array([0.0, 0.0, -9.81])

        states.append(FlightState(
            quat=quat.copy(),
            velocity=vel.copy(),
            position=pos.copy(),
            gyro=gyro.copy(),
            accel=accel_meas.copy(),
            timestamp=t,
        ))

    return states


def inject_attitude_anomaly(
    states: list[FlightState],
    inject_at: float,
    duration: float = 2.0,
    magnitude: float = 1.0,
) -> list[FlightState]:
    """Inject sudden attitude anomaly (e.g., wind gust flipping the drone)."""
    result = [FlightState(
        quat=s.quat.copy(), velocity=s.velocity.copy(),
        position=s.position.copy(), gyro=s.gyro.copy(),
        accel=s.accel.copy(), timestamp=s.timestamp
    ) for s in states]

    for s in result:
        if inject_at <= s.timestamp < inject_at + duration:
            t_local = (s.timestamp - inject_at) / duration
            # Sudden roll and pitch deviation that grows
            deviation = magnitude * np.sin(np.pi * t_local) * 0.8
            r = Rotation.from_euler('xyz', [deviation, deviation * 0.5, 0.0])
            q_orig = Rotation.from_quat([s.quat[1], s.quat[2], s.quat[3], s.quat[0]])
            q_new = r * q_orig
            q = q_new.as_quat()
            s.quat = np.array([q[3], q[0], q[1], q[2]])
            s.gyro += np.array([deviation * 2.0, deviation, 0.0])

    return result


def inject_gps_glitch(
    states: list[FlightState],
    inject_at: float,
    duration: float = 3.0,
    jump_magnitude: float = 10.0,
) -> list[FlightState]:
    """Inject GPS glitch: sudden position jump with velocity inconsistency."""
    result = [FlightState(
        quat=s.quat.copy(), velocity=s.velocity.copy(),
        position=s.position.copy(), gyro=s.gyro.copy(),
        accel=s.accel.copy(), timestamp=s.timestamp
    ) for s in states]

    offset_applied = np.zeros(3)

    for s in result:
        if s.timestamp >= inject_at:
            if s.timestamp < inject_at + duration:
                t_local = (s.timestamp - inject_at) / duration
                # Position jumps suddenly then slowly returns
                jump = jump_magnitude * np.exp(-2 * t_local) * np.array([1.0, 0.5, 0.3])
                s.position = s.position + jump
                # Velocity becomes inconsistent with position change
                s.velocity += jump * 0.5
            else:
                # After glitch, position is offset
                s.position += np.array([jump_magnitude * 0.1, jump_magnitude * 0.05, jump_magnitude * 0.03])

    return result


def inject_motor_failure(
    states: list[FlightState],
    inject_at: float,
    duration: float = 5.0,
) -> list[FlightState]:
    """Inject motor failure: asymmetric thrust causing yaw spin and altitude drop."""
    result = [FlightState(
        quat=s.quat.copy(), velocity=s.velocity.copy(),
        position=s.position.copy(), gyro=s.gyro.copy(),
        accel=s.accel.copy(), timestamp=s.timestamp
    ) for s in states]

    for s in result:
        if inject_at <= s.timestamp < inject_at + duration:
            t_local = (s.timestamp - inject_at) / duration

            # Yaw spin increases over time
            yaw_rate = 2.0 * t_local  # rad/s, increasing
            yaw_accum = yaw_rate * 0.02

            r_yaw = Rotation.from_euler('z', yaw_accum)
            q_orig = Rotation.from_quat([s.quat[1], s.quat[2], s.quat[3], s.quat[0]])
            q_new = r_yaw * q_orig
            q = q_new.as_quat()
            s.quat = np.array([q[3], q[0], q[1], q[2]])

            # High gyro on yaw axis
            s.gyro += np.array([0.1 * t_local, 0.05 * t_local, yaw_rate])

            # Altitude drop (lost thrust)
            s.velocity[2] += 0.5 * t_local  # falling faster
            s.position[2] += s.velocity[2] * 0.02

            # Asymmetric acceleration
            s.accel += np.array([0.3 * t_local, 0.2 * t_local, 0.5 * t_local])

    return result
