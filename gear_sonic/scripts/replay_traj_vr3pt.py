"""Replay a pre-recorded wrist trajectory through SONIC's VR 3-point tracking
(sim2sim) and report tracking metrics using the **same** ``TrackingErrorMonitor``
/ ``CommandSmoothnessMonitor`` classes as the IK baseline in
``rl_ik_solver/.../trajectory.py`` (vendored in
``gear_sonic/utils/eval/trajectory_metrics.py``).

Usage (three terminals):

  # Terminal 1 - MuJoCo sim
  python -m gear_sonic.scripts.run_sim_loop

  # Terminal 2 - C++ policy deploy, using ZMQ manager as input
  cd gear_sonic_deploy && bash deploy.sh --input-type zmq_manager sim

  # Terminal 3 - replay trajectory (this script)
  python -m gear_sonic.scripts.replay_traj_vr3pt \
      --csv /home/rail/rail-unitree/rl_ik_solver/trajectory/recorded_trajectories/traj1.csv \
      --output-log /tmp/sonic_traj1_log.npz \
      --metrics-csv /tmp/sonic_traj1_metrics.csv

The CSV schema is:
    t, left_x,left_y,left_z,left_roll,left_pitch,left_yaw,
       left_qw,left_qx,left_qy,left_qz,
       right_x,right_y,right_z,right_roll,right_pitch,right_yaw,
       right_qw,right_qx,right_qy,right_qz

The wrist pose is ``*_wrist_yaw_link`` expressed in the ``torso_link`` frame,
using scalar-first (wxyz) quaternions.

The IK baseline's ``TrackingErrorMonitor.update()`` wants the target as
``(xyz, rpy)`` length-6 in the torso frame and the actual as ``(xyz, qwxyz)``
length-7 in the torso frame, so this script feeds it exactly that.

The script:
  1. Uses Pinocchio FK on the G1 URDF at the zero-waist default pose to
     resolve the fixed ``pelvis <- torso_link`` transform and the reference
     torso/neck keypoint.
  2. For each CSV frame, converts the wrist pose from the torso frame to the
     pelvis frame, applies the wrist local offset
     (``left: [0.18,-0.025,0]``, ``right: [0.18,0.025,0]``), and packs the 9
     position + 12 orientation floats into a ``planner`` ZMQ message.  The
     ``mode`` field is set to ``LocomotionMode::IDLE = 0`` so the C++
     ``ZMQManager`` keeps the lower body in the default standing pose (see
     ``localmotion_kplanner.hpp``).  VR 3-point upper-body tracking is
     activated by simply populating ``vr_3point_position`` /
     ``vr_3point_orientation`` in the planner message -- the ``mode`` field
     selects the *locomotion* style, not whether VR tracking is on.
     Sending ``mode = 5`` here would map to ``IDEL_KNEEL_TWO_LEGS`` and make
     the robot kneel; sending ``mode = 0`` (IDLE) is what the real pico
     teleop does when the joystick is in its deadzone.
  3. Re-asserts ``command {start=True, planner=True}`` every 250 ms so the
     C++ ZMQManager doesn't time out into IDLE.
  4. Before starting metric collection, runs an optional ``--ramp-in-sec``
     phase that reads the robot's current wrist pose (FK on
     ``body_q_measured``), linearly interpolates position and SLERP-s
     orientation to CSV frame 0, and sends those as VR 3-point targets.
     This avoids the initial large step between the robot's rest pose and
     the first CSV frame (especially important on the real robot).  No
     samples from this phase are fed to the monitors -- the tracking /
     smoothness clock (``elapsed`` = 0) starts only after the ramp.
 5. Subscribes to ``g1_debug`` (port 5557) and records ``body_q_measured``
    (for tracking-error FK), ``last_action`` (the policy's per-step
    joint-target command after scale + default-angle offset, used for
    command-smoothness metrics -- this is the SONIC analog of the IK
    controller's ``joint_target_q``), and two per-tick wall-clock
    inference-time fields ``encoder_infer_us`` / ``policy_infer_us``
    emitted by the C++ deploy around the TensorRT ``Encode()`` / ``Infer()``
    calls.  Note: ``body_q_target`` in the feedback is the *reference
    motion* joint positions, NOT the policy's commanded action, so we
    deliberately do not use it for smoothness.
 6. On exit prints / logs the same metric block the IK repo produces,
    appends a policy-inference-time summary aligned with the IK
    baseline's ``infer_times_ms`` (encoder / policy / total, mean/std/
    p50/p95/max), and optionally saves per-frame data plus the
    inference-time arrays to ``--output-log``.
"""

from __future__ import annotations

import argparse
import csv
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import Slerp

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.eval.trajectory_metrics import (
    CommandSmoothnessMonitor,
    TrackingErrorMonitor,
    matrix_to_quaternion_wxyz,
)
from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import (
    G1_KEY_FRAME_OFFSETS,
    G1_LEFT_WRIST_FRAME,
    G1_RIGHT_WRIST_FRAME,
    G1_TORSO_FRAME,
    get_g1_key_frame_poses,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)

# ``mode`` field of the planner msgpack maps to
# ``LocomotionMode`` in ``gear_sonic_deploy/.../localmotion_kplanner.hpp``,
# not to ``StreamMode`` in pico_manager_thread_server.py.  Use 0 = IDLE to
# keep the lower body in the default standing pose while VR 3-point tracks
# the upper body.  (Sending 5 here would mean ``IDEL_KNEEL_TWO_LEGS`` and
# the robot would kneel.)
LOCOMOTION_MODE_IDLE = 0

FEEDBACK_TOPIC = "g1_debug"
FEEDBACK_DEFAULT_PORT = 5557
PLANNER_DEFAULT_PORT = 5556

# SONIC / MuJoCo XML joint order for the two 7-DoF arms. This is what Pinocchio
# sees in the G1 URDF (mirrored in gear_sonic_deploy/g1/g1_29dof.xml).
SONIC_LEFT_ARM_IDX = np.array([15, 16, 17, 18, 19, 20, 21], dtype=np.int64)
SONIC_RIGHT_ARM_IDX = np.array([22, 23, 24, 25, 26, 27, 28], dtype=np.int64)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryFrame:
    t: float
    left_pos: np.ndarray         # (3,) in torso frame
    left_rpy: np.ndarray         # (3,) roll/pitch/yaw, torso frame
    left_quat_wxyz: np.ndarray   # (4,) torso frame
    right_pos: np.ndarray
    right_rpy: np.ndarray
    right_quat_wxyz: np.ndarray


def load_trajectory(csv_path: Path) -> list[TrajectoryFrame]:
    frames: list[TrajectoryFrame] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(
                TrajectoryFrame(
                    t=float(row["t"]),
                    left_pos=np.array(
                        [float(row["left_x"]), float(row["left_y"]), float(row["left_z"])],
                        dtype=np.float64,
                    ),
                    left_rpy=np.array(
                        [
                            float(row["left_roll"]),
                            float(row["left_pitch"]),
                            float(row["left_yaw"]),
                        ],
                        dtype=np.float64,
                    ),
                    left_quat_wxyz=np.array(
                        [
                            float(row["left_qw"]),
                            float(row["left_qx"]),
                            float(row["left_qy"]),
                            float(row["left_qz"]),
                        ],
                        dtype=np.float64,
                    ),
                    right_pos=np.array(
                        [float(row["right_x"]), float(row["right_y"]), float(row["right_z"])],
                        dtype=np.float64,
                    ),
                    right_rpy=np.array(
                        [
                            float(row["right_roll"]),
                            float(row["right_pitch"]),
                            float(row["right_yaw"]),
                        ],
                        dtype=np.float64,
                    ),
                    right_quat_wxyz=np.array(
                        [
                            float(row["right_qw"]),
                            float(row["right_qx"]),
                            float(row["right_qy"]),
                            float(row["right_qz"]),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
    if not frames:
        raise ValueError(f"Trajectory file {csv_path} has no data rows")
    return frames


# ---------------------------------------------------------------------------
# FK / SE3 utilities
# ---------------------------------------------------------------------------


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=q_xyzw.dtype)


def se3_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    R = sRot.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def pos_quat_from_se3(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the same Shepperd-style conversion as the IK repo so quaternion
    sign conventions match bit-for-bit."""
    pos = T[:3, 3].copy()
    quat_wxyz = matrix_to_quaternion_wxyz(T[:3, :3])
    return pos, quat_wxyz


def _se3(placement) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = placement.rotation
    T[:3, 3] = placement.translation
    return T


class G1FKHelper:
    """Small wrapper around Pinocchio FK for the transforms we need.

    ``instantiate_g1_robot_model()`` loads ``g1_29dof_with_hand.urdf`` as a
    **fixed-base** model (no 7-DoF floating base), so the Pinocchio
    configuration vector ``q`` has length ``num_body_dofs + num_hand_dofs``
    laid out according to URDF joint declaration order — NOT simply
    ``[body_29, hand_k]``.  We therefore use
    ``supplemental_info.body_actuated_joints`` to recover the exact
    Pinocchio dof indices for the 29 body joints (in
    IsaacLab = MuJoCo-XML body order, which is the order the C++ deploy
    publishes ``body_q_measured`` in).
    """

    def __init__(self):
        self.robot_model = instantiate_g1_robot_model()
        self.num_dofs = self.robot_model.num_dofs
        body_indices = self.robot_model.get_body_actuated_joint_indices()
        self._body_dof_indices = np.asarray(body_indices, dtype=np.int64)
        self.num_joints = int(self._body_dof_indices.size)
        if self.num_joints != 29:
            raise RuntimeError(
                f"Expected 29 body actuated joints in G1, got {self.num_joints}"
            )

    def full_q_from_joints(self, joints_29: np.ndarray) -> np.ndarray:
        """Embed a 29-DoF body-joint vector (IsaacLab / MuJoCo-XML body order)
        into a full Pinocchio ``q`` vector, leaving hand joints at their
        default values."""
        q = self.robot_model.default_body_pose.copy()
        q[self._body_dof_indices] = np.asarray(joints_29, dtype=np.float64)
        return q

    def frame_T_in_pelvis(self, frame_name: str, joints_29: np.ndarray | None) -> np.ndarray:
        q = (
            self.robot_model.default_body_pose.copy()
            if joints_29 is None
            else self.full_q_from_joints(np.asarray(joints_29, dtype=np.float64))
        )
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        return _se3(self.robot_model.frame_placement(frame_name))

    def wrist_poses_in_torso(
        self, joints_29: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lw_pos, lw_quat_wxyz, rw_pos, rw_quat_wxyz)`` of the
        ``*_wrist_yaw_link`` frames expressed in the ``torso_link`` frame --
        the same geometric quantity that ``larm_forward`` / ``rarm_forward``
        produces in the IK baseline."""
        q_full = self.full_q_from_joints(np.asarray(joints_29, dtype=np.float64))
        self.robot_model.cache_forward_kinematics(q_full, auto_clip=False)
        T_p_lw = _se3(self.robot_model.frame_placement(G1_LEFT_WRIST_FRAME))
        T_p_rw = _se3(self.robot_model.frame_placement(G1_RIGHT_WRIST_FRAME))
        T_p_torso = _se3(self.robot_model.frame_placement(G1_TORSO_FRAME))
        T_torso_p = np.linalg.inv(T_p_torso)
        T_torso_lw = T_torso_p @ T_p_lw
        T_torso_rw = T_torso_p @ T_p_rw
        lw_pos, lw_quat = pos_quat_from_se3(T_torso_lw)
        rw_pos, rw_quat = pos_quat_from_se3(T_torso_rw)
        return lw_pos, lw_quat, rw_pos, rw_quat


# ---------------------------------------------------------------------------
# VR 3-point target assembly
# ---------------------------------------------------------------------------


def build_vr3pt_targets(
    frame: TrajectoryFrame,
    T_pelvis_torso: np.ndarray,
    neck_pos_pelvis: np.ndarray,
    neck_quat_wxyz_pelvis: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Assemble the 9-float position + 12-float orientation buffers that
    SONIC's ``vr_3point_local_target`` observation expects.

    Positions are in the pelvis frame with the wrist local offset applied;
    orientations are the link rotation in the pelvis frame (wxyz)."""

    T_torso_lw = se3_from_pos_quat(frame.left_pos, frame.left_quat_wxyz)
    T_torso_rw = se3_from_pos_quat(frame.right_pos, frame.right_quat_wxyz)
    T_pelvis_lw = T_pelvis_torso @ T_torso_lw
    T_pelvis_rw = T_pelvis_torso @ T_torso_rw

    lw_pos_pelvis, lw_quat_wxyz = pos_quat_from_se3(T_pelvis_lw)
    rw_pos_pelvis, rw_quat_wxyz = pos_quat_from_se3(T_pelvis_rw)

    lw_pos_pelvis = lw_pos_pelvis + T_pelvis_lw[:3, :3] @ G1_KEY_FRAME_OFFSETS["left_wrist"]
    rw_pos_pelvis = rw_pos_pelvis + T_pelvis_rw[:3, :3] @ G1_KEY_FRAME_OFFSETS["right_wrist"]

    vr_position = np.concatenate([lw_pos_pelvis, rw_pos_pelvis, neck_pos_pelvis]).astype(np.float32)
    vr_orientation = np.concatenate(
        [lw_quat_wxyz, rw_quat_wxyz, neck_quat_wxyz_pelvis]
    ).astype(np.float32)
    return vr_position.tolist(), vr_orientation.tolist()


# ---------------------------------------------------------------------------
# Ramp-in helper
# ---------------------------------------------------------------------------


def _send_vr3pt_for_pose(
    pub: "zmq.Socket",
    lw_pos: np.ndarray,
    lw_quat_wxyz: np.ndarray,
    rw_pos: np.ndarray,
    rw_quat_wxyz: np.ndarray,
    T_pelvis_torso: np.ndarray,
    neck_pos_pelvis: np.ndarray,
    neck_quat_wxyz: np.ndarray,
) -> None:
    synth = TrajectoryFrame(
        t=0.0,
        left_pos=lw_pos,
        left_rpy=np.zeros(3),
        left_quat_wxyz=lw_quat_wxyz,
        right_pos=rw_pos,
        right_rpy=np.zeros(3),
        right_quat_wxyz=rw_quat_wxyz,
    )
    vr_pos, vr_orn = build_vr3pt_targets(
        synth,
        T_pelvis_torso=T_pelvis_torso,
        neck_pos_pelvis=neck_pos_pelvis,
        neck_quat_wxyz_pelvis=neck_quat_wxyz,
    )
    pub.send(
        build_planner_message(
            mode=LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
            vr_3pt_position=vr_pos,
            vr_3pt_orientation=vr_orn,
        )
    )


def run_ramp_in(
    pub: "zmq.Socket",
    fk: "G1FKHelper",
    feedback: "FeedbackLogger",
    target_frame: TrajectoryFrame,
    T_pelvis_torso: np.ndarray,
    neck_pos_pelvis: np.ndarray,
    neck_quat_wxyz: np.ndarray,
    ramp_in_sec: float,
    rate_hz: float,
    stop_flag: dict,
    feedback_wait_sec: float = 3.0,
) -> bool:
    """Smoothly blend the VR 3-point target from the robot's *current* wrist
    pose (recovered via FK on the latest ``body_q_measured``) to the first
    CSV frame over ``ramp_in_sec`` seconds.  Does *not* feed the tracking /
    smoothness monitors -- this phase happens before ``elapsed = 0``.

    Returns ``True`` if the ramp completed normally, ``False`` if it was
    skipped (e.g. no feedback arrived in time)."""

    if ramp_in_sec <= 0.0:
        return False

    # 1. Wait for at least one valid body_q_measured sample so the FK source
    #    pose reflects the robot's actual state, not the URDF default.
    t_wait_start = time.time()
    q_meas: np.ndarray | None = None
    while time.time() - t_wait_start < feedback_wait_sec and not stop_flag["value"]:
        q_meas, _ = feedback.latest()
        if (
            q_meas is not None
            and q_meas.size >= 29
            and np.all(np.isfinite(q_meas[:29]))
        ):
            break
        time.sleep(0.02)
    if q_meas is None or q_meas.size < 29:
        print(
            f"[ramp-in] WARN: no body_q_measured within {feedback_wait_sec:.1f}s; "
            f"skipping ramp-in and starting replay directly."
        )
        return False

    lw0_pos, lw0_quat, rw0_pos, rw0_quat = fk.wrist_poses_in_torso(q_meas[:29])
    lw1_pos = target_frame.left_pos
    lw1_quat = target_frame.left_quat_wxyz
    rw1_pos = target_frame.right_pos
    rw1_quat = target_frame.right_quat_wxyz

    print(
        f"[ramp-in] Blending wrists in torso frame over {ramp_in_sec:.2f}s:\n"
        f"  left : {np.array2string(lw0_pos, precision=3)} -> "
        f"{np.array2string(lw1_pos, precision=3)}\n"
        f"  right: {np.array2string(rw0_pos, precision=3)} -> "
        f"{np.array2string(rw1_pos, precision=3)}"
    )

    slerp_left = Slerp(
        [0.0, 1.0],
        sRot.from_quat(np.stack([lw0_quat, lw1_quat]), scalar_first=True),
    )
    slerp_right = Slerp(
        [0.0, 1.0],
        sRot.from_quat(np.stack([rw0_quat, rw1_quat]), scalar_first=True),
    )

    dt = 1.0 / max(rate_hz, 1e-3)
    n_steps = max(1, int(round(ramp_in_sec / dt)))
    t_start = time.time()
    last_cmd_pulse = 0.0
    for k in range(n_steps + 1):
        if stop_flag["value"]:
            return False
        alpha = min(1.0, k / n_steps)
        target_t = k * dt
        now = time.time() - t_start
        if now < target_t:
            time.sleep(max(target_t - now, 0.0))

        lw_pos = (1.0 - alpha) * lw0_pos + alpha * lw1_pos
        rw_pos = (1.0 - alpha) * rw0_pos + alpha * rw1_pos
        lw_quat = slerp_left([alpha]).as_quat(scalar_first=True)[0]
        rw_quat = slerp_right([alpha]).as_quat(scalar_first=True)[0]

        _send_vr3pt_for_pose(
            pub=pub,
            lw_pos=lw_pos,
            lw_quat_wxyz=lw_quat,
            rw_pos=rw_pos,
            rw_quat_wxyz=rw_quat,
            T_pelvis_torso=T_pelvis_torso,
            neck_pos_pelvis=neck_pos_pelvis,
            neck_quat_wxyz=neck_quat_wxyz,
        )

        now2 = time.time() - t_start
        if now2 - last_cmd_pulse > 0.25:
            pub.send(build_command_message(start=True, stop=False, planner=True))
            last_cmd_pulse = now2

    print("[ramp-in] Done; starting metric clock at elapsed = 0.")
    return True


# ---------------------------------------------------------------------------
# Feedback subscriber
# ---------------------------------------------------------------------------


class FeedbackLogger:
    """Background thread reading ``g1_debug`` msgpack frames and caching the
    latest ``body_q_measured`` (measured joint positions), ``last_action``
    (the policy's per-step joint-target command after scale + default-angle
    offset, i.e. the SONIC analog of the IK controller's ``joint_target_q``),
    and the two per-tick inference-time fields ``encoder_infer_us`` /
    ``policy_infer_us`` that the C++ deploy emits (see
    ``zmq_output_handler.hpp::pack_combined_state``).

    The two joint arrays are 29-DoF in MuJoCo joint order.  The inference
    timings are raw wall-clock microseconds measured around each TensorRT
    ``Encode()`` / ``Infer()`` call (H2D + compute + D2H + sync)."""

    def __init__(self, host: str, port: int):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, FEEDBACK_TOPIC)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()
        self._latest_q_measured: np.ndarray | None = None
        self._latest_action: np.ndarray | None = None
        # Latest per-tick inference times (microseconds).  None until the
        # first g1_debug message with the new schema arrives; 0 means the
        # engine was not wired up (e.g. encoder disabled) but the field is
        # still present, so we treat 0 as "no data" for that engine.
        self._latest_encoder_infer_us: int | None = None
        self._latest_policy_infer_us: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="FeedbackLogger", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._sock.poll(timeout=100):
                    data = self._sock.recv(zmq.NOBLOCK)
                else:
                    continue
            except zmq.ZMQError:
                continue

            payload = data[len(FEEDBACK_TOPIC) :]
            try:
                unpacked = msgpack.unpackb(payload, raw=False)
            except Exception:
                continue
            body_q_meas = unpacked.get("body_q_measured")
            last_action = unpacked.get("last_action")
            enc_us = unpacked.get("encoder_infer_us")
            pol_us = unpacked.get("policy_infer_us")
            with self._lock:
                if body_q_meas is not None:
                    self._latest_q_measured = np.asarray(body_q_meas, dtype=np.float64)
                if last_action is not None:
                    self._latest_action = np.asarray(last_action, dtype=np.float64)
                if enc_us is not None:
                    self._latest_encoder_infer_us = int(enc_us)
                if pol_us is not None:
                    self._latest_policy_infer_us = int(pol_us)

    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            return (
                None if self._latest_q_measured is None else self._latest_q_measured.copy(),
                None if self._latest_action is None else self._latest_action.copy(),
            )

    def latest_infer_us(self) -> tuple[int | None, int | None]:
        """Return the most recent ``(encoder_infer_us, policy_infer_us)``
        values from the C++ deploy.  Either component is ``None`` until at
        least one ``g1_debug`` message carrying the new schema has been
        received.  A value of 0 means the C++ side published the field but
        the corresponding engine was not wired up (no encoder, or policy
        hasn't run yet); callers should treat 0 as "no measurement"."""
        with self._lock:
            return self._latest_encoder_infer_us, self._latest_policy_infer_us


# ---------------------------------------------------------------------------
# Inference-time summary
# ---------------------------------------------------------------------------


def _one_stream_summary(name: str, samples: list[float]) -> None:
    """Print IK-baseline-compatible stats for one stream of per-tick
    inference times (ms).  Mirrors the ``compute_summary`` block in
    ``rl_ik_solver/.../trajectory.py::get_summary`` (sample_count /
    mean / std / p50 / p95 / max) so the numbers are directly comparable.
    """
    if not samples:
        print(f"  {name:<9} : no samples collected")
        return
    arr = np.asarray(samples, dtype=np.float64)
    print(
        f"  {name:<9} : n={arr.size:5d}  "
        f"mean={arr.mean():6.3f}  std={arr.std(ddof=0):6.3f}  "
        f"p50={np.percentile(arr, 50):6.3f}  p95={np.percentile(arr, 95):6.3f}  "
        f"max={arr.max():6.3f}  (ms)"
    )


def print_inference_summary(
    total_ms: list[float],
    encoder_ms: list[float],
    policy_ms: list[float],
    warmup_sec: float,
) -> None:
    """Print a per-tick inference-time block aligned with the IK baseline's
    ``infer_times_ms`` summary.  The C++ deploy measures:

    * ``encoder_ms``: wall-clock around the encoder TensorRT engine (H2D +
      kernel + D2H + cudaStreamSynchronize).  Present only when an encoder
      is configured.
    * ``policy_ms``: wall-clock around the decoder / policy TensorRT
      engine, same H2D+compute+D2H+sync scope.
    * ``total_ms``: ``encoder_ms + policy_ms``.  This is the closest analog
      to the IK baseline's scalar ``infer_ms`` (which times its single
      policy forward pass).  Compare the summary's ``total`` line against
      the IK baseline.

    Only samples collected after ``warmup_sec`` are included, matching the
    IK baseline's behaviour (``infer_times_ms`` is filled only once the
    control loop has been running past the warmup window).
    """
    print("===== Policy inference time =====")
    print(f"  (warmup_sec={warmup_sec:.2f}s, TensorRT FP32 unless --use-fp16 was set)")
    _one_stream_summary("encoder", encoder_ms)
    _one_stream_summary("policy", policy_ms)
    _one_stream_summary("total", total_ms)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    frames = load_trajectory(Path(args.csv))
    print(
        f"Loaded {len(frames)} frames from {args.csv} "
        f"(duration {frames[-1].t - frames[0].t:.2f}s)"
    )

    fk = G1FKHelper()

    T_pelvis_torso = fk.frame_T_in_pelvis(G1_TORSO_FRAME, joints_29=None)
    print(f"pelvis<-torso translation: {T_pelvis_torso[:3, 3]}")

    default_poses = get_g1_key_frame_poses(fk.robot_model, q=None, apply_offset=True)
    neck_pos_pelvis = default_poses["torso"]["position"].astype(np.float32)
    neck_quat_wxyz = default_poses["torso"]["orientation_wxyz"].astype(np.float32)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.bind(f"tcp://*:{args.planner_port}")

    feedback = FeedbackLogger(host=args.feedback_host, port=args.feedback_port)
    feedback.start()

    print("Waiting 1.5s for ZMQ subscribers to connect...")
    time.sleep(1.5)

    # Metric monitors -- same classes and arguments as the IK baseline.
    tracking_monitor = TrackingErrorMonitor(
        print_hz=args.print_hz,
        log_csv_path=Path(args.metrics_csv) if args.metrics_csv else None,
        warmup_sec=args.warmup_sec,
        lag_search_max_sec=args.lag_search_max_sec,
        spatial_resample_count=args.spatial_resample_count,
    )
    smoothness_monitor = CommandSmoothnessMonitor(
        print_hz=args.print_hz,
        warmup_sec=args.warmup_sec,
        fixed_dt=args.control_dt,
    )

    log_t: list[float] = []
    log_left_pos: list[np.ndarray] = []
    log_left_quat: list[np.ndarray] = []
    log_right_pos: list[np.ndarray] = []
    log_right_quat: list[np.ndarray] = []
    log_body_q_measured: list[np.ndarray] = []
    log_last_action: list[np.ndarray] = []
    # Per-tick C++-measured inference times (ms) collected only after the
    # smoothness / tracking warmup window, so the summary matches the IK
    # baseline's ``infer_times_ms`` stat (see trajectory.py :: get_summary).
    log_infer_ms_total: list[float] = []
    log_infer_ms_encoder: list[float] = []
    log_infer_ms_policy: list[float] = []
    # De-dup: the C++ atomic only updates once per control tick (~50 Hz),
    # while this loop also runs at 50 Hz but asynchronously, so we can
    # easily sample the same (encoder_us, policy_us) pair twice.  Track
    # the last seen pair and only append when it changes.
    last_infer_pair: tuple[int | None, int | None] = (None, None)

    stop_requested = {"value": False}

    def handle_sigint(_sig, _frame):
        print("\nInterrupt received, stopping...")
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    print("Sending start command (planner mode) ...")
    for _ in range(5):
        pub.send(build_command_message(start=True, stop=False, planner=True))
        time.sleep(0.01)

    # Ramp-in: smoothly move the VR 3-point target from the robot's current
    # wrist pose to CSV frame 0.  This is especially important on the real
    # robot, where the initial step could be several cm.  Metrics are NOT
    # collected during this phase (the monitor clock starts afterwards).
    run_ramp_in(
        pub=pub,
        fk=fk,
        feedback=feedback,
        target_frame=frames[0],
        T_pelvis_torso=T_pelvis_torso,
        neck_pos_pelvis=neck_pos_pelvis,
        neck_quat_wxyz=neck_quat_wxyz,
        ramp_in_sec=args.ramp_in_sec,
        rate_hz=args.rate_hz,
        stop_flag=stop_requested,
    )

    dt = 1.0 / args.rate_hz
    start_wall = time.time()
    traj_t0 = frames[0].t
    last_cmd_pulse = 0.0

    idx = 0
    while idx < len(frames) and not stop_requested["value"]:
        now = time.time() - start_wall
        target_t = frames[idx].t - traj_t0
        if now < target_t:
            sleep_t = min(dt, target_t - now)
            time.sleep(max(sleep_t, 0.0))
            continue

        frame = frames[idx]

        # --- Send VR 3-point target ---
        vr_pos, vr_orn = build_vr3pt_targets(
            frame,
            T_pelvis_torso=T_pelvis_torso,
            neck_pos_pelvis=neck_pos_pelvis,
            neck_quat_wxyz_pelvis=neck_quat_wxyz,
        )
        msg = build_planner_message(
            mode=LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
            vr_3pt_position=vr_pos,
            vr_3pt_orientation=vr_orn,
        )
        pub.send(msg)

        if now - last_cmd_pulse > 0.25:
            pub.send(build_command_message(start=True, stop=False, planner=True))
            last_cmd_pulse = now

        q_meas, q_action = feedback.latest()
        elapsed = frame.t - traj_t0

        # --- Feed the IK-style monitors ---
        if q_meas is not None and np.all(np.isfinite(q_meas)) and q_meas.size >= 29:
            lw_pos_m, lw_quat_m, rw_pos_m, rw_quat_m = fk.wrist_poses_in_torso(q_meas[:29])
            left_actual_pose = np.concatenate([lw_pos_m, lw_quat_m])  # (7,)
            right_actual_pose = np.concatenate([rw_pos_m, rw_quat_m])
            left_target_euler = np.concatenate([frame.left_pos, frame.left_rpy])  # (6,)
            right_target_euler = np.concatenate([frame.right_pos, frame.right_rpy])
            tracking_monitor.update(
                elapsed,
                left_target_euler,
                right_target_euler,
                left_actual_pose,
                right_actual_pose,
            )

        if q_action is not None and np.all(np.isfinite(q_action)) and q_action.size >= 29:
            # 14 arm-joint commands (7 left + 7 right) from the policy's
            # last_action (scale + default-angle-adjusted PD targets, MuJoCo
            # order), to match the IK baseline's joint_target_q input.
            arm_targets = np.concatenate(
                [q_action[SONIC_LEFT_ARM_IDX], q_action[SONIC_RIGHT_ARM_IDX]]
            )
            smoothness_monitor.update(elapsed, arm_targets)

        # --- Sample inference-time telemetry ---
        # Same warmup gate as the other monitors so the final summary is
        # directly comparable to the IK baseline's infer_times_ms block.
        enc_us, pol_us = feedback.latest_infer_us()
        if (
            enc_us is not None
            and pol_us is not None
            and (enc_us, pol_us) != last_infer_pair
            and elapsed >= args.warmup_sec
        ):
            last_infer_pair = (enc_us, pol_us)
            enc_ms = enc_us / 1000.0
            pol_ms = pol_us / 1000.0
            log_infer_ms_encoder.append(enc_ms)
            log_infer_ms_policy.append(pol_ms)
            log_infer_ms_total.append(enc_ms + pol_ms)

        # --- Raw log (for offline inspection) ---
        log_t.append(elapsed)
        log_left_pos.append(frame.left_pos.copy())
        log_left_quat.append(frame.left_quat_wxyz.copy())
        log_right_pos.append(frame.right_pos.copy())
        log_right_quat.append(frame.right_quat_wxyz.copy())
        log_body_q_measured.append(
            q_meas.copy() if q_meas is not None else np.full(fk.num_joints, np.nan)
        )
        log_last_action.append(
            q_action.copy() if q_action is not None else np.full(fk.num_joints, np.nan)
        )

        idx += 1

    # --- Shutdown ---
    print("Sending stop command ...")
    for _ in range(5):
        pub.send(build_command_message(start=False, stop=True, planner=True))
        time.sleep(0.02)

    feedback.stop()
    pub.close(linger=0)

    tracking_monitor.print_summary()
    smoothness_monitor.print_summary()
    tracking_monitor.close()

    print_inference_summary(
        total_ms=log_infer_ms_total,
        encoder_ms=log_infer_ms_encoder,
        policy_ms=log_infer_ms_policy,
        warmup_sec=args.warmup_sec,
    )

    if args.output_log:
        out = Path(args.output_log)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            t=np.array(log_t),
            cmd_left_pos_torso=np.array(log_left_pos),
            cmd_left_quat_torso=np.array(log_left_quat),
            cmd_right_pos_torso=np.array(log_right_pos),
            cmd_right_quat_torso=np.array(log_right_quat),
            body_q_measured=np.array(log_body_q_measured),
            last_action=np.array(log_last_action),
            infer_ms_total=np.array(log_infer_ms_total),
            infer_ms_encoder=np.array(log_infer_ms_encoder),
            infer_ms_policy=np.array(log_infer_ms_policy),
        )
        print(f"Saved log to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to a trajectory CSV (wrist_yaw_link in torso frame, wxyz quats).",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Replay publish rate in Hz (default: 50, matches CSV 20ms cadence).",
    )
    p.add_argument(
        "--planner-port",
        type=int,
        default=PLANNER_DEFAULT_PORT,
        help=f"ZMQ port for publishing planner/command topics (default: {PLANNER_DEFAULT_PORT}).",
    )
    p.add_argument(
        "--feedback-host",
        default="localhost",
        help="Host for the g1_debug feedback subscriber (default: localhost).",
    )
    p.add_argument(
        "--feedback-port",
        type=int,
        default=FEEDBACK_DEFAULT_PORT,
        help=f"Port for g1_debug feedback (default: {FEEDBACK_DEFAULT_PORT}).",
    )
    p.add_argument(
        "--output-log",
        default=None,
        help="Optional .npz path to save per-frame commanded + measured data.",
    )
    p.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional CSV path for per-sample tracking errors (TrackingErrorMonitor log).",
    )
    # Knobs aligned with rl_ik_solver/.../trajectory.py's CLI defaults.
    p.add_argument("--print-hz", type=float, default=2.0, help="Live-print rate (Hz, 0 disables).")
    p.add_argument(
        "--warmup-sec",
        type=float,
        default=0.0,
        help="Skip the first N seconds when computing tracking / smoothness "
        "stats.  Default is 0.0 because --ramp-in-sec already drives the "
        "robot onto CSV frame 0 before the metric clock starts, so the "
        "catch-up transient is no longer inside the measurement window.  "
        "Set to 0.5 (the IK baseline's --error_warmup_sec default) if you "
        "disable ramp-in with --ramp-in-sec 0.",
    )
    p.add_argument(
        "--ramp-in-sec",
        type=float,
        default=2.0,
        help="Duration of the pre-replay ramp-in that blends the VR 3-point "
        "target from the robot's current wrist pose (FK on body_q_measured) "
        "to CSV frame 0.  Metrics are NOT collected during this phase.  "
        "Set to 0 to disable.",
    )
    p.add_argument(
        "--lag-search-max-sec",
        type=float,
        default=1.0,
        help="Max time shift (seconds) to search over for lag-compensated RMSE "
        "(matches IK baseline's --error_lag_search_sec default).",
    )
    p.add_argument(
        "--spatial-resample-count",
        type=int,
        default=200,
        help="Number of arc-length-resampled points for the spatial RMSE summary.",
    )
    p.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="Policy control dt (seconds) used for ddq units in the smoothness monitor.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
