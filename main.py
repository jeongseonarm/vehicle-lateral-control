import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.tracks import create_offroad_track
from utils.geometry import calc_heading
from utils.visualization import animate_trajectories

from models.kinematic_bicycle import KinematicBicycle
from controllers.mpc import MPCController
from controllers.pure_pursuit import PurePursuitController


# ==================================
# Reference track & Velocity Profile 생성
# ==================================

x_ref, y_ref = create_offroad_track(radius=100, noise=50)
theta_ref = calc_heading(x_ref, y_ref)

# 목표 속도 프로필 생성 (기본 10 m/s, 종단 지점 감속)
target_velocity = 10.0  # [m/s]
v_ref = np.full_like(x_ref, target_velocity)
v_ref[-20:] = 0.0  # 도착 지점 20개 트랙 포인트 전부터 감속


# ==================================
# Vehicle model 생성
# ==================================

wheelbase = 2.5
dt = 0.1

model = KinematicBicycle(wheelbase=wheelbase, dt=dt)


# ==================================
# Controller 생성 및 범용 래퍼 함수
# ==================================

horizon = 10
mpc = MPCController(model=model, horizon=horizon)
pp = PurePursuitController(wheelbase=wheelbase, lookahead_distance=5.0)


# 종적 속도 P-제어기 (Pure Pursuit / Stanley 용)
def speed_control(target_v, current_v, Kp=1.0):
    return Kp * (target_v - current_v)


# ==================================
# 4D 초기 상태 설정 [x, y, theta, v]
# ==================================

offset_xy = np.random.uniform(-3.0, 3.0, size=2)
offset_theta = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))

initial_state = np.array(
    [
        x_ref[0] + offset_xy[0],
        y_ref[0] + offset_xy[1],
        theta_ref[0] + offset_theta,
        0.0,  # 초기 속도 v = 0.0 m/s
    ]
)

state_mpc = initial_state.copy()
state_pp = initial_state.copy()

trajectory_mpc = []
trajectory_pp = []


# ==================================
# Helper: MPC용 N+1 Reference 추출
# ==================================

def get_mpc_reference(nearest_idx, horizon, x_ref, y_ref, theta_ref, v_ref):
    ref = []
    total_pts = len(x_ref)
    for j in range(horizon + 1):  # N+1 개 포인트 추출
        idx = min(nearest_idx + j, total_pts - 1)
        ref.append([x_ref[idx], y_ref[idx], theta_ref[idx], v_ref[idx]])
    return np.array(ref)  # Shape: (N+1, 4)


# ==================================
# MPC Simulation
# ==================================

previous_index = 0

for _ in tqdm(range(500), desc="MPC"):
    trajectory_mpc.append(state_mpc.copy())

    # Nearest waypoint 계산
    dists = np.hypot(x_ref - state_mpc[0], y_ref - state_mpc[1])
    nearest_idx = max(np.argmin(dists), previous_index)
    previous_index = nearest_idx

    # N+1 차원 레퍼런스 생성 [x, y, theta, v]
    reference = get_mpc_reference(
        nearest_idx, horizon, x_ref, y_ref, theta_ref, v_ref
    )

    # MPC 제어 입력 계산 [a, delta]
    control = mpc.compute_control(state_mpc, reference)

    # 차량 상태 업데이트
    state_mpc = model.update(state_mpc, control)

trajectory_mpc = np.array(trajectory_mpc)


# ==================================
# Pure Pursuit Simulation
# ==================================

previous_index_pp = 0

for _ in tqdm(range(500), desc="Pure Pursuit"):
    trajectory_pp.append(state_pp.copy())

    dists = np.hypot(x_ref - state_pp[0], y_ref - state_pp[1])
    nearest_idx = max(np.argmin(dists), previous_index_pp)
    previous_index_pp = nearest_idx

    # 1. 횡적 제어 (Steering angle delta)
    delta = pp.compute_control(state_pp, x_ref, y_ref)

    # 2. 종적 제어 (Acceleration a)
    target_v = v_ref[nearest_idx]
    a = speed_control(target_v, state_pp[3])

    # 통일된 입력 형식 [a, delta]
    control = np.array([a, delta])

    # 차량 상태 업데이트
    state_pp = model.update(state_pp, control)

trajectory_pp = np.array(trajectory_pp)


# ==================================
# Animation & Plotting
# ==================================

trajectories = {"MPC": trajectory_mpc, "Pure Pursuit": trajectory_pp}
colors = {"MPC": "red", "Pure Pursuit": "blue"}

animate_trajectories(x_ref, y_ref, trajectories, colors, initial_state)