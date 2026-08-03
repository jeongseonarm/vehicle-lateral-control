import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.tracks import create_offroad_track
from utils.geometry import calc_heading
from utils.visualization import draw_vehicle
from utils.visualization import animate_trajectories

from models.kinematic_bicycle import KinematicBicycle

from controllers.mpc import MPCController
from controllers.pure_pursuit import PurePursuitController


# ==================================
# Reference track 생성
# ==================================

x_ref, y_ref = create_offroad_track(
    radius=100,
    noise=50
)

theta_ref = calc_heading(
    x_ref,
    y_ref
)


# ==================================
# Vehicle model 생성
# ==================================

model = KinematicBicycle(
    wheelbase=5,
    dt=0.1
)


# ==================================
# Controller 생성
# ==================================

horizon = 10

mpc = MPCController(
    model=model,
    horizon=horizon
)

pp = PurePursuitController(
    wheelbase= 5,
    velocity= 30.0,
    constant= 0.5
)


# ==================================
# 초기 상태
# ==================================

offset_xy = np.random.uniform(
    -5.0,
    5.0,
    size=2
)

offset_theta = np.random.uniform(
    -np.deg2rad(30),
    np.deg2rad(30)
)


initial_state = np.array(
    [
        x_ref[0] + offset_xy[0],
        y_ref[0] + offset_xy[1],
        theta_ref[0] + offset_theta
    ]
)

state_mpc = initial_state.copy()
state_pp = initial_state.copy()

trajectory_mpc = []
trajectory_pp = []


# ==================================
# MPC Simulation
# ==================================

previous_index = 0

for _ in tqdm(range(500), desc="MPC"):

    trajectory_mpc.append(
        state_mpc.copy()
    )

    # ------------------------------
    # nearest waypoint
    # ------------------------------

    distance = np.sqrt(
        (x_ref - state_mpc[0]) ** 2 +
        (y_ref - state_mpc[1]) ** 2
    )

    nearest_index = np.argmin(distance)

    nearest_index = max(
        nearest_index,
        previous_index
    )

    previous_index = nearest_index

    # ------------------------------
    # MPC reference 생성
    # ------------------------------

    reference = []

    for j in range(horizon):

        index = nearest_index + j

        if index >= len(x_ref):
            index = len(x_ref) - 1

        reference.append(
            [
                x_ref[index],
                y_ref[index],
                theta_ref[index]
            ]
        )

    reference = np.array(reference)

    # ------------------------------
    # MPC control
    # ------------------------------

    control = mpc.compute_control(
        state_mpc,
        reference
    )

    # ------------------------------
    # Vehicle update
    # ------------------------------

    state_mpc = model.update(
        state_mpc,
        control
    )

trajectory_mpc = np.array(
    trajectory_mpc
)


# ==================================
# Pure Pursuit Simulation
# ==================================

for _ in tqdm(range(500), desc="Pure Pursuit"):

    trajectory_pp.append(
        state_pp.copy()
    )

    control = pp.compute_control(
        state_pp,
        x_ref,
        y_ref
    )

    state_pp = model.update(
        state_pp,
        control
    )

trajectory_pp = np.array(
    trajectory_pp
)


# ==================================
# Animation
# ==================================

trajectories = {
    "MPC": trajectory_mpc,
    "Pure Pursuit": trajectory_pp
}


colors = {
    "MPC": "red",
    "Pure Pursuit": "blue"
}


animate_trajectories(
    x_ref,
    y_ref,
    trajectories,
    colors,
    initial_state
)