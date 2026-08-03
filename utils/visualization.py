# utils/visualization.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D



def draw_vehicle(ax, state, color):

    x, y, theta = state

    length = 5.0
    width = 2.2

    vehicle = Rectangle(
        (-length/2, -width/2),
        length,
        width,
        linewidth=2,
        edgecolor=color,
        facecolor="none"
    )

    transform = (
        Affine2D()
        .rotate(theta)
        .translate(x, y)
        + ax.transData
    )

    vehicle.set_transform(transform)

    ax.add_patch(vehicle)

    return vehicle



def animate_trajectories(
        x_ref,
        y_ref,
        trajectories,
        colors,
        initial_state
):

    fig, ax = plt.subplots(
        figsize=(8,8)
    )


    # ----------------------
    # reference
    # ----------------------

    ax.plot(
        x_ref,
        y_ref,
        "k--",
        linewidth=1.5,
        label="Reference"
    )


    # ----------------------
    # trajectory line
    # ----------------------

    lines = {}

    for name, traj in trajectories.items():

        line, = ax.plot(
            [],
            [],
            "--",
            linewidth=2,
            color=colors[name],
            label=name
        )

        lines[name] = line



    # ----------------------
    # vehicle
    # ----------------------

    vehicles = {}

    for name, traj in trajectories.items():

        vehicles[name] = draw_vehicle(
            ax,
            traj[0],
            colors[name]
        )


    ax.scatter(
        initial_state[0],
        initial_state[1],
        c="green",
        s=50,
        label="Start"
    )


    ax.axis("equal")
    ax.grid(True)


    ax.set_xlim(
        min(x_ref)-20,
        max(x_ref)+20
    )

    ax.set_ylim(
        min(y_ref)-20,
        max(y_ref)+20
    )


    ax.legend()



    def update(i):

        for name, traj in trajectories.items():

            lines[name].set_data(
                traj[:i+1,0],
                traj[:i+1,1]
            )


            vehicles[name].remove()


            vehicles[name] = draw_vehicle(
                ax,
                traj[i],
                colors[name]
            )


        ax.set_title(
            f"step={i}"
        )


        return (
            list(lines.values())
            +
            list(vehicles.values())
        )



    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(
            list(trajectories.values())[0]
        ),
        interval=50,
        blit=False,
        repeat=False
    )


    plt.show()