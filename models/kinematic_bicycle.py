import numpy as np


class KinematicBicycle:
    """
    Kinematic Bicycle Model

    State:
        x     : position x [m]
        y     : position y [m]
        theta : heading angle [rad]

    Control:
        v     : velocity [m/s]
        delta : steering angle [rad]

    Model:
        x_dot     = v*cos(theta)
        y_dot     = v*sin(theta)
        theta_dot = v/L*tan(delta)
    """

    def __init__(
            self,
            wheelbase,
            dt
    ):
        """
        Parameters
        ----------
        wheelbase : float
            Wheelbase L [m]

        dt : float
            Sampling time [s]
        """

        self.L = wheelbase
        self.dt = dt


    def update(
            self,
            state,
            control
    ):
        """
        One step state update

        Parameters
        ----------
        state : numpy.ndarray
            Current state

            [
                x,
                y,
                theta
            ]

        control : numpy.ndarray
            Control input

            [
                v,
                delta
            ]

        Returns
        -------
        next_state : numpy.ndarray
            Next state

            [
                x_next,
                y_next,
                theta_next
            ]
        """

        x, y, theta = state
        v, delta = control


        # continuous model
        x_dot = v * np.cos(theta)

        y_dot = v * np.sin(theta)

        theta_dot = (
            v / self.L
            * np.tan(delta)
        )


        # Euler integration
        x_next = (
            x
            + x_dot * self.dt
        )

        y_next = (
            y
            + y_dot * self.dt
        )

        theta_next = (
            theta
            + theta_dot * self.dt
        )


        # normalize heading
        theta_next = (
            theta_next + np.pi
        ) % (
            2 * np.pi
        ) - np.pi


        return np.array(
            [
                x_next,
                y_next,
                theta_next
            ]
        )