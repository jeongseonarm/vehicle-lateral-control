import numpy as np


class PurePursuitController:

    def __init__(
            self,
            wheelbase,
            velocity=10.0,
            constant=0.5
    ):
        """
        Pure Pursuit Controller

        Parameters
        ----------
        wheelbase : float
            차량 wheelbase [m]

        velocity : float
            Constant velocity [m/s]
        """

        self.L = wheelbase
        self.v = velocity
        self.k = constant

        self.delta_min = np.deg2rad(-30)
        self.delta_max = np.deg2rad(30)


    def compute_control(
            self,
            state,
            x_ref,
            y_ref
    ):
        """
        Parameters
        ----------
        state :
            [x, y, theta]

        x_ref :
            reference x

        y_ref :
            reference y

        Returns
        -------
        control :
            [v, delta]
        """

        x = state[0]
        y = state[1]
        theta = state[2]

        Ld = self.k * self.v

        Px = x + Ld * np.cos(theta)
        Py = y + Ld * np.sin(theta)

        distance = np.sqrt(
                            (x_ref - Px) ** 2 +
                            (y_ref - Py) ** 2
                            )

        idx = np.argmin(distance)

        L_bar = distance[idx]  
        Cx = x_ref[idx]  # Close x        
        Cy = y_ref[idx]  # Close y

        # P1(x,y), P2(px,py), C(cx,cy)
        a = y - Py
        b = Px - x
        c = x * Py - Px * y

        y2 = (a * Cx + b * Cy + c) / np.sqrt(a**2 + b**2)

        delta = (2 * self.L * y2) / (L_bar**2)

        delta = np.clip(
                    delta,
                    self.delta_min,
                    self.delta_max
                )


        return np.array(
            [
                self.v,
                delta
            ]
        )