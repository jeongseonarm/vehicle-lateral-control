import numpy as np
from scipy.optimize import minimize


class MPCController:

    def __init__(
            self,
            model,
            horizon=10
    ):

        self.model = model

        self.N = horizon


        # ==================================
        # State cost
        # ==================================

        self.Q = np.diag(
            [
                10.0,   # x error
                10.0,   # y error
                1.0     # heading error
            ]
        )


        # ==================================
        # Input cost
        # ==================================

        self.R = np.diag(
            [
                1.0,    # velocity
                10.0    # steering
            ]
        )


        # ==================================
        # Input rate cost
        #
        # (u_k - u_k-1)^T R_delta (u_k-u_k-1)
        # ==================================

        self.R_delta = np.diag(
            [
                1.0,    # velocity change
                50.0    # steering change
            ]
        )


        # ==================================
        # Previous input
        # ==================================

        self.previous_control = None


        # ==================================
        # Input limit
        # ==================================

        self.v_min = 0.0
        self.v_max = 30.0


        self.delta_min = np.deg2rad(-30)
        self.delta_max = np.deg2rad(30)



    def predict(
            self,
            state,
            controls
    ):

        states = []

        x = state.copy()


        for u in controls:

            x = self.model.update(
                x,
                u
            )

            states.append(
                x.copy()
            )


        return np.array(states)



    def cost_function(
            self,
            u_flat,
            state,
            reference
    ):

        controls = u_flat.reshape(
            self.N,
            2
        )


        predicted_states = self.predict(
            state,
            controls
        )


        cost = 0.0


        for k in range(self.N):


            # ==================================
            # State error
            # ==================================

            error = (
                predicted_states[k]
                -
                reference[k]
            )


            # heading error wrap
            error[2] = np.arctan2(
                np.sin(error[2]),
                np.cos(error[2])
            )


            cost += (
                error.T
                @
                self.Q
                @
                error
            )


            # ==================================
            # Input magnitude cost
            # ==================================

            u = controls[k]


            cost += (
                u.T
                @
                self.R
                @
                u
            )


            # ==================================
            # Input rate cost
            # (u_k - u_k-1)
            # ==================================

            if k == 0:

                if self.previous_control is None:

                    du = np.zeros(2)

                else:

                    du = (
                        u
                        -
                        self.previous_control
                    )

            else:

                du = (
                    u
                    -
                    controls[k-1]
                )


            cost += (
                du.T
                @
                self.R_delta
                @
                du
            )


        return cost



    def compute_control(
            self,
            state,
            reference
    ):


        # ==================================
        # Initial guess
        # ==================================

        u0 = np.zeros(
            self.N * 2
        )


        for i in range(self.N):

            u0[2*i] = 20.0
            u0[2*i+1] = 0.0



        # ==================================
        # Input bounds
        # ==================================

        bounds = []


        for _ in range(self.N):

            bounds.append(
                (
                    self.v_min,
                    self.v_max
                )
            )


            bounds.append(
                (
                    self.delta_min,
                    self.delta_max
                )
            )



        # ==================================
        # Optimization
        # ==================================

        result = minimize(
            self.cost_function,
            u0,
            args=(
                state,
                reference
            ),
            bounds=bounds,
            method="SLSQP"
        )



        optimal_controls = result.x.reshape(
            self.N,
            2
        )


        # ==================================
        # Receding Horizon
        # ==================================

        control = optimal_controls[0]


        # 다음 step에서 사용할 이전 입력 저장
        self.previous_control = control.copy()


        return control