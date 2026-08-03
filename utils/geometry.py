import numpy as np


def calc_heading(x_ref, y_ref):
    """
    Reference path의 heading angle 계산

    Parameters
    ----------
    x_ref : numpy.ndarray
        Reference path의 x 좌표 [m]

    y_ref : numpy.ndarray
        Reference path의 y 좌표 [m]

    Returns
    -------
    theta_ref : numpy.ndarray
        각 waypoint에서의 heading angle [rad]
    """

    n = len(x_ref)
    theta_ref = np.zeros(n)

    # 각 waypoint의 heading 계산
    for i in range(n - 1):
        dx = x_ref[i + 1] - x_ref[i]
        dy = y_ref[i + 1] - y_ref[i]

        theta_ref[i] = np.arctan2(dy, dx)

    # 마지막 waypoint는 이전 heading 사용
    theta_ref[-1] = theta_ref[-2]

    return theta_ref