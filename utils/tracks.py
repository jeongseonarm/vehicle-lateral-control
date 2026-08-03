import numpy as np

def create_running_track(
        straight_length=100,
        radius=20,
        n_straight=100,
        n_curve=100
):
    """
    100m 육상 트랙 형태의 reference path 생성 함수

    Parameters
    ----------
    straight_length : float, optional
        직선 구간의 길이 [m]

    radius : float, optional
        곡선 구간(반원)의 반지름 [m]

    n_straight : int, optional
        직선 구간을 표현할 waypoint 개수

    n_curve : int, optional
        곡선 구간을 표현할 waypoint 개수

    Returns
    -------
    x_ref : numpy.ndarray
        Reference path의 x 좌표 배열 [m]

    y_ref : numpy.ndarray
        Reference path의 y 좌표 배열 [m]
    """

    # -----------------------
    # 아래쪽 직선
    # -----------------------
    x1 = np.linspace(
        0,
        straight_length,
        n_straight
    )

    y1 = np.zeros(
        n_straight
    )

    # -----------------------
    # 오른쪽 반원
    # -----------------------
    theta1 = np.linspace(
        0,
        np.pi,
        n_curve
    )

    x2 = (
        straight_length
        + radius * np.sin(theta1)
    )

    y2 = (
        radius
        - radius * np.cos(theta1)
    )

    # -----------------------
    # 위쪽 직선
    # -----------------------
    x3 = np.linspace(
        straight_length,
        0,
        n_straight
    )

    y3 = np.ones(
        n_straight
    ) * (2 * radius)

    # -----------------------
    # 왼쪽 반원
    # -----------------------
    theta2 = np.linspace(
        np.pi,
        2 * np.pi,
        n_curve
    )

    x4 = (
        radius * np.sin(theta2)
    )

    y4 = (
        radius
        - radius * np.cos(theta2)
    )

    # -----------------------
    # 전체 경로 합치기
    # (구간 연결부의 중복 waypoint 제거)
    # -----------------------
    x_ref = np.concatenate(
        [
            x1[:-1],
            x2[:-1],
            x3[:-1],
            x4
        ]
    )

    y_ref = np.concatenate(
        [
            y1[:-1],
            y2[:-1],
            y3[:-1],
            y4
        ]
    )

    return x_ref, y_ref

from scipy.interpolate import splprep, splev

def create_offroad_track(
        radius=50,
        noise=12,
        n_control=12,
        n_points=600,
        seed=0
):
    """
    자연스러운 오프로드 형태의 폐곡선 생성
    """

    np.random.seed(seed)

    # 원 위에 control point 생성
    theta = np.linspace(
        0,
        2*np.pi,
        n_control,
        endpoint=False
    )

    r = radius + np.random.uniform(
        -noise,
        noise,
        n_control
    )

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 폐곡선
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Cubic spline
    tck, _ = splprep(
        [x, y],
        s=20,
        per=True
    )

    u = np.linspace(
        0,
        1,
        n_points
    )

    x_ref, y_ref = splev(
        u,
        tck
    )

    return np.array(x_ref), np.array(y_ref)