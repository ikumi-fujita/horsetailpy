import numpy as np
from numba import jit


@jit(nopython=True)
def horsetail(
    l: float = 14.0,
    n_mt_oneside: int = 3,
    v: tuple[float, float, float, float] = (3.3, 3.3, 4.2, 3.7),
    f: tuple[float, float, float, float] = (0.001, 0.03, 0, 0),
    r: float = 2.0,
    eta: float = 1.0,
    F_push: float = 4.0,
    F_pull: float = 2.0,
    dt: float = 1,
    rs: float = 60,
    min: float = 240,
) -> np.ndarray:
    """
    Horsetail Simulation
    Python version, numba accelerated

    HTsim simulates the positions of the SPB-nucleus and MT tips.

    Parameters
    ----------
    l: float
        cell length [um]
    n_mt_oneside: int
        number of MT on each side of the SPB
    v: tuple[float, float, float, float]
        MT velocity [um/min]
        v[1],v_plus(cyt); v[2],v_plus(ctx); v[3],v_minus(cyt); v[4],v_minus(ctx)
    f: tuple[float, float, float, float]
        catastroph or rescue frequency [/s]
        f[1],f_cat(cyt); f[2],f_cat(ctx); f[3],f_res(cyt); f[4],f_res(ctx)
    r: float
        Stokes radius of the nucleus [um]
    eta: float
        viscosity of cytosol [kg/m s]
    F_push: float
        Pushing force [pN]
    F_pull: float
        Pulling force [pN]
    dt: float
        one step of simultion [s]
        output results at every 'rs' steps
    min: float
        duration of simulation [min]

    Returns
    -------
    res: np.ndarray
    """

    cortex = (-l / 2, l / 2)
    init_position = 0
    st = 60 * min / dt
    n_mt = n_mt_oneside * 2
    microtubules = [i for i in range(n_mt)]
    sign = np.array([1, -1] * n_mt_oneside)
    v_plus = np.array([v[0], v[1]]) / 60 * dt
    v_minus = np.array([v[2], v[3]]) / 60 * dt
    v_mode = np.array([v_plus[0], -v_minus[0], v_plus[1], -v_minus[1], 0])

    f_cat = np.array([f[0], f[1]]) * dt
    f_res = np.array([f[2], f[3]]) * dt

    C_drag = 6.0 * np.pi * r * eta
    C_buckle = 25 * np.pi**2
    push_length = (C_buckle / F_push) ** (1 / 2)

    position = init_position
    mt_length = np.zeros(n_mt)
    mt_tip = np.array([position] * n_mt)
    mt_mode = np.zeros(n_mt).astype(np.int64)
    # 0: Growing in the cytoplasm
    # 1: Shrinking in the cytoplasm
    # 2: Growing along the cortex
    # 3: Shrinking along the cortex and generating pulling force
    # 4: Pushing the cortex

    ###### Initialization ######
    result = np.zeros(shape=(int(st / rs + 1), n_mt + 1))

    result[0, 0] = position
    result[0, 1:] = mt_tip

    times = [i for i in range(int(st))]

    ###### Run Simulation ######
    for t in times:
        ### Force Generation
        Force = 0
        for x in microtubules:
            # Pulling force
            if mt_mode[x] == 3:
                Force = Force + sign[x] * F_pull
            # Pushing force
            if mt_mode[x] == 4:
                Force = Force - sign[x] * F_push

        ### SPB movement
        # Change the length of each MT according to the mt_mode
        mt_length += v_mode[mt_mode]

        # Change the SPB position according to the Force
        position = position + Force / C_drag * dt
        if position < cortex[0]:
            position = cortex[0]
        elif position > cortex[1]:
            position = cortex[0]
        # Change the tip position of each MT
        mt_tip = position + sign * mt_length

        for x in microtubules:
            ### Mode change depending on the position
            # Growing in the cytoplasm
            if mt_mode[x] == 0:
                if (mt_tip[x] <= cortex[0]) or (mt_tip[x] >= cortex[1]):
                    if mt_length[x] < push_length:
                        # changed to Pushing the cortex
                        mt_mode[x] = 4
                    else:
                        # changed to Growing along the cortex
                        mt_mode[x] = 2

            # Shrinking in the cytoplasm
            elif mt_mode[x] == 1:
                if (mt_tip[x] <= cortex[0]) or (mt_tip[x] >= cortex[1]):
                    # changed to Shrinking along the cortex
                    mt_mode[x] = 3

            # Growing along the cortex
            elif mt_mode[x] == 2:
                if mt_length[x] < push_length:
                    # changed to Pushing the cortex
                    mt_mode[x] = 4
                if (mt_tip[x] > cortex[0]) and (mt_tip[x] < cortex[1]):
                    # changed to Growing in the cytoplasm
                    mt_mode[x] = 0

            # Shrinking along the cortex
            elif mt_mode[x] == 3:
                if (mt_tip[x] > cortex[0]) and (mt_tip[x] < cortex[1]):
                    # changed to Shrinking in the cytoplasm
                    mt_mode[x] = 1

            # Pushing the cortex
            elif mt_mode[x] == 4:
                if (mt_tip[x] > cortex[0]) and (mt_tip[x] < cortex[1]):
                    # changed to Growing in the cytoplasm
                    mt_mode[x] = 0

            ### catasrophe or rescue of each MT
            # Growing in the cytoplasm
            if mt_mode[x] == 0:
                if np.random.uniform() < f_cat[0]:
                    # changed to Shrinking in the cytoplasm
                    mt_mode[x] = 1

            # Shrinking in the cytoplasm
            elif mt_mode[x] == 1:
                if (mt_length[x] < 0.1) or (np.random.uniform() < f_res[0]):
                    # changed to Growing in the cytoplasm
                    mt_mode[x] = 0

            # Growing along the cortex
            elif mt_mode[x] == 2:
                if np.random.uniform() < f_cat[1]:
                    # changed to Shrinking along the cortex
                    mt_mode[x] = 3

            # Shrinking along the cortex
            elif mt_mode[x] == 3:
                if (mt_length[x] < 0.1) or (np.random.uniform() < f_res[1]):
                    # changed to Growing along the cortex
                    mt_mode[x] = 2

            # Pushing the cortex
            elif mt_mode[x] == 4:
                if np.random.uniform() < f_cat[1]:
                    # changed to Shrinking along the cortex
                    mt_mode[x] = 3

        ### Output results once in every 'rs' step
        if t % rs == 0:
            result[int(t / rs) + 1, 0] = position
            result[int(t / rs) + 1, 1:] = mt_tip

    return result
