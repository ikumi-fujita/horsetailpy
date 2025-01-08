import numpy as np
from numba import jit


@jit(nopython=True)
def horsetail(
    cell_length: float = 14.0,
    n_mt_oneside: int = 3,
    v: tuple[float, float, float, float] = (3.3, 3.3, 4.2, 3.7),
    f: tuple[float, float, float, float] = (0.001, 0.03, 0, 0),
    r: float = 2.0,
    eta: float = 1.0,
    F_push: float = 4.0,
    F_pull: float = 2.0,
    dt: float = 1.0,
    T_total: float = 240.0,
    output_interval: int = 60,
    seed: float | None = None,
) -> np.ndarray:
    """
    Horsetail Simulation
    Python version, numba accelerated

    HTsim simulates the positions of the SPB-nucleus and MT tips.

    Parameters
    ----------
    cell_length: float
        cell length [um].
    n_mt_oneside: int
        number of MT on each side of the SPB.
    v: tuple[float, float, float, float]
        MT velocity [um/min].
        v[1], v_plus(cyt); v[2], v_plus(ctx); v[3], v_minus(cyt); v[4], v_minus(ctx).
    f: tuple[float, float, float, float]
        Catastroph or rescue frequency [/s].
        f[1], f_cat(cyt); f[2], f_cat(ctx); f[3], f_res(cyt); f[4], f_res(ctx).
    r: float
        Stokes radius of the nucleus [um].
    eta: float
        viscosity of cytosol [kg/m s].
    F_push: float
        Pushing force [pN].
    F_pull: float
        Pulling force [pN].
    dt: float
        Step of simultion [s].
    T_total: float
        Total time of simulation [min].
    output_interval: int
        Output results every this interval.

    Returns
    -------
    res: np.ndarray
    """

    if seed is not None:
        np.random.seed(seed)

    ctx = (-cell_length / 2, cell_length / 2)
    x_0 = 0
    n_steps = int(60 * T_total / dt)
    n_mt = n_mt_oneside * 2
    sign = np.array([1, -1] * n_mt_oneside)
    dx = np.array(v) / 60 * dt
    dx_in_mode = np.array([dx[0], -dx[2], dx[1], -dx[3], 0])

    f_cat = np.array([f[0], f[1]]) * dt
    f_res = np.array([f[2], f[3]]) * dt

    C_drag = 6.0 * np.pi * r * eta
    C_buckle = 25 * np.pi**2
    len_push = (C_buckle / F_push) ** (1 / 2)

    x = x_0
    length = np.zeros(n_mt).astype(np.float64)
    plus_end = np.array([x] * n_mt)

    ### MT modes ###
    # 0: Growing in the cytoplasm
    # 1: Shrinking in the cytoplasm
    # 2: Growing along the cortex
    # 3: Shrinking along the cortex and generating pulling force
    # 4: Pushing the cortex
    mode = np.zeros(n_mt).astype(np.int64)

    ### Initialization ###
    res = np.zeros(shape=(int(n_steps / output_interval + 1), n_mt + 1))

    res[0, 0] = x
    res[0, 1:] = plus_end

    ### Run Simulation ###
    for t in range(n_steps):
        ## Force Generation
        F = 0
        for mt in range(n_mt):
            # Pulling force
            if mode[mt] == 3:
                F += +sign[mt] * F_pull
            # Pushing force
            if mode[mt] == 4:
                F += -sign[mt] * F_push

        ## SPB movement
        # Change the length of each MT according to the mode
        length += dx_in_mode[mode]

        # Change the SPB position according to the Force
        x = x + F / C_drag * dt
        if x < ctx[0]:
            x = ctx[0]
        elif x > ctx[1]:
            x = ctx[0]

        # Change the plue end position of each MT
        plus_end = x + sign * length

        ### Change the mode of MT depending on the current mode and plus end position
        for mt in range(n_mt):
            # Mode 0: Growing in the cytoplasm
            if mode[mt] == 0:
                if (plus_end[mt] <= ctx[0]) or (plus_end[mt] >= ctx[1]):
                    if length[mt] < len_push:
                        # changed to Pushing the cortex
                        mode[mt] = 4
                    else:
                        # changed to Growing along the cortex
                        mode[mt] = 2

            # Mode 1: Shrinking in the cytoplasm
            elif mode[mt] == 1:
                if (plus_end[mt] <= ctx[0]) or (plus_end[mt] >= ctx[1]):
                    # changed to Shrinking along the cortex
                    mode[mt] = 3

            # Mode 2: Growing along the cortex
            elif mode[mt] == 2:
                if length[mt] < len_push:
                    # changed to Pushing the cortex
                    mode[mt] = 4
                if (plus_end[mt] > ctx[0]) and (plus_end[mt] < ctx[1]):
                    # changed to Growing in the cytoplasm
                    mode[mt] = 0

            # Mode 3: Shrinking along the cortex
            elif mode[mt] == 3:
                if (plus_end[mt] > ctx[0]) and (plus_end[mt] < ctx[1]):
                    # changed to Shrinking in the cytoplasm
                    mode[mt] = 1

            # Mode 4: Pushing the cortex
            elif mode[mt] == 4:
                if (plus_end[mt] > ctx[0]) and (plus_end[mt] < ctx[1]):
                    # changed to Growing in the cytoplasm
                    mode[mt] = 0

            ### Catasrophe or Rescue of MT
            # Mode 0: Growing in the cytoplasm
            if mode[mt] == 0:
                if np.random.uniform() < f_cat[0]:
                    # changed to Shrinking in the cytoplasm
                    mode[mt] = 1

            # Mode 1: Shrinking in the cytoplasm
            elif mode[mt] == 1:
                if (length[mt] < 0.1) or (np.random.uniform() < f_res[0]):
                    # changed to Growing in the cytoplasm
                    mode[mt] = 0

            # Mode 2: Growing along the cortex
            elif mode[mt] == 2:
                if np.random.uniform() < f_cat[1]:
                    # changed to Shrinking along the cortex
                    mode[mt] = 3

            # Mode 3: Shrinking along the cortex
            elif mode[mt] == 3:
                if (length[mt] < 0.1) or (np.random.uniform() < f_res[1]):
                    # changed to Growing along the cortex
                    mode[mt] = 2

            # Mode 4: Pushing the cortex
            elif mode[mt] == 4:
                if np.random.uniform() < f_cat[1]:
                    # changed to Shrinking along the cortex
                    mode[mt] = 3

        ### Output the results every output_interval
        if t % output_interval == 0:
            res[t // output_interval + 1, 0] = x
            res[t // output_interval + 1, 1:] = plus_end

    return res
