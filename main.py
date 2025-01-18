import time

from joblib import Parallel, delayed, cpu_count
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from horsetail import horsetail


def main():
    num_simulations = 10000

    print(f"[Test run]")
    num_jobs = -1
    start_time = time.time()
    res = Parallel(n_jobs=num_jobs)(delayed(horsetail)() for _ in tqdm(range(num_simulations)))
    end_time = time.time()
    n_core = cpu_count()
    print(f"Number of simulations: {num_simulations}")
    print(f"Number of cores: {n_core}")
    print(f"Elapsed time: {np.round(end_time - start_time, 3)} s\n")

    print("[Draw a figure]")
    res = horsetail(seed=1)
    _, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(res[:, 0], c="k", lw=2, zorder=10)
    cmap = cm.terrain
    c = [cmap(i / 5) for i in range(6)]
    for i in range(1, 7, 1):
        ax.plot(res[:, i], lw=1, c=c[i - 1])
    ax.hlines(y=[-7, 7], xmin=0, xmax=240, linestyle="--", colors="gray", lw=1)
    ax.set_xlim(0, 240)
    plt.tight_layout()
    plt.savefig("result.png")
    plt.close()
    print("Exported: result.png")


if __name__ == "__main__":
    main()
