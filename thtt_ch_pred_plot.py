import numpy as np
import matplotlib.pyplot as plt

def plot_iq_from_H(H: np.ndarray, sample_idx: int | None = None, rx_idx: int | None = None):
    # H shape: (n_samples, n_rx, n_tx, n_time_steps), complex64
    i = np.random.randint(H.shape[0]) if sample_idx is None else sample_idx
    r = np.random.randint(H.shape[1]) if rx_idx is None else rx_idx

    plt.figure(dpi=150)
    lim = 0.0
    for t in range(H.shape[2]):
        z = H[i, r, t, :]  # complex time series
        lim = max(lim, np.max(np.abs(z)))
        plt.plot(z.real, z.imag, "*-", markersize=3, label=f"Tx antenna {t+1}")
    lim = float(lim) * 1.1
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.title("Channel Gain")
    plt.show()

    return i, r  # return sample and rx index
