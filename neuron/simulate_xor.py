from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def optim(out: np.ndarray) -> np.float32:
    assert out.shape == (4,)
    return (
        (10 - out[0]) ** 2
        + (20 - out[1]) ** 2
        + (20 - out[2]) ** 2
        + (10 - out[3]) ** 2
    )


def soma(vx: float, ux: float, ix: float, dtx: float) -> tuple:
    """Excitatory neuron"""
    # Model parameters
    dt = 0.5
    a = 0.02
    b = 0.2
    c = -65  # mV
    d = 8

    # Copy values
    v = vx
    u = ux
    i = ix

    if v < 35:
        dv = 0.04 * v**2 + 5 * v + 140 - u
        vv = v + (dv + i) * dt

        # Accelerate computation
        if vv > 20:
            vv = 35

        du = a * (b * v - u)
        uu = u + dt * du
    else:
        v = 35
        vv = c
        uu = u + d

    return vv, uu, i


def simulation(
    simulation_time: int,
    dt: float,
    i: Tuple[np.ndarray, np.ndarray],
    v: np.ndarray,
    u: np.ndarray,
    w1: np.float32,
    w2: np.float32,
) -> None:
    """Simulate Izhikevich neuron model"""
    i1, i2 = i
    v1 = v
    u1 = u

    out = np.zeros(shape=(4,), dtype=np.float32)

    for t in range(simulation_time - 1):
        # Input signals
        if t < 20 / dt:
            i1[t] = 10  # pA
            i2[t] = 10

            if out[0] == 0 and v1[t] >= 0:
                out[0] = 20 - t
        elif t < 40 / dt:
            i1[t] = 20
            i2[t] = 10

            if out[1] == 0 and v1[t] >= 0:
                out[1] = 40 - t
        elif t < 60 / dt:
            i1[t] = 10
            i2[t] = 20

            if out[2] == 0 and v1[t] >= 0:
                out[2] = 60 - t
        elif t < 80 / dt:
            i1[t] = 20
            i2[t] = 20

            if out[3] == 0 and v1[t] >= 0:
                out[3] = 80 - t
        else:
            i1[t] = 10
            i2[t] = 10

        i_a = w1 * i1[t] + w2 * i2[t]

        # Network architecture
        state_b = soma(v1[t], u1[t], i_a, dt)
        v1[t + 1] = state_b[0]
        u1[t + 1] = state_b[1]

    # Replace last value to boosted value
    i1[-1] = 10
    i2[-1] = 10

    print(f"Time response vector: {out}")
    return


def plot_simulation(potential: np.ndarray, current: Tuple[np.ndarray, ...], time: int) -> None:
    assert len(current) == 2
    
    time_x = np.arange(0, time)
    
    i1, i2 = current

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(9.5, 5.5, forward=True)
    fig.tight_layout(h_pad=3, w_pad=3)

    ax1.plot(time_x, potential)
    ax1.set_title("Membrane voltage")
    # ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("v1 [mV]")

    ax2.plot(time_x, i1, color="red")
    ax2.set_title("Input current 1")
    # ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("i1 [pA]")

    ax3.plot(time_x, i2, color="red")
    ax3.set_title("Input current 2")
    ax3.set_xlabel("Time [ms]")
    ax3.set_ylabel("i2 [pA]")

    fig.savefig("xor.png")
    plt.show()
    return


def main() -> None:
    # Simulation settings
    dt = 1  # ms
    time = 80  # units
    simulation_time = int(np.ceil(time / dt))

    # Weights
    w1 = 0.5227
    w2 = 0.8092

    # Allocate mem
    i1 = np.zeros((simulation_time,))
    i2 = np.zeros((simulation_time,))

    v1 = np.zeros((simulation_time,))

    u1 = np.zeros((simulation_time,))

    # Electric potential
    v1[0] = -70 + np.random.rand(1) / 10
    # Steady state
    u1[0] = -14

    simulation(simulation_time, dt, (i1, i2), v1, u1, w1, w2)
    plot_simulation(v1, (i1, i2), simulation_time)
    return


if __name__ == "__main__":
    main()
