from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


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
    simulation_time: int, dt: float, i1: np.ndarray, v1: np.ndarray, u1: np.ndarray
) -> None:
    """Simulate Izhikevich neuron model"""
    for t in range(simulation_time - 1):
        # Input signals
        if t < 200 / dt:
            i1[t] = 0
        elif t < 600 / dt:
            i1[t] = 20  # pA
        else:
            i1[t] = 0

        # Network architecture
        state_b = soma(v1[t], u1[t], i1[t], dt)
        v1[t + 1] = state_b[0]
        u1[t + 1] = state_b[1]
    return


def plot_simulation(potential: np.ndarray, current: np.ndarray, time: int) -> None:
    time_x = np.arange(0, time)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(9.5, 5.5, forward=True)
    fig.tight_layout(h_pad=3)

    ax1.plot(time_x, potential)
    ax1.set_title("Membrane voltage")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("v1 [mV]")

    ax2.plot(time_x, current, color="red")
    ax2.set_title("Soma current")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("i1 [pA]")

    fig.savefig("izhikevich_neuron_model.png")
    plt.show()
    return


def main() -> None:
    # Simulation settings
    dt = 1  # ms
    time = 1000  # units
    simulation_time = int(np.ceil(time / dt))

    # Allocate mem
    i1 = np.zeros((simulation_time,))
    v1 = np.zeros((simulation_time,))
    u1 = np.zeros((simulation_time,))

    # Electric potential
    v1[0] = -70 + np.random.rand(1) / 10
    # Steady state
    u1[0] = -14

    simulation(simulation_time, dt, i1, v1, u1)
    plot_simulation(v1, i1, simulation_time)
    return


if __name__ == "__main__":
    main()
