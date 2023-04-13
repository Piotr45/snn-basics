from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def get_Izhikevich_params(
    a=0.02, b=0.20, c=-65.0, d=8.0, mode=None
) -> Tuple[float, ...]:
    """Izhikevich two-variable neuron model.
    Parameters
    ----------
    mode : optional, str
        The neuron spiking mode.
    a : float
        It determines the time scale of the recovery variable :math:`u`.
    b : float
        It describes the sensitivity of the recovery variable :math:`u` to
        the sub-threshold fluctuations of the membrane potential :math:`v`.
    c : float
        It describes the after-spike reset value of the membrane potential
        :math:`v` caused by the fast high-threshold :math:`K^{+}` conductance.
    d : float
        It describes after-spike reset of the recovery variable :math:`u` caused
        by slow high-threshold :math:`Na^{+}` and :math:`K^{+}` conductance.
    t_refractory : float
        Refractory period length. [ms]
    noise : float
        The noise fluctuation.
    V_th : float
        The membrane potential threshold.
    """

    if mode in ["Regular Spiking", "RS"]:
        a, b, c, d = [0.02, 0.2, -65, 8]
    elif mode in ["Intrinsically Bursting", "IB"]:
        a, b, c, d = [0.02, 0.2, -55, 4]
    elif mode in ["Chattering", "CH"]:
        a, b, c, d = [0.02, 0.2, -50, 2]
    elif mode in ["Fast Spiking", "FS"]:
        a, b, c, d = [0.1, 0.2, -65, 2]
    elif mode in ["Thalamo-cortical", "TC"]:
        a, b, c, d = [0.02, 0.25, -65, 0.05]
    elif mode in ["Resonator", "RZ"]:
        a, b, c, d = [0.1, 0.26, -65, 2]
    elif mode in ["Low-threshold Spiking", "LTS"]:
        a, b, c, d = [0.02, 0.25, -65, 2]

    return a, b, c, d


class SNN_classificator:
    """"""

    def __init__(
        self,
        offset: int,
        weights: np.ndarray,
        encoding_method: str,
        sampling_time: float,
        capacity: float,
    ) -> None:
        # Hyperparameters
        self.offset: int = offset
        self.weights: np.ndarray = weights
        self.encoding_method: str = encoding_method
        self.sampling_time: float = sampling_time
        self.capacity: float = capacity

        # NN
        self.mode = "RS"
        self.input_currents = None
        self.output_voltages = None
        self.neuron_potentials = None
        self.output_spikes = np.zeros(
            shape=(2, 2)
        )  # Two output voltages, first two spikes

    def run_simulation(
        self, simulation_time: int, dt: float, reset_time: float = 200
    ) -> None:
        """"""
        # Check if simulation time is long enough
        assert simulation_time >= (reset_time + self.sampling_time * 2)

        # Init simulation
        self._init_simulation(simulation_time)

        # Simulation
        for t in range(simulation_time - 1):
            if t < reset_time:
                self.input_currents[:, t] = 10  # pA
                # TODO self._encode_information()
            elif t < reset_time + self.sampling_time:
                self.input_currents[:, t] = 40  # pA
            elif t < reset_time + 2 * self.sampling_time:
                self.input_currents[:, t] = 30  # pA
            else:
                self.input_currents[:, t] = 10  # pA

            self._update(t)
        return

    def _init_simulation(self, simulation_time: int) -> None:
        self.input_currents = np.zeros(shape=(4, simulation_time))
        self.output_voltages = np.zeros(shape=(2, simulation_time))
        self.neuron_potentials = np.zeros(shape=(2, simulation_time))

        self.output_voltages[0, 0] = -70 + np.random.rand(1) / 10
        self.output_voltages[1, 0] = -70 + np.random.rand(1) / 10
        self.neuron_potentials[:, 0] = -14
        return

    def _encode_information(self, neuron_idx: int, dt: float, t: int) -> None:
        if self.encoding_method == "Latency Coding":  # Time between first two spikes
            self._latency_encoding(neuron_idx, dt, t)
        else:
            raise ValueError("Wrong encoding method!")

    def _latency_encoding(self, neuron_idx: int, dt: float, t: int) -> None:
        """"""
        # Namespaces
        out_spike = self.output_spikes[neuron_idx]
        out_voltage = self.output_voltages[neuron_idx]

        # Check if classificator has two neurons
        assert out_spike.shape[0] == 2 and out_voltage.shape[0] == 2

        # Update delta time values
        if out_spike[0] == 0 and out_voltage[t] >= 0:
            self.output_spikes[neuron_idx][0] = dt
            return
        elif out_spike[0] != 0 and out_spike[1] == 0 and out_voltage[t] >= 0:
            self.output_spikes[neuron_idx][1] = dt
            return
        return

    def _update(self, t: int) -> None:
        """"""
        i_a = np.dot(self.weights, self.input_currents[:, t])

        neuron_a = self._soma(
            self.output_voltages[0][t], self.neuron_potentials[0][t], i_a, 1
        )
        neuron_b = self._soma(
            self.output_voltages[1][t], self.neuron_potentials[1][t], i_a, 1
        )

        self.output_voltages[0][t + 1], self.neuron_potentials[0][t + 1] = neuron_a
        self.output_voltages[1][t + 1], self.neuron_potentials[1][t + 1] = neuron_b

    def _soma(self, v: float, u: float, i: float, dt: float) -> Tuple[float, float]:
        """"""
        a, b, c, d = get_Izhikevich_params(mode=self.mode)
        # v = vx

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

        return vv, uu


def main() -> None:
    simulation_time = 1200  # ms

    snn_classifier = SNN_classificator(
        10, np.random.rand(4,), "Latency Encoding", 500.0, 40
    )

    snn_classifier.run_simulation(simulation_time, 1)


if __name__ == "__main__":
    main()
