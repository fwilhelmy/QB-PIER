import numpy as np
import matplotlib.pyplot as plt


def qubits_amplitude_encoding(input_sizes: np.ndarray) -> np.ndarray:
    """
    Amplitude encoding:
    To encode d features, we need n qubits such that 2^n >= d.
    Therefore:
        n = ceil(log2(d))
    """
    return np.ceil(np.log2(input_sizes)).astype(int)


def qubits_angle_encoding(input_sizes: np.ndarray) -> np.ndarray:
    """
    Angle encoding:
    Typically 1 feature -> 1 qubit.
    Therefore:
        n = d
    """
    return input_sizes.astype(int)


def main():
    # Range of input dimensions to compare
    input_sizes = np.arange(1, 1000)  # from 1 to 1000 features

    amp_qubits = qubits_amplitude_encoding(input_sizes)
    angle_qubits = qubits_angle_encoding(input_sizes)

    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, amp_qubits, label="Amplitude encoding")
    plt.plot(input_sizes, angle_qubits, label="Angle encoding")

    plt.xlabel("Input size (number of features)")
    plt.ylabel("Required qubits")
    plt.title("Dimensional tradeoff: Amplitude encoding vs Angle encoding")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()