import numpy as np
import matplotlib.pyplot as plt


def plot_signal(signal, title: str):
    signal_length = len(signal)
    t = np.linspace(0, signal_length, signal_length)
    plt.figure(figsize=(20, 6))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.plot(t, signal)


def plot_two_signals(signals, title: str):
    signal_length = len(signals[0])
    t = np.linspace(0, signal_length, signal_length)
    plt.figure(figsize=(20, 6))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.plot(t, signals[0])
    plt.plot(t, signals[1])