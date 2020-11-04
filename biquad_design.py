"""
Biquad filter design functions
http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
"""

import numpy as np


def pure_gain(gain: float, f0: float, Q: float, fs: float):
    return np.array([gain, 0, 0]), np.array([1, 0, 0])


def low_pass_variable_q_second_order(gain: float, f0: float, Q: float, fs: float):
    A = gain
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([A * (1 - np.cos(w0)) / 2, A *
                       (1 - np.cos(w0)), A * (1 - np.cos(w0)) / 2])
    aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def high_pass_variable_q_second_order(gain: float, f0: float, Q: float, fs: float):
    A = gain
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([A * (1 + np.cos(w0)) / 2, -A *
                       (1 + np.cos(w0)), A * (1 + np.cos(w0)) / 2])
    aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def bandpass(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([np.sin(w0) / 2, 0, - np.sin(w0) / 2])
    aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def notch(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    if abs(w0 - np.pi) < 1e-7:
        # notch filter converges to this:
        bCoeff = np.array([1, 2, 1])
        aCoeff = np.array([1, 2, 1])
    else:
        alpha1 = np.sin(w0) / (2 * Q)
        bCoeff = np.array([1, - 2 * np.cos(w0), 1])
        aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def allpass(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([1 - alpha1, - 2 * np.cos(w0), 1 + alpha1])
    aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def parametric_equalizer(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    A = np.sqrt(gain)
    bCoeff = np.array([1 + alpha1 * A, - 2 * np.cos(w0), 1 - alpha1 * A])
    aCoeff = np.array([1 + alpha1 / A, - 2 * np.cos(w0), 1 - alpha1 / A])
    return bCoeff, aCoeff


def low_shelve(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    A = np.sqrt(gain)
    alpha1 = np.sqrt(max(2 + (1 / Q - 1) * (A + 1 / A), 0)) * \
        np.sin(w0) / 2
    bCoeff = np.array([A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha1),
                       2 * A * ((A - 1) - (A + 1) * np.cos(w0)),
                       A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha1)])
    aCoeff = np.array([(A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha1,
                       - 2 * ((A - 1) + (A + 1) * np.cos(w0)),
                       (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha1])
    return bCoeff, aCoeff


def high_shelve(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    A = np.sqrt(gain)
    alpha1 = np.sqrt(max(2 + (1 / Q - 1) * (A + 1 / A), 0)) * \
        np.sin(w0) / 2
    bCoeff = np.array([A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha1),
                       - 2 * A * ((A - 1) + (A + 1) * np.cos(w0)),
                       A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha1)])
    aCoeff = np.array([(A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha1,
                       2 * ((A - 1) - (A + 1) * np.cos(w0)), (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha1])
    return bCoeff, aCoeff


def butterworth_low_pass_order_two(gain: float, f0: float, Q: float, fs: float):
    A = gain
    Q = 1 / np.sqrt(2)
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([A * (1 - np.cos(w0)) / 2, A *
                       (1 - np.cos(w0)), A * (1 - np.cos(w0)) / 2])
    aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def butterworth_high_pass_order_two(gain: float, f0: float, Q: float, fs: float):
    A = gain
    Q = 1 / np.sqrt(2)
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([A * (1 + np.cos(w0)) / 2, -A *
                       (1 + np.cos(w0)), A * (1 + np.cos(w0)) / 2])
    aCoeff = np.array([1 + alpha1, - 2 * np.cos(w0), 1 - alpha1])
    return bCoeff, aCoeff


def first_order_low_pass(gain: float, f0: float, Q: float, fs: float):
    A = gain
    w0 = 2 * np.pi * f0 / fs
    bCoeff = np.array([A * (np.sin(w0)), A * (np.sin(w0)), 0])
    aCoeff = np.array([np.sin(w0) + np.cos(w0) + 1,
                       (np.sin(w0) - np.cos(w0) - 1), 0])
    return bCoeff, aCoeff


def first_order_high_pass(gain: float, f0: float, Q: float, fs: float):
    A = gain
    w0 = 2 * np.pi * f0 / fs
    bCoeff = np.array([A * (1 + np.cos(w0)), - A * (1 + np.cos(w0)), 0])
    aCoeff = np.array([np.sin(w0) + np.cos(w0) + 1,
                       (np.sin(w0) - np.cos(w0) - 1), 0])
    return bCoeff, aCoeff


def bypass(gain: float, f0: float, Q: float, fs: float):
    return np.array([1, 0, 0]), np.array([1, 0, 0])


def bandpass_zero_dB_peak(gain: float, f0: float, Q: float, fs: float):
    w0 = 2 * np.pi * f0 / fs
    alpha1 = np.sin(w0) / (2 * Q)
    bCoeff = np.array([alpha1, 0, -alpha1])
    aCoeff = np.array([1+alpha1, -2*np.cos(w0), 1-alpha1])
    gain1 = gain
    bCoeff = gain1 * bCoeff
    return bCoeff, aCoeff


BIQUAD_DESIGN_LIBRARY = {
    'gain': pure_gain,
    'variable Q lp2': low_pass_variable_q_second_order,
    'variable q hp2': high_pass_variable_q_second_order,
    'bandpass': bandpass,
    'notch': notch,
    'allpass': allpass,
    'parametric eq': parametric_equalizer,
    'low shelve': low_shelve,
    'high shelve': high_shelve,
    'butterworth lp2': butterworth_low_pass_order_two,
    'butterworth hp2': butterworth_high_pass_order_two,
    'first order lp': first_order_low_pass,
    'first order hp': first_order_high_pass,
    'bypass': bypass,
    'bandpass_0dbpeak': bandpass_zero_dB_peak,
}
