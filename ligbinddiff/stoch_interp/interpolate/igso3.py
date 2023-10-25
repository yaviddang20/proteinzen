import torch
import numpy as np


def f_igso3_approx(omega, t):
    return (
        np.sqrt(np.pi)
        * np.power(t, -3/2)
        * np.exp(t - omega ** 2 / t)
        * (omega -
            np.exp(-np.pi ** 2 / t) * (
               (omega - 2 * np.pi) * np.exp(np.pi * omega / t)
               + (omega + 2 * np.pi) * np.exp(-np.pi * omega / t)
            )
        ) / (2 * np.sin(omega/2))
    )


def igso3_density():
    pass
