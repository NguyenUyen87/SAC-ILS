"""Mersenne-Twister stream shaped like MATLAB's rand/randi, taken from the Python ports of the
MATLAB code of MCS, AS and LTIU; also used by TBLS and SAC-ILS for tie-breaking."""
import math

import numpy as np


class MatlabRNG:
    """Mersenne Twister stream shaped like MATLAB's rand/randi.

    For the same seed, rand() reproduces MATLAB's ``rng(seed,'twister'); rand``
    and rand_matrix() fills column-major as MATLAB does. randi() is
    ceil(rand*imax), which is not guaranteed to match MATLAB's internal randi.
    """

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        # MATLAB's rng(0,'twister') is Mersenne Twister seeded with 5489
        self._rs = np.random.RandomState(5489 if seed == 0 else seed)

    def rand(self):
        return self._rs.random_sample()

    def rand_matrix(self, rows, cols):
        return self._rs.random_sample(rows * cols).reshape((rows, cols), order="F")

    def randi(self, imax):
        if imax < 1:
            raise ValueError("randi: imax must be >= 1")
        return max(1, math.ceil(self._rs.random_sample() * imax))
