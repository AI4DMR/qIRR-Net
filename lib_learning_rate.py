# 2024/08/16
# public


import numpy as np
import tensorflow as tf


class adam:
    # https://doi.org/10.48550/arXiv.1412.6980

    def __init__(
        self,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._is_built = False

    def build(
        self,
        list_gradient: list,
    ) -> None:
        self._list_m = []
        self._list_v = []
        self._list_ret = []
        self._t = 0
        self._beta_1_t = 1.0
        self._beta_2_t = 1.0
        for gradient in list_gradient:
            self._list_m.append(tf.Variable(tf.zeros_like(gradient), trainable=False))
            self._list_v.append(tf.Variable(tf.zeros_like(gradient), trainable=False))
            self._list_ret.append(tf.Variable(tf.zeros_like(gradient), trainable=False))
        self._is_built = True

    def update(
        self,
        list_gradient: list,
    ) -> list:
        if not self._is_built:
            self.build(list_gradient)
        for ret, m, v, g in zip(self._list_ret, self._list_m, self._list_v, list_gradient):
            m.assign(self._beta_1 * m + (1 - self._beta_1) * g)
            v.assign(self._beta_2 * v + (1 - self._beta_2) * g**2)
            self._t += 1
            self._beta_1_t *= self._beta_1
            self._beta_2_t *= self._beta_2
            ret.assign(self._alpha * (m / (1 - self._beta_1_t)) / ((v / (1 - self._beta_2_t)) ** 0.5 + self._epsilon))
        return self._list_ret

    def timestep(self):
        return self._t
