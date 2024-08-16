# 2024/08/16
# public

import typing
import tensorflow as tf

from lib_layers import *


class transformer_encooder(tf.Module):

    def __init__(
        self,
        dict_args_ma: dict,
        dict_args_mlp: dict,
        name=None,
    ):
        """
        dict_args_ma: {
            num_head: int,
            dim_key: int,
            dim_value: int,
            dim_out: int,
            use_relative_position_matrix: bool = False,
        }
        dict_args_mlp: {
            list_dim_out: typing.List,
            list_activation: typing.List,
        }
        """
        super().__init__(name)
        self._is_built = False
        self._dict_args_ma = dict_args_ma
        self._dict_args_mlp = dict_args_mlp

    def build(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor = None,
        m: tf.Tensor = None,
    ) -> None:
        self._is_built = True
        self._ln_1 = layer_normalization([-1, -2], name="ln_1", use_beta=False, use_gamma=False)
        self._ma = multihead_attention(**self._dict_args_ma, name="ma")
        self._ln_2 = layer_normalization([-1, -2], name="ln_2", use_beta=False, use_gamma=False)
        self._mlp = multilayer_perceptron(**self._dict_args_mlp, name="mlp")

    def __call__(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor = None,
        m: tf.Tensor = None,
    ) -> tf.Tensor:
        if not self._is_built:
            self.build(x1, x2, m)

        y = x1 if (x2 is None) else x2

        y1 = self._ma(x1, x2, m)
        y = tf.add(y, y1)
        y = self._ln_1(y)

        y2 = self._mlp(y)
        y = tf.add(y, y2)
        y = self._ln_2(y)

        return y


class encoder(tf.Module):

    def __init__(
        self,
        dict_args_ebd: dict,
        num_tes: int,
        dict_args_tes: dict,
        dim_out: int,
        num_tec: int,
        dict_args_tec: dict,
        name=None,
    ):
        super().__init__(name)
        self._is_built = False
        self._dict_args_ebd = dict_args_ebd
        self._num_tes = num_tes
        self._dict_args_tes = dict_args_tes
        self._dim_out = dim_out
        self._num_tec = num_tec
        self._dict_args_tec = dict_args_tec

    def build(
        self,
        x: tf.Tensor,
        m: tf.Tensor,
    ):
        self._is_built = True
        self._ebd = multilayer_perceptron(**self._dict_args_ebd, name="ebd")
        self._list_tes = []
        for i in range(self._num_tes):
            self._list_tes.append(
                transformer_encooder(**self._dict_args_tes, name="tes_%d" % (i)),
            )
        self._query = tf.Variable(tf.random.normal([1, self._dim_out]))
        self._list_tec = []
        for i in range(self._num_tec):
            self._list_tec.append(
                transformer_encooder(**self._dict_args_tec, name="tec_%d" % (i)),
            )

    def __call__(
        self,
        x: tf.Tensor,  # [B,L,D_in]
        m: tf.Tensor = None,  # [B,L,1]
    ):

        if not self._is_built:
            self.build(x, m)

        if m is not None:
            m_tes = tf.einsum("...ik,...jk->...ij", m, m)  # [B,L,L]
            m_tec = m  # [B,L,1]
        else:
            m_tes = None
            m_tec = None

        y1 = self._ebd(x)  # [B,L,D]
        for tes in self._list_tes:
            y1 = tes(y1, m=m_tes)

        y2 = tf.broadcast_to(self._query, (x.shape[0],) + self._query.shape)  # [B,1,O]
        for tec in self._list_tec:
            y2 = tec(y1, y2, m=m_tec)

        y = tf.reshape(y2, [y2.shape[0], -1])  # [B,O]

        return y


class decoder(tf.Module):

    def __init__(
        self,
        dict_args_mlp: dict,
        name=None,
    ):
        super().__init__(name)
        self._is_built = False
        self._dict_args_mlp = dict_args_mlp

    def build(
        self,
        x: tf.Tensor,
    ):
        self._is_built = True
        self._mlp = MLP(
            **self._dict_args_mlp,
            name="mlp",
        )

    def __call__(
        self,
        x: tf.Tensor,
    ):
        if not self._is_built:
            self.build(x)
        return self._mlp(x)
