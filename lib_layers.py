# 2024/08/16
# public

import typing
import tensorflow as tf


class dense(tf.Module):
    """全连接"""

    def __init__(
        self,
        dim_out: int,
        activation: typing.Callable | None = None,
        name="dense",
    ):
        super().__init__(name)
        self._is_built = False
        assert dim_out > 0
        self._dim_out = dim_out
        self._activation = activation
        if self._activation is None:
            self._activation = lambda x: x
        elif self._activation == "tanh":
            self._activation = tf.math.tanh
        elif self._activation == "relu":
            self._activation = tf.nn.relu
        elif self._activation == "sigmoid":
            self._activation = tf.nn.sigmoid

    def build(
        self,
        x: tf.Tensor,
    ) -> None:
        self._is_built = True
        self._dim_in = x.shape[-1]
        self._w = tf.Variable(
            tf.random.normal([self._dim_in, self._dim_out]) / tf.sqrt(float(self._dim_in)),
            name="w",
        )
        self._b = tf.Variable(tf.zeros([self._dim_out]), name="b")

    def __call__(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:
        if not self._is_built:
            self.build(x)
        y = tf.einsum("...ij,...jk->...ik", x, self._w)
        y = y + self._b
        y = self._activation(y)
        return y


class multihead_self_attention(tf.Module):
    """多头自注意力"""

    def __init__(
        self,
        num_head: int,
        dim_key: int,
        dim_value: int,
        dim_out: int,
        use_relative_position_matrix: bool = False,
        name="msa",
    ):
        super().__init__(name)
        self._is_built = False
        self._num_head = num_head
        self._dim_key = dim_key
        self._dim_value = dim_value
        self._dim_out = dim_out
        self._use_relative_position_matrix = use_relative_position_matrix

    def build(
        self,
        x: tf.Tensor,
        m: tf.Tensor = None,
    ) -> None:
        self._is_built = True
        self._dim_in = x.shape[-1]
        self._list_dense_key = []
        self._list_dense_value = []
        self._list_dense_query = []

        for i in range(self._num_head):
            self._list_dense_key.append(
                dense(
                    self._dim_key,
                    activation=tf.nn.tanh,
                    name="dense_key_head%d" % (i),
                )
            )
            self._list_dense_value.append(
                dense(
                    self._dim_value,
                    activation=tf.nn.tanh,
                    name="dense_value_head%d" % (i),
                )
            )
            self._list_dense_query.append(
                dense(
                    self._dim_key,
                    activation=tf.nn.tanh,
                    name="dense_query_head%d" % (i),
                )
            )
        self._dense_out = dense(self._dim_out, activation=None, name="dense_out")

        if self._use_relative_position_matrix:
            self._len_in = x.shape[-2]
            self._list_rp_mat = []
            for i in range(self._num_head):
                self._list_rp_mat.append(tf.Variable(tf.random.normal([self._len_in, self._len_in])))

    def __call__(
        self,
        x: tf.Tensor,
        m: tf.Tensor = None,
    ) -> tf.Tensor:
        if not self._is_built:
            self.build(x, m)
        list_s = []
        for i in range(self._num_head):
            key_i = self._list_dense_key[i](x)
            value_i = self._list_dense_value[i](x)
            query_i = self._list_dense_query[i](x)
            attention_i = tf.einsum("...ij,...kj->...ik", key_i, query_i)
            if self._use_relative_position_matrix:
                attention_i = tf.add(attention_i, self._list_rp_mat[i])
            attention_i = tf.divide(attention_i, tf.sqrt(float(self._dim_value)))
            if m is not None:
                attention_i -= (1.0 - m) * 1e9
            # attention_i = tf.nn.softmax(attention_i, [-2, -1])
            attention_i = tf.exp(attention_i)
            attention_i = tf.divide(
                attention_i,
                tf.math.reduce_sum(attention_i, [-1, -2], keepdims=True) + 1e-6,
            )
            sum_i = tf.einsum("...ij,...jk->...ik", attention_i, value_i)
            list_s.append(sum_i)
        s = tf.concat(list_s, -1)
        y = self._dense_out(s)
        return y


MSA = multihead_self_attention


class multihead_attention(tf.Module):
    """
    y=f(x1,x2) \\
    x1 is used as Value and Key. \\
    x2 is used as Query if it is not None, otherwise x1 is used as Query.
    """

    def __init__(
        self,
        num_head: int,
        dim_key: int,
        dim_value: int,
        dim_out: int,
        use_relative_position_matrix: bool = False,
        activate_kqv=None,
        name="ma",
    ):
        super().__init__(name)
        self._is_built = False
        self._num_head = num_head
        self._dim_key = dim_key  # D
        self._dim_value = dim_value  # V
        self._dim_out = dim_out
        self._use_relative_position_matrix = use_relative_position_matrix
        self._activate_kqv = activate_kqv

    def build(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor = None,
        m: tf.Tensor = None,
    ) -> None:
        self._is_built = True
        self._dim_in_1 = x1.shape[-1]
        self._dim_in_2 = x2.shape[-1]
        self._list_dense_key = []
        self._list_dense_value = []
        self._list_dense_query = []

        for i in range(self._num_head):
            self._list_dense_key.append(
                dense(
                    self._dim_key,
                    activation=self._activate_kqv,
                    name="dense_key_head%d" % (i),
                )
            )
            self._list_dense_value.append(
                dense(
                    self._dim_value,
                    activation=self._activate_kqv,
                    name="dense_value_head%d" % (i),
                )
            )
            self._list_dense_query.append(
                dense(
                    self._dim_key,
                    activation=self._activate_kqv,
                    name="dense_query_head%d" % (i),
                )
            )
        self._dense_out = dense(self._dim_out, activation=None, name="dense_out")

        if self._use_relative_position_matrix:
            self._len_in_1 = x1.shape[-2]
            self._len_in_2 = x2.shape[-2]
            self._list_rp_mat = []
            for i in range(self._num_head):
                self._list_rp_mat.append(tf.Variable(tf.random.normal([self._len_in_1, self._len_in_2])))

    def __call__(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor = None,
        m: tf.Tensor = None,
    ) -> tf.Tensor:
        if x2 is None:
            x2 = x1
        if not self._is_built:
            self.build(x1, x2, m)
        list_s = []
        for i in range(self._num_head):
            key_i = self._list_dense_key[i](x1)  # [...,K,D]
            value_i = self._list_dense_value[i](x1)  # [...,K,V]
            query_i = self._list_dense_query[i](x2)  # [...,Q,D]
            attention_i = tf.einsum("...kd,...qd->...kq", key_i, query_i)  # [...,K,Q]
            if self._use_relative_position_matrix:
                attention_i = tf.add(attention_i, self._list_rp_mat[i])
            attention_i = tf.divide(attention_i, tf.sqrt(float(self._dim_key)))
            if m is not None:
                attention_i = attention_i - (1.0 - m) * 1e9
            # attention_i = tf.nn.softmax(attention_i, [-2, -1])
            # attention_i = tf.exp(attention_i)
            attention_i = tf.exp(attention_i - tf.math.reduce_max(attention_i, -2, keepdims=True))
            attention_i = tf.divide(
                attention_i,
                tf.math.reduce_sum(attention_i, -2, keepdims=True) + 1e-6,
            )
            sum_i = tf.einsum("...kq,...kd->...qd", attention_i, value_i)
            list_s.append(sum_i)
        s = tf.concat(list_s, -1)
        y = self._dense_out(s)
        return y


MA = multihead_attention


class multilayer_perceptron(tf.Module):
    """多层感知机，连续多个全连接层"""

    def __init__(
        self,
        list_dim_out: typing.List,
        list_activation: typing.List,
        name=None,
    ):
        super().__init__(name)
        self._is_built = False
        self._list_dim_out = list_dim_out
        self._list_activation = list_activation
        self._depth = len(self._list_dim_out)

    def build(
        self,
        x: tf.Tensor,
    ) -> None:
        self._is_built = True
        self._list_dense = []
        for i in range(self._depth):
            self._list_dense.append(
                dense(
                    self._list_dim_out[i],
                    self._list_activation[i],
                    name="dense_%d" % (i),
                )
            )

    def __call__(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:
        if not self._is_built:
            self.build(x)
        y = x
        for i in range(self._depth):
            y = self._list_dense[i](y)
        return y


MLP = multilayer_perceptron


class layer_normalization(tf.Module):
    """层归一化"""

    def __init__(self, axis: int | typing.List = -1, use_gamma=True, use_beta=True, name=None):
        self._is_built = False
        if isinstance(axis, int):
            self._list_axis = [axis]
        else:
            self._list_axis = axis
        self._use_gamma = use_gamma
        self._use_beta = use_beta
        super().__init__(name)

    def build(self, x: tf.Tensor) -> None:
        self._is_built = True
        shape_x = x.shape
        num_dim = len(shape_x)
        shape_norm = [1] * num_dim
        for axis in self._list_axis:
            shape_norm[axis] = shape_x[axis]
        if self._use_gamma:
            self._gamma = tf.Variable(tf.ones(shape_norm), name="gamma")
        if self._use_beta:
            self._beta = tf.Variable(tf.zeros(shape_norm), name="beta")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if not self._is_built:
            self.build(x)
        mean = tf.math.reduce_mean(x, self._list_axis, True)
        std = tf.math.reduce_std(x, self._list_axis, True) + 1e-6
        x = tf.divide(tf.subtract(x, mean), std)
        if self._use_gamma:
            x = tf.multiply(x, self._gamma)
        if self._use_beta:
            x = tf.add(x, self._beta)
        return x


LN = layer_normalization
