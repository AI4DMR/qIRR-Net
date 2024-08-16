# 2024/08/16
# public

import numpy as np


# import time # DEBUG

path_dwi = "path/to/dwi"
path_metric = "path/to/metric"
path_b_value = "path/to/b_value"
path_b_vector = "path/to/b_vector"
path_mask = "path/to/mask"


class DataLoader_SingleSubject:
    def __init__(
        self,
        subject: str,
    ) -> None:

        self.dwi = np.load("%s/%s.npy" % (path_dwi, subject))
        self.metric = np.load("%s/%s.npy" % (path_metric, subject))
        self.b_value = np.load("%s/%s.npy" % (path_b_value, subject))
        self.b_vector = np.load("%s/%s.npy" % (path_b_vector, subject))
        self.mask = np.load("%s/%s.npy" % (path_mask, subject)).astype(bool)

        self.dwi = self.dwi[self.mask, :]
        self.metric = self.dwi[self.mask, :]

    def voxel_generator(
        self,
        num_voxel_selected: int = None,
        range_b_value: tuple[float, float] = None,
    ):
        dwi = self.dwi
        metric = self.metric
        b_value = self.b_value
        b_vector = self.b_vector
        if range_b_value is not None:
            list_index_selected = np.logical_and(b_value >= range_b_value[0], b_value < range_b_value[1])
            dwi = dwi[:, list_index_selected]
            b_value = b_value[list_index_selected]
            b_vector = b_vector[list_index_selected, :]

        if num_voxel_selected is None:
            for i in range(dwi.shape[0]):
                yield dwi[i], metric[i], b_value, b_vector
        else:
            for i in np.random.choice(dwi.shape[0], num_voxel_selected, False):
                yield dwi[i], metric[i], b_value, b_vector
        return

    def num_voxel(self):
        return self.dwi.shape[0]


class DataLoader_MultipleSubject:
    def __init__(
        self,
        list_subject: str,
    ) -> None:
        self._list_subject = list_subject

    def voxel_generator(self, num_voxel_selected: int = None, range_b_value=None):
        for subject in self._list_subject:
            for dwi, metric, b_value, b_vector in DataLoader_SingleSubject(subject).voxel_generator(num_voxel_selected, range_b_value):
                yield dwi, metric, b_value, b_vector


def non_uniform_b1(list_factor: np.ndarray, list_prob: np.ndarray = None):
    def new_generator(old_generator):
        for dwi, metric, b_value, b_vector in old_generator:
            factor = np.random.choice(list_factor, size=dwi.shape[0], p=list_prob)
            yield dwi * factor, metric, b_value, b_vector

    return new_generator


def rotation():
    _2pi = 2 * np.pi

    def new_generator(old_generator):
        for dwi, metric, b_value, b_vector in old_generator:
            h, alpha, beta = np.random.rand(3)
            sin_2pi_alpha = np.sin(_2pi * alpha)
            sin_2pi_beta = np.sin(_2pi * beta)
            cos_2pi_alpha = np.cos(_2pi * alpha)
            cos_2pi_beta = np.cos(_2pi * beta)
            g = np.sqrt(1 - h**2)
            matrix_rotation = np.array(
                [
                    [h, -g * sin_2pi_alpha, -g * cos_2pi_alpha],
                    [
                        g * sin_2pi_beta,
                        -cos_2pi_alpha * cos_2pi_beta + h * sin_2pi_alpha * sin_2pi_beta,
                        sin_2pi_alpha * cos_2pi_beta + h * cos_2pi_alpha * sin_2pi_beta,
                    ],
                    [
                        g * cos_2pi_beta,
                        cos_2pi_alpha * sin_2pi_beta + h * sin_2pi_alpha * cos_2pi_beta,
                        -sin_2pi_alpha * sin_2pi_beta + h * cos_2pi_alpha * cos_2pi_beta,
                    ],
                ]
            )
            yield dwi, metric, b_value, np.einsum("ij,kj->ki", matrix_rotation, b_vector)

    return new_generator


def undersample(
    list_num_sample: np.ndarray | int,
    list_prob: np.ndarray = None,
):
    def new_generator(old_generator):
        if isinstance(list_num_sample, int):
            num_sample = list_num_sample
            for dwi, metric, b_value, b_vector in old_generator:
                list_index_selected = np.random.choice(dwi.shape[0], num_sample, False)
                yield dwi[list_index_selected], metric, b_value[list_index_selected], b_vector[list_index_selected, :]
        else:
            for dwi, metric, b_value, b_vector in old_generator:
                num_sample = np.random.choice(list_num_sample, p=list_prob)
                list_index_selected = np.random.choice(dwi.shape[0], num_sample, False)
                yield dwi[list_index_selected], metric, b_value[list_index_selected], b_vector[list_index_selected, :]

    return new_generator


def repeat(
    time: int,
):
    def new_generator(old_generator):
        for dwi, metric, b_value, b_vector in old_generator:
            for _ in range(time):
                yield dwi, metric, b_value, b_vector

    return new_generator


def b0_norm(
    threshold_b0,
):
    def new_generator(old_generator):
        for dwi, metric, b_value, b_vector in old_generator:
            s0_mean = np.mean(dwi[b_value < threshold_b0])
            yield dwi / s0_mean, metric, b_value, b_vector

    return new_generator


def batch(
    batchsize: int,
):
    assert batchsize > 0

    def new_generator(old_generator):
        try:
            dwi, metric, b_value, b_vector = next(old_generator)
        except StopIteration:
            return
        dwi_batch = np.empty((batchsize,) + dwi.shape)
        dwi_batch[0, ...] = dwi
        metric_batch = np.empty((batchsize,) + metric.shape)
        metric_batch[0, ...] = metric
        b_value_batch = np.empty((batchsize,) + b_value.shape)
        b_value_batch[0, ...] = b_value
        b_vector_batch = np.empty((batchsize,) + b_vector.shape)
        b_vector_batch[0, ...] = b_vector

        if batchsize == 1:
            yield dwi_batch, metric_batch, b_value_batch, b_value_batch
            i = 0
        else:
            i = 1

        for dwi, metric, b_value, b_vector in old_generator:
            dwi_batch[i, ...] = dwi
            metric_batch[i, ...] = metric
            b_value_batch[i, ...] = b_value
            b_vector_batch[i, ...] = b_vector
            i += 1
            if i == batchsize:
                yield dwi_batch, metric_batch, b_value_batch, b_vector_batch
                i = 0
        if i != 0:
            yield dwi_batch[:i, ...], metric_batch[:i, ...], b_value_batch[:i, ...], b_vector_batch[:i, ...]

    return new_generator


def aqDL_batch(
    batchsize: int,
):
    assert batchsize > 0

    def new_generator(old_generator):

        list_dwi = []
        list_metric = []
        list_b_value = []
        list_b_vector = []
        list_len = []
        i = 0
        for dwi, metric, b_value, b_vector in old_generator:
            list_dwi.append(dwi)
            list_metric.append(metric)
            list_b_value.append(b_value)
            list_b_vector.append(b_vector)
            list_len.append(dwi.shape[0])
            i += 1
            if i == batchsize:
                max_len = max(list_len)
                x = np.zeros((batchsize, max_len, 7))
                y = np.zeros((batchsize, 8))
                m = np.zeros((batchsize, max_len, 1))
                for j in range(batchsize):

                    x[j, : list_len[j], 0] = np.log(list_dwi[j] + 1e-6)
                    x[j, : list_len[j], 1][list_b_value[j] < 100] = 1
                    x[j, : list_len[j], 2][(list_b_value[j] > 900) & (list_b_value[j] < 1100)] = 1
                    x[j, : list_len[j], 3][(list_b_value[j] > 1900) & (list_b_value[j] < 2100)] = 1
                    x[j, : list_len[j], 4:] = list_b_vector[j]
                    y[j, :] = np.clip(
                        list_metric[j] / np.array([0.004, 0.004, 0.004, 1, 4, 4, 4, 1])[None, ...],
                        0,
                        1,
                    )
                    m[j, : list_len[j], 0] = 1.0
                yield x, m, y
                list_dwi = []
                list_metric = []
                list_b_value = []
                list_b_vector = []
                list_len = []
                i = 0
        if i != 0:
            max_len = max(list_len)
            x = np.zeros((i, max_len, 7))
            y = np.zeros((i, 8))
            m = np.zeros((i, max_len, 1))
            for j in range(i):

                x[j, : list_len[j], 0] = np.log(list_dwi[j] + 1e-6)
                x[j, : list_len[j], 1][list_b_value[j] < 100] = 1
                x[j, : list_len[j], 2][(list_b_value[j] > 900) & (list_b_value[j] < 1100)] = 1
                x[j, : list_len[j], 3][(list_b_value[j] > 1900) & (list_b_value[j] < 2100)] = 1
                x[j, : list_len[j], 4:] = list_b_vector[j]
                y[j, :] = np.clip(
                    list_metric[j] / np.array([0.004, 0.004, 0.004, 1, 4, 4, 4, 1])[None, ...],
                    0,
                    1,
                )
                m[j, : list_len[j], 0] = 1.0
            yield x, m, y

    return new_generator
