import abc
import numpy as np


class BaseSolution(abc.ABC):
    """Base solution."""

    @abc.abstractmethod
    def get_output(self, inputs, update_filter):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params_from_layer(self, layer_index):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_params_to_layer(self, params, layer_index):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_num_params_per_layer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, log_dir, iter_count, best_so_far):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, filename):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    def get_l2_penalty(self):
        if not hasattr(self, '_l2_coefficient'):
            raise ValueError('l2_coefficient not specified.')
        params = self.get_params()
        return self._l2_coefficient * np.sum(params ** 2)

    def add_noise(self, noise):
        self.set_params(self.get_params() + noise)

    def add_noise_to_layer(self, noise, layer_index):
        layer_params = self.get_params_from_layer(layer_index)
        assert layer_params.size == noise.size, '#params={}, #noise={}'.format(
            layer_params.size, noise.size)
        self.set_params_to_layer(
            params=layer_params + noise, layer_index=layer_index)
