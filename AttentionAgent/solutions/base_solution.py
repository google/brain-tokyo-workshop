import abc


class BaseSolution(abc.ABC):
    """Base solution."""

    @abc.abstractmethod
    def get_action(self, obs):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_num_params(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, filename):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, filename):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()
