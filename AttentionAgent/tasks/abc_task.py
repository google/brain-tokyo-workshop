import abc


class BaseTask(abc.ABC):
    """Abstract base task, this serves as a template."""

    @abc.abstractmethod
    def create_task(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def seed(self, seed):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def roll_out(self, solution, evaluate):
        raise NotImplementedError()
