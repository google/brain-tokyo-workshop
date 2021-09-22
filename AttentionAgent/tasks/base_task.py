import abc


class BaseTask(abc.ABC):
    """Base task."""

    @abc.abstractmethod
    def reset_for_rollout(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def seed(self, seed=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def rollout(self, solution, evaluation=False):
        raise NotImplementedError()
