import gin
import logging
import os
import shutil
from solutions.base_solution import BaseSolution
from tasks.base_task import BaseTask


@gin.configurable
def create_task(task_loader, **kwargs):
    """Load and return a task."""

    if isinstance(task_loader, BaseTask):
        return task_loader
    else:
        return task_loader(**kwargs)


@gin.configurable
def create_solution(solution_loader, **kwargs):
    """Create a solution."""

    if isinstance(solution_loader, BaseSolution):
        return solution_loader
    else:
        return solution_loader(**kwargs)


def save_config(log_dir, config):
    """Create a log directory and save config in it.

    Create a log directory and save configurations.

    Args:
        log_dir: str. Path of the log directory.
        config: str. Path to configuration file.
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.copy(config, os.path.join(log_dir, 'config.gin'))


def create_logger(name, log_dir=None, debug=False):
    """Create a logger.

    Create a logger that logs to log_dir.

    Args:
        name: str. Name of the logger.
        log_dir: str. Path to log directory.
        debug: bool. Whether to set debug level logging.
    Returns:
        logging.logger.
    """

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format=log_format)
    logger = logging.getLogger(name)
    if log_dir:
        log_file = os.path.join(log_dir, '{}.txt'.format(name))
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    return logger
