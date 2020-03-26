import gin
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import solutions.abc_solution
import seaborn as sns
import shutil
import time


MAX_INT = (1 << 31) - 1


def create_shared_noise(seed=42, count=250000000):
    noise = np.random.RandomState(seed).randn(count).astype(np.float64)
    return noise


class SharedNoiseTable(object):
    """A precomputed noise generator that is shared by multiple workers."""

    def __init__(self, noise, seed=42):
        self._rnd = np.random.RandomState(seed)
        self.noise = noise

    def sample_idx(self, dim):
        return self._rnd.randint(0, len(self.noise) - dim + 1)

    def get_noise_by_idx(self, idx, dim):
        start_index = idx
        end_index = start_index + dim
        noise = self.noise[start_index:end_index]
        return noise.copy()


@gin.configurable
def create_task(task_loader, **kwargs):
    """Load and return a task.

    This function loads a task using the given task_loader.

    Args:
        task_loader: A callable object, or an object with 'create_task' method.
        kwargs: dict. Task configurations.
    Returns:
        A tasks.* object.
    """

    if hasattr(task_loader, 'create_task'):
        return task_loader.create_task(**kwargs)
    elif hasattr(task_loader, '__call__'):
        return task_loader(**kwargs)
    else:
        raise ValueError(
            'task_loader should have create_task method or __call__ method.')


@gin.configurable
def create_solution(solution_loader, **kwargs):
    """Create a solution.

    Create and return a solution.

    Args:
        solution_loader: A callable object, or an object with 'create_solution'
            method.
        kwargs: dict. Solution configurations.
    Return:
        A solution.* object.
    """

    if isinstance(solution_loader, solutions.abc_solution.BaseSolution):
        return solution_loader
    elif hasattr(solution_loader, 'create_solution'):
        return solution_loader.create_solution(**kwargs)
    elif hasattr(solution_loader, '__call__'):
        return solution_loader(**kwargs)
    else:
        raise ValueError(
            'solution_loader should have create_solution or __call__ method.')


@gin.configurable
def get_es_master(es_algorithm, logger, log_dir,
                  stubs, bucket, experiment_name, credential):
    """Instantiate an evolution strategic algorithm master.

    Return an ES algorithm master from the algorithms package.

    Args:
        es_algorithm: algorithms.*. ES algorithm.
        logger: logging.Logger. Logger.
        log_dir: str. Log directory.
        stubs: list. grpc stubs.
        bucket: str. GCS bucket name.
        experiment_name: str. Experiment name.
        credential: str. Credential JSON file path.
    Returns:
            algorithms.*Master.
        """

    return es_algorithm(
        logger=logger,
        log_dir=log_dir,
        workers=stubs,
        bucket_name=bucket,
        experiment_name=experiment_name,
        credential_json=credential,
    )


@gin.configurable
def get_es_worker(es_algorithm, logger, **kwargs):
    """Instantiate an evolution strategic algorithm worker.

    Return an ES algorithm worker from the algorithms package.

    Args:
        es_algorithm: algorithms.*. ES algorithm.
        logger: logging.Logger. Logger.
    Returns:
        algorithms.*Worker.
    """

    return es_algorithm(logger=logger, **kwargs)


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


def save_scores(log_dir, n_iter, scores):
    """Save scores to file system.

    Save scores to file system as a csv file.

    Args:
        log_dir: str. Path to log directory.
        n_iter: int. Current iteration number.
        scores: np.array. Scores.
    """
    # save scores for analysis in the future
    filename = os.path.join(log_dir, 'scores.csv')
    df = pd.DataFrame({'Time': [int(time.time())] * scores.size,
                       'Iteration': [n_iter] * scores.size,
                       'Reward': scores})
    need_header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=need_header, index=False)
    # draw graphs
    df = pd.read_csv(filename)
    sns.set()
    # plot 1: reward vs iteration
    sns_plot = sns.lineplot(x='Iteration', y='Reward', data=df, ci='sd')
    im_filename = os.path.join(log_dir, 'reward_vs_iteration.png')
    sns_plot.get_figure().savefig(im_filename)
    plt.clf()
    # plot 2: reward vs time
    start_time = df.Time.values[0]
    df.Time = (df.Time - start_time) / 3600  # convert to hours
    sns_plot = sns.lineplot(x='Time', y='Reward', data=df, ci='sd')
    sns_plot.set_xlabel('Time (hour)')
    im_filename = os.path.join(log_dir, 'reward_vs_time.png')
    sns_plot.get_figure().savefig(im_filename)
    plt.clf()


def log_scores(logger, iter_cnt, scores, evaluate=False):
    """Log scores.

    Log scores.

    Args:
        logger: A logger.
        iter_cnt: int. Iteration number.
        scores: list. List of scores.
        evaluate: bool. Whether these scores are from evaluation roll-outs.
    """
    msg = ('Iter {0}: size(scores)={1}, '
           'max(scores)={2:.2f}, '
           'mean(scores)={3:.2f}, '
           'min(scores)={4:.2f}, '
           'sd(scores)={5:.2f}'.format(iter_cnt,
                                       scores.size,
                                       np.max(scores),
                                       np.mean(scores),
                                       np.min(scores),
                                       np.std(scores)))
    if evaluate:
        msg = '[TEST] ' + msg
    logger.info(msg)


def get_gcs_bucket(logger, bucket_name, credential_json_path):
    """Create a Google Cloud Storage bucket object.

    Create a GCS bucket object for uploading files.

    Args:
        logger: A logger.
        bucket_name: str. Bucket name.
        credential_json_path: str. Path to the JSON file.
    Returns:
        GCS bucket object.
    """
    logger.info(
        'Bucket: {}, JSON: {}'.format(bucket_name, credential_json_path))
    from google.cloud import storage
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_json_path
    storage_client = storage.Client()
    return storage_client.get_bucket(bucket_name)


def upload_file_to_gcs(logger, bucket, src_filename, tar_filename):
    """Upload a file to GCS.

    Upload a file with specified path to GCS.

    Args:
        logger: A logger.
        bucket: GCS bucket object.
        src_filename: str. File name of the local file.
        tar_filename: str. File name of the GCS file.
    """
    if bucket is not None:
        blob = bucket.blob(tar_filename)
        blob.upload_from_filename(src_filename)
        logger.info('Uploaded {} to GCS {}.'.format(src_filename, tar_filename))


def save_logs_to_gcs(logger, log_dir, iter_cnt, bucket, folder):
    """Upload log files to GCS.

    Upload the following log files to GCS in the specified folder:
    1. config.gin
    2. scores.csv
    3. model_{iter_cnt}.[npz|p]
    4. best_model.[npz|p]
    5. es_master.txt
    6. reward_vs_iteration.png
    7. reward_vs_time.png

    Args:
        logger: A logger.
        log_dir: str. Log directory.
        iter_cnt: int. Iteration count.
        bucket: GCS bucket object.
        folder: str. Folder name.
    """

    # Upload log file.
    log_files = ['config.gin',
                 'scores.csv',
                 'es_master.txt',
                 'reward_vs_iteration.png',
                 'reward_vs_time.png']
    for f in log_files:
        src_file = os.path.join(log_dir, f)
        tar_file = os.path.join(folder, f)
        upload_file_to_gcs(logger=logger,
                           bucket=bucket,
                           src_filename=src_file,
                           tar_filename=tar_file)

    # Upload model files.
    model_file_suffix = '.npz'
    model_file = 'best_model' + model_file_suffix
    if not os.path.exists(os.path.join(log_dir, model_file)):
        model_file_suffix = '.p'
    model_files = ['best_model' + model_file_suffix,
                   'model_{}'.format(iter_cnt) + model_file_suffix]
    for f in model_files:
        src_file = os.path.join(log_dir, f)
        tar_file = os.path.join(folder, f)
        upload_file_to_gcs(logger=logger,
                           bucket=bucket,
                           src_filename=src_file,
                           tar_filename=tar_file)
