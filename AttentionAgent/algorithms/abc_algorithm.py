import abc
import misc.communication
import misc.utility
import numpy as np
import protobuf.roll_out_service_pb2_grpc
import time


class EvolutionAlgorithm(abc.ABC):
    """Evolution algorithm."""

    @abc.abstractmethod
    def get_population(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def evolve(self, fitness):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_parameters(self):
        raise NotImplementedError()


class BaseESMaster(abc.ABC):
    """Base ES master."""

    def __init__(self,
                 logger,
                 log_dir,
                 workers,
                 bucket_name,
                 experiment_name,
                 credential_json,
                 seed,
                 n_repeat,
                 max_iter,
                 eval_every_n_iter,
                 n_eval_roll_outs):
        """Initialization."""

        self._logger = logger
        self._log_dir = log_dir
        self._n_repeat = n_repeat
        self._max_iter = max_iter
        self._eval_every_n_iter = eval_every_n_iter
        self._n_eval_roll_outs = n_eval_roll_outs
        self._communication_helper = misc.communication.CommunicationHelper(
            workers=workers, logger=logger)
        self._solution = misc.utility.create_solution()
        self._rnd = np.random.RandomState(seed=seed)
        self._algorithm = None

        self._gcs_folder = experiment_name
        try:
            self._bucket = misc.utility.get_gcs_bucket(
                logger=logger,
                bucket_name=bucket_name,
                credential_json_path=credential_json,
            )
            self._logger.info(
                'Master will upload logs to GCS {}/{} too'.format(
                    bucket_name, self._gcs_folder))
        except Exception:
            self._bucket = None

    def train(self):
        """Train for max_iter iterations."""

        # Evaluate before train.
        eval_scores = self._evaluate()
        misc.utility.log_scores(
            logger=self._logger, iter_cnt=0, scores=eval_scores, evaluate=True)
        misc.utility.save_scores(
            log_dir=self._log_dir, n_iter=0, scores=eval_scores)
        best_eval_score = -float('Inf')

        self._logger.info(
            'Start training for {} iterations.'.format(self._max_iter))
        for iter_cnt in range(self._max_iter):

            # Training.
            start_time = time.time()
            scores = self._train_once()
            time_cost = time.time() - start_time
            self._logger.info('1-step training time: {}s'.format(time_cost))
            misc.utility.log_scores(
                logger=self._logger, iter_cnt=iter_cnt + 1, scores=scores)

            # Evaluate periodically.
            if (iter_cnt + 1) % self._eval_every_n_iter == 0:

                # Evaluate.
                start_time = time.time()
                eval_scores = self._evaluate()
                time_cost = time.time() - start_time
                self._logger.info('Evaluation time: {}s'.format(time_cost))

                # Record results and save the model.
                mean_score = eval_scores.mean()
                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_so_far = True
                else:
                    best_so_far = False
                misc.utility.log_scores(logger=self._logger,
                                        iter_cnt=iter_cnt + 1,
                                        scores=eval_scores,
                                        evaluate=True)
                misc.utility.save_scores(log_dir=self._log_dir,
                                         n_iter=iter_cnt + 1,
                                         scores=eval_scores)
                self._save_solution(iter_count=iter_cnt + 1,
                                    best_so_far=best_so_far)
                misc.utility.save_logs_to_gcs(logger=self._logger,
                                              log_dir=self._log_dir,
                                              iter_cnt=iter_cnt + 1,
                                              bucket=self._bucket,
                                              folder=self._gcs_folder)

    def _evaluate(self):
        if self._algorithm is None:
            raise NotImplementedError()

        requests = self._create_rpc_requests(evaluate=True)
        fitness = self._communication_helper.collect_fitness_from_workers(
            requests=requests,
        )
        return fitness

    def _train_once(self):
        if self._algorithm is None:
            raise NotImplementedError()

        requests = self._create_rpc_requests(evaluate=False)

        fitness = self._communication_helper.collect_fitness_from_workers(
            requests=requests,
        )
        fitness = fitness.reshape([-1, self._n_repeat]).mean(axis=1)
        self._algorithm.evolve(fitness)

        return fitness

    def _save_solution(self, iter_count, best_so_far):
        if self._algorithm is None:
            raise NotImplementedError()
        self._update_solution()
        self._solution.save(self._log_dir, iter_count, best_so_far)

    @abc.abstractmethod
    def _create_rpc_requests(self, evaluate):
        raise NotImplementedError()

    @abc.abstractmethod
    def _update_solution(self):
        raise NotImplementedError()


class BaseESWorker(abc.ABC,
                   protobuf.roll_out_service_pb2_grpc.RollOutServiceServicer):
    """Base ES worker."""

    def __init__(self, logger):
        """Initialization."""

        self._logger = logger
        self._task = misc.utility.create_task(logger=logger)
        self._solution = misc.utility.create_solution()
        self._communication_helper = misc.communication.CommunicationHelper(
            logger=logger)

    @abc.abstractmethod
    def _handle_master_request(self, request):
        raise NotImplementedError()

    def performRollOut(self, request, context):

        fitness = self._handle_master_request(request)

        return self._communication_helper.report_fitness(
            roll_out_index=request.roll_out_index,
            fitness=fitness,
        )

