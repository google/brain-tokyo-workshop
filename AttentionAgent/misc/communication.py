import grpc
import numpy as np
import pickle
import protobuf.roll_out_service_pb2
import sys
import traceback


class CommunicationHelper(object):
    """A helper for communications between the master and the workers."""

    def __init__(self, logger, workers=None, timeout=3600):
        self._workers = workers
        self._num_workers = 0 if workers is None else len(workers)
        self._next_worker_index = 0
        self._logger = logger
        self._timeout = timeout

    def collect_fitness_from_workers(self, requests):
        """Make workers do roll-outs and collect fitness scores."""

        num_roll_outs = len(requests)
        unfinished_tasks = list(range(num_roll_outs))
        fitness_scores = np.zeros(num_roll_outs)

        while unfinished_tasks:

            # Send requests to workers.
            futures = []
            for task_id in unfinished_tasks:
                request = requests[task_id]
                worker = self._workers[self._next_worker_index]
                future = worker.performRollOut.future(
                    request, timeout=self._timeout)
                futures.append(future)
                self._next_worker_index = (
                    (self._next_worker_index + 1) % self._num_workers)

            # Collect results.
            for future in futures:
                try:
                    result = future.result()
                    roll_out_index = result.roll_out_index
                    fitness_scores[roll_out_index] = result.fitness
                    unfinished_tasks.remove(roll_out_index)
                except grpc.RpcError as e:
                    self._logger.error(e)
                    exc_info = sys.exc_info()
                    traceback.print_exception(*exc_info)

        return fitness_scores

    def report_fitness(self, roll_out_index, fitness):
        """Report fitness to the master."""

        try:
            return protobuf.roll_out_service_pb2.RollOutResponse(
                roll_out_index=roll_out_index,
                fitness=fitness,
            )
        except Exception as e:
            self._logger.error('Error in report_fitness(): {}'.format(e))
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            return None

    @staticmethod
    def create_cma_request(roll_out_index,
                           env_seed,
                           evaluate,
                           parameters):
        """Create a CMA request message."""

        cma_msg = protobuf.roll_out_service_pb2.CMAParameters(
            parameters=parameters,
        )
        return protobuf.roll_out_service_pb2.RollOutRequest(
            roll_out_index=roll_out_index,
            env_seed=env_seed,
            evaluate=evaluate,
            cma_parameters=cma_msg,
        )

