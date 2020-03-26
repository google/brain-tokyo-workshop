import cma
import gin
import numpy as np
import algorithms.abc_algorithm
from misc.utility import MAX_INT


@gin.configurable
class CMA(algorithms.abc_algorithm.EvolutionAlgorithm):
    """CMA algorithm."""

    def __init__(self,
                 logger,
                 seed,
                 population_size,
                 init_sigma,
                 init_params):
        """Create a wrapper of cmapy."""

        self._logger = logger
        self._algorithm = cma.CMAEvolutionStrategy(
            x0=init_params,
            sigma0=init_sigma,
            inopts={
                'popsize': population_size,
                'seed': seed if seed > 0 else 42,  # ignored if seed is 0
                'randn': np.random.randn,
            },
        )
        self._population = None

    def get_population(self):
        self._population = self._algorithm.ask()
        return self._population

    def evolve(self, fitness):
        self._algorithm.tell(self._population, -fitness)

    def get_current_parameters(self):
        return self._algorithm.result.xfavorite


@gin.configurable
class CMAMaster(algorithms.abc_algorithm.BaseESMaster):
    """CMA master."""

    def __init__(self,
                 logger,
                 log_dir,
                 bucket_name,
                 experiment_name,
                 credential_json,
                 workers,
                 seed,
                 n_repeat,
                 max_iter,
                 eval_every_n_iter,
                 n_eval_roll_outs):
        """Initialize the master."""

        super(CMAMaster, self).__init__(
            logger=logger,
            log_dir=log_dir,
            workers=workers,
            bucket_name=bucket_name,
            experiment_name=experiment_name,
            credential_json=credential_json,
            seed=seed,
            n_repeat=n_repeat,
            max_iter=max_iter,
            eval_every_n_iter=eval_every_n_iter,
            n_eval_roll_outs=n_eval_roll_outs,
        )
        self._algorithm = CMA(
            logger=logger, seed=seed, init_params=self._solution.get_params())
        self._logger.info(
            'Master initialized, algorithm={}'.format(self._algorithm))

    def _create_rpc_requests(self, evaluate):
        """Create gRPC requests."""

        if evaluate:
            n_repeat = 1
            num_roll_outs = self._n_eval_roll_outs
            params_list = [self._algorithm.get_current_parameters()]
        else:
            n_repeat = self._n_repeat
            params_list = self._algorithm.get_population()
            num_roll_outs = len(params_list) * n_repeat

        env_seed_list = self._rnd.randint(
            low=0, high=MAX_INT, size=num_roll_outs)

        requests = []
        for i, env_seed in enumerate(env_seed_list):
            ix = 0 if evaluate else i // n_repeat
            requests.append(self._communication_helper.create_cma_request(
                roll_out_index=i,
                env_seed=env_seed,
                parameters=params_list[ix],
                evaluate=evaluate,
            ))
        return requests

    def _update_solution(self):
        if self._algorithm is None:
            raise NotImplementedError()
        self._solution.set_params(self._algorithm.get_current_parameters())


@gin.configurable
class CMAWorker(algorithms.abc_algorithm.BaseESWorker):
    """CMA worker."""

    def __init__(self, logger):
        """Initialize the worker."""

        super(CMAWorker, self).__init__(logger=logger)
        self._logger.info('CMAWorker initialized.')

    def _handle_master_request(self, request):
        params = np.asarray(request.cma_parameters.parameters)
        self._solution.set_params(params)
        self._task.seed(request.env_seed)
        score = self._task.roll_out(self._solution, request.evaluate)
        penalty = 0 if request.evaluate else self._solution.get_l2_penalty()
        return score - penalty

