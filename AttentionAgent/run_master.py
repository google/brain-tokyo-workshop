import argparse
import protobuf.roll_out_service_pb2_grpc
from protobuf.roll_out_service_pb2_grpc import add_ParameterSyncServiceServicer_to_server
from concurrent import futures
import gin
import grpc
import time
import misc.utility


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MAX_MSG_LEN = 40 * 1024 * 1024


def main(config):
    """Start the master."""

    gin.parse_config_file(config.config)
    misc.utility.save_config(log_dir=config.log_dir, config=config.config)
    logger = misc.utility.create_logger(
        name='es_master', log_dir=config.log_dir)

    # Create worker stubs.
    if config.run_on_gke:
        workers = ['{}'.format(address)
                   for address in config.worker_addresses.split(',')]
    else:
        workers = ['127.0.0.1:{}'.format(i + config.worker_port)
                   for i in range(config.num_workers)]
    logger.info('Workers: {}'.format(workers))

    stubs = []
    for worker in workers:
        if config.run_on_gke:
            channel = grpc.insecure_channel(
                worker, [('grpc.lb_policy_name', 'round_robin'),
                         ("grpc.max_send_message_length", _MAX_MSG_LEN),
                         ("grpc.max_receive_message_length", _MAX_MSG_LEN)])
        else:
            channel = grpc.insecure_channel(worker)
        grpc.channel_ready_future(channel).result()
        stubs.append(
            protobuf.roll_out_service_pb2_grpc.RollOutServiceStub(channel)
        )

    master = misc.utility.get_es_master(
        logger=logger,
        log_dir=config.log_dir,
        stubs=stubs,
        bucket=config.gcs_bucket,
        experiment_name=config.gcs_experiment_name,
        credential=config.gcs_credential,
    )

    # Start the parameters synchronization service.
    if config.start_param_sync_service:
        port = config.port
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[("grpc.max_send_message_length", _MAX_MSG_LEN),
                     ("grpc.max_receive_message_length", _MAX_MSG_LEN)])
        add_ParameterSyncServiceServicer_to_server(master, server)
        server.add_insecure_port('[::]:{}'.format(port))
        server.start()

    # Start the master.
    logger.info('Start to train.')
    master.train()

    if config.run_on_gke:
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            logger.info('Job done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to the config file.')
    parser.add_argument(
        '--log-dir', help='Path to the log directory.', default='./log')
    parser.add_argument(
        '--num-workers', help='Number of workers.', type=int, default=1)
    parser.add_argument(
        '--worker-addresses', help='Worker addresses, separated by comma.')
    parser.add_argument(
        '--worker-port', help='Worker start port.', type=int, default=20000)
    parser.add_argument(
        '--port', help='Master port.', type=int, default=10000)
    parser.add_argument(
        '--start-param-sync-service', default=False, action='store_true')
    parser.add_argument(
        '--run-on-gke', help='Whether run this on GKE.', default=False,
        action='store_true')
    parser.add_argument(
        '--gcs-bucket', help='GCS bucket name.')
    parser.add_argument(
        '--gcs-experiment-name', help='GCS folder name.')
    parser.add_argument(
        '--gcs-credential', help='GCS credential JSON file path.')
    args, _ = parser.parse_known_args()

    main(args)
