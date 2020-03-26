import argparse
import protobuf.roll_out_service_pb2_grpc
import gin
import grpc
import time
import misc.utility
from concurrent import futures


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MAX_MSG_LEN = 40 * 1024 * 1024


def main(config):
    """Start the worker."""

    gin.parse_config_file(config.config)
    logger = misc.utility.create_logger(
        name='es_worker{}'.format(config.worker_id), log_dir=config.log_dir)

    if config.master_address is not None:
        logger.info('master_address: {}'.format(config.master_address))
        channel = grpc.insecure_channel(
            config.master_address,
            [("grpc.max_receive_message_length", _MAX_MSG_LEN)])
        stub = protobuf.roll_out_service_pb2_grpc.ParameterSyncServiceStub(
            channel)
        worker = misc.utility.get_es_worker(logger=logger, master=stub)
    else:
        worker = misc.utility.get_es_worker(logger=logger)

    if config.run_on_gke:
        port = config.port
    else:
        port = config.port + config.worker_id
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[("grpc.max_send_message_length", _MAX_MSG_LEN),
                 ("grpc.max_receive_message_length", _MAX_MSG_LEN)])

    # Start the RPC server.
    protobuf.roll_out_service_pb2_grpc.add_RollOutServiceServicer_to_server(
        worker, server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    logger.info('Listening to port {} ...'.format(port))

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('Worker quit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port', help='Port to start the service.', type=int, default=20000)
    parser.add_argument(
        '--config', help='Path to the config file.')
    parser.add_argument(
        '--log-dir', help='Path to the log directory.', default='./log')
    parser.add_argument(
        '--worker-id', help='Worker ID.', type=int, default=0)
    parser.add_argument(
        '--master-address', help='Master address.')
    parser.add_argument(
        '--run-on-gke', help='Whether run this on GKE.', default=False,
        action='store_true')
    args, _ = parser.parse_known_args()

    main(args)
