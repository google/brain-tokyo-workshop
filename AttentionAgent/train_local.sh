#!/usr/bin/env bash


START_PARAM_SYNC_SERVICE=0
NUM_WORKERS=8
CONFIG_FILE="configs/CarRacing.gin"
JSON_FILE="gcs.json"
BUCKET_NAME="es_experiments"
EXPERIMENT_NAME="test"

# parse command line args
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -c|--config)
      CONFIG_FILE=$2
      shift 2
      ;;
    -n|--num-workers)
      NUM_WORKERS=$2
      shift 2
      ;;
    -s|--start-params-sync-service)
      START_PARAM_SYNC_SERVICE=1
      shift
      ;;
    -e|--experiment-name)
      EXPERIMENT_NAME=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS --$1"
      shift
      ;;
  esac
done

declare -A worker_ids

if (($START_PARAM_SYNC_SERVICE==1))
then
    WORKER_OPTION="--master-address=127.0.0.1:10000"
    MASTER_OPTION="--start-param-sync-service"
else
    WORKER_OPTION=""
    MASTER_OPTION=""
fi

for ((i = 0; i < $NUM_WORKERS; i++))
do
    python run_worker.py --config=$CONFIG_FILE --worker-id=$i $WORKER_OPTION &
    worker_ids[$i]=$!
    echo "Worker $i started."
done

python run_master.py \
--config=$CONFIG_FILE \
--num-workers=$NUM_WORKERS \
--gcs-bucket=$BUCKET_NAME \
--gcs-experiment-name=$EXPERIMENT_NAME \
--gcs-credential=$JSON_FILE \
$MASTER_OPTION \

echo "Terminate all workers ..."
for wid in "${worker_ids[@]}"
do
    kill $wid
done
echo "Done"
