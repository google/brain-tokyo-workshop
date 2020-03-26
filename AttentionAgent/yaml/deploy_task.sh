#!/usr/bin/env bash


# parse command line args
EXPERIMENT_NAME=""
IMAGE="docker.io\/braintok\/self-attention-agent:CarRacing"
GIN_FILE="configs/CarRacing.gin"

WAIT_TIME=60
NUM_REPLICA=256

while (( "$#" )); do
  case "$1" in
    -c|--config)
      GIN_FILE=$2
      shift 2
      ;;
    -n|--num-workers)
      NUM_REPLICA=$2
      shift 2
      ;;
    -w|--wait-time)
      WAIT_TIME=$2
      shift 2
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
  esac
done

if [ -z "${EXPERIMENT_NAME}" ]
then
    echo "You must set --experiment-name"
    exit
fi

# Common settings.
echo ${EXPERIMENT_NAME}
CURRENT_TIME=`date +%Y%m%d_%H%M%S`
GCS_DIR_NAME=${EXPERIMENT_NAME}"_"${CURRENT_TIME}

COMMON_CMD="\
sed -i 's/n_repeat = 16/n_repeat = 16/g' ${GIN_FILE} \&\& \
export DISPLAY=:0 \&\&"

WORKER_OPTION=""
MASTER_OPTION=""
MEMORY="2Gi"
if [ `uname` == "Darwin" ]
then
  EXTRA_SED="''"
else
  EXTRA_SED=""
fi

# Worker specific settings.
MEMORY_LIMIT="7Gi"
CPU="1200m"
CPU_LIMIT="1500m"
WORKER_CMD="${COMMON_CMD} xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' \
python3 run_worker.py \
--run-on-gke \
--log-dir=/var/log/es/log \
--config=${GIN_FILE} ${WORKER_OPTION}"
echo ${WORKER_CMD}

sed "s/%EXPERIMENT%/${EXPERIMENT_NAME}/g; \
s/%IMAGE%/${IMAGE}/g; \
s/%CPU%/${CPU}/g; \
s/%CPU_LIMIT%/${CPU_LIMIT}/g; \
s/%MEM%/${MEMORY}/g; \
s/%MEM_LIMIT%/${MEMORY_LIMIT}/g; \
s/%REPLICA%/${NUM_REPLICA}/g;" deploy_workers.yaml > deploy.yaml
sed -i ${EXTRA_SED} 's,%CMD%,'\""${WORKER_CMD}"\"',g' deploy.yaml
kubectl apply -f deploy.yaml

# Wait for the workers to deploy.
echo "Wait for ${WAIT_TIME} seconds ..."
sleep ${WAIT_TIME}

# Master specific settings.
MASTER_CMD="${COMMON_CMD} chmod 644 ${GIN_FILE} \&\& \
cp index.html /var/log/es/ \&\& \
xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' \
python3 run_master.py \
--run-on-gke \
--worker-addresses=worker-${EXPERIMENT_NAME}-service:20000 \
--log-dir=/var/log/es/log \
--config=${GIN_FILE} \
--gcs-bucket=es_experiments \
--gcs-credential=gcs.json \
--gcs-experiment-name=${GCS_DIR_NAME} ${MASTER_OPTION}"
echo ${MASTER_CMD}

sed "s/%EXPERIMENT%/${EXPERIMENT_NAME}/g; \
s/%IMAGE%/${IMAGE}/g; " deploy_master.yaml > deploy.yaml
sed -i ${EXTRA_SED} 's,%CMD%,'\""${MASTER_CMD}"\"',g' deploy.yaml
kubectl apply -f deploy.yaml

rm deploy.yaml
