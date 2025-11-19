#!/usr/bin/bash

source "../config.sh"

CHANNEL_SCRIPT_DIR="${SCRIPT_DIR}/verification"

export PYTHONPATH="${CHANNEL_SCRIPT_DIR}:${PYTHONPATH}"

cd "${CHANNEL_SCRIPT_DIR}" || exit

if [ -z "${SLURM_JOB_ID}" ]; then
  SLURM_JOB_ID="000000"
fi

if [ $# -eq 0 ]; then
  SCRIPT_NAME="verify_cont.py"
  CONFIG_FILE="confs/config.yaml"
  ARGS="${SCRIPT_NAME} ${CONFIG_FILE} ${SLURM_JOB_ID}"
else
  ARGS="${@}"
fi

if [ -z "${SLURM_JOB_ID}" ]; then
  echo "Running interactively without SLURM"
  MASTER_NODE="localhost"
  NNODES=1
  GPUS_PER_NODE=1
  run_torch
  exit 0
else
  echo "Running in SLURM job ${SLURM_JOB_ID}"
  NNODES=${SLURM_NNODES}
  GPUS_PER_NODE=$(scontrol show job ${SLURM_JOB_ID} | grep -oP 'gres/gpu:\K[^ ]+')
fi

# get node list and set MASTER_NODE/master_address
declare -A node_global_rank
node_list=$(scontrol show hostnames "${SLURM_NODELIST}")
index=0
for node in ${node_list[@]}; do
    node_global_rank["${node}"]=${index}
    index=$((index+1))
done

echo "node_list: ${node_list[@]}"

MASTER_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)"

for node in ${node_list[@]}; do
    echo "==> Launching on node: ${node}"
    run_torch
done