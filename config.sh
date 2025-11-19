# repo_root/config.sh
# Central configuration for every Bash script in this repo.
# -----------------------------------------------------------------
# Only *export* a variable if you need child processes to inherit it.
# Otherwise keep it local to the script that sources this file.

# ---- paths -------------------------------------------------------
export SCRIPT_DIR="${HOME}/wacv2026_public"

export SINGULARITY_DIR="${SCRIPT_DIR}/singularity"
export SINGULARITY_IMAGE_NAME="scent"
export SINGULARITY_FILEPATH="${SINGULARITY_DIR}/${SINGULARITY_IMAGE_NAME}.sif"

export DATA_DIR="/this/is/path/to/data_dir/"  # <-- CHANGE THIS PATH TO YOUR DATA DIRECTORY
export DATASET_DIR="${DATA_DIR}/dataset/wacv2026_public/"
export EXPERIMENTS_DIR="${DATA_DIR}/experiments/wacv2026_public/"
export CACHE_DIR="${DATA_DIR}/cache/wacv2026_public/"

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

run_torch() {
  if [ -f "${SINGULARITY_FILEPATH}" ]; then
    singularity exec --nv \
      -B "${DATA_DIR}:${DATA_DIR}" \
      -B "${SINGULARITY_DIR}:${SINGULARITY_DIR}" \
      -B "${DATASET_DIR}:${DATASET_DIR}" \
      -B "${EXPERIMENTS_DIR}:${EXPERIMENTS_DIR}" \
      -B "${CACHE_DIR}:${CACHE_DIR}" \
      "${SINGULARITY_FILEPATH}" \
      torchrun \
        --nnodes="${NNODES}" \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --rdzv-id="${SLURM_JOB_ID}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_NODE}:29529" \
        ${ARGS}
  else
    echo "Error: Singularity image not found at ${SINGULARITY_FILEPATH}"
    exit 1
  fi
}


