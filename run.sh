#!/bin/bash
# Train, decode and build the submission file as one GPU batch job.
#
#   sbatch run.sh                                   # monolingual (Checkpoint 1): English-only data, no LID tag
#   sbatch --export=ALL,MODE=multilingual run.sh    # bilingual (Checkpoints 3/4): [ENG]/[ITA] tag prepended
#
# Log: slurm-<jobid>.log. Needs ./setup_env.sh done once beforehand.
#SBATCH --account=bjdh-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=coding4
#SBATCH --output=slurm-%j.log

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="${HOME}/.pixi/bin:${PATH}"
eval "$(pixi shell-hook)"      # same as `pixi shell`, for a non-interactive job
nvidia-smi -L

MODE=${MODE:-monolingual}            # monolingual | multilingual
case "${MODE}" in
    monolingual)  SET=monolingual ;;
    multilingual) SET=bilingual ;;
    *) echo "unknown MODE=${MODE}" >&2; exit 1 ;;
esac
TAG=${TAG:-${MODE}}
EXP=exp/train_${TAG}

# 1. train
python train.py \
    --tag "${TAG}" \
    --mode "${MODE}" \
    --train_json dump/raw/train_${SET}/data.json \
    --valid_json dump/raw/dev_${SET}/data.json

# 2. decode with the best checkpoint (the only one kept in ckpts/)
CKPT=$(ls -t "${EXP}/ckpts" | head -n 1)
echo "checkpoint: ${CKPT}"
for split in dev test; do
    python decode.py \
        --exp_dir "${EXP}" --ckpt_name "${CKPT}" --decode_tag "${split}" \
        --recog_json dump/raw/${split}_${SET}/data.json --mode "${MODE}"
done
grep "wer=" "${EXP}/decode_dev_${CKPT}/decode.log" | tail -n 1     # dev WER

# 3. the file to submit (see prepare_submission.sh)
cp "${EXP}/decode_test_${CKPT}/decoded_hyp.txt" "decoded_hyp_${SET}.txt"
echo "wrote decoded_hyp_${SET}.txt"
echo ALL_STAGES_OK
