#!/usr/bin/env bash
# Run speedup experiments sequentially, each in its own container.
#
# Each argument is an experiment branch name. For every branch the script
# checks out a clean git worktree (detached at the branch tip, so it works
# even if the branch is checked out in the main repo), mounts the dataset
# and a per-experiment results dir, and runs training in the docker image.
# All container output is written to $RESULTS_DIR/{exp_name}/train.log.
# TensorBoard event files (Profiling/*, Test/*, Loss/*) land in the same dir.
#
# Usage:
#   ./run_experiments.sh                         # run the whole stack in order
#   ./run_experiments.sh exp0-baseline           # run a single experiment
#   EPOCHS=12 ./run_experiments.sh exp2-ema-foreach exp3-persistent-workers
#
# Environment overrides:
#   EPOCHS          total training epochs (default 6 -> 5 stage1 + 1 stage2,
#                   epoch 0 is warmup, exclude it when comparing branches)
#   RES             training resolution (default "512 512")
#   IMAGE           docker image (default quay.io/logivations/ml_all:LS_dfine_latest)
#   DATASET_DIR     host dataset root in the OLD layout: train/, test/ and
#                   coco_annotations.json (default /data/GM_dataset)
#   RESULTS_DIR     host results root (default /data/GM_results)
#   WORKTREES_DIR   where per-branch worktrees are created (default /data/DEIM_worktrees)
#   STG1_PERC       optional stage-2 share override, e.g. 0.5 for a 12+12 split
#                   with EPOCHS=24 (passed as -u stg1_epochs_perc=...)
#   EXTRA_UPDATES   extra -u overrides appended to the train command
#   CONTINUE_ON_FAIL=1  keep running remaining experiments after a failure
#
# NOTE: only experiment branches (exp0-baseline and later) understand the
# profile_sync flag; `main` does not. The baseline run is exp0-baseline.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCHS="${EPOCHS:-6}"
RES="${RES:-512 512}"
IMAGE="${IMAGE:-quay.io/logivations/ml_all:LS_dfine_latest}"
DATASET_DIR="${DATASET_DIR:-/data/GM_dataset}"
RESULTS_DIR="${RESULTS_DIR:-/data/GM_results}"
WORKTREES_DIR="${WORKTREES_DIR:-/data/DEIM_worktrees}"
STG1_PERC="${STG1_PERC:-}"
EXTRA_UPDATES="${EXTRA_UPDATES:-}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"

DEFAULT_STACK=(
  exp0-baseline
  exp1-tf32-nan-gate
  exp2-ema-foreach
  exp3-persistent-workers
  exp4-criterion-desync
  exp5-decode-backend
)

if [ "$#" -gt 0 ]; then
  BRANCHES=("$@")
else
  BRANCHES=("${DEFAULT_STACK[@]}")
fi

# Old dataset layout (the experiment branches revert the RTDT-7618 new-layout PR)
if [ ! -d "$DATASET_DIR/train" ] || [ ! -f "$DATASET_DIR/coco_annotations.json" ]; then
  echo "ERROR: $DATASET_DIR must contain train/, test/ and coco_annotations.json (old dataset layout)" >&2
  exit 1
fi

UPDATES="profile_sync=True"
if [ -n "$STG1_PERC" ]; then
  UPDATES="$UPDATES stg1_epochs_perc=$STG1_PERC"
fi
if [ -n "$EXTRA_UPDATES" ]; then
  UPDATES="$UPDATES $EXTRA_UPDATES"
fi

mkdir -p "$WORKTREES_DIR" \
  "$REPO_DIR/.cache/torch" "$REPO_DIR/.cache/triton" "$REPO_DIR/.cache/nv"

FAILED=()
for BRANCH in "${BRANCHES[@]}"; do
  EXP_NAME="${BRANCH//\//-}"
  WT="$WORKTREES_DIR/$EXP_NAME"
  EXP_RESULTS="$RESULTS_DIR/$EXP_NAME"

  echo ""
  echo "================================================================"
  echo "=== Experiment: $EXP_NAME (branch: $BRANCH, epochs: $EPOCHS)"
  echo "================================================================"

  # Resolve the branch: local first, then origin/<branch> (fresh clones have
  # the experiment branches only as remote-tracking refs).
  if git -C "$REPO_DIR" rev-parse --verify --quiet "refs/heads/$BRANCH" >/dev/null; then
    REF="$BRANCH"
  elif git -C "$REPO_DIR" rev-parse --verify --quiet "refs/remotes/origin/$BRANCH" >/dev/null; then
    REF="origin/$BRANCH"
  else
    echo "ERROR: branch '$BRANCH' not found in $REPO_DIR (try: git fetch origin)" >&2
    FAILED+=("$EXP_NAME")
    [ "$CONTINUE_ON_FAIL" = "1" ] && continue || break
  fi

  # Fresh detached worktree at the branch tip: guarantees the container sees
  # exactly the committed branch code, never a dirty checkout.
  if [ -d "$WT" ]; then
    git -C "$REPO_DIR" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
  fi
  git -C "$REPO_DIR" worktree add --force --detach "$WT" "$REF" || {
    echo "ERROR: failed to create worktree for $BRANCH" >&2
    FAILED+=("$EXP_NAME")
    [ "$CONTINUE_ON_FAIL" = "1" ] && continue || break
  }
  git -C "$WT" log --oneline -1

  mkdir -p "$EXP_RESULTS"

  docker run --name "deim_train_$EXP_NAME" --rm --gpus all --shm-size=16g \
    --network host -w /DEIM \
    -v "$WT":/DEIM \
    -v "$DATASET_DIR":/dataset:ro \
    -v "$EXP_RESULTS":/result \
    -v "$REPO_DIR/.cache/torch":/root/.cache/torch \
    -v "$REPO_DIR/.cache/triton":/root/.triton/cache \
    -v "$REPO_DIR/.cache/nv":/root/.nv \
    "$IMAGE" bash -c "torchrun --master_port=7777 --nproc_per_node=1 train.py \
      -c configs/label_studio/ls_dfine_hgnetv2_s_coco.yml \
      --train-epochs $EPOCHS --training-res $RES --use-amp --seed=0 \
      -u $UPDATES" 2>&1 | tee "$EXP_RESULTS/train.log"
  STATUS=${PIPESTATUS[0]}

  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: experiment $EXP_NAME failed with exit code $STATUS (see $EXP_RESULTS/train.log)" >&2
    FAILED+=("$EXP_NAME")
    [ "$CONTINUE_ON_FAIL" = "1" ] || break
  else
    echo "=== Experiment $EXP_NAME finished OK, results in $EXP_RESULTS"
  fi
done

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED experiments: ${FAILED[*]}" >&2
  exit 1
fi
echo "All experiments finished. Compare TensorBoard files under $RESULTS_DIR/"
echo "  tensorboard --logdir $RESULTS_DIR"
