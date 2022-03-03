#!/bin/bash
# MIT License
#
# Copyright (c) 2022, Lawrence Livermore National Security, LLC
# Written by Zachariah Carmichael et al.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

if [ $# -eq 1 ]; then
  TASK="$1"
  # we default to multi-objective search
  SINGLE_OBJECTIVE="false"
elif [ $# -eq 2 ]; then
  TASK="$1"
  SINGLE_OBJECTIVE="$2"
else
  echo "Usage: $(basename "$0") <task>" >&2
  exit 1
fi

echo "$(basename "$0") parsed arguments:"
echo "              TASK=$TASK"
echo "  SINGLE_OBJECTIVE=$SINGLE_OBJECTIVE"

# Hard-coded flags
CPU_ONLY=0

# Common HW info
CPUS_PER_NODE=$(nproc --all)

if [ $CPU_ONLY -eq 0 ]; then
    if [ -z $SLURM_GPUS ]; then
        if [ -z $SLURM_JOB_GPUS ]; then
            N_GPU=$(ls /proc/driver/nvidia/gpus | wc -l)
        else
            N_GPU=$(echo $SLURM_JOB_GPUS | awk -F',' '{print NF}')
        fi
    else
        N_GPU=$SLURM_GPUS
    fi
    echo "Discovered $N_GPU GPUs on machine."
    CPUS_PER_TASK=$(($CPUS_PER_NODE / $((N_GPU < 1 ? 1 : N_GPU))))
    GPUS_PER_NODE=$N_GPU
    GPUS_PER_TASK=1
    NUM_WORKERS=$N_GPU
else
    echo "CPU only!!!"
    export CUDA_VISIBLE_DEVICES=''
    CPUS_PER_TASK=$CPUS_PER_NODE
    GPUS_PER_NODE=0
    GPUS_PER_TASK=0
    NUM_WORKERS=1
fi

echo ""
echo "     hostname=$(hostname -f)"
echo "CPUS_PER_NODE=$CPUS_PER_NODE"
echo "CPUS_PER_TASK=$CPUS_PER_TASK"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "GPUS_PER_TASK=$GPUS_PER_TASK"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo ""

# In case we need to do GRPC debugging
# export GRPC_VERBOSITY=debug
# export GRPC_TRACE=all

# script

ACTIVATE_PYTHON_ENV="xnas"
echo "Activate conda env: $ACTIVATE_PYTHON_ENV"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/etc/profile.d/conda.sh" ]; then
        . "/usr/etc/profile.d/conda.sh"
    else
        export PATH="/usr/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate $ACTIVATE_PYTHON_ENV

head_node=$(squeue -j ${SLURM_ARRAY_JOB_ID}_0 | tail -1 | rev | cut -d' ' -f1 | rev)
head_node_ip=$(ping -c1 "$head_node" | head -1 | sed 's/^[^()]*(\([0-9\.]\+\)).*$/\1/')

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Starting the Ray Head Node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# https://github.com/ray-project/ray/issues/1269
ulimit -n 65536

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "Starting HEAD at $head_node"

  ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --include-dashboard="false" --num-cpus $CPUS_PER_NODE \
    --num-gpus $GPUS_PER_NODE --block &

  # optional, though may be useful in certain versions of Ray < 1.0.
  sleep 5

  echo "Start deephyper"

  projdir=$(realpath .)
  workdir=results/"$TASK"/$(date +"%Y-%m-%dT%H%M%S")/
  mkdir -p "$workdir"
  cd "$workdir" || exit
  echo -e "\nSave results and logs to ${workdir}\n"

  # set explainability type according to task
  case "$TASK" in
    imagenet) EXPLAINABILITY_TYPE="activations-imagenet" ;;
    *) EXPLAINABILITY_TYPE="activations" ;;
  esac

  PYTHONPATH="$projdir:$PYTHONPATH" "$projdir"/deephyper_xnas nas "nsga2" \
    --problem "experiments.$TASK.problem.Problem" \
    --run "xnas.nas_deephyper.nas_run.run" \
    --multiobjective-explainability "true" \
    --record-mo-xai-only "$SINGLE_OBJECTIVE" \
    --explainability-type "$EXPLAINABILITY_TYPE" \
    --max-evals 16000 \
    --evaluator "ray" \
    --ray-address "auto" \
    --num-cpus-per-task $CPUS_PER_TASK \
    --num-gpus-per-task $GPUS_PER_TASK \
    --num-workers $NUM_WORKERS

  ray stop
else
  sleep 5

  ray start --address "$ip_head" --include-dashboard="false" \
    --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block
fi
