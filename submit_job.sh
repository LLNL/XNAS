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

# https://medium.com/@Drew_Stokes/bash-argument-parsing-54f3b81a6a8f
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
while (("$#")); do
  case "$1" in
  -h | --help)
    echo "Usage: $(basename "$0") [OPTION]..."
    echo ""
    echo "Submit a slurm batch job for the given task and other options."
    echo ""
    echo "  -h, --help                     Print this message"
    echo "  -s, --single-objective         Single-objective search (task performance only)"
    echo "  -t, --task TASK                The task/dataset name for which to run the"
    echo "                                 experiment (required)"
    echo "  -r, --reservation RESERVATION  The reservation name, if applicable"
    echo "  -n, --reservation-nodes RESERVATION_NODES"
    echo "                                 The reserved number of nodes, if applicable"
    exit 0
    ;;
  -s | --single-objective)
    if [ -n "$SINGLE_OBJECTIVE" ]; then
      echo "Error: flag $1 specified multiple times" >&2
      exit 2
    fi
    SINGLE_OBJECTIVE="true"
    shift
    ;;
  -t | --task)
    if [ -n "$TASK" ]; then
      echo "Error: flag $1 specified multiple times" >&2
      exit 2
    elif [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
      TASK=$2
      shift 2
    else
      echo "Error: argument for $1 is missing" >&2
      exit 2
    fi
    ;;
  -r | --reservation)
    if [ -n "$RESERVATION" ]; then
      echo "Error: flag $1 specified multiple times" >&2
      exit 2
    elif [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
      RESERVATION=$2
      shift 2
    else
      echo "Error: argument for $1 is missing" >&2
      exit 2
    fi
    ;;
  -n | --reservation-nodes)
    if [ -n "$RESERVATION_NODES" ]; then
      echo "Error: flag $1 specified multiple times" >&2
      exit 2
    elif [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
      RESERVATION_NODES=$2
      shift 2
    else
      echo "Error: argument for $1 is missing" >&2
      exit 2
    fi
    ;;
  *)
    echo "$(basename "$0"): invalid option -- $1" >&2
    echo "Try '$(basename "$0") --help' for more information." >&2
    exit 2
    ;;
  esac
done

if [ -z "$TASK" ]; then
  echo "Error: missing required argument: -t, --task TASK" >&2
  echo "Try '$(basename "$0") --help' for more information." >&2
  exit 2
fi

if [ -n "$RESERVATION" ] && [ -z "$RESERVATION_NODES" ]; then
  echo "Error: if a reservation, -n, --reservation-nodes RESERVATION_NODES must" >&2
  echo "also be specified." >&2
  echo "Try '$(basename "$0") --help' for more information." >&2
  exit 2
fi

if ! command -v sbatch &> /dev/null; then
  echo "You do not appear to be on a slurm-based job submission system!" >&2
  echo "Details: sbatch is not a command on \$PATH." >&2
  echo "Exiting without submitting a job :(" >&2
  exit 1
fi

# ensure cleanup happens in case of user keyboard interrupt
trap rm_tmp INT

function rm_tmp() {
  rm -rf "$batch_script_dir"
  # https://tldp.org/LDP/abs/html/exitcodes.html#EXITCODESREF
  exit 130
}

# make temporary directory for job to run within
batch_script_dir=$(mktemp --directory)
batch_script="$batch_script_dir/xnas_batch_script_$TASK.sh"

# populate the script
cat > "$batch_script" <<- EOM
#!/bin/bash
#SBATCH --output=logs_batch/slurm-xnas-$TASK-%A_%a.out
#SBATCH --job-name=xnas-$TASK
EOM

# reservation-specific lines
if [ -n "$RESERVATION" ]; then
  cat >> "$batch_script" <<- EOM
# start: reservation-specific directives
#SBATCH --array=0-$(( RESERVATION_NODES-1 ))
#SBATCH --time=0
#SBATCH --qos=exempt
#SBATCH --reservation=$RESERVATION
# end: reservation-specific directives
EOM
else
  cat >> "$batch_script" <<- EOM
#SBATCH --array=0-15
#SBATCH --time=1-00:00:00
EOM
fi

# host-specific lines
if [[ "$(hostname -f)" =~ ^flash[0-9]+\.llnl\.gov$ ]]; then
  cat >> "$batch_script" <<- EOM
# start: flash-specific directives
#SBATCH --partition=psummer
#SBATCH --exclusive
#SBATCH -G 4
# end: flash-specific directives
EOM
else
  cat >> "$batch_script" <<- EOM
#SBATCH --partition=pbatch
EOM
fi

cat >> "$batch_script" <<- EOM
echo "Started at \$(date)"
# symlink to "latest" log file
ln -fs \
"slurm-xnas-$TASK-\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.out" \
"logs_batch/slurm-xnas-$TASK-latest_\${SLURM_ARRAY_TASK_ID}.out"
env | grep SLURM
srun ./xnas-ray-job.sh $TASK $SINGLE_OBJECTIVE
status=\$?
echo "Finished at \$(date) with status \$status"
exit \$status
EOM

echo "Submitting the following job:"
echo "============================================================================="
# print script contents
cat "$batch_script"
echo "============================================================================="

sbatch "$batch_script"

# cleanup
rm_tmp
