# eXplainable Neural Architecture Search (XNAS)
Code for the paper
[_Learning Interpretable Models Through Multi-Objective Neural Architecture Search_](https://arxiv.org/abs/2112.08645)
by
[Zachariah Carmichael](https://github.com/craymichael),
[Tim Moon](https://github.com/timmoon10), and
[Sam Ade Jacobs](https://github.com/samadejacobs).

Paper abstract:
> Monumental advances in deep learning have led to unprecedented achievements across a
> multitude of domains. While the performance of deep neural networks is indubitable, the
> architectural design and interpretability of such models are nontrivial. Research has been
> introduced to automate the design of neural network architectures through
> _neural architecture search (NAS)_. Recent progress has made these methods more pragmatic by
> exploiting distributed computation and novel optimization algorithms. However, there is
> little work in optimizing architectures for interpretability. To this end, we propose a
> multi-objective distributed NAS framework that optimizes for both task performance and
> introspection. We leverage the non-dominated sorting genetic algorithm (NSGA-II) and
> explainable AI (XAI) techniques to reward architectures that can be better comprehended by
> humans. The framework is evaluated on several image classification datasets. We demonstrate
> that jointly optimizing for introspection ability and task error leads to more disentangled
> architectures that perform within tolerable error.

# Installation
Python 3.6 or newer is required. Python 3.8 is highly recommended as it was used in all experiments.
For a minimally sufficient environment to run the code:

```shell
pip install -r requirements.txt
```

To reproduce our environment exactly as used in experiments (using conda):
```shell
conda env create -f environment.yml
```

# Running

Slurm Workload Manager which will use Ray for multi-node:
```text
$ ./submit_job.sh -h
Usage: submit_job.sh [OPTION]...

Submit a slurm batch job for the given task and other options.

  -h, --help                     Print this message
  -s, --single-objective         Single-objective search (task performance only)
  -t, --task TASK                The task/dataset name for which to run the
                                 experiment (required)
  -r, --reservation RESERVATION  The reservation name, if applicable
  -n, --reservation-nodes RESERVATION_NODES
                                 The reserved number of nodes, if applicable
```
Before submitting a job, make sure you update the variable in the script `ACTIVATE_PYTHON_ENV`
to the name of your conda environment. If not using conda, the script will need minor
refactoring to activate your environment rather than conda.

For running on a single node:
```shell
TASK=mnist
SINGLE_OBJECTIVE=false
EXPLAINABILITY_TYPE=activations
CPUS_PER_TASK=16
GPUS_PER_TASK=1
PYTHONPATH="$PWD" ./deephyper_xnas nas "nsga2" \
    --problem "experiments.$TASK.problem.Problem" \
    --run "xnas.nas_deephyper.nas_run.run" \
    --multiobjective-explainability "true" \
    --record-mo-xai-only "$SINGLE_OBJECTIVE" \
    --explainability-type "$EXPLAINABILITY_TYPE" \
    --max-evals 16000 \
    --evaluator "ray" \
    --ray-address "auto" \
    --num-cpus-per-task $CPUS_PER_TASK \
    --num-gpus-per-task $GPUS_PER_TASK
```

# Custom Experiments
See the directory structure of each task within `experiments/`. You must create the files
`load_data.py`, `problem.py`, and `search_space.py` following the existing experiments. You
can also modify the parameters for the existing experiments in these files.

## `load_data.py`

This file requires that a function be defined with the name `load_data()`.
See `experiments/mnist/load_data.py` for an example of the expected return structure.
The role of the function is to load a dataset (ideally) into TensorFlow `Dataset` objects,
split into train and validation sets. The return value provides the data, the data and
label types, and the data shapes and sizes. This is where you should define how your custom
dataset should be loaded.

## `search_space.py`

This file requires that a function be defined with the name `create_search_space()`.
See `experiments/mnist/search_space.py` for an example of the function signature.
It is best practice to specify the input and output shapes of the associated data
set with the experiment as defaults. It is also important that the first conditional
statement block starting with `if isinstance(...` be included in your code; the input
shape is expected to be a `list` of `tuple` objects in the case that architectures
are expected to have multiple inputs, e.g., multiple images per example. This is not
the case in any of our experiments. In this file, you can specify a custom search space
and whether your task is classification or regression (note that introspectability is
classification-only in our paper).

If you would like to create a custom search space, good examples to follow are defined
in `xnas/search_space_common.py`. We use the function within called `nas_bench_201_search_space`
to produce the NASBench201 cell-based search space.
In addition, read the `deephyper` [0.2.5 docs](https://github.com/deephyper/deephyper/tree/0.2.5/docs)
for how to define search spaces of any complexity. For instance:
[Defining the search space](https://github.com/deephyper/deephyper/blob/0.2.5/docs/user_guides/nas/problem.rst#defining-the-search-space=)
[polynome2 example search space](https://github.com/deephyper/deephyper/blob/0.2.5/docs/tutorials/polynome2_nas/search_space.py)


## `problem.py`

This file requires that a `deephyper.problem.NaProblem` object be defined with the
variable name `Problem`. See `experiments/mnist/problem.py` for an example. The object
provides the `load_data()` function for the problem, the search space, the
hyperparameters, the loss function, the evaluation metrics, and the task-related
objective (differentiated from an XAI-related objective, e.g., introspectability)
to use in the NAS algorithm.


# Note on hierarchical_imagenet.py
The file `hierarchical_imagenet.py` is used to generate the label similarities in
ImageNet-derived datasets. This is used when `--explainability-type` is "activations-imagenet"
and the file `imagenet_resized_label_distances_9aca0f6d9df6da4fa8c2f33ee7fb8fd9.csv` is already
included (generated by the script). However, you can regenerate this file yourself if you want to
make changes or make changes to use the WordNet weighting in another dataset.


# Results
Results are stored with the following tree structure (`submit_job.sh` script):
```text
results/
|-- <task>/
|   |-- <timestamp>/
|   |   |-- deephyper.log (DeepHyper logging output)
|   |   |-- init_infos.json (DeepHyper problem definition)
|   |   |-- results.csv (pertinent results you are likely interested in)
|   |   |-- activations/ (activations for intra-gen Pareto-optimal solutions)
|   |   |   |-- <UID>.npz
|   |   |   `-- ...
|   |   `-- save/
|   |       |-- config/ (configuration for each instantiated model)
|   |       |   |-- <UUID>.json
|   |       |   `-- ...
|   |       |-- history/
|   |       |   |-- <UUID>.json
|   |       `-- model/ (empty)
|   |-- ...
|-- ...
```
Logs are stored in `logs_batch/`.

# Citing XNAS

BibTeX:

```text
@article{carmichael2021xnas,
  title   = {Learning Interpretable Models Through Multi-Objective Neural Architecture Search},
  author  = {Carmichael, Zachariah and Moon, Tim and Jacobs, Sam Ade},
  journal = {arXiv},
  volume  = {arXiv/2112.08645},
  year    = {2021},
  url     = {https://arxiv.org/abs/2112.08645}, 
}
```

# License

XNAS is distributed under the terms of the MIT license.

SPDX-License-Identifier: MIT

LLNL-CODE-831992