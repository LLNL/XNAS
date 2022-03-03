"""
MIT License

Copyright (c) 2022, Lawrence Livermore National Security, LLC
Written by Zachariah Carmichael et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from abc import ABC

from deephyper.search.nas import NeuralArchitectureSearch  # noqa
from deephyper.core.parser import add_arguments_from_signature
from deephyper.core.parser import str2bool
from deephyper.evaluator.evaluate import Evaluator
from deephyper.problem.neuralarchitecture import NaProblem

from xnas.utils import get_logger

logger = get_logger(__name__)


class XNAS(NeuralArchitectureSearch, ABC):
    """XNAS base class
    Args:
        problem (str): Module path to the Problem instance you want to use for
            `the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the
            search (e.g. deephyper.nas.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool',
            'threadPool'].
    """
    # Type hints
    evaluator: Evaluator
    problem: NaProblem

    def __init__(
            self, problem, run='xnas.nas_deephyper.nas_run.run',
            evaluator='ray',
            multiobjective_explainability=True, record_mo_xai_only=False,
            explainability_type='activations',
            weight_perf=0.5, weight_xai=5.0, **kwargs
    ):
        multiobjective_explainability = str2bool(multiobjective_explainability)
        record_mo_xai_only = str2bool(record_mo_xai_only)
        if (run == 'deephyper.nas.run.alpha.run' and
                multiobjective_explainability):
            logger.warning(f'Running with both {run} and '
                           f'multiobjective_explainability! This is known '
                           f'to cause runtime issues as {run} does not '
                           f'handle the multi-objective case...')
        super().__init__(problem=problem, run=run, evaluator=evaluator,
                         **kwargs)

        # Setup
        self.free_workers = self.evaluator.num_workers
        self.pb_dict: dict = self.problem.space
        self.pb_dict['multiobjective_explainability'] = \
            multiobjective_explainability
        self.pb_dict['explainability_type'] = explainability_type
        self.record_mo_xai_only = record_mo_xai_only

        self.weight_perf = weight_perf
        self.weight_xai = weight_xai

        logger.info(f'Running with multiobjective_explainability='
                    f'{multiobjective_explainability}, record_mo_xai_only='
                    f'{record_mo_xai_only}, weight_perf={weight_perf}, '
                    f'weight_xai={weight_xai}')

    def handle_score(self, x):
        scores = x[1]
        if self.pb_dict['multiobjective_explainability']:
            assert isinstance(scores, dict), f'x: {x} | scores: {scores}'
            assert len(scores) >= 3, f'x: {x} | scores: {scores}'
            # 3rd: layerwise
            perf = scores['score']
            xai_fit = scores['xai_fitness']
            if xai_fit is None or self.record_mo_xai_only:
                return perf
            else:
                # cosine: no tanh
                return self.weight_perf * perf + self.weight_xai * xai_fit
        else:
            return scores

    @classmethod
    def _extend_parser(cls, parser):
        NeuralArchitectureSearch._extend_parser(parser)
        add_arguments_from_signature(parser, cls)
        if cls is not XNAS:
            add_arguments_from_signature(parser, XNAS)
        return parser

    def saved_keys(self, val: dict):
        res = {
            'id': val['id'],
            'arch_seq': str(val['arch_seq'])
        }
        return res
