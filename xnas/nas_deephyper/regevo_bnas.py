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
import collections
import numpy as np

from deephyper.search import util
from deephyper.core.logs.logging import JsonMessage as jm

from xnas.nas_deephyper.xnas import XNAS

dhlogger = util.conf_logger('xnas.nas_deephyper.regevo_xnas')


class RegularizedEvolutionXNAS(XNAS):
    """Regularized evolution with multiobjective XNAS.
    https://arxiv.org/abs/1802.01548
    Args:
        problem (str): Module path to the Problem instance you want to use for
            `the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the
            search (e.g. deephyper.nas.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool',
            'threadPool'].
        population_size (int, optional): the number of individuals to keep in
            the population. Defaults to 100.
        sample_size (int, optional): the number of individuals that should
            participate in each tournament. Defaults to 10.
    """

    def __init__(
            self, problem, run, evaluator, population_size=10, sample_size=5,
            **kwargs
    ):

        super().__init__(problem=problem, run=run, evaluator=evaluator,
                         **kwargs)

        # Setup
        search_space = self.problem.build_search_space()
        self.space_list = [
            (0, vnode.num_ops - 1) for vnode in search_space.variable_nodes
        ]
        self.population_size = int(population_size)
        self.sample_size = int(sample_size)

    def main(self):

        num_evals_done = 0
        population = collections.deque(maxlen=self.population_size)

        # Filling available nodes at start
        self.evaluator.add_eval_batch(
            self.gen_random_batch(size=self.free_workers))

        # Main loop
        while num_evals_done < self.max_evals:

            # Collecting finished evaluations
            new_results = list(self.evaluator.get_finished_evals())

            if len(new_results) > 0:
                population.extend(new_results)
                stats = {
                    'num_cache_used': self.evaluator.stats['num_cache_used']}
                dhlogger.info(jm(type='env_stats', **stats))
                self.evaluator.dump_evals(saved_keys=self.saved_keys)

                num_received = len(new_results)
                num_evals_done += num_received

                if num_evals_done >= self.max_evals:
                    break

                # If the population is big enough evolve the population
                if len(population) == self.population_size:
                    children_batch = []
                    # For each new parent/result we create a child from it
                    for _ in range(len(new_results)):
                        # select_sample
                        indexes = np.random.choice(
                            self.population_size, self.sample_size,
                            replace=False
                        )
                        sample = [population[i] for i in indexes]
                        # select_parent
                        parent = self.select_parent(sample)
                        # copy_mutate_parent
                        child = self.copy_mutate_arch(parent)
                        # add child to batch
                        children_batch.append(child)
                    # submit_children
                    if len(new_results) > 0:
                        self.evaluator.add_eval_batch(children_batch)
                else:  # If the population is too small keep increasing it
                    self.evaluator.add_eval_batch(
                        self.gen_random_batch(size=len(new_results))
                    )

    def select_parent(self, sample: list) -> dict:
        # score
        cfg, _ = max(sample, key=self.handle_score)
        return cfg

    def gen_random_batch(self, size: int) -> list:
        batch = []
        for _ in range(size):
            cfg = self.pb_dict.copy()
            cfg['arch_seq'] = self.random_search_space()
            cfg['parent_id'] = ''
            batch.append(cfg)
        return batch

    def random_search_space(self) -> list:
        return [np.random.choice(b + 1) for (_, b) in self.space_list]

    def copy_mutate_arch(self, parent: dict) -> dict:
        """
        # ! Time performance is critical because called sequentially
        Args:
            parent (dict): [description]
        Returns:
            dict: [description]
        """
        parent_arch: list = parent['arch_seq']
        i = np.random.choice(len(parent_arch))
        child_arch = parent_arch[:]

        range_upper_bound = self.space_list[i][1]
        elements = [j for j in range(range_upper_bound + 1)
                    if j != child_arch[i]]

        # The mutation has to create a different search_space!
        sample = np.random.choice(elements, 1)[0]

        child_arch[i] = sample
        cfg = self.pb_dict.copy()
        cfg['arch_seq'] = child_arch
        cfg['parent_id'] = parent['id']
        return cfg

    def saved_keys(self, val: dict):
        res = super().saved_keys(val)
        res['parent_id'] = val['parent_id']
        return res


if __name__ == '__main__':
    args = RegularizedEvolutionXNAS.parse_args()
    search = RegularizedEvolutionXNAS(**vars(args))
    search.main()
