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
import os

import numpy as np

from deephyper.search import util
from deephyper.core.logs.logging import JsonMessage as jm

from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_mutation
from pymoo.factory import get_sampling
from pymoo.factory import get_termination
from pymoo.util.display import MultiObjectiveDisplay

from xnas.nas_deephyper.xnas import XNAS
from xnas.pareto import is_pareto_efficient

dhlogger = util.conf_logger('xnas.nas_deephyper.nsga_ii')


class NSGAII(XNAS):
    """"""

    def __init__(self, *args, population_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self._reset_pymoo_vars()

    def _reset_pymoo_vars(self):
        self._non_dupe_mask = None
        self._actual_parents = None

    def handle_score(self, x):
        """Negate scores so we maximize instead of minimize (pymoo)"""
        scores = x[1]
        if self.pb_dict['multiobjective_explainability']:
            perf = scores['score']
            xai_fit = scores['xai_fitness']
            if self.record_mo_xai_only:
                return [-perf]
            elif xai_fit is None:
                dhlogger.warning('xai_fit is None! setting to zero...')
                return [-perf, 0]
            else:
                return [-perf, -xai_fit]
        else:
            return -scores

    def pymoo_problem(self):
        # base config dict
        pb_dict = self.pb_dict
        # deephyper evaluator
        evaluator = self.evaluator
        # saved_keys
        saved_keys = self.saved_keys
        # handle_score
        handle_score = self.handle_score

        search_space = self.problem.build_search_space()
        # lower bound (all 0)
        xl = 0
        # upper bounds
        variable_nodes = [*search_space.variable_nodes]
        xu = [(vnode.num_ops - 1) for vnode in search_space.variable_nodes]

        # number of variables
        n_var = len(variable_nodes)
        # number of objectives
        n_obj = (2 if (self.pb_dict['multiobjective_explainability'] and
                       not self.record_mo_xai_only) else 1)
        self_outer = self

        class ProblemCompat(Problem):
            def __init__(self):
                super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0,
                                 type_var=int, xl=xl, xu=xu)
                self._generation = 0

            def _evaluate(self, x, out, *args, **kwargs):
                jobs = []
                # iterate over members of population, map Pymoo pop to DH pop
                if self_outer._actual_parents is not None:
                    if (len(self_outer._actual_parents) >
                            self_outer.population_size):
                        # truncate as pymoo does from the RHS
                        dhlogger.warning(
                            f'Truncating actual_parents from '
                            f'{len(self_outer._actual_parents)} '
                            f'to {self_outer.population_size}'
                        )
                        self_outer._actual_parents = (
                            self_outer._actual_parents[
                            :self_outer.population_size])

                    assert (len(self_outer._actual_parents) >=
                            self_outer.population_size), (
                        len(self_outer._actual_parents),
                        self_outer.population_size)
                for i, x_i in enumerate(x):
                    cfg = pb_dict.copy()
                    cfg['arch_seq'] = x_i
                    cfg['pop_sort_key'] = i
                    cfg['generation'] = self._generation
                    # phylo stuff
                    if self_outer._actual_parents is None:
                        cfg['parent_ids'] = []  # no parents
                    else:
                        cfg['parent_ids'] = [
                            self._parent_ids[p_idx]
                            for p_idx in self_outer._actual_parents[i]
                        ]
                    jobs.append(cfg)
                # reset for next gen
                self_outer._reset_pymoo_vars()

                n_jobs = len(jobs)

                # submit jobs
                evaluator.add_eval_batch(jobs)
                # fetch ALL results
                results = []

                dhlogger.info(f'Begin evaluation population of size {n_jobs}')

                from time import time

                t_start = time()
                while len(results) != n_jobs:
                    finished_evals = list(evaluator.get_finished_evals())
                    if finished_evals:
                        results.extend(finished_evals)

                        dhlogger.info(f'Now have {len(results)} / {n_jobs} '
                                      f'members of population evaluated')
                    elif (time() - t_start) >= 120:  # seconds
                        t_start = time()
                        dhlogger.info(f'{len(evaluator.pending_evals)} pending '
                                      f'evals left')
                        if results:
                            best_acc = max(
                                results, key=lambda r: r[1]['score']
                            )[1]['score']
                            dhlogger.info(f'Best accuracy in generation='
                                          f'{best_acc * 100}%')
                        dhlogger.info(f'{evaluator.counter} evals total')

                # ensure results in proper order
                # each result: (cfg, scores)
                results = sorted(results, key=lambda r: r[0]['pop_sort_key'])

                # fetch scores - one score per column
                out['F'] = np.stack([handle_score(r) for r in results])

                # check if activations exist for runs that are not on the Pareto
                #  front for this generation and delete them from disk to save
                #  disk space. note we say minimize here as `handle_score`
                #  used to process `out['F']` negates objectives
                non_dominated_indices = is_pareto_efficient(
                    out['F'], maximize=False, return_mask=True)
                for i, nd_flag in enumerate(non_dominated_indices):
                    if nd_flag:
                        continue  # non-dominated
                    r_conf = results[i][0]
                    save_dir = os.path.join(r_conf.get('log_dir', ''),
                                            'activations')
                    save_path = os.path.join(save_dir,
                                             str(r_conf['id']) + '.npz')
                    if os.path.exists(save_path):
                        dhlogger.info(f'Removing {save_path} (not in front)')
                        os.remove(save_path)
                    else:
                        dhlogger.info(f'Cannot remove {save_path} as it does '
                                      f'not exist!!!')

                self._parent_ids = [r[0]['id'] for r in results]

                # DH log stuff
                stats = {
                    'num_cache_used': evaluator.stats['num_cache_used']}
                dhlogger.info(jm(type='env_stats', **stats))
                evaluator.dump_evals(saved_keys=saved_keys)  # noqa

                # increment generation (of evoAlg)
                self._generation += 1

        return ProblemCompat()

    def saved_keys(self, val: dict):
        res = super().saved_keys(val)
        res['generation'] = val['generation']
        res['parent_ids'] = str(val['parent_ids'])
        return res

    def pymoo_algorithm(self):

        from pymoo.model.population import Population
        from pymoo.operators.integer_from_float_operator import (
            apply_float_operation)
        from pymoo.operators.repair.to_bound import \
            set_to_bounds_if_outside_by_problem
        from pymoo.model.duplicate import default_attr
        from pymoo.util.misc import cdist

        self_outer = self

        class XNASCrossover:
            """
            copied from pymoo==0.4.2.2

            The crossover combines parents to offsprings. Some crossover are
            problem specific and use additional information. This class must be
            inherited from to provide a crossover method to an algorithm.
            """

            def __init__(self, n_parents, n_offsprings, prob=0.9):
                self.prob = prob
                self.n_parents = n_parents
                self.n_offsprings = n_offsprings

            def do(self, problem, pop, parents, **kwargs):
                """

                This method executes the crossover on the parents. This class
                wraps the implementation of the class that implements the
                crossover.

                Parameters
                ----------
                problem: class
                    The problem to be solved. Provides information such as lower
                    and upper bounds or feasibility conditions for custom
                    crossovers.

                pop : Population
                    The population as an object

                parents: numpy.array
                    The select parents of the population for the crossover

                kwargs : dict
                    Any additional data that might be necessary to perform the
                    crossover. E.g. constants of an algorithm.

                Returns
                -------
                offsprings : Population
                    The off as a matrix. n_children rows and the number of
                    columns is equal to the variable length of the problem.
                """
                if self.n_parents != parents.shape[1]:
                    raise ValueError(
                        'Exception during crossover: Number of parents differs '
                        'from defined at crossover.')

                # get the design space matrix form the population and parents
                X = pop.get("X")[parents.T].copy()

                # now apply the crossover probability
                do_crossover = np.random.random(len(parents)) < self.prob

                # execute the crossover
                _X = self._do(problem, X, **kwargs)

                # shape: n_offspring x
                #        parents (usually n_pop / n_offspring) x
                #        n_vars
                X[:, do_crossover, :] = _X[:, do_crossover, :]

                # flatten the array to become a 2d-array
                X = X.reshape(-1, X.shape[-1])

                # create a population object
                off = Population.new("X", X)

                # process things for phylo
                assert self.n_offsprings == 2
                assert self.n_parents == 2
                assert self.n_offsprings == self.n_parents
                # find new/merged parents
                actual_parents = parents.tolist() * self.n_offsprings
                # for each offspring:
                for i in range(self.n_offsprings):
                    ja = i * len(parents)
                    jb = ja + len(parents)
                    # for each set of parents:
                    for jp, j in enumerate(range(ja, jb)):
                        if not do_crossover[jp]:
                            # no crossover was done - select the parent this
                            #  offspring came from
                            actual_parents[j] = [parents[jp, i]]
                        # else: the parents are already correct
                # store phylo info
                if self_outer._actual_parents is not None:
                    actual_parents_old = self_outer._actual_parents
                    # this must be set (by XNAS duplicate handler) if
                    #  _actual_parents has been set (otherwise something bad has
                    #  happened)
                    assert self_outer._non_dupe_mask is not None
                    assert (len(self_outer._non_dupe_mask) ==
                            self._true_actual_parents_old_len, (
                                (len(self_outer._non_dupe_mask),
                                 self._true_actual_parents_old_len)))
                    actual_parents_cur = actual_parents_old[
                                         :-self._true_actual_parents_old_len]
                    actual_parents_cur += [
                        parents
                        for parents, keep_flag in zip(
                            actual_parents_old[
                            -self._true_actual_parents_old_len:],
                            self_outer._non_dupe_mask)
                        if keep_flag
                    ]
                    if len(actual_parents_cur) > self_outer.population_size:
                        # truncate as pymoo does from the RHS
                        actual_parents_cur = actual_parents_cur[
                                             :self_outer.population_size]

                    self._true_actual_parents_old_len = len(actual_parents)
                    actual_parents = actual_parents_cur + actual_parents
                else:
                    self._true_actual_parents_old_len = len(actual_parents)

                self_outer._actual_parents = actual_parents

                return off

        class XNASIntegerFromFloatCrossover(XNASCrossover):
            """copied from pymoo==0.4.2.2"""

            def __init__(self, clazz=None, **kwargs):
                if clazz is None:
                    raise Exception(
                        "Please define the class of the default crossover to "
                        "use XNASIntegerFromFloatCrossover.")

                self.crossover = clazz(**kwargs)
                super().__init__(self.crossover.n_parents,
                                 self.crossover.n_offsprings,
                                 prob=self.crossover.prob)

            def _do(self, problem, X, **kwargs):
                def fun():
                    return self.crossover._do(problem, X, **kwargs)

                return apply_float_operation(problem, fun)

        class XNASSimulatedBinaryCrossover(XNASCrossover):
            """copied from pymoo==0.4.2.2"""

            def __init__(self, eta, n_offsprings=2, prob_per_variable=0.5,
                         **kwargs):
                super().__init__(2, n_offsprings, **kwargs)
                self.eta = float(eta)
                self.prob_per_variable = prob_per_variable

            def _do(self, problem, X, **kwargs):
                X = X.astype(float)
                _, n_matings, n_var = X.shape

                # boundaries of the problem
                xl, xu = problem.xl, problem.xu

                # crossover mask that will be used in the end
                do_crossover = np.full(X[0].shape, True)

                # per variable the probability is then 50%
                do_crossover[
                    np.random.random((n_matings, problem.n_var))
                    > self.prob_per_variable] = False
                # also if values are too close no mating is done
                do_crossover[np.abs(X[0] - X[1]) <= 1.0e-14] = False

                # assign y1 the smaller and y2 the larger value
                y1 = np.min(X, axis=0)
                y2 = np.max(X, axis=0)

                # random values for each individual
                rand = np.random.random((n_matings, problem.n_var))

                def calc_betaq(beta):
                    alpha = 2.0 - np.power(beta, -(self.eta + 1.0))

                    mask, mask_not = (rand <= (1.0 / alpha)), (
                            rand > (1.0 / alpha))

                    betaq = np.zeros(mask.shape)
                    betaq[mask] = \
                        np.power((rand * alpha), (1.0 / (self.eta + 1.0)))[mask]
                    betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)),
                                               (1.0 / (self.eta + 1.0)))[
                        mask_not]

                    return betaq

                # difference between all variables
                delta = (y2 - y1)

                # now just be sure not dividing by zero (these cases will be
                # filtered later anyway)
                delta[delta < 1.0e-10] = 1.0e-10

                beta = 1.0 + (2.0 * (y1 - xl) / delta)
                betaq = calc_betaq(beta)
                c1 = 0.5 * ((y1 + y2) - betaq * delta)

                beta = 1.0 + (2.0 * (xu - y2) / delta)
                betaq = calc_betaq(beta)
                c2 = 0.5 * ((y1 + y2) + betaq * delta)

                # do randomly a swap of variables
                b = np.random.random((n_matings, problem.n_var)) <= 0.5
                val = np.copy(c1[b])
                c1[b] = c2[b]
                c2[b] = val

                # take the parents as _template
                c = np.copy(X)

                # copy the positions where the crossover was done
                c[0, do_crossover] = c1[do_crossover]
                c[1, do_crossover] = c2[do_crossover]

                c[0] = set_to_bounds_if_outside_by_problem(problem, c[0])
                c[1] = set_to_bounds_if_outside_by_problem(problem, c[1])

                if self.n_offsprings == 1:
                    # Randomly select one offspring
                    c = c[
                        np.random.choice(2, X.shape[1]), np.arange(X.shape[1])]
                    c = c.reshape((1, X.shape[1], X.shape[2]))

                return c

        class XNASDuplicateElimination:

            def __init__(self, func=None) -> None:
                super().__init__()
                self.func = func

                if self.func is None:
                    self.func = default_attr

            def do(self, pop, *args, return_indices=False, to_itself=True):
                original = pop

                if len(pop) == 0:
                    self_outer._non_dupe_mask = None
                    return pop

                if to_itself:
                    keep_mask = (~self._do(pop, None, np.full(len(pop), False)))
                    pop = pop[keep_mask]
                    running_keep_mask = keep_mask
                else:
                    running_keep_mask = np.full(len(pop), True)

                for arg in args:
                    if len(arg) > 0:
                        if len(pop) == 0:
                            break
                        elif len(arg) == 0:
                            continue
                        else:
                            keep_mask = (
                                ~self._do(pop, arg, np.full(len(pop), False)))
                            pop = pop[keep_mask]
                            running_keep_mask[running_keep_mask] = keep_mask

                # expose
                self_outer._non_dupe_mask = running_keep_mask

                if return_indices:
                    no_duplicate, is_duplicate = [], []
                    H = set(pop)

                    for i, ind in enumerate(original):
                        if ind in H:
                            no_duplicate.append(i)
                        else:
                            is_duplicate.append(i)

                    return pop, no_duplicate, is_duplicate
                else:
                    return pop

            def _do(self, pop, other, is_duplicate):
                raise NotImplementedError

        class XNASDefaultDuplicateElimination(XNASDuplicateElimination):

            def __init__(self, epsilon=1e-16, **kwargs) -> None:
                super().__init__(**kwargs)
                self.epsilon = epsilon

            def calc_dist(self, pop, other=None):
                X = self.func(pop)

                if other is None:
                    D = cdist(X, X)
                    D[np.triu_indices(len(X))] = np.inf
                else:
                    _X = self.func(other)
                    D = cdist(X, _X)

                return D

            def _do(self, pop, other, is_duplicate):
                D = self.calc_dist(pop, other)
                D[np.isnan(D)] = np.inf

                is_duplicate[np.any(D < self.epsilon, axis=1)] = True
                return is_duplicate

        return NSGA2(
            pop_size=self.population_size,
            sampling=get_sampling('int_random'),
            crossover=XNASIntegerFromFloatCrossover(
                clazz=XNASSimulatedBinaryCrossover, prob=0.9, eta=3.0
            ),
            mutation=get_mutation('int_pm', eta=3.0),
            eliminate_duplicates=XNASDefaultDuplicateElimination(),
            n_offsprings=None,
            repair=None,  # NOTE: repair unsupported by phylo trees
            mating=None,
            min_infeas_pop_size=0,
        )

    def main(self):

        res = minimize(
            problem=self.pymoo_problem(),
            algorithm=self.pymoo_algorithm(),
            termination=get_termination('n_eval', self.max_evals),
            callback=None,
            display=MultiObjectiveDisplay(),
            seed=None,
            verbose=False,
            save_history=False,
            return_least_infeasible=False,
            pf=True,
            evaluator=None,
        )

        dhlogger.info(f'Result `res`:\n{res}')


if __name__ == '__main__':
    args_ = NSGAII.parse_args()
    search = NSGAII(**vars(args_))
    search.main()
