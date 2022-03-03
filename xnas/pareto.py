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
import numpy as np


def is_pareto_efficient(
        costs: np.ndarray,
        return_mask: bool = True,
        maximize: bool = True,
) -> np.ndarray:
    """
    Find the pareto-efficient points
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :param maximize: True if maximizing
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    assert costs.ndim == 2
    costs_ = np.copy(costs)
    costs_[np.isnan(costs_)] = np.inf
    if maximize:
        costs_ = -costs_
    is_efficient, n_points = np.arange(costs_.shape[0]), costs_.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_):
        nondominated_point_mask = np.any(costs_ < costs_[next_point_index],
                                         axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_ = costs_[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient  # else


def nondominating(
        costs: np.ndarray,
        maximize: bool = True,
) -> np.ndarray:
    mask = is_pareto_efficient(costs, maximize=maximize, return_mask=True)
    return costs[mask]
