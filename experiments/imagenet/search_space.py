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
from xnas.search_space_common import nas_bench_201_search_space
from xnas.search_space_common import summarize_search_space_arch


def create_search_space(
        input_shape=(16, 16, 3), output_shape=(120,),
        filter_scale_factor=1,
        *args, **kwargs
):
    if isinstance(input_shape[0], int):
        # format given: (int, int, ...)
        #      desired: [(int, int, ...), (...)]
        input_shape = [input_shape]

    return nas_bench_201_search_space(
        input_shape,
        output_shape,
        regression=False,
        n_nodes=4,
        stack_size=5,
        filter_scale_factor=filter_scale_factor,
    )


def test_create_search_space(*args):
    """Generate a random neural network from the search_space definition"""
    from random import random

    search_space = create_search_space()

    if args:
        ops = [int(arg) for arg in args]
        print('parsed:', ops)
    else:  # random
        ops = [random() for _ in range(search_space.num_nodes)]

    summarize_search_space_arch(search_space, ops)


if __name__ == '__main__':
    import sys

    test_create_search_space(*sys.argv[1:])
