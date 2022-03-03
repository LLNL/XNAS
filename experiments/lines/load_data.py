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
from math import floor
from math import ceil

import numpy as np

rng = np.random.default_rng(42)


def load_data(height=12, width=12, line_len=5, train_size=0.75, size=5000):
    """
    Dummy synthetic "lines" dataset...

    Args:
        height:
        width:
        line_len:
        train_size:
        size:

    Returns:

    """
    assert line_len <= min(height, width)
    # balanced set 4 classes:
    #   0: no line
    #   1: horizontal line
    #   2: vertical line
    #   3: diagonal line
    n_classes = 4
    per_class, remainder = divmod(size, n_classes)
    x = []
    y = []
    for c in range(n_classes):
        has_remainder = bool(remainder)
        size_class = per_class + int(has_remainder)
        if has_remainder:
            remainder -= 1
        # Single grayscale channel
        x_class = np.zeros((size_class, height, width, 1), dtype=np.float32)
        # Add lines of the appropriate class
        if c != 0:  # no lines needed for class no line
            padf = (line_len - 1) / 2
            pad1, pad2 = floor(padf), ceil(padf)

            for x_i in x_class:
                if c == 1:  # horizontal
                    i = rng.choice(range(height))
                    j = rng.choice(range(pad1, width - pad2))
                    x_i[i, j - pad1:j + pad2 + 1] = 1
                elif c == 2:  # vertical
                    i = rng.choice(range(pad1, height - pad2))
                    j = rng.choice(range(width))
                    x_i[i - pad1:i + pad2 + 1, j] = 1
                elif c == 3:  # diagonal
                    i = rng.choice(range(pad1, height - pad2))
                    j = rng.choice(range(pad1, width - pad2))
                    i_idx = range(i - pad1, i + pad2 + 1)
                    if rng.choice(2):  # select which diagonal
                        i_idx = [*reversed(i_idx)]
                    j_idx = range(j - pad1, j + pad2 + 1)
                    x_i[i_idx, j_idx] = 1
                else:
                    raise NotImplementedError(c)
        x.append(x_class)
        y_c = [0] * n_classes
        y_c[c] = 1
        y.extend([y_c] * size_class)

    x = np.concatenate(x, axis=0)
    y = np.asarray(y)

    sep_index = int(train_size * size)
    X_train, y_train = x[:sep_index], y[:sep_index]
    X_val, y_val = x[sep_index:], y[sep_index:]

    return (X_train, y_train), (X_val, y_val)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    (X, Y), _ = load_data(
        height=12, width=12, line_len=5, train_size=1., size=4)
    f, axes = plt.subplots(2, 2)
    for c_, (x_, ax) in enumerate(zip(X, axes.flat)):
        ax.imshow(x_, cmap='gray')
        ax.set_title(str(c_))
    plt.show()
