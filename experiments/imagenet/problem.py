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
from deephyper.problem import NaProblem

from experiments.imagenet.load_data import load_data
from experiments.imagenet.search_space import create_search_space

Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    num_epochs=200,
    batch_size=512, learning_rate=0.1, optimizer='sgd', momentum=0.9,
    weight_decay=0.0005, nesterov=True,
)

Problem.loss('categorical_crossentropy')

# top_k_categorical_accuracy: k=5 by default
Problem.metrics(['acc', 'top_k_categorical_accuracy'])

Problem.objective('val_acc__last')

if __name__ == '__main__':
    print(Problem)
