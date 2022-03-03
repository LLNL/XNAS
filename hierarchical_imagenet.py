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
from queue import Queue

import pandas as pd

from nltk.corpus import wordnet as wn

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import matplotlib.pyplot as plt

IMAGENET_16_120 = False  # for the 16-120 labels only
DEBUG = 0

with open('imagenet_resized.labels', 'r') as f:
    labels = f.read().splitlines()

if IMAGENET_16_120:
    labels = labels[:120]
if DEBUG:
    labels = labels[:15]


def hyp(s):
    return s.hypernyms()


edges = []
leaves = []

print('put together graph')
for label in labels:
    pos = label[0]
    offset = int(label[1:])

    s = wn.synset_from_pos_and_offset(pos, offset)
    tree = s.tree(hyp)

    leaves.append(tree[0])

    q = Queue()
    q.put(tree)
    while not q.empty():
        line = q.get()
        child, parents = line[0], line[1:]
        for p in parents:
            edges.append((p[0].name(), child.name()))
            q.put(p)

g = nx.DiGraph(edges)

leaf_names = {leaf.name() for leaf in leaves}
g_leaves = set(n for n in g.nodes() if g.out_degree(n) == 0)
print('This should be empty (non-leaf labels):',
      g_leaves ^ leaf_names)

# stats: shortest distances between labels
g_hash = nx.weisfeiler_lehman_graph_hash(g)
prefix = 'imagenet_resized'
if IMAGENET_16_120:
    prefix += '_120'
dist_filename = f'{prefix}_label_distances_{g_hash}.csv'

if os.path.isfile(dist_filename):
    print('read cached df')
    dist_df = pd.read_csv(dist_filename)
else:
    print('measure similarities')
    distances = []
    g_undirected = g.to_undirected()
    idx_is = []  # always less than j
    idx_js = []
    for i, leaf_i in enumerate(leaves):
        for leaf_j in leaves[i + 1:]:
            length = nx.shortest_path_length(g_undirected, leaf_i.name(),
                                             leaf_j.name())
            distances.append({
                'label_i': leaf_i.name(),
                'label_j': leaf_j.name(),
                'nx_distance': length,
                'path_sim': leaf_i.path_similarity(leaf_j),
                'lch_sim': leaf_i.lch_similarity(leaf_j),
                'wup_sim': leaf_i.wup_similarity(leaf_j),
            })
            idx_is.append(labels.index(
                leaf_i.pos() + f'{leaf_i.offset():08}'))
            idx_js.append(labels.index(
                leaf_j.pos() + f'{leaf_j.offset():08}'))

    mi = pd.MultiIndex.from_arrays([idx_is, idx_js], names=['idx_i', 'idx_j'])
    dist_df = pd.DataFrame(distances, index=mi)
    dist_df.to_csv(dist_filename, index=True)

print('sort')
dist_df.sort_values(by='nx_distance', ignore_index=True, inplace=True)
print('ok printing')
print(dist_df.head(25).to_string(index=False))
print('...')
print(dist_df.tail(25).to_string(index=False))
print('Pearson Correlation')
print(dist_df.corr().to_string())
print('Spearman Correlation')
print(dist_df.corr('spearman').to_string())

pos = graphviz_layout(g, prog='twopi')

non_leaf = {*g.nodes()} - leaf_names
nx.draw_networkx_nodes(g, pos, nodelist=leaf_names, node_shape='^',
                       node_size=25, node_color='g')
nx.draw_networkx_nodes(g, pos, nodelist=non_leaf, node_shape='o',
                       node_size=25, node_color='b')
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, verticalalignment='center',
                        horizontalalignment='right',
                        font_size=8)

plt.show()
