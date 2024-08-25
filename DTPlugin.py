#!/usr/bin/env python
# coding: utf-8

# # Decision Tree
# 
# Binary tree, minimize error in each leaf
# 
# ## Lectures
# - https://www.youtube.com/watch?v=p17C9q2M00Q&list=PLD0F06AA0D2E8FFBA&index=7 (ML 2.1 - ML 2.8) mathematicalmonk

# ## Topics
# - CART (Classification Trees), Breiman
# - Regression Tree
# - Growing a tree ("Greedy")
# - Bootsrap aggregation (Bagging)
# - Random forest
# - AdaBoost

# ## CART (Classification Trees), Breiman

# Based on http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html Implementing Decision Trees in Python. It is ID3/C4.5, isn't CART algorithm

# In[1]:


import numpy as np
from pprint import pprint
from sklearn.model_selection import cross_val_score, train_test_split

import PyPluMA
import PyIO

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y


    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res['x_%d = %d' % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, 'rt')
    lines = reader(file)
    # convert str -> float
    dataset = [list(map(float, row)) for row in lines]
    return dataset

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))

class MyDecisionTree:
    def __init__(self, depth=1, min_size=1):
        self.depth = depth
        self.min_size = min_size
        self.tree = None
        
    def fit(self, X, y, **kwargs):
        self.tree = build_tree(np.column_stack((X, y)), self.depth, self.min_size)
        
    # Make a prediction with a decision tree
    def predict(self, row, node = None):
        if node is None:
            node = self.tree
            
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(row, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(row, node['right'])
            else:
                return node['right']
    
    def get_params(self, deep = False):
        return {'depth':self.depth}
    
def score_decision_tree(estimator, X, y):
    return sum([estimator.predict(x_row) == y_row for x_row, y_row in zip(X, y)])/len(X)

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# x1 is weather type (0 = partly cloudy, 1 = cloudy, 2 = sunny)
#x1 = [0, 1, 1, 2, 2, 2]
# x2 is atmospheric pressure (0 = low, 1 = high)
#x2 = [0, 0, 1, 1, 1, 0]
# y is rain 1 or not rain 0
#y = np.array([0, 0, 0, 1, 1, 0])
#X = np.array([x1, x2]).T
#pprint(recursive_split(X, y))
# Base on http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ How To Implement The Decision Tree Algorithm From Scratch In Python (Classification and regression tree algorithm - CART).
# 
# Used dataset from here http://archive.ics.uci.edu/ml/datasets/banknote+authentication
# ### Get Dataset
# In[2]:
# ### Train Model
# In[3]:

class DTPlugin:
 def input(self, inputfile):
   self.inputfile = inputfile
 def run(self):
     pass
 def output(self, outputfile):
  # load and prepare data
  filename = self.inputfile#'./dataset/data_banknote_authentication.csv'
  dataset = load_csv(filename)
  dataset_array = np.array(dataset)
  dataset_train, dataset_test = train_test_split(dataset_array, test_size=0.33, random_state=42)

  # train model
  dataset_train_x = dataset_train[:,0:-1]
  dataset_train_y = dataset_train[:,-1]

  # evaluate
  cv_scores = []
  cv_scores_std = []


  for depth in list(range(1, 4)):
    print('training depth {}'.format(depth))
    tree_estimator = MyDecisionTree(depth=depth, min_size=1)
    scores = cross_val_score(tree_estimator, 
                             X=dataset_train_x, y=dataset_train_y, 
                             cv=5, scoring=score_decision_tree)
    print('mean {}'.format(scores.mean()))
    cv_scores.append(scores.mean())
    print('std {}'.format(scores.std()))
    cv_scores_std.append(scores.std())


  # ### Make prediction and estimate accuracy
  # - **TODO**: show dynamic of accuracy based on parameters
  # - **TODO**: k-fold cross validation

  # In[5]:


  dataset = [[2.771244718, 1.784783929, 0],
            [1.728571309, 1.169761413, 0],
            [3.678319846, 2.81281357, 0],
            [3.961043357, 2.61995032, 0],
            [2.999208922, 2.209014212, 0],
            [7.497545867, 3.162953546, 1],
            [9.00220326, 3.339047188, 1],
            [7.444542326, 0.476683375, 1],
            [10.12493903, 3.234550982, 1],
            [6.642287351, 3.319983761, 1]]
  tree = build_tree(dataset, 5, 1)
  #print_tree(tree)
  # #### Train model on banknote auth 
  # In[4]:
        
  accuracy = 0
  for row in dataset_test:
    prediction = predict(tree, row)
    if row[-1] == prediction:
        accuracy += 1
    
  accuracy = accuracy/len(dataset_test)
    
  print('Accuracy: {}'.format(accuracy))


  # ### Visualize data set

  # In[6]:


  import seaborn as sns
  import matplotlib
  # use color map, otherwise it will be grayscale
  from matplotlib import cm
  import matplotlib.pyplot as plt
  # can choose different styles
  # print(plt.style.available)
  plt.style.use('fivethirtyeight')
  # list available fonts: [f.name for f in matplotlib.font_manager.fontManager.ttflist]
  matplotlib.rc('font', family='DejaVu Sans') 


  # draw dataset

  np_dataset = np.array(dataset)
  y = np_dataset[:, -1]
  # actually we could use label_to_idx=y, because we have label as number here 
  label_to_idx = [list(set(y)).index(y_value) for y_value in y]
  plt.scatter(x=np_dataset[:, 0], y=np_dataset[:,1], c=label_to_idx, cmap=cm.jet)
  plt.savefig(outputfile)
  #plt.show()


  # ## Random forest
  # base on: http://machinelearningmastery.com/implement-random-forest-scratch-python/

# 
