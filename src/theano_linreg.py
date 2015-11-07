# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:58:37 2015

@author: xapharius
"""
import time
import numpy as np
from numpy import random as rnd
from sklearn import preprocessing

from visualizer import Visualizer

import theano
from theano import tensor as T
floatX = theano.config.floatX


class linReg(object):
    '''
    Linear Regression using mini-batch stochastic gradient descent
    '''

    def __init__(self, model_complexity):
        self.model_complexity = model_complexity + 1  # bias

        # theano expression
        rnd_array = rnd.random(self.model_complexity)
        self.weights = theano.shared(rnd_array.astype(floatX))
        learning_rate = T.scalar("learing_rate")
        X = T.matrix("X")  # batch
        true_y = T.vector("y")
        pred_y = T.dot(X, self.weights)  # vector
        cost = ((pred_y - true_y)**2).mean() + (self.weights**2).sum()
        grads = T.grad(cost=cost, wrt=self.weights)
        updates = [[self.weights, self.weights - learning_rate * grads]]

        self.theano_train = theano.function(inputs=[X, true_y, learning_rate],
                                            outputs=cost,
                                            updates=updates)
        self.theano_predict = theano.function(inputs=[X], outputs=pred_y)

    def _make_features(self, data_points):
        dupl_cols = np.repeat(data_points, self.model_complexity, axis=1)
        x = np.power(dupl_cols, range(self.model_complexity))
        return x

    def train(self, data_points, targets, iterations=10000,
              learning_rate=1e-6, batch_size=100, vis=None):

        dataset_size = len(data_points)

        # prepare features
        dataset_inputs = self._make_features(data_points)

        for i in range(iterations):
            start = time.time()
            # get a batch for each iteration
            batch_indices = rnd.choice(dataset_size, batch_size)
            batch_inputs = dataset_inputs[batch_indices]
            batch_targets = dataset_targets[batch_indices]
            err = self.theano_train(batch_inputs, batch_targets.ravel(),
                                    learning_rate)
            if vis:
                vis.plot(self, err)

            dt = time.time() - start  # error plotting takes up most time
            print "Iteration: {}, Error: {:.2f} ({:.2f}s)".format(i, \
                np.asscalar(err), dt)

    def predict(self, data_points):
        dataset_inputs = self._make_features(data_points)
        return self.theano_predict(dataset_inputs)


# define generating process and dataset
def true_f(x): return x**3 - 10*x**2 + 12
def proc_f(x): return true_f(x) + rnd.normal(scale=15*max(abs(x)), size=(len(x), 1))
domain = (-10, 10)
dataset_size = 100

# create data
x = rnd.random((dataset_size, 1))
x = (domain[1] - domain[0])*x + domain[0]
dataset_targets = proc_f(x)
# TODO Data Normalization

vis = Visualizer(true_f, x, dataset_targets)

model = linReg(model_complexity=3)
model.train(x, dataset_targets, iterations=1000, learning_rate=1e-8,
            batch_size=20, vis=vis)
