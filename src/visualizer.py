# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:09:08 2015

@author: xapharius
"""
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(object):

    def __init__(self, true_f, x_points, y_points):
        """
        true_f: function of true unlderlying process
        x_points: the x-coords of the points in the dataset
        y_points: the y-coords of the points in the dataset
        """
        self.true_f = true_f
        self.data_x = x_points
        self.data_y = y_points
        self.domain_min = min(self.data_x)
        self.domain_max = max(self.data_x)
        self.domain = np.linspace(self.domain_min-2, self.domain_max+2, 1000)
        self.domain = self.domain[np.newaxis].T  # as row vec
        self.true_y = self.true_f(self.domain)
        self.errors = []
        self.iterations = []

        # model plot
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Function Approximation")
        self.ax1.scatter(self.data_x, self.data_y)
        self.ax1.plot(self.domain, self.true_y, color="g")
        self.model_line, = self.ax1.plot(self.domain, [0]*len(self.domain),
                                         color="b")

        # error plot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Error")
        self.err_line, = self.ax2.plot([0], [1], color="b")
        self.ax2.set_yscale("log")

    def plot(self, model, error):
        """
        model: model at a certain point in time
        error: error of last iteration
        """
        predictions = model.predict(self.domain)
        self.model_line.set_ydata(predictions)

        # ultra slow, but quite helpful
        self.errors.append(error)
        self.iterations.append(len(self.errors))
        self.err_line.set_xdata(self.iterations)
        self.err_line.set_ydata(self.errors)
        self.ax2.set_ylim([0, max(self.errors)])
        self.ax2.set_xlim([0, len(self.errors)])

        self.fig.canvas.draw()
        plt.pause(1e-10)
