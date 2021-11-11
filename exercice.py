#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
from typing import Generator
import numpy as np
from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from scipy.integrate import quad

# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    
    
    return np.array([np.linspace(-1.3, 2.5, num=64)])


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    vector_list = list()

    
    for vector in cartesian_coordinates:
        rho = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        phi = np.arctan2(vector[0], vector[1])
        vector_list.append((rho, phi))

    return np.array([vector_list])


def find_closest_index(values: np.ndarray, number: float) -> int:
    
   
    
    
    print(values)

    index = (np.abs(values - number )).argmin()

    return index

def graphic()-> None:

    x = np.linspace(-1, 1, 250)

    y = x ** 2 * np.sin(1 / (x ** 2)) + x

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x,y, 'g')

    # show the plot
    plt.show()


def monte_carlo(num_point : int)-> None:
    
    x = np.random.rand(num_point)
    y = np.random.rand(num_point)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')


    red = 0
    blue = 0
    for i in range(num_point):
        if x[i] ** 2 + y[i] ** 2 < 1:
            plt.plot(x[i],y[i], 'ro', color="b")
            blue+=1
        else:
            plt.plot(x[i],y[i], 'ro', color="r")
            red+=1

    # show the plot
    plt.show()
    return 4 * (blue/num_point)
    

def fct(x):
    return np.exp(-x ** 2)

def graph_inte():


    x = np.linspace(-4, 4, 200)

    plt.plot(x, fct(x), "r")

    plt.show()


    

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())
    print(coordinate_conversion(cartesian_coordinates = np.random.rand(5, 2)))
    print(find_closest_index(values= np.random.rand(1, 10), number = 5))
    graphic()
    print(monte_carlo(50))
    Ih, err = quad(fct, -np.inf, np.inf)
    print("intégrale =" , Ih)
    graph_inte()
    print("ok")