#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:50:43 2020

@author: charles
"""
import numpy as np
import matplotlib.pyplot as plt

#sigmoid
# return 1/(1+ np.exp(-x_lst))

x_lst = np.arange(-13,13,0.1)
y_lst = 1/(1+ np.exp(-x_lst))
plt.xlim((-10,10))
plt.xlabel("z")
plt.ylabel(r'$\sigma(z)$')
plt.title("sigmoid function")
plt.grid()
plt.plot(x_lst,y_lst,color="blue")

plt.savefig('Figure_1.png', dpi=300)
