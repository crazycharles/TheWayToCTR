#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:04:16 2020

@author: charles
"""

'''
f=(a*ctr1+N*ctr2)/(a+N)
'''
from matplotlib import pyplot as plt
import numpy as np

plt.subplot(1,2,1)
x = [i for i in np.arange(0,1,0.025)]
y = [x[i] for i in range(len(x))]
plt.ylim(0,1)
plt.plot(x,y)


plt.subplot(1,2,2)
x = [i for i in np.arange(0,1,0.025)]
y = [(0.5+2*x[i])/3 for i in range(len(x))]
plt.ylim(0,1)
plt.plot(x,y)
plt.show()