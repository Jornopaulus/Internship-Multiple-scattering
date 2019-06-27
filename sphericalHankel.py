# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:22:24 2019

@author: Jorn
"""
import math as m
import scipy.special as sp
import numpy as np
import cmath as cm


def SPhank1(mt,z):
    h = cm.exp(-1j*z)/(z**(mt+1))
    q = 0.0
    for n in range(0,mt):
        q = q+1j**(mt-n+1)*(m.factorial(mt+n)/(m.factorial(n)*m.factorial(mt-n)*2**n))*z**(mt-n)
    return h*q


