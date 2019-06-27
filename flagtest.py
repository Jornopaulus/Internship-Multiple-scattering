# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:57:01 2019

@author: Jorn
"""


import numpy as np
from scipy.optimize import newton
rho_ratio = 0.16


def func(KT):
    return 1j*KT**2*(2*rho_ratio + 1) - 9*KT- 1j*9 

def derfunc(KT):
    return 2j*KT*(2*rho_ratio + 1) - 9

F = 100
KT = 10000 + 40j
while abs(F) >0.001:
    F = func(KT)
    DF = derfunc(KT)
    KT = KT - F/DF
    print(abs(KT))
print(abs(KT))




plt.figure(2)
for t in range(1,Nn):
    plt.subplot(311)    
    plt.plot(f,abs(T1_LLstore[t,:]),label=str(t))
    plt.legend()
    plt.subplot(312)   
    plt.plot(f,abs(T0_LLstore[t,:]),label = str(t))
    plt.legend()
    plt.subplot(313)
    plt.plot(f,abs(R[t,:]),label = str(t))
    plt.legend()

plt.figure(4)
plt.subplot(211)
plt.plot(f,abs(A_00[1,:]),label="zero")
plt.legend()
plt.subplot(212)
plt.plot(f,abs(A_10[1,:]),label = "zero")
plt.legend()