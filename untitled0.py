# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:10:22 2019

@author: Jorn
"""

import math as m

Re = 6378.136
H = 0.1

def Func(lamda):
    q = m.tan(7.5*m.pi/180) - (Re/(Re+H)*m.sin(lamda))/(1-(Re/(Re+H))*m.cos(lamda))
    return q

def DerFunc(lamda):
    return m.tan(7.5*m.pi/180)*(Re/(Re+H)*m.sin(lamda)) - Re/(Re+H)*m.cos(lamda)


Q = 100
lamda = 0
while abs(Q)>0.000000001:
    Q = Func(lamda)
    DQ = DerFunc(lamda)
    lamda = lamda - Q/DQ
    print(Q)
print(lamda*180/m.pi)


rho_0 = m.acos(Re/(Re+H))*180/m.pi
print("rho_0 equals " + str(rho_0))
    
    


   