# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:49:07 2019

@author: Jorn
"""

import math as m
import scipy.special as sp
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sympy import *

plt.close("all")
## Define spherical Hankel function
def spherical_hn1(n,z):

    return sp.spherical_jn(n,z,derivative=False)+1j*sp.spherical_yn(n,z,derivative=False)

Nf = 1000
r = 0.2
f = np.linspace(5.0e4,3.0e6,Nf)


#Numerical
#Hankel1 = spherical_hn1(1,z)
#DerivHankel1 = np.gradient(Hankel1, 0.0001 )
#Bessel0 = sp.spherical_jn(0,z)
#DerivBessel0 = np.gradient(Bessel0, 0.0001)
#Bessel1 = sp.spherical_jn(1,z)
#DerivBessel1 = np.gradient(Bessel1, 0.0001)
#Hankel0 = spherical_hn1(0,z)
#DerivHankel0 = np.gradient(Hankel0, 0.0001 )
#
##Symbolic (using wolframalpha)
#DH1n = 0.5*(spherical_hn1(0,z) - (spherical_hn1(1,z) + z*spherical_hn1(2,z))/z)
#DJ0 = -sp.spherical_jn(1,z)
#DJ1 = 0.5*(sp.spherical_jn(0,z) - (sp.spherical_jn(1,z) + z * sp.spherical_jn(2,z))/z)
#
#
#DH0 = np.zeros((1,Nf),dtype=complex)
#for i in range(1,Nf):
#    DH0[:,i] = cm.exp(1j*z[i])*(z[i]+1j)/(z[i]**2)
#


#plot figures to check
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
##ax1.plot(z, DerivFunction.real)
#ax1.plot(z, DH1n.real)
##ax2.plot(z,DerivFunction.imag)
#ax2.plot(z, DH1n.imag)
#ax3.plot(z, DH1n.real - DerivHankel1.real )
#ax3.plot(z, DH1n.imag - DerivHankel1.imag )
#
#
#
#fig, (ax4, ax5, ax6, ax7) = plt.subplots(1, 4, sharey=True)
##ax1.plot(z, DerivFunction.real)
#ax4.plot(z, DJ0.real)
##ax2.plot(z,DerivFunction.imag)
#ax5.plot(z, DJ0.imag)
#ax6.plot(z, DJ0.real - DerivBessel0.real )
#ax6.plot(z, DJ0.imag - DerivBessel0.imag )
#ax7.plot(z, abs(DJ0) - abs(DerivBessel0))
#
#fig, (ax8, ax9, ax10) = plt.subplots(1,3,sharey = True)
#ax8.plot(z,DJ1.real)
#ax9.plot(z,DJ1.imag)
#ax10.plot(z, abs(DJ1) - abs(DerivBessel1))


#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
#ax1.plot(z[1:], abs(DH0[0,1:]))
#ax2.plot(z, abs(DerivHankel0))
#ax3.plot(z[1:], abs(DH0[0,1:]) - abs(DerivHankel0[1:]))

# create potential function

A0 = 1
A1 = 1
theta = 0


def potentiallongitudinal(f,r,A0,A1,theta):
    k = f*r
    S_0p = 0.0
    S_1p = 0.0
    S_1s = 0.0
    d = 0.02
    integerII = np.linspace(-50,50,101)
    for integerI in range(-50,50):
        
        if integerI==0:
            integerII = np.linspace(-50,51,102)
            integerII = integerII[integerII !=0]
          
        S_0p += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*d*k*np.sqrt(integerI**2 + (integerII)**2)))
        S_1p += np.sum((np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/((k*d)**2*(integerI**2 +(integerII)**2)))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))))
        S_1s += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))-1/((k*d)**2*(integerI**2 + (integerII)**2))))

    result = sp.spherical_jn(0,k) + 3j*sp.spherical_jn(1,k)*m.cos(theta) + A0*spherical_hn1(0,k) + A1*spherical_hn1(1,k)*m.cos(theta) + A0*S_0p*sp.spherical_jn(0,k) + 3*A1*S_1p*sp.spherical_jn(1,k)*m.cos(theta)
    return result


def potentialtransverse(f,r,B1,theta):
    k = f*r
    S_0p = 0.0
    S_1p = 0.0
    S_1s = 0.0
    d = 0.02
    integerII = np.linspace(-50,50,101)
    for integerI in range(-50,50):
        
        if integerI==0:
            integerII = np.linspace(-50,51,102)
            integerII = integerII[integerII !=0]
          
        S_0p += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*d*k*np.sqrt(integerI**2 + (integerII)**2)))
        S_1p += np.sum((np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/((k*d)**2*(integerI**2 +(integerII)**2)))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))))
        S_1s += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))-1/((k*d)**2*(integerI**2 + (integerII)**2))))

    result = B1*spherical_hn1(1,k)*m.cos(theta) + 3/2*B1*S_1s*sp.spherical_jn(1,k)
    return result

derivativeself = np.zeros((1,Nf), dtype = complex)
Potential = np.zeros((1,Nf), dtype = complex)
for i in range(1,Nf):
    k = r*f[i]
    S_0p = 0.0
    S_1p = 0.0
    S_1s = 0.0
    d = 0.02
    integerII = np.linspace(-50,50,101)
    for integerI in range(-50,50):
        
        if integerI==0:
            integerII = np.linspace(-50,51,102)
            integerII = integerII[integerII !=0]
          
        S_0p += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*d*k*np.sqrt(integerI**2 + (integerII)**2)))
        S_1p += np.sum((np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/((k*d)**2*(integerI**2 +(integerII)**2)))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))))
        S_1s += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))-1/((k*d)**2*(integerI**2 + (integerII)**2))))

    derivativeself[:,i] = -sp.spherical_jn(1,k)*f[i] + 3j*0.5**f[i]*(sp.spherical_jn(0,k) - (sp.spherical_jn(1,k) + k * sp.spherical_jn(2,k))/k)*m.cos(theta) + A0*f[i]*cm.exp(1j*k)*(k+1j)/(k**2) + A1*0.5*f[i]*(spherical_hn1(0,k) - (spherical_hn1(1,k) + k*spherical_hn1(2,k))/k)*m.cos(theta) - A0*f[i]*S_0p*sp.spherical_jn(1,k) + 3*A1*S_1p*0.5*f[i]*(sp.spherical_jn(0,k) - (sp.spherical_jn(1,k) + k * sp.spherical_jn(2,k))/k) 
    Potential[:,i] = potentiallongitudinal(f[i],r,A0,A1,theta)
    
    
    
error  = abs(derivativeself) - abs(np.gradient(Potential[0,:], 0.01)  ) 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.plot(f,abs(derivativeself[0,:]))
ax2.plot(f, abs(np.gradient(Potential[0,:], 0.01)))
ax3.plot(f, error[0,:])

derivativeselftransverse = np.zeros((1,Nf), dtype = complex)
Potentialtransverse = np.zeros((1,Nf), dtype = complex)
B1 = 1
for i in range(1,Nf):
    k = f[i]*r
    S_0p = 0.0
    S_1p = 0.0
    S_1s = 0.0
    d = 0.02
    integerII = np.linspace(-50,50,101)
    for integerI in range(-50,50):
        
        if integerI==0:
            integerII = np.linspace(-50,51,102)
            integerII = integerII[integerII !=0]
          
        S_1s += np.sum(np.exp(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))*(1-1/(1j*k*d*np.sqrt(integerI**2 + (integerII)**2))-1/((k*d)**2*(integerI**2 + (integerII)**2))))

    derivativeselftransverse[:,i] =  B1*f[i]*0.5*(spherical_hn1(0,k) - (spherical_hn1(1,k) + k*spherical_hn1(2,k))/k)
    Potentialtransverse[:,i] = potentialtransverse(f[i],r,B1,theta)
    
    
    
    
