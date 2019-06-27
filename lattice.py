# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:31:41 2019

@author: Jorn
"""


import math as m
import scipy.special as sp
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

plt.close("all")


def spherical_hn1(n,z):

    return sp.spherical_jn(n,z)+1j*sp.spherical_yn(n,z)

##ambient
rho = 1220
Nf = 1000
f = np.linspace(5.0e4,1.0e6,Nf)



KKTp = np.zeros((1,Nf))
KKLp = np.zeros((1,Nf))
KKTi = np.zeros((1,Nf))
KKLi = np.zeros((1,Nf))
R_omega = np.zeros((1,Nf),dtype=complex)
T_omega = np.zeros((1,Nf),dtype=complex)
omegastore = np.zeros((1,Nf))
Omega = np.zeros((1,Nf))
Omega_T = np.zeros((1,Nf))
S_1s_store = np.zeros((1,Nf))
S_1s_phase_store = np.zeros((1,Nf))
S_1p_store = np.zeros((1,Nf))
S_1p_phase_store = np.zeros((1,Nf))
S_0p_store = np.zeros((1,Nf))
S_0p_phase_store = np.zeros((1,Nf))

R_0 = 5.85*10**-4  #"m"

U_N = np.zeros((1,Nf),dtype=complex)
c_L_i = 2490.0
c_T_i = 1180.0
lamda_p = 55.0e9#Pa Al
rho_p = 7800.0 #kg/m^3
#mhu_p = 25.0e9 #Pa
rho_ratio = rho/rho_p
#rho_ratio = 0.16
#mhu_ratio = mhu_p/mhu
c_L_p = 5940.0
c_T_p = 3200.0
R_0_dratio = 0.15
d = R_0/R_0_dratio

for iterator in range(0,Nf):
    if iterator%100==0:
        print(iterator)
    omega = 2*m.pi*f[iterator]
    omegastore[:,iterator] = omega

    
    alphaL_f = 17/800000*f[iterator]
    alphaS_f = 35/800000*f[iterator]

    
    k_L_i = omega/c_L_i + 1j*alphaL_f
    k_T_i = omega/c_T_i + 1j*alphaS_f
    K_L = k_L_i * R_0
    K_T = k_T_i * R_0

    S_0p = 0.0
    S_1p = 0.0
    S_1s = 0.0
    integerII = np.linspace(-50,50,101)
    for integerI in range(-50,50):
        
        if integerI==0:
            integerII = np.linspace(-50,51,102)
            integerII = integerII[integerII !=0]
          
        S_0p = np.sum(np.exp(1j*k_L_i*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*d*k_L_i*np.sqrt(integerI**2 + (integerII)**2)))
        S_1p = np.sum((np.exp(1j*k_L_i*d*np.sqrt(integerI**2 +(integerII)**2))/((k_L_i*d)**2*(integerI**2 +(integerII)**2)))*(1-1/(1j*k_L_i*d*np.sqrt(integerI**2 + (integerII)**2))))
        S_1s = np.sum(np.exp(1j*k_T_i*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*k_T_i*d*np.sqrt(integerI**2 + (integerII)**2))*(1-1/(1j*k_T_i*d*np.sqrt(integerI**2 + (integerII)**2))-1/((k_T_i*d)**2*(integerI**2 + (integerII)**2))))

       
    Omega[:,iterator] = abs(k_T_i*d/(m.pi*2))
    Omega_T[:,iterator] = abs(k_L_i*R_0)
    
    S_1s_store[:,iterator] = abs(S_1s)
    S_1s_phase_store[:,iterator] = cm.phase(S_1s)
    S_1p_store[:,iterator] = abs(S_1p)
    S_1p_phase_store[:,iterator] = cm.phase(S_1p)
    S_0p_store[:,iterator] = abs(S_0p)
    S_0p_phase_store[:,iterator] = cm.phase(S_0p)    


integerII = np.linspace(-Np,Np,2*Np+1)
        for integerI in range(-Np,Np):
            
            if integerI==0:
                integerII = np.linspace(-Np,Np+1,2*Np+2)
                integerII = integerII[integerII !=0]
              
            S_0p += np.sum(np.exp(1j*k_L_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(2*d*k_L_i*np.sqrt(integerI**2 + (integerII)**2)))
            S_1p += np.sum(np.exp(1j*k_L_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(2*d*k_L_i*np.sqrt(integerI**2 + (integerII)**2))*(1/3 + 1j*4/(k_L_i*2*d*np.sqrt(integerI**2 + (integerII)**2)) - 4/((k_L_i*2*d)**2*(integerI**2 + (integerII)**2))))
            S_1s += np.sum(np.exp(1j*k_T_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(2*d*k_T_i*np.sqrt(integerI**2 + (integerII)**2))*(1/3 + 1j*4/(k_T_i*2*d*np.sqrt(integerI**2 + (integerII)**2)) - 4/((k_T_i*2*d)**2*(integerI**2 + (integerII)**2))))
            
            
        S_0p = -1j*S_0p
        S_1p = 1j*S_1p
        S_1s = 1j*S_1s
    
plt.style.use('ggplot')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)   
    

plt.figure(2)
plt.subplot(211)
plt.suptitle("Lattice sum S_1s due to Longitudinal wave", fontsize=40)
plt.plot(Omega[0,:], abs(S_1s_store[0,:]), linewidth=3.0)
plt.axvline(x=1.0)
plt.axvline(x=m.sqrt(2))
plt.axvline(x=2.0)
plt.axvline(x=m.sqrt(5))
plt.axvline(x=m.sqrt(8))
plt.subplot(212)
plt.plot(Omega[0,:], S_1s_phase_store[0,:], linewidth=3.0)


plt.figure(4)
plt.subplot(211)
plt.suptitle("Lattice sum S_0p due to Longitudinal wave", fontsize=40)
plt.plot(Omega[0,:], abs(S_0p_store[0,:]), linewidth=3.0)
plt.axvline(x=1.0)
plt.axvline(x=m.sqrt(2))
plt.axvline(x=2.0)
plt.axvline(x=m.sqrt(5))
plt.axvline(x=m.sqrt(8))
plt.subplot(212)
plt.plot(Omega[0,:], S_0p_phase_store[0,:], linewidth=3.0)

plt.figure(5)
plt.subplot(211)
plt.suptitle("Lattice sum S_1p due to Longitudinal wave", fontsize=40)
plt.plot(Omega[0,:], abs(S_1p_store[0,:]), linewidth=3.0)
plt.axvline(x=1.0)
plt.axvline(x=m.sqrt(2))
plt.axvline(x=2.0)
plt.axvline(x=m.sqrt(5))
plt.axvline(x=m.sqrt(8))
plt.subplot(212)
plt.plot(Omega[0,:], S_1p_phase_store[0,:], linewidth=3.0)

