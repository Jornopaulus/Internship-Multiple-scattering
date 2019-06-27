# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:43:05 2019

@author: Jorn
"""

import math as m
import scipy.special as sp
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

plt.close("all")

## Define spherical Hankel function
def spherical_hn1(n,z):

    return sp.spherical_jn(n,z)+1j*sp.spherical_yn(n,z)

Nn = 5
Nf = 1000
R_omega = np.zeros((1,Nf),dtype=complex)
T_omega = np.zeros((1,Nf),dtype=complex)
Omega = np.zeros((1,Nf))
Omega_T = np.zeros((1,Nf))
R_omegastore = np.zeros((Nn,Nf),dtype=complex)
T_omegastore = np.zeros((Nn,Nf),dtype=complex)
A_00 = np.zeros((Nn,Nf),dtype=complex)
A_10 = np.zeros((Nn,Nf),dtype=complex)
V_f = np.zeros((Nn,Nf),dtype=complex)
R = np.zeros((Nn,Nf),dtype=complex)
R2 = np.zeros((Nn,Nf),dtype=complex)
T_self = np.zeros((Nn,Nf),dtype=complex)
R_self = np.zeros((Nn,Nf),dtype=complex)
T1_LLstore = np.zeros((Nn,Nf),dtype=complex)
T0_LLstore = np.zeros((Nn,Nf),dtype=complex)
KKLi = np.zeros((1,Nf),dtype=complex)
S_1s_store = np.zeros((Nn,Nf))
S_1s_phase_store = np.zeros((Nn,Nf))
S_1p_store = np.zeros((Nn,Nf))
S_1p_phase_store = np.zeros((Nn,Nf))
S_0p_store = np.zeros((Nn,Nf))
S_0p_phase_store = np.zeros((Nn,Nf))
D = np.zeros((Nn,Nf))
rho = 1100
Np = 50
f = np.linspace(1.0e5,3.0e6,Nf)
A0 = np.zeros((Nn,Nf),dtype=complex)
A0p = np.zeros((Nn,Nf),dtype=complex)
A1 = np.zeros((Nn,Nf),dtype=complex)
B1 = np.zeros((Nn,Nf),dtype=complex)
E1 = np.zeros((Nn,Nf),dtype=complex)
D1 = np.zeros((Nn,Nf),dtype=complex)

U_N = np.zeros((Nn,Nf),dtype=complex)
c_L_i = 2490.0
c_T_i = 1250.0


Flag = "Tungsten Carbine" # Steel, Lead, Glass


if Flag == "Tungsten Carbine":
    rho_p = 15250.0 #kg/m^3
    R_0 = 200*10**-6  #"m"
    mhu_p = 243.0e9
    lamda_p = 162.0e9
    
if Flag == "Lead":
    rho_p = 11310.0 #kg/m^3
    R_0 = 200*10**-6  #"m"
    mhu_p = 4.0e9
    lamda_p = 26.76923e9
    
if Flag == "Glass":
    rho_p = 2490.0 #kg/m^3
    R_0 = 5.6*10**-4  #"m"


rho_ratio = rho/rho_p



wavelength_res = (1/((3*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio+1))))*c_L_i/c_T_i

Beta = np.array([0.5 , 1 , 2 , 10 , 100])
for N in range(Nn):
    d = R_0*(1+Beta[N])
    print("Iteration" ,  N , "started")
    for iterator in range(0,Nf):
        #print("Iteration" , iterator , "started"
        # Create some constants for particle(sphere) and ambient
        omega = 2*m.pi*f[iterator]
        lame1 = 3.6e9 + 0.1e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.3e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) +0.25e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2) + 0.3e9*((omega*8e-9)**2 - 1j*omega*8e-9)/(1+(omega*8e-9)**2)
        lame2 = 1.05e9 + 0.2e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.15e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) - 0.1e9*((omega*14e-9)**2 - 1j*omega*14e-9)/(1+(omega*14e-9)**2) + 0.15e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2)
    
        c_L_i = cm.sqrt((lame1+2*lame2)/rho)
        c_T_i = cm.sqrt(lame2/rho)
        mhu = lame2
        lamda = lame1
        alphaL_f = 0.1 + 0.25/2000000*f[iterator]
        alphaS_f = 0.05 + 0.1/2000000*f[iterator]
    
        
        k_L_i = omega/c_L_i + 1j*alphaL_f
        k_T_i = omega/c_T_i + 1j*alphaS_f
        K_L = k_L_i * R_0
        K_T = k_T_i * R_0
        
        
        c_L_p = m.sqrt((lamda_p + 2*mhu_p)/rho_p)
        c_T_p = m.sqrt(mhu_p/rho_p)
        k_L_p = omega/c_L_p
        k_T_p = omega/c_T_p
        K_L_p = k_L_p * R_0
        K_T_p = k_T_p * R_0
        
        for n in range(2):
            hnL_n = spherical_hn1(n,K_L)
            hnT_n = spherical_hn1(n, K_T)
            hnL_1n = spherical_hn1((n+1),K_L)
            hnT_1n = spherical_hn1((n+1), K_T)
            
            jnL_n = sp.spherical_jn(n,K_L)
            jnT_n = sp.spherical_jn(n,K_T)
            jnL_np = sp.spherical_jn(n,K_L_p) 
            jnT_np = sp.spherical_jn(n,K_T_p) 
            
            jnL_1n = sp.spherical_jn((n+1),K_L)
            jnT_1n = sp.spherical_jn((n+1),K_T)
            jnL_1np = sp.spherical_jn((n+1),K_L_p) 
            jnT_1np = sp.spherical_jn((n+1),K_T_p) 
            
            U_L_p = n*jnL_np - K_L_p*jnL_1np
            U_T_p = n*(n+1)*jnT_np
           
            
            V_L_p = jnL_np
            V_T_p = (1+n)*jnT_np - K_T*jnT_1np  
            
            W_S_p = 1j*(K_T_p*jnT_np)
    
            SS_L_p = (2*n*(n-1)*mhu_p - (lamda_p + 2*mhu_p)*K_L_p**2)*jnL_np + 4*mhu_p*K_L_p*jnL_1np
            SS_T_p = 2*n*(n + 1)*mhu_p*((n-1)*jnT_np - K_T_p*jnT_1np )
           
            Tau_L_p = 2*mhu_p*((n-1)*jnL_np - K_L_p*jnL_1np)
            Tau_T_p = mhu_p*((2*n**2-2 - K_T_p**2)*jnT_np + 2*K_T_p*jnT_1np)
           
            Sig_S_p = 1j*mhu_p*K_T_p*((n-1)*jnT_np- K_T_p*jnT_1np)
            
            
            dj1dr_KL = k_L_i*((1/(k_L_i*R_0)*sp.spherical_jn(1,k_L_i*R_0) - sp.spherical_jn(2,k_L_i*R_0)))
            
            dj1dr_KT = k_T_i*((1/(k_T_i*R_0)*sp.spherical_jn(1,k_T_i*R_0) - sp.spherical_jn(2,k_T_i*R_0)))
            
            dh1dr_KL = k_L_i*((1/(k_L_i*R_0)*spherical_hn1(1,k_L_i*R_0) - spherical_hn1(2,k_L_i*R_0)))
            
            dh1dr_KT = k_T_i*((1/(k_T_i*R_0)*spherical_hn1(1,k_T_i*R_0) - spherical_hn1(2,k_T_i*R_0)))
            
            
            S_0p = 0.0
            S_1p = 0.0
            S_1s = 0.0
            integerII = np.linspace(-50,50,101)
            for integerI in range(-50,50):
                
                if integerI==0:
                    integerII = np.linspace(-50,51,102)
                    integerII = integerII[integerII !=0]
                  
                S_0p += np.sum(np.exp(1j*k_L_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*2*d*k_L_i*np.sqrt(integerI**2 + (integerII)**2)))
                S_1p += np.sum((np.exp(1j*k_L_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/((k_L_i*2*d)**2*(integerI**2 +(integerII)**2)))*(1-1/(1j*k_L_i*2*d*np.sqrt(integerI**2 + (integerII)**2))))
                S_1s += np.sum(np.exp(1j*k_T_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(1j*k_T_i*2*d*np.sqrt(integerI**2 + (integerII)**2))*(1-1/(1j*k_T_i*2*d*np.sqrt(integerI**2 + (integerII)**2))-1/((k_T_i*2*d)**2*(integerI**2 + (integerII)**2))))
    
           
            S_1p = -S_1p
            
            g_Ls = hnL_n + 3*S_1p*jnL_n
            g_Li = 3j*jnL_n
            dg_Ls = dh1dr_KL + 3*S_1p*dj1dr_KL
            dg_Li = 3j*dj1dr_KL
            
            g_T = hnT_n + 3/2*S_1s*jnT_n
            dg_T = dh1dr_KT + 3/2*S_1s*dj1dr_KT
            
            f0s = hnL_n + S_0p*jnL_n
            df0s = -hnL_1n + S_0p*dj1dr_KL
            f0i = jnL_n
            df0i = -jnL_1n
            
            SS_L = -(k_T_i**2)/2*g_Ls - 2/R_0*dg_Ls + (2/R_0**2)*g_Ls
            SS_Li = -(k_T_i**2)/2*g_Li - 2/R_0*dg_Li + (2/R_0**2)*g_Li
            SS_T = -2/R_0*dg_T + 2/R_0*g_T
            
            Tau_L = -2/R_0*dg_Ls + (1/R_0**2)*g_Ls
            Tau_Li = -2/R_0*dg_Li + (1/R_0**2)*g_Li
            Tau_T =  -(k_T_i**2)*g_T - 2/R_0*dg_T + (2/R_0**2)*g_T
            if n==1:
                DM = np.zeros((4,4), dtype = complex)
                
                DM[0,0] = +dg_Ls
                DM[0,1] = -2*g_T/R_0
                DM[0,2] = -U_L_p/R_0
                DM[0,3] = -U_T_p/R_0
                
                DM[1,0] = -g_Ls/R_0
                DM[1,1] = +g_T/R_0 + dg_T
                DM[1,2] = V_L_p/R_0#- put minus here to get "nicer graph"
                DM[1,3] = V_T_p/R_0#- put minus here to get "nicer graph" (not sure if its correct but resuts are more compliant with kinra)
                
                DM[2,0] = 2*mhu*SS_L
                DM[2,1] = 2*mhu*SS_T
                DM[2,2] = -SS_L_p/R_0**2
                DM[2,3] = -SS_T_p/R_0**2
                
                DM[3,0] = +2*mhu*Tau_L
                DM[3,1] = +2*mhu*Tau_T 
                DM[3,2] = Tau_L_p/R_0**2#
                DM[3,3] = Tau_T_p/R_0**2#
                
                P = np.zeros((4,1), dtype = complex)
                P[0,0] = -dg_Li
                P[1,0] = +g_Li/R_0
                P[2,0] = -2*mhu*SS_Li
                P[3,0] = -2*mhu*Tau_Li
                
                solution = np.linalg.solve(DM,P)
                
                A1[N,iterator] = solution[0]
                B1[N,iterator] = solution[1]
                E1[N,iterator] = solution[2]
                D1[N,iterator] = solution[3]
                
            elif n==0:
                DM = np.zeros((2,2), dtype = complex)
                DM[0,0] = df0s
                DM[0,1] = -U_L_p/R_0
                DM[1,0] = (-0.5*k_T_i**2*f0s -2/R_0*df0s)*2*mhu
                DM[1,1] = -SS_L_p/R_0**2
                
                P = np.zeros((2,1), dtype = complex)
                P[0,0] = -df0i
                P[1,0] = -2*mhu*(-0.5*k_T_i**2*f0i -  2/R_0*df0i)
                solution = np.linalg.solve(DM,P)
                
                A0[N,iterator] = solution[0]
                A0p[N,iterator] = solution[1]
                
        R_self[N,iterator] = 2*m.pi*(A0[N,iterator] + 1j*A1[N,iterator])/(k_L_i**2*4*d**2)
        T_self[N,iterator] = 1 + 2*m.pi*(A0[N,iterator] - 1j*A1[N,iterator])/(k_L_i**2*4*d**2)
    f_res = (3*c_T_i*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio+1))

plt.style.use('bmh')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


plt.figure(1)
for ii in range(Nn):
    plt.suptitle("$A_1$ Elastic sphere , $d = R(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(A1[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("A1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    

plt.figure(2)
for ii in range(Nn):
    plt.suptitle("$B_1$ Elastic sphere, $d = R(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(B1[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("B1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    
plt.figure(3)
for ii in range(Nn):
    plt.suptitle("$E_1$ Elastic sphere, $d = R(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(E1[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("E1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    

plt.figure(4)
for ii in range(Nn):
    plt.suptitle("$D_1$ Elastic sphere, $d = R(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(D1[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("D1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))

plt.figure(5)
for ii in range(Nn):
    plt.suptitle("$A_0$ Elastic sphere, $d = R(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(A0[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("$A_0$", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))

plt.figure(6)
for ii in range(Nn):
    plt.suptitle("$A_{0p}$ Elastic sphere, $d = R(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(A0p[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("$A_{0p}$", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))

plt.figure(7)
for ii in range(Nn):
    #plt.suptitle("Transmission and reflection coefficients due to Longitudinal wave for " + Flag +" radius of particle = 0.2mm", fontsize=50)
    plt.subplot(211)
    plt.suptitle("Transmission and reflection for elastic spheres, "+ Flag +" , $d = R(1+\\beta)$" , fontsize=40)
    plt.plot(f/f_res,abs(T_self[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.tick_params(labelsize=40)
    #plt.xlim(0,2.1)
    plt.ylabel("T($\omega$)", fontsize=60)
    plt.xlabel("$f/f_{res}$", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
    plt.subplot(212)
    plt.plot( f/f_res ,abs(R_self[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.tick_params(labelsize=40)
    #plt.xlim(0,2.4)
    plt.xlabel("$f/f_{res}$", fontsize=60)
    plt.ylabel("R($\omega$)", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
plt.savefig("Reflec_transmis_Elastic_Spheres"+ Flag +".jpg")

