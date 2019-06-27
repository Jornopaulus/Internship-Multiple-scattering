# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:46:18 2019

@author: Jorn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:31:56 2019

@author: Jorn
"""
import math as m
import scipy.special as sp
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
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
A1 = np.zeros((Nn,Nf),dtype=complex)
B1 = np.zeros((Nn,Nf),dtype=complex)

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

if Flag == "Glass":
    rho_p = 2490.0 #kg/m^3
    R_0 = 5.6*10**-4  #"m"


rho_ratio = rho/rho_p



wavelength_res = (1/((3*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio+1))))*c_L_i/c_T_i

Beta = np.array([0.5 , 1 , 2 , 10 , 100])
for N in range(Nn):
    d = R_0*(1+Beta[N])
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
        
        n  = 1
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
        
        #modeling with system of equations:
        dj1dr_KL = k_L_i*((1/(k_L_i*R_0)*sp.spherical_jn(1,k_L_i*R_0) - sp.spherical_jn(2,k_L_i*R_0)))
        
        j1_KL = sp.spherical_jn(1,k_L_i*R_0)
        
        dj1dr_KT = k_T_i*((1/(k_T_i*R_0)*sp.spherical_jn(1,k_T_i*R_0) - sp.spherical_jn(2,k_T_i*R_0)))
        
        j1_KT = sp.spherical_jn(1,k_T_i*R_0)
        
        dh1dr_KL = k_L_i*((1/(k_L_i*R_0)*spherical_hn1(1,k_L_i*R_0) - spherical_hn1(2,k_L_i*R_0)))
        
        h1_KL = spherical_hn1(1,k_L_i*R_0)
        
        dh1dr_KT = k_T_i*((1/(k_T_i*R_0)*spherical_hn1(1,k_T_i*R_0) - spherical_hn1(2,k_T_i*R_0)))
        
        h1_KT = spherical_hn1(1,k_T_i*R_0)
        
        g_Ls = h1_KL + 3*S_1p*j1_KL
        g_Li = 3j*j1_KL
        dg_Ls = dh1dr_KL + 3*S_1p*dj1dr_KL
        dg_Li = 3j*dj1dr_KL
        
        g_T = h1_KT + 3/2*S_1s*j1_KT
        dg_T = dh1dr_KT + 3/2*S_1s*dj1dr_KT
        
        SS_L = -(k_T_i**2)/2*g_Ls - 2/R_0*dg_Ls + (2/R_0**2)*g_Ls
        SS_Li = -(k_T_i**2)/2*g_Li - 2/R_0*dg_Li + (2/R_0**2)*g_Li
        SS_T = -2/R_0*dg_T + 2/(R_0**2)*g_T
        
        Tau_L = -2/R_0*dg_Ls + (1/R_0**2)*g_Ls
        Tau_Li = -2/R_0*dg_Li + (1/R_0**2)*g_Li
        Tau_T =  -(k_T_i**2)*g_T - 2/R_0*dg_T + (2/R_0**2)*g_T

        
        DM = np.zeros((2,2), dtype = complex)
        
        DM[0,0] = mhu/(R_0*rho_p*omega**2)*(-2*SS_L + 4*Tau_L) - dg_Ls
        
        DM[0,1] = mhu/(R_0*rho_p*omega**2)*(-2*SS_T + 4*Tau_T) + 2*g_T/R_0
        
        DM[1,0] = -mhu/(R_0*rho_p*omega**2)*(-2*SS_L + 4*Tau_L) + g_Ls/R_0
        
        DM[1,1] = -mhu/(R_0*rho_p*omega**2)*(-2*SS_T + 4*Tau_T) - g_T/R_0 - dg_T
        
        P = np.zeros((2,1), dtype = complex)
        
        P[0,0] = mhu/(R_0*rho_p*omega**2)*(+2*SS_Li - 4*Tau_Li) + dg_Li
        
        P[1,0] = -mhu/(R_0*rho_p*omega**2)*(+2*SS_Li - 4*Tau_Li) - g_Li/R_0

        solution = np.linalg.solve(DM,P)

        A1[N,iterator] = solution[0]
        
        B1[N,iterator] = solution[1]
        
        S_1s_store[N,iterator] = abs(S_1s)
        S_1s_phase_store[N,iterator] = cm.phase(S_1s)
        S_1p_store[N,iterator] = abs(S_1p)
        S_1p_phase_store[N,iterator] = cm.phase(S_1p)
        S_0p_store[N,iterator] = abs(S_0p)
        S_0p_phase_store[N,iterator] = cm.phase(S_0p)    
        
        T0_LL = -(sp.spherical_jn(1,K_L))/( spherical_hn1(1,K_L) + S_0p*sp.spherical_jn(1,K_L))
        
        A0[N,iterator] = -(sp.spherical_jn(1,K_L))/( spherical_hn1(1,K_L) + S_0p*sp.spherical_jn(1,K_L))
        
        T0_LLstore[N,iterator] = T0_LL
        E0 = K_L*(3*S_1p*sp.spherical_jn(0,K_L) + spherical_hn1(0,K_L))    
        E1 = 3*S_1p*sp.spherical_jn(1,K_L) + spherical_hn1(1,K_L)
        E10 = (3*S_1s*sp.spherical_jn(1,K_T) + 2*spherical_hn1(1,K_T))/(K_T*(3*S_1s*sp.spherical_jn(0,K_T) + 2*spherical_hn1(0,K_T)))
        
        T1_LL = -3j*((((9*rho_ratio*E10 - (rho_ratio + 2))*sp.spherical_jn(1,K_L) - ((2*rho_ratio + 1)*E10 - 1)*K_L*sp.spherical_jn(0,K_L))/((9*rho_ratio*E10 - (rho_ratio + 2))*E1 - ((2*rho_ratio + 1)*E10 - 1)*E0)))
        T1_LLstore[N,iterator] = T1_LL
        R_omega[:,iterator] = 2*m.pi*(T0_LL + 1j*T1_LL)/(k_L_i**2*4*d**2)
        T_omega[:,iterator] = 1 + 2*m.pi*(T0_LL - 1j*T1_LL)/(k_L_i**2*4*d**2)
        
        R_self[N,iterator] = 2*m.pi*(A0[N,iterator] + 1j*A1[N,iterator])/(k_L_i**2*4*d**2)
        T_self[N,iterator] = 1 + 2*m.pi*(A0[N,iterator] - 1j*A1[N,iterator])/(k_L_i**2*4*d**2)
        
        Omega[:,iterator] = abs(k_T_i*d/(m.pi*2))
        Omega_T[:,iterator] = abs(k_L_i*R_0)
        
        A_00[N,iterator] = -(1j*(K_L)**3)/3
        A_10[N,iterator] = (rho_ratio - 1)*(K_L**3)/(3*rho_ratio)
        V_f[N,iterator] = (4/3*m.pi*R_0**3)/d**2
        R[N,iterator] = -1j*k_L_i*V_f[N,iterator]*rho_p/(2*rho)
        R2[N,iterator] = 2*m.pi*(A_00[N,iterator] + 1j*A_10[N,iterator])/(k_L_i**2*d**2)
        U_N[N,iterator] = -1j*((rho_ratio/(K_L**2))*(3*(3*E10-1)/((9*rho_ratio*E10 - (rho_ratio+2))*E1 - ((2*rho_ratio+1)*E10-1)*E0)))
        
    f_res = (3*c_T_i*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio+1))
    KPA_res = R_0*f_res*2*m.pi/c_L_i
    
    R_omegastore[N,:] = R_omega
    T_omegastore[N,:] = T_omega

    
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
    #plt.suptitle("Transmission and reflection coefficients due to Longitudinal wave for " + Flag +" radius of particle = 0.2mm", fontsize=50)
    plt.subplot(211)
    plt.suptitle("Transmission and reflection for rigid spheres "+ Flag +" , $d = R*(1+\\beta)$" , fontsize=40)
    plt.plot(f/f_res,abs(T_omegastore[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.tick_params(labelsize=40)
    #plt.xlim(0,2.1)
    plt.ylabel("T($\omega$)", fontsize=60)
    plt.xlabel("$f/f_{res}$", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
    plt.subplot(212)
    plt.plot( f/f_res ,abs(R_omegastore[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.tick_params(labelsize=40)
    #plt.xlim(0,2.4)
    plt.xlabel("$f/f_{res}$", fontsize=60)
    plt.ylabel("R($\omega$)", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
plt.savefig("Reflec_transmis_collectiontransmission"+ Flag +".jpg")



 
plt.figure(2)
for q in range(1,Nn):
    plt.title("Rigid body translation", fontsize=40)
    plt.plot(f, abs(U_N[q,:]),label=str(2*q/10)+"$\lambda_{res}$", linewidth=3.0)
    plt.tick_params(labelsize=40)
    plt.xlabel("$\Omega$", fontsize=40)
    plt.legend()

if os.path.exists("./RigidBodyTrans.jpg"):
    os.remove("./RigidBodyTrans.jpg")
    
plt.figure(3)
for z in range(Nn):   
    plt.suptitle("Lattice sum S_1s due to Longitudinal wave, $d = R*(1+\\beta)$", fontsize=50)
    plt.plot(abs(f/f_res), abs(S_1s_store[z,:]), label=str(Beta[z])+ "$ = \\beta $", linewidth=3.0)
    plt.ylim(-0.5,3.5)
    plt.ylabel("Magnitude", fontsize=60)
    plt.xlabel("$f/f_{res}$", fontsize=60)
    plt.tick_params(labelsize=40)
    plt.legend( fontsize=40, loc='best')
plt.savefig("LatticeSums.jpg") 



plt.figure(4)
for z in range(1,Nn):
    plt.subplot(311)
    plt.suptitle("Reigleight limit", fontsize=40)
    plt.plot(Omega[0,:], abs(R[z,:]), linewidth=3.0)
    plt.ylabel("R($\omega$)", fontsize=40)
    plt.subplot(312)
    plt.plot(Omega[0,:], abs(A_00[1,:]), linewidth=3.0)
    plt.ylabel("A0_Reileight", fontsize=30)
    plt.subplot(313)
    plt.plot(Omega[0,:], abs(A_10[1,:]), linewidth=3.0)
    plt.ylabel("A1_Reileight", fontsize=30)
    
    
plt.figure(5)
for z in range(1,Nn):
    plt.subplot(311)
    plt.suptitle("A0 and A1", fontsize=40)
    plt.plot(Omega[0,:], abs(T0_LLstore[z,:]), linewidth=3.0)
    plt.ylabel("A0", fontsize=30)
    plt.subplot(312)
    plt.plot(Omega[0,:], abs(T1_LLstore[z,:]), linewidth=3.0)
    plt.ylabel("A1", fontsize=30)
    plt.subplot(313)
    plt.plot(Omega[0,:],abs(R2[z,:]))

R_min, R_max = -np.abs(R_omegastore).max(), np.abs(R_omegastore).max()

plt.figure(6)
for ii in range(1,2):
    plt.plot(f/f_res, ((abs(T_omegastore[ii,:]))**2 + (abs(R_omegastore[ii,:]))**2 ),label=str(1/4*ii)+"$\lambda_{res}$", linewidth=3.0)
    plt.ylabel("$E_{Loss}$", fontsize=60)
    plt.xlabel("$f/f_{res}$", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
    
plt.figure(7)
for ii in range(Nn):
   
    plt.suptitle("$A_n$ , $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(T0_LLstore[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("An", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    

plt.figure(8)
for ii in range(Nn):
    plt.suptitle("$A_1$ Kinra, $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs( T1_LLstore[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("A1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
#plt.figure(6)
#plt.subplot(211)
#plt.title("T($\omega$)", fontsize=40)
#plt.pcolormesh(abs(Omega_T), D/wavelength_res, abs(T_omegastore),cmap='RdBu', vmin=0, vmax=1)
#plt.label("d $\lambda_{Res}")
#plt.axvline(x=abs(KPA_res), linewidth=3.0)
#plt.colorbar()
#plt.subplot(212)
#plt.title("R($\omega$)", fontsize=40)
#plt.pcolormesh(abs(Omega_T), D/wavelength_res, abs(R_omegastore),cmap='RdBu', vmin=0, vmax=1)
#plt.axvline(x=abs(KPA_res), linewidth=3.0)
#plt.colorbar()
#plt.savefig("collormeshdetailed2.jpg") 
#
#
##
#plt.figure(5)
#plt.subplot(211)
#plt.suptitle("Lattice sum S_1p due to Longitudinal wave", fontsize=40)
#plt.plot(Omega[0,:], abs(S_1p_store[0,:]), linewidth=3.0)
#plt.axvline(x=1.0)
#plt.axvline(x=m.sqrt(2))
#plt.axvline(x=2.0)
#plt.axvline(x=m.sqrt(5))
#plt.axvline(x=m.sqrt(8))
#plt.subplot(212)
#plt.plot(Omega[0,:], S_1p_phase_store[0,:], linewidth=3.0)

plt.figure(9)
for ii in range(Nn):
    plt.suptitle("$A_1$ Self , $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(A1[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("A1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    

plt.figure(10)
for ii in range(Nn):
    plt.suptitle("$B_n$, $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(B1[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("Bn", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))

plt.figure(11)
for ii in range(Nn):
    #plt.suptitle("Transmission and reflection coefficients due to Longitudinal wave for " + Flag +" radius of particle = 0.2mm", fontsize=50)
    plt.subplot(211)
    plt.suptitle("Transmission and reflection for rigid spheres, $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(f/f_res,abs(T_self[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.tick_params(labelsize=40)
    #plt.xlim(0,2.1)
    plt.ylabel("T($\omega$)", fontsize=60)
    plt.xlabel("f/f_res", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
    plt.subplot(212)
    plt.plot( f/f_res ,abs(R_self[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.tick_params(labelsize=40)
    #plt.xlim(0,2.4)
    plt.xlabel("f/f_res", fontsize=60)
    plt.ylabel("R($\omega$)", fontsize=60)
    plt.legend( fontsize=40, loc = 'best')
plt.savefig("Reflec_transmis_collectiontransmission"+ Flag +".jpg")  



