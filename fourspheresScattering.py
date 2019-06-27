# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:21:53 2019

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

    return sp.spherical_jn(n,z,derivative=False)+1j*sp.spherical_yn(n,z,derivative=False)




def scatteringcoefficients(mhu, mhu_p , lamda , lamda_p, rho , rho_p, Flag ):
    rho_ratio = rho_p/rho
    R_0 = 200e-6
    Nf = 1000
    f = np.linspace(1.0e5,3.0e6,Nf)

    #c_L_i = m.sqrt((lamda + 2*mhu)/rho)
    #c_T_i = m.sqrt(mhu/rho)
    T1 = np.zeros((2,Nf),dtype=complex)
    T2 = np.zeros((4,Nf),dtype=complex)
    KKTi = np.zeros((1,Nf),dtype=complex)
    KKLi = np.zeros((1,Nf),dtype=complex)
    resonance = np.zeros((1,Nf),dtype=complex)
    
    for iterator in range(0,Nf):
        omega = 2*m.pi*f[iterator]
        lame1 = 3.6e9 + 0.1e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.3e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) +0.25e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2) + 0.3e9*((omega*8e-9)**2 - 1j*omega*8e-9)/(1+(omega*8e-9)**2)
        lame2 = 1.05e9 + 0.2e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.15e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) - 0.1e9*((omega*14e-9)**2 - 1j*omega*14e-9)/(1+(omega*14e-9)**2) + 0.15e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2)
        lamda = lame1
        mhu = lame2
        c_L_i = cm.sqrt((lame1+2*lame2)/rho)
        c_T_i = cm.sqrt(lame2/rho)
        
        alphaL_f = 0.1 + 0.25/2000000*f[iterator]
        alphaS_f = 0.05 + 0.1/2000000*f[iterator]
    
        
        k_L_i = omega/c_L_i + 1j*alphaL_f
        k_T_i = omega/c_T_i + 1j*alphaS_f
        K_L = k_L_i * R_0
        K_T = k_T_i * R_0
        k_L_p = omega*m.sqrt(rho_p/(lamda_p + 2*mhu_p))
        k_T_p = omega*m.sqrt(rho_p/mhu_p)
        K_L_p = k_L_p * R_0
        K_T_p = k_T_p * R_0
        

        KKTi[:,iterator] = K_T
        KKLi[:,iterator] = K_L
        
        if Flag == "T":
            n = 1
            hnL_n = spherical_hn1(n,K_L)
            hnT_n = spherical_hn1(n, K_T)
            hnL_1n = spherical_hn1(n+1,K_L)
            hnT_1n = spherical_hn1(n+1, K_T)
            
            jnL_n = sp.spherical_jn(n,K_L)
            jnT_n = sp.spherical_jn(n,K_T)
            jnL_np = sp.spherical_jn(n,K_L_p) 
            jnT_np = sp.spherical_jn(n,K_T_p) 
            
            jnL_1n = sp.spherical_jn(n+1,K_L)
            jnT_1n = sp.spherical_jn(n+1,K_T)
            jnL_1np = sp.spherical_jn(n+1,K_L_p) 
            jnT_1np = sp.spherical_jn(n+1,K_T_p) 
            
            U_L_p = n*jnL_np - K_L_p*jnL_1np
            U_T_p = n*(n+1)*jnT_np
            U_L   = n*hnL_n - K_L*hnL_1n
            U_T   = n*(n+1)*hnT_n
            U_I   = n*(n+1)*jnT_n
            
            V_L_p = jnL_np
            V_T_p = (1+n)*jnT_np - K_T*jnT_1np  
            V_L   = hnL_n
            V_T   = (1+n)*hnT_n - K_T*hnT_1n      
            V_I   = (1+n)*jnT_n - K_T*jnT_1n
            
            W_S   = 1j*(K_T*hnT_n)
            W_S_p = 1j*(K_T_p*jnT_np)
            W_I   = 1j*(K_T*jnT_n)
            
            
            SS_L_p = (2*n*(n-1)*mhu_p - (lamda_p + 2*mhu_p)*K_L_p**2)*jnL_np + 4*mhu_p*K_L_p*jnL_1np
            SS_T_p = 2*n*(n + 1)*mhu_p*((n-1)*jnT_np - K_T_p*jnT_1np )
            SS_L   = (2*n*(n-1)*mhu - (lamda + 2*mhu)*K_L**2)*hnL_n + 4*mhu*K_L*hnL_1n
            SS_T   = 2*n*(n + 1)*mhu*((n-1)*hnT_n - K_T*hnT_1n)
            SS_I   = 2*n*(n + 1)*mhu*((n-1)*jnT_n - K_T*jnT_1n)
            
            Tau_L_p = 2*mhu_p*((n-1)*jnL_np - K_L_p*jnL_1np)
            Tau_L   = 2*mhu*((n-1)*hnL_n - K_L*hnL_1n)
            Tau_T_p = mhu_p*((2*n**2-2 - K_T_p**2)*jnT_np + 2*K_T_p*jnT_1np)
            Tau_T   = mhu*((2*n**2-2 - K_T**2)*hnT_n + 2*K_T*hnT_1n)
            Tau_I   = mhu*((2*n**2-2 - K_T**2)*jnT_n + 2*K_T*jnT_1n )
            
            Sig_S_p = 1j*mhu_p*K_T_p*((n-1)*jnT_np- K_T_p*jnT_1np)
            Sig_S   = 1j*mhu*K_T*((n-1)*hnT_n - K_T*hnT_1n)
            Sig_I   = 1j*mhu*K_T*((n-1)*jnT_n - K_T*jnT_1n)
            
            #Build system of equations
            Designmatrix_LHS = np.array([[U_L , U_T , -U_L_p , -U_T_p],
                                        [V_L , V_T , -V_L_p , -V_T_p],
                                        [SS_L , SS_T , -SS_L_p , -SS_T_p],
                                        [Tau_L , Tau_T , -Tau_L_p , -Tau_T_p]])
            Designmatrix_RHS = np.array([U_I , V_I , SS_I , Tau_I])
            
            t1 = np.linalg.solve(Designmatrix_LHS, -Designmatrix_RHS)
            T2[:,iterator] = t1  
            
            Designmatrix2_LHS = np.array([[W_S , -W_S_p],
                                          [Sig_S , -Sig_S_p]])
            Designmatrix2_RHS = np.array([W_I , Sig_I ])
            t2 = np.linalg.solve(Designmatrix2_LHS, -Designmatrix2_RHS)
            T1[:,iterator] = t2
        
        
        if Flag == "L":
            
            f_res = (3*c_T_i*m.sqrt(8*rho_ratio))/(4*m.pi*R_0*(2*rho_ratio + 1))
            omega_res = 2*m.pi*f_res
            K_L_i_res = R_0*omega_res/c_L_i
            #K_T_i_res = R_0*omega_res/c_T_i
            resonance[:,iterator] = K_L_i_res
                
            for n in range(0,2):
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
                U_L   = n*hnL_n - K_L*hnL_1n
                U_T   = n*(n+1)*hnT_n
                U_I   = n*jnL_n - K_L*jnL_1n
                
                V_L_p = jnL_np
                V_T_p = (1+n)*jnT_np - K_T*jnT_1np  
                V_L   = hnL_n
                V_T   = (1+n)*hnT_n - K_T*hnT_1n      
                V_I   = jnL_n
                
                
                SS_L_p = (2*n*(n-1)*mhu_p - (lamda_p + 2*mhu_p)*K_L_p**2)*jnL_np + 4*mhu_p*K_L_p*jnL_1np
                SS_T_p = 2*n*(n + 1)*mhu_p*((n-1)*jnT_np - K_T_p*jnT_1np )
                SS_L   = (2*n*(n-1)*mhu - (lamda + 2*mhu)*K_L**2)*hnL_n + 4*mhu*K_L*hnL_1n
                SS_T   = 2*n*(n + 1)*mhu*((n-1)*hnT_n - K_T*hnT_1n)
                SS_I   = (2*n*(n-1)*mhu - (lamda + 2*mhu)*K_L**2)*jnL_n + 4*mhu*K_L*jnL_1n
                
                Tau_L_p = 2*mhu_p*((n-1)*jnL_np - K_L_p*jnL_1np)
                Tau_L   = 2*mhu*((n-1)*hnL_n - K_L*hnL_1n)
                Tau_T_p = mhu_p*((2*n**2-2 - K_T_p**2)*jnT_np + 2*K_T_p*jnT_1np)
                Tau_T   = mhu*((2*n**2-2 - K_T**2)*hnT_n + 2*K_T*hnT_1n)
                Tau_I   = 2*mhu*((n-1)*jnL_n - K_L*jnL_1n)
                
                if n == 0:
                    LHS = np.array([[U_L , -U_L_p],
                                   [SS_L, -SS_L_p]])
                    RHS = np.array([U_I , SS_I] )
                
                    a0 = np.linalg.solve(LHS, -RHS)
                    T1[:,iterator] = a0
                elif n == 1:
                    LHS1 = np.array([[U_L , U_T , -U_L_p , -U_T_p],
                                    [V_L , V_T , -V_L_p , -V_T_p],
                                    [SS_L , SS_T , -SS_L_p , -SS_T_p],
                                    [Tau_L , Tau_T , -Tau_L_p , -Tau_T_p]])
                    
                    RHS1 = np.array([U_I , V_I , SS_I , Tau_I] )
                    
                    a1 = np.linalg.solve(LHS1, -RHS1)
                    
                    T2[:,iterator] = a1

    if Flag== "T":
        coefficients = np.vstack((KKTi[0,:], T1[0,:],T2[0,:],T2[1,:]))
        print("Transverse incident wave, scatteringcoefficients are ordered as [K_T , T1_SS , T1_TL , T1_TT] in np.array")
    if Flag=="L":
        coefficients = np.vstack((KKLi[0,:], T1[0,:] ,T2[0,:] ,T2[1,:] ,resonance[0,:]))
        print("Longitudinal incident wave, scatteringcoefficients are ordered as [K_L , T0_LL , T1_LL , T1_LT, K_Lires] in np.array")
        
  

    return coefficients




    
    
rho = 1100
lamda = 2.426470588e9
mhu = 1.25e9

lamda_thung = 162.0e9#Pa Al
rho_thung = 15250.0 #g/cm^3
mhu_thung = 243.0e9 #Pa
rho_ratioWC = rho_thung/rho


#Thungsten carbine

B_thungsten_T = scatteringcoefficients(mhu, mhu_thung, lamda , lamda_thung, rho , rho_thung, "T" )
print("\n")

B_thungsten_L = scatteringcoefficients(mhu, mhu_thung, lamda , lamda_thung, rho , rho_thung, "L" )
print("\n")



T0_LL = B_thungsten_L[1,:]
T1_TL = B_thungsten_T[2,:]
T1_LL = B_thungsten_L[2,:]
T1_TT = B_thungsten_T[3,:]
T1_LT = B_thungsten_L[3,:]

c_L_i = 2490.0
c_T_i = 1250.0


Nf = 1000
f = np.linspace(1.0e5,3.0e6,Nf)
R_0 = 200e-6

Nn = 5

A1 = np.zeros((Nn,Nf),dtype=complex)
A2 = np.zeros((Nn,Nf),dtype=complex)
A3 = np.zeros((Nn,Nf),dtype=complex)
A4 = np.zeros((Nn,Nf),dtype=complex)
B1 = np.zeros((Nn,Nf),dtype=complex)
B2 = np.zeros((Nn,Nf),dtype=complex)
C1 = np.zeros((Nn,Nf),dtype=complex)
C2 = np.zeros((Nn,Nf),dtype=complex)
B3 = np.zeros((Nn,Nf),dtype=complex)
B4 = np.zeros((Nn,Nf),dtype=complex)
C3 = np.zeros((Nn,Nf),dtype=complex)
C4 = np.zeros((Nn,Nf),dtype=complex)

lamda_p = lamda_thung
rho_p = rho_thung
mhu_p = mhu_thung
rho_ratio = rho/rho_p
wavelength_res = (1/((3*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio+1))))#*c_L_i/c_T_i
#d = (wavelength_res/2)
Beta = np.array([0.5 , 1 , 2 , 10 , 100])
for N in range(Nn):
    d = R_0*(1+Beta[N])
    for iterator in range(0,Nf):
        omega = 2*m.pi*f[iterator]
        lame1 = 3.6e9 + 0.1e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.3e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) + 0.25e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2) + 0.3e9*((omega*8e-9)**2 - 1j*omega*8e-9)/(1+(omega*8e-9)**2)
        lame2 = 1.05e9 + 0.2e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.15e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) - 0.1e9*((omega*14e-9)**2 - 1j*omega*14e-9)/(1+(omega*14e-9)**2) + 0.15e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2)
        lamda = lame1
        mhu = lame2
        c_L_i = cm.sqrt((lame1+2*lame2)/rho)
        c_T_i = cm.sqrt(lame2/rho)
        
        alphaL_f = 0.1 + 0.25/2000000*f[iterator]
        alphaS_f = 0.05 + 0.1/2000000*f[iterator]
        
        
        k_L_i = omega/c_L_i + 1j*alphaL_f
        k_T_i = omega/c_T_i + 1j*alphaS_f
        
        DM1 = np.zeros((4,4), dtype = complex)
        DM1[0,0] = 1.0
        DM1[0,1] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        DM1[0,2] =  -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d*m.sqrt(2))
        DM1[0,3] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        
        DM1[1,0] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        DM1[1,1] = 1.0
        DM1[1,2] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        DM1[1,3] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d*m.sqrt(2))
        
        DM1[2,0] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d*m.sqrt(2))
        DM1[2,1] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        DM1[2,2] =  1.0
        DM1[2,3] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        
        DM1[3,0] = -T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        DM1[3,1] =-T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d*m.sqrt(2))
        DM1[3,2] =-T0_LL[iterator]*spherical_hn1(0,k_L_i*2*d)
        DM1[3,3] = 1.0 
        
        
        DMRHS1 = np.array([T0_LL[iterator], T0_LL[iterator], T0_LL[iterator], T0_LL[iterator]])                 
        solutionA  = np.linalg.solve(DM1, DMRHS1)
        
        A1[N,iterator]  = solutionA[0]
        A2[N,iterator]  = solutionA[1]
        A3[N,iterator]  = solutionA[2]
        A4[N,iterator]  = solutionA[3]
       
        DM2 = np.zeros((8,8),dtype=complex)
        DM2[0,0] = 1.0 
        DM2[0,1] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[0,2] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[0,3] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[0,4] =  0.0 
        DM2[0,5] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[0,6] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        DM2[0,7] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        
        DM2[1,0] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[1,1] = 1.0
        DM2[1,2] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[1,3] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[1,4] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[1,5] = 0.0
        DM2[1,6] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[1,7] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        
        DM2[2,0] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[2,1] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[2,2] = 1.0
        DM2[2,3] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[2,4] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        DM2[2,5] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[2,6] = 0.0
        DM2[2,7] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        
        DM2[3,0] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[3,1] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[3,2] = -T1_LL[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[3,3] = 1.0 
        DM2[3,4] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[3,5] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        DM2[3,6] = -T1_TL[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[3,7] = 0.0
        
        DM2[4,0] = 0.0
        DM2[4,1] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d)) 
        DM2[4,2] =  -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[4,3] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d)) 
        DM2[4,4] = 1.0 
        DM2[4,5] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[4,6] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        DM2[4,7] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        
        DM2[5,0] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[5,1] = 0.0
        DM2[5,2] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[5,3] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[5,4] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[5,5] = 1.0
        DM2[5,6] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[5,7] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
      
        DM2[6,0] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[6,1] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[6,2] =  0.0 
        DM2[6,3] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[6,4] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        DM2[6,5] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d)) 
        DM2[6,6] = 1.0
        DM2[6,7] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        
        DM2[7,0] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[7,1] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_L_i*2*d*m.sqrt(2)))
        DM2[7,2] = -T1_LT[iterator]*(spherical_hn1(0,k_L_i*2*d) + 4/3*spherical_hn1(2,k_L_i*2*d))
        DM2[7,3] = 0.0
        DM2[7,4] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[7,5] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d*m.sqrt(2)) + 4/3*spherical_hn1(2,k_T_i*2*d*m.sqrt(2)))
        DM2[7,6] = -T1_TT[iterator]*(spherical_hn1(0,k_T_i*2*d) + 4/3*spherical_hn1(2,k_T_i*2*d))
        DM2[7,7] = 1.0 
        
        
       
        RHSDM2 = np.array([T1_LL[iterator] , T1_LL[iterator] ,T1_LL[iterator] , T1_LL[iterator], T1_LT[iterator] , T1_LT[iterator] ,  T1_LT[iterator] , T1_LT[iterator]])
        
        
        solutionBC  = np.linalg.solve(DM2, RHSDM2)
        B1[N,iterator] = solutionBC[0]
        B2[N,iterator] = solutionBC[1]
        B3[N,iterator] = solutionBC[2]
        B4[N,iterator] = solutionBC[3]
        C1[N,iterator] = solutionBC[4]
        C2[N,iterator] = solutionBC[5]
        C3[N,iterator] = solutionBC[6]
        C4[N,iterator] = solutionBC[7]
        
    f_res = (3*c_T_i*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio + 1))

        
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


plt.figure(1)
for ii in range(Nn):
    plt.subplot(411)
    plt.suptitle("$A_n$ , $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(A1[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("A1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(412)
    plt.plot(abs(f/f_res) , abs(A2[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("A2", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(413)
    plt.plot(abs(f/f_res) , abs(A3[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("A3", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(414)
    plt.plot(abs(f/f_res) , abs(A4[ii,:]), label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.ylabel("A4", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))

plt.figure(2)
for ii in range(Nn):
    plt.subplot(411)
    plt.suptitle("$B_n$, $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(B1[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("B1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(412)
    plt.plot(abs(f/f_res) , abs(B2[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("B2", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(413)
    plt.plot(abs(f/f_res) , abs(B3[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("B3", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(414)
    plt.plot(abs(f/f_res) , abs(B4[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("B4", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))

plt.figure(3)
for ii in range(Nn):
    plt.subplot(411)
    plt.suptitle("$C_n$, $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(C1[ii,:]),label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.ylabel("C1", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.subplot(412)
    plt.plot(abs(f/f_res) , abs(C2[ii,:]),label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("C2", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(413)
    plt.plot(abs(f/f_res) , abs(C3[ii,:]),label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("C3", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
    plt.subplot(414)
    plt.plot(abs(f/f_res) , abs(C4[ii,:]),label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("C4", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))
        
plt.figure(4)
for ii in range(Nn):
    plt.suptitle("$B_n$, $d = R*(1+\\beta)$", fontsize=40)
    plt.plot(abs(f/f_res) , abs(B1[ii,:]), label=str(Beta[ii])+ "$= \\beta $", linewidth=3.0)
    plt.ylabel("Bn", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.xlabel("$f/f_{res}$", fontsize=40)
    plt.legend( fontsize=40, loc=(1.0,0))