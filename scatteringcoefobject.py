# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:46:44 2019

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



def scatteringcoefficients_rigid(mhu, mhu_p , lamda , lamda_p, rho , rho_p, Flag ):
    rho_ratio = rho_p/rho
    R_0 = 200e-6
    Nf = 1000
    f = np.linspace(1.0e5,3.0e6,Nf)
    #c_L_i = m.sqrt((lamda + 2*mhu)/rho)
    #c_T_i = m.sqrt(mhu/rho)
    T0_L_ri = np.zeros((1,Nf),dtype=complex)
    T_SS_ri = np.zeros((1,Nf),dtype=complex)
    T1_LL_ri = np.zeros((1,Nf),dtype=complex)
    T1_LT_ri = np.zeros((1,Nf),dtype=complex)
    T1_TL_ri = np.zeros((1,Nf),dtype=complex)
    T1_TT_ri = np.zeros((1,Nf),dtype=complex)
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
            Y_I = 5*Sig_I/(rho_p*(omega*R_0)**2)
            Y_S = 5*Sig_S/(rho_p*(omega*R_0)**2)
            
            T_SS_ri[:,iterator] = -(W_I + Y_I)/(W_S + Y_S)
            
            
            
            
            X_IT = (SS_I + 2*Tau_I)/(rho_p*(omega*R_0)**2)
            X_SL = (SS_L + 2*Tau_L)/(rho_p*(omega*R_0)**2)
            X_ST = (SS_T + 2*Tau_T)/(rho_p*(omega*R_0)**2)
    
            LHS1_ri = U_L + X_SL
            LHS2_ri = U_T + X_ST
            LHS3_ri = V_L + X_SL
            LHS4_ri = V_T + X_ST
            
            RHS1_ri = U_I + X_IT
            RHS2_ri = V_I + X_IT
            
            LHS_ri =  np.array([[LHS1_ri , LHS2_ri],
                                [LHS3_ri , LHS4_ri]])
            
            RHS_ri =  np.array([RHS1_ri , RHS2_ri])
            
            solution  = np.linalg.solve(LHS_ri, -RHS_ri)
            
            T1_TL_ri[:,iterator] = solution[0]
            T1_TT_ri[:,iterator] = solution[1]
        
        
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
                    T0_L_ri[:,iterator] = -U_I/U_L
                elif n == 1:
                    X_IL = (SS_I + 2*Tau_I)/(rho_p*(omega*R_0)**2)
                    X_SL = (SS_L + 2*Tau_L)/(rho_p*(omega*R_0)**2)
                    X_ST = (SS_T + 2*Tau_T)/(rho_p*(omega*R_0)**2)
            
                    LHS1_ri = U_L + X_SL
                    LHS2_ri = U_T + X_ST
                    LHS3_ri = V_L + X_SL
                    LHS4_ri = V_T + X_ST
                    
                    RHS1_ri = U_I + X_IL
                    RHS2_ri = V_I + X_IL
                    
                    LHS_ri =  np.array([[LHS1_ri , LHS2_ri],
                                        [LHS3_ri , LHS4_ri]])
                    
                    RHS_ri =  np.array([RHS1_ri , RHS2_ri])
                    
                    solution  = np.linalg.solve(LHS_ri, -RHS_ri)
                    
                    T1_LL_ri[:,iterator] = solution[0]
                    T1_LT_ri[:,iterator] = solution[1]
    if Flag== "T":
        coefficients = np.vstack((KKTi[0,:], T_SS_ri,T1_TL_ri,T1_TT_ri))
        print("Transverse incident wave, scatteringcoefficients are ordered as [K_T , T1_SS , T1_TL , T1_TT] in np.array")
    if Flag=="L":
        coefficients = np.vstack((KKLi[0,:], T0_L_ri ,T1_LL_ri,T1_LT_ri ,resonance[0,:]))
        print("Longitudinal incident wave, scatteringcoefficients of rigid sphere are ordered as [K_L , T0_LL , T1_LL , T1_LT, K_Lires] in np.array")
        
  

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

B_thungsten_T_rigid = scatteringcoefficients_rigid(mhu, mhu_thung, lamda , lamda_thung, rho , rho_thung, "T" )
print("\n")

B_thungsten_L_rigid = scatteringcoefficients_rigid(mhu, mhu_thung, lamda , lamda_thung, rho , rho_thung, "L" )
print("\n")

lamda_Lead = 26.76923e9 #Pa  26.76923_48.54545
rho_Lead = 11310 #kg/m^3 11310_11390
rho_ratiolead = rho_Lead/rho

mhu_Lead = 4.0e9 #Pa    4_6
B_Lead_L = scatteringcoefficients(mhu, mhu_Lead, lamda , lamda_Lead, rho , rho_Lead, "L" )
print("\n")

B_Lead_T = scatteringcoefficients(mhu, mhu_Lead, lamda , lamda_Lead, rho , rho_Lead, "T" )
print("\n")



def funcWC(KTresWC):
    return 1j*KTresWC**2*(2*rho_ratioWC + 1) - 9*KTresWC- 1j*9 

def derfuncWC(KTresWC):
    return 2j*KTresWC*(2*rho_ratioWC + 1) - 9


def func2WC(KT2resWC):
    return rho_ratioWC*KT2resWC**3 + 1j*KT2resWC**2*(rho_ratioWC + 5) - 15*KT2resWC -1j*15

def derfunc2WC(KT2resWC):
    return rho_ratioWC*3*KT2resWC**2 + 2j*KT2resWC*(rho_ratioWC + 5) - 15




FWC = 10000
F2WC = 10000
KTresWC = 10000 + 40j
KT2resWC = 10000 + 40j

while abs(FWC) > 0.00001:
    FWC = funcWC(KTresWC)
    DFWC = derfuncWC(KTresWC)
    KTresWC = KTresWC - FWC/DFWC
#print(abs(KTresWC))
#print(abs(KTres))

while abs(F2WC) >0.00001:
    F2WC = func2WC(KT2resWC)
    DF2WC = derfunc2WC(KT2resWC)
    KT2resWC = KT2resWC - F2WC/DF2WC
#print(abs(KT2resWC))




def funclead(KTreslead):
    return 1j*KTreslead**2*(2*rho_ratiolead + 1) - 9*KTreslead- 1j*9 

def derfunclead(KTreslead):
    return 2j*KTreslead*(2*rho_ratiolead + 1) - 9


def func2lead(KT2reslead):
    return rho_ratiolead*KT2reslead**3 + 1j*KT2reslead**2*(rho_ratiolead + 5) - 15*KT2reslead -1j*15

def derfunc2lead(KT2reslead):
    return rho_ratiolead*3*KT2reslead**2 + 2j*KT2reslead*(rho_ratiolead + 5) - 15

Flead = 10000
F2lead = 10000
KTreslead = 10000 + 40j
KT2reslead = 10000 + 40j

while abs(Flead) >0.00001:
    Flead = funclead(KTreslead)
    DFlead = derfunclead(KTreslead)
    KTreslead = KTreslead - Flead/DFlead
#print(abs(KTreslead))
#print(abs(KTres))

while abs(F2lead) >0.00001:
    F2lead = func2lead(KT2reslead)
    DF2lead = derfunc2lead(KT2reslead)
    KT2reslead = KT2reslead - F2lead/DF2lead
#print(abs(KT2reslead))











params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


#
plt.figure(1)
plt.subplot(311)
plt.tick_params(labelsize=40)
plt.suptitle("Scattering coefficients due to longitudinal wave on single particle", fontsize=40)
plt.plot(B_thungsten_L[0,:], abs(B_thungsten_L[1,:]),label= "Tungsten Carbine" , linewidth=3.0)
plt.plot(B_thungsten_L[0,:], abs(B_Lead_L[1,:]),label= "Lead" , linewidth=3.0)
#plt.plot(abs(B_Lead_L[0,:]), abs(B_Lead_L[1,:]),label= "Lead" , linewidth=3.0)
plt.ylabel("$T_0^{LL}$", fontsize=60)
plt.legend(fontsize = 30 ,loc = "best")
#plt.axvline(x=B_thungsten_L[4,0])
plt.subplot(312)
plt.tick_params(labelsize=40)
plt.plot(B_thungsten_L[0,:], abs(B_thungsten_L[2,:]),label= "Tungsten Carbine", linewidth=3.0 )
plt.plot(B_thungsten_L[0,:], abs(B_Lead_L[2,:]),label= "Lead", linewidth=3.0 )
#plt.plot(abs(B_Lead_L[0,:]), abs(B_Lead_L[2,:]), label= "Lead", linewidth=3.0 )
plt.ylabel("$T_1^{LL}$", fontsize=60)
plt.axvline(x=abs(B_thungsten_L[4,0]),label= "$K_{L-res}$ Tungsten Carbine")
plt.axvline(x=abs(B_Lead_L[4,0]),linestyle='dashed',label= "$K_{L-res}$ Lead")
plt.legend( fontsize = 30 ,loc = "best")
plt.subplot(313)
plt.tick_params(labelsize=40)
plt.plot(B_thungsten_L[0,:], abs(B_thungsten_L[3,:]),label= "Tungsten Carbine", linewidth=3.0 )
plt.plot(B_thungsten_L[0,:], abs(B_Lead_L[3,:]),label= "Lead", linewidth=3.0 )
#plt.plot(abs(B_Lead_L[0,:]), abs(B_Lead_L[3,:]),label= "Lead", linewidth=3.0 )
plt.ylabel("$T_1^{LT}$", fontsize=60)
plt.axvline(x=abs(B_thungsten_L[4,0]),label= "$K_{L-res}$ Tungsten Carbine")
plt.axvline(x=abs(B_Lead_L[4,0]),linestyle='dashed',label= "$K_{L-res}$ Lead")
plt.legend( fontsize = 30 ,loc = "best")
plt.xlabel("$K_L$", fontsize=60)
plt.savefig('LongiScatteringCoefsingleparticle.jpg')


plt.figure(2)
plt.subplot(311)
plt.suptitle("Scattering coefficients due to shear wave on single particle", fontsize=40)
plt.axvline(x=abs(KTresWC),label= "$K_{T-res}$ Tungsten Carbine")
plt.axvline(x=abs(KTreslead),linestyle='dashed',label= "$K_{T-res}$ Lead")
plt.tick_params(labelsize=40)
plt.plot(B_thungsten_T[0,:], abs(B_thungsten_T[2,:]),label= "Tungsten Carbine", linewidth=3.0 )
plt.plot(B_Lead_T[0,:], abs(B_Lead_T[2,:]),label= "Lead", linewidth=3.0 )
#plt.plot(abs(B_Lead_T[0,:]), abs(B_Lead_T[2,:]),label= "Lead", linewidth=3.0 )
plt.ylabel("$T_1^{TL}$", fontsize=60)
plt.legend( fontsize = 30 ,loc = "best")
plt.subplot(312)
plt.tick_params(labelsize=40)
plt.plot(B_thungsten_T[0,:], abs(B_thungsten_T[3,:]),label= "Tungsten Carbine" , linewidth=3.0)
plt.plot(B_Lead_T[0,:], abs(B_Lead_T[3,:]),label= "Lead" , linewidth=3.0)
#plt.plot(abs(B_Lead_T[0,:]), abs(B_Lead_T[3,:]), label= "Lead", linewidth=3.0 )
plt.ylabel("$T_1^{TT}$", fontsize=60)
plt.axvline(x=abs(KTresWC),label= "$K_{T-res}$ Tungsten Carbine")
plt.axvline(x=abs(KTreslead),linestyle='dashed',label= "$K_{T-res}$ Lead")
plt.legend( fontsize = 30 ,loc = "right")
plt.subplot(313)
plt.axvline(x=abs(KT2resWC),label= "$K_{S-res}$ Tungsten Carbine")
plt.axvline(x=abs(KT2reslead),linestyle='dashed',label= "$K_{S-res}$ Lead")
plt.tick_params(labelsize=40)
plt.plot(B_thungsten_T[0,:], abs(B_thungsten_T[1,:]),label= "Tungsten Carbine", linewidth=3.0 )
plt.plot(B_Lead_T[0,:], abs(B_Lead_T[1,:]),label= "Lead", linewidth=3.0 )
#plt.plot(abs(B_Lead_T[0,:]), abs(B_Lead_T[1,:]),label= "Lead", linewidth=3.0 )
plt.ylabel("$T_1^{SS}$", fontsize=60)
plt.xlabel("$K_T$", fontsize=60)
plt.legend(fontsize = 30 ,loc = "best")
plt.savefig('TransScatteringCoefsingleparticle.jpg')



#
#T0_LL = B_thungsten_L[1,:]
#T1_TL = B_thungsten_T[2,:]
#T1_LL = B_thungsten_L[2,:]
#T1_TT = B_thungsten_T[3,:]
#T1_LT = B_thungsten_L[3,:]
#
#lamda_p = lamda_thung
#rho_p = rho_thung
#mhu_p = mhu_thung
#rho_ratio = rho/rho_p
#R_0 = 200e-6
#
#
#Np = 50
#Nf = 1000
#Nn = 5
#KKTi = np.zeros((1,Nf),dtype=complex)
#KKLi = np.zeros((1,Nf),dtype=complex)
#
#
#R_omegastore = np.zeros((Nn,Nf),dtype=complex)
#T_omegastore = np.zeros((Nn,Nf),dtype=complex)
#
#c_L_i = 2490.0
#c_T_i = 1250.0
#
#Omega = np.zeros((1,Nf))
#Omega_T = np.zeros((1,Nf))
#
##R0_omegastore = np.zeros((Nn,Nf),dtype=complex)
##T0_omegastore = np.zeros((Nn,Nf),dtype=complex)
#
#
#
#wavelength_res = (1/((3*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio+1))))*c_L_i/c_T_i
#
#
#
#AA0 = np.zeros((Nn,Nf),dtype=complex)
#AA1BB1 = np.zeros((Nn,Nf),dtype=complex)
#R = np.zeros((1,Nf),dtype=complex)
#T = np.zeros((1,Nf),dtype=complex)
##R0 = np.zeros((1,Nf),dtype=complex)
##T0 = np.zeros((1,Nf),dtype=complex)
#AA0_store = np.zeros((Nn,Nf),dtype=complex)
#AA1_store = np.zeros((Nn,Nf),dtype=complex)
#BB1_store = np.zeros((Nn,Nf),dtype=complex)
#S_1s_store = np.zeros((1,Nf))
#S_1s_phase_store = np.zeros((1,Nf))
#S_1p_store = np.zeros((1,Nf))
#S_1p_phase_store = np.zeros((1,Nf))
#S_0p_store = np.zeros((1,Nf))
#S_0p_phase_store = np.zeros((1,Nf))
#
#
##AA0_different_store = np.zeros((Nn,Nf),dtype=complex)
##AA1_different_store = np.zeros((Nn,Nf),dtype=complex)
##BB1_different_store = np.zeros((Nn,Nf),dtype=complex)
#
#f = np.linspace(1.0e5,3.0e6,Nf)
#Beta = np.array([0.5 , 1 , 2 , 10 , 100])
#for N in range(Nn):
#    d = R_0*(1+Beta[N])
#    print("Iteration" , N , "started")
#     
#       
#    for iterator in range(0,Nf):
##        if iterator%100==0:
##            print(iterator)
#        omega = 2*m.pi*f[iterator]
#        lame1 = 3.6e9 + 0.1e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.3e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) + 0.25e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2) + 0.3e9*((omega*8e-9)**2 - 1j*omega*8e-9)/(1+(omega*8e-9)**2)
#        lame2 = 1.05e9 + 0.2e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.15e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) - 0.1e9*((omega*14e-9)**2 - 1j*omega*14e-9)/(1+(omega*14e-9)**2) + 0.15e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2)
#        lamda = lame1
#        mhu = lame2
#        c_L_i = cm.sqrt((lame1+2*lame2)/rho)
#        c_T_i = cm.sqrt(lame2/rho)
#        
#        alphaL_f = 0.1 + 0.25/2000000*f[iterator]
#        alphaS_f = 0.05 + 0.1/2000000*f[iterator]
#        
#        
#        k_L_i = omega/c_L_i + 1j*alphaL_f
#        k_T_i = omega/c_T_i + 1j*alphaS_f
#        K_L = k_L_i * R_0
#        K_T = k_T_i * R_0
#        k_L_p = omega*m.sqrt(rho_p/(lamda_p + 2*mhu_p))
#        k_T_p = omega*m.sqrt(rho_p/mhu_p)
#        K_L_p = k_L_p * R_0
#        K_T_p = k_T_p * R_0
#        
#        
#        KKTi[:,iterator] = K_T
#        KKLi[:,iterator] = K_L
#    
#    
#        S_0p = 0.0
#        S_1p = 0.0
#        S_1s = 0.0
#
#
#        integerII = np.linspace(-Np,Np,2*Np+1)
#        for integerI in range(-Np,Np):
#            
#            if integerI==0:
#                integerII = np.linspace(-Np,Np+1,2*Np+2)
#                integerII = integerII[integerII !=0]
#              
#            S_0p += np.sum(np.exp(1j*k_L_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(2*d*k_L_i*np.sqrt(integerI**2 + (integerII)**2)))
#            S_1p += np.sum(np.exp(1j*k_L_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(2*d*k_L_i*np.sqrt(integerI**2 + (integerII)**2))*(1/3 + 1j*4/(k_L_i*2*d*np.sqrt(integerI**2 + (integerII)**2)) - 4/((k_L_i*2*d)**2*(integerI**2 + (integerII)**2))))
#            S_1s += np.sum(np.exp(1j*k_T_i*2*d*np.sqrt(integerI**2 +(integerII)**2))/(2*d*k_T_i*np.sqrt(integerI**2 + (integerII)**2))*(1/3 + 1j*4/(k_T_i*2*d*np.sqrt(integerI**2 + (integerII)**2)) - 4/((k_T_i*2*d)**2*(integerI**2 + (integerII)**2))))
#            
#            
#        S_0p = -1j*S_0p
#        S_1p = 1j*S_1p
#        S_1s = 1j*S_1s
#        
#        S_1s_store[:,iterator] = abs(S_1s)
#        S_1s_phase_store[:,iterator] = cm.phase(S_1s)
#        S_1p_store[:,iterator] = abs(S_1p)
#        S_1p_phase_store[:,iterator] = cm.phase(S_1p)
#        S_0p_store[:,iterator] = abs(S_0p)
#        S_0p_phase_store[:,iterator] = cm.phase(S_0p)   
#        
#        #DM1 = spherical_hn1(1,K_L) - T1_LL[iterator]*S_1p
#        DM1 = -T1_LL[iterator]*S_1p + 1
#        #DM2 = -T1_TL[iterator]*S_1s
#        DM2 = -T1_TL[iterator]*S_1s 
#        #DM3 = -T1_LT[iterator]*S_1p
#        DM3 = -T1_LT[iterator]*S_1p
#        #DM4 = spherical_hn1(1,K_T) - T1_TT[iterator]*S_1s
#        DM4 = -T1_TT[iterator]*S_1s + 1
#        
#       # RHS1 = T1_LL[iterator]*sp.spherical_jn(1,K_L)
#        #RHS2 = T1_LT[iterator]*sp.spherical_jn(1,K_L)
#        RHS1 = T1_LL[iterator]
#        RHS2 = T1_LT[iterator]
#        
#        LHS = np.array([[DM1 , DM2],
#                        [DM3 , DM4]])
#        RHS = np.array([[RHS1],
#                        [RHS2]])
#       # LHS0 = np.array([[DM10 , DM20],
#                      #  [DM30 , DM40]])
#        #RHS0 = np.array([[RHS10],
#                       # [RHS20]])
#        
#        A1B1 = np.linalg.solve(LHS, RHS)
#       # A1B10 = np.linalg.solve(LHS0, RHS0)
#        
#        #A0 = -T0_LL[iterator]*sp.spherical_jn(0,K_L)/(T0_LL[iterator]*S_0p - spherical_hn1(0,K_L))
#        A0 = T0_LL[iterator]/(1 - T0_LL[iterator]*S_0p)
#        AA0[N,iterator] = A0
#        AA0_store[N,iterator] = A0
#        #AA0_different_store[N,iterator] = A0_different
#        AA1_store[N,iterator] = A1B1[0]
#        BB1_store[N,iterator] = A1B1[1]
#        #AA1_different_store[N,iterator] = A1B10[0]
#        #BB1_different_store[N,iterator] = A1B10[1]
#        #AA1BB1[N,iterator] = A1B1[]
#        
#        R[:,iterator] = 2*m.pi*(A0 + A1B1[0]/3)/(k_L_i**2*4*d**2)
#        T[:,iterator] = 1 + 2*m.pi*(A0 - A1B1[0]/3)/(k_L_i**2*4*d**2)
#        #R0[:,iterator] = 2*m.pi*(A0 + 1j*A1B1[0])/(k_L_i**2*d**2)
#        #T0[:,iterator] = 1 + 2*m.pi*(A0 - 1j*A1B1[0])/(k_L_i**2*d**2)
#        
#        Omega[:,iterator] = abs(k_T_i*d/(m.pi*2))
#        Omega_T[:,iterator] = abs(k_L_i*R_0)
#        
#    f_res = (3*c_T_i*m.sqrt(8*1/rho_ratio - 5))/(4*m.pi*R_0*(2*1/rho_ratio + 1))
#    KPA_res = R_0*f_res*2*m.pi/c_L_i   
#        
#    R_omegastore[N,:] = R
#    T_omegastore[N,:] = T
#    #R0_omegastore[N,:] = R0
#    #T0_omegastore[N,:] = T0
#    
#params = {'legend.fontsize': 'x-large',
#          'figure.figsize': (25, 20),
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#         'xtick.labelsize':'x-large',
#         'ytick.labelsize':'x-large'}
#pylab.rcParams.update(params)
#
#plt.figure(3)
#for ii in range(Nn):
#    #plt.suptitle("Transmission and reflection coefficients due to Longitudinal wave for " + Flag +" radius of particle = 0.2mm", fontsize=50)
#    plt.subplot(211)
#    plt.suptitle("Transmission and reflection, $d = R*(1+\\beta)$", fontsize=40)
#    plt.plot(abs(f/f_res),abs(T_omegastore[ii,:]),label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
#    plt.tick_params(labelsize=40)
#    #plt.xlim(0,2.1)
#    plt.ylabel("T($\omega$)", fontsize=60)
#    plt.xlabel("$f/f_{res}$", fontsize=60)
#    plt.legend( fontsize=40, loc = 'best')
#    plt.subplot(212)
#    plt.plot(abs(f/f_res) ,abs(R_omegastore[ii,:]),label=str(Beta[ii])+ "$ = \\beta $", linewidth=3.0)
#    plt.tick_params(labelsize=40)
#    #plt.xlim(0,2.4)
#    plt.xlabel("$f/f_{res}$", fontsize=60)
#    plt.ylabel("R($\omega$)", fontsize=60)
#    plt.legend( fontsize=40, loc = 'best')
#    
##plt.figure(5)
##for ii in range(Nn):
##    plt.plot(f/f_res, ((abs(T_omegastore[ii,:]))**2 + (abs(R_omegastore[ii,:]))**2 ),label=str(1/4*ii)+"$\lambda_{res}$", linewidth=3.0)
##    plt.ylabel("$E_{Loss}$", fontsize=60)
##    plt.xlabel("$f/f_{res}$", fontsize=60)
##    plt.legend( fontsize=40, loc = 'best')
###plt.figure(4)
##for ii in range(1,Nn):
##    #plt.suptitle("Transmission and reflection coefficients due to Longitudinal wave for " + Flag +" radius of particle = 0.2mm", fontsize=50)
##    plt.subplot(211)
##    plt.plot(Omega_T[0,:] ,abs(T0_omegastore[ii,:]),label=str(5*ii/10)+"$\lambda_{res}$", linewidth=3.0)
##    plt.tick_params(labelsize=40)
##    #plt.xlim(0,2.1)
##    plt.ylabel("T0($\omega$)", fontsize=60)
##    plt.xlabel("$k_pa$", fontsize=60)
##    plt.legend( fontsize=40, loc = 'best')
##    plt.subplot(212)
##    plt.plot( Omega[0,:] ,abs(R0_omegastore[ii,:]), label=str(5*ii/10)+"$\lambda_{res}$", linewidth=3.0)
##    plt.tick_params(labelsize=40)
##    #plt.xlim(0,2.4)
##    plt.xlabel("$\Omega$", fontsize=60)
##    plt.ylabel("R0($\omega$)", fontsize=60)
##    plt.legend( fontsize=40, loc = 'best')
#
##
#plt.figure(4)
#plt.subplot(211)
#plt.suptitle("Lattice sum S_1s due to Longitudinal wave", fontsize=40)
#plt.plot(Omega[0,:], abs(S_1s_store[0,:]), linewidth=3.0)
#plt.axvline(x=1.0)
#plt.axvline(x=m.sqrt(2))
#plt.axvline(x=2.0)
#plt.axvline(x=m.sqrt(5))
#plt.axvline(x=m.sqrt(8))
#plt.subplot(212)
#plt.plot(Omega[0,:], S_1s_phase_store[0,:], linewidth=3.0)
#









