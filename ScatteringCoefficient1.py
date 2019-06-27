# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math as m
import scipy.special as sp
import numpy as np
import cmath
import matplotlib.pyplot as plt
##ambient
rho = 1
f = np.linspace(5.0e4,5.0e7,10000)

A0 = np.zeros((2,1))
A1 = np.zeros((4,1))


# Main loop over frequency domain
for i in range(0,10000):
    
    # Create some constants for particle(sphere) and ambient
    omega = 2*m.pi*f[i]
    c_0 = 1500 #m/s
    n_v = 0.024 #Pa 
    n_s = 0.001    # Pa
    
    b = -omega*(n_v - 2/3*n_v)
    c = rho*c_0**2
    lamda_i =  complex(c,b)
    mhu_i = complex(0,-omega*n_s)
    c_L_i = cmath.sqrt((lamda_i + 2*mhu_i)/rho)
    c_T_i = cmath.sqrt(mhu_i/rho)
    R_0 = 10**-6  #"m"
    k_L_i = omega/c_L_i
    k_T_i = omega/c_T_i
    K_L_i = k_L_i * R_0
    K_T_i = k_T_i * R_0
    
    
    
    
    rho_0 = 1000  #"m/s"
    lamda_p = 55e9#Pa
    rho_p = 2700.0 #g/cm^3
    mhu_p = 25.0e9 #Pa
    c_L_p = m.sqrt((lamda_p + 2*mhu_p)/rho_p)
    c_T_p = m.sqrt(mhu_p/rho_p)
    k_L_p = omega/c_L_p
    k_T_p = omega/c_T_p
    K_L_p = k_L_p * R_0
    K_T_p = k_T_p * R_0
    
#Loop over two values of n
    
    for n in range(0,2):
    
        U_L_p = n*sp.jv(n,K_L_p) - K_L_p*sp.jv((n+1),K_L_p)
        U_T_p = n*(n+1)*sp.jv(0, K_T_p)
        V_L_p = sp.jv(n,K_L_p)
        V_T_p = (1+n)*sp.jv(n,K_T_p) - K_T_p*sp.jv((n+1),K_T_p)
        SS_L_p = (2*n*(n-1)*mhu_p - (lamda_p + 2*mhu_p)*K_L_p**2)*sp.jv(n,K_L_p) + 4*mhu_p*K_L_p*sp.jv((n+1),K_L_p)
        SS_T_p = 2*n*(n**2 - 1)*mhu_p*sp.jv(n,K_T_p) - 2*n*(n+1)*mhu_p*K_T_p*sp.jv(n+1,K_T_p)
        T_L_p = 2*mhu_p*((n-1)*sp.jv(n,K_L_p) - K_L_p*sp.jv(n+1,K_L_p))
        T_T_p = mhu_p*(2*(n**2-1) - K_T_p**2)*sp.jv(n,K_T_p) + 2*mhu_p*K_T_p*sp.jv(n+1,K_T_p)
        
        
        
        U_L_i = n*sp.jv(n,K_L_i) - K_L_i*sp.jv((n+1),K_L_i)
        V_L_i = sp.jv(n,K_L_i)
        SS_L_i = (2*n*(n-1)*mhu_i - (lamda_i + 2*mhu_i)*K_L_i**2)*sp.jv(n,K_L_i) + 4*mhu_i*K_L_i*sp.jv((n+1),K_L_i)
        T_L_i = 2*mhu_i*((n-1)*sp.jv(n,K_L_i) - K_L_i*sp.jv(n+1,K_L_i))
        
        U_L_s = n*sp.hankel1(n,K_L_i) - K_L_i*sp.hankel1((n+1),K_L_i)
        U_T_s = n*(n+1)*sp.hankel1(0, K_T_i)
        V_L_s = sp.hankel1(n,K_L_i)
        V_T_s = (1+n)*sp.hankel1(n,K_T_i) - K_T_i*sp.hankel1((n+1),K_T_i)
        SS_L_s = (2*n*(n-1)*mhu_i - (lamda_i + 2*mhu_i)*K_L_i**2)*sp.hankel1(n,K_L_i) + 4*mhu_i*K_L_i*sp.hankel1((n+1),K_L_i)
        SS_T_s = 2*n*(n**2 - 1)*mhu_i*sp.hankel1(n,K_T_i) - 2*n*(n+1)*mhu_i*K_T_i*sp.hankel1(n+1,K_T_i)
        T_L_s = 2*mhu_i*((n-1)*sp.hankel1(n,K_L_i) - K_L_i*sp.hankel1(n+1,K_L_i))
        T_T_s = mhu_i*(2*(n**2-1) - K_T_i**2)*sp.hankel1(n,K_T_i) + 2*mhu_i*K_T_i*sp.hankel1(n+1,K_T_i)
        
        
        if n == 0:
            LHS = np.array([[U_L_s , -U_L_p],
                       [SS_L_s, SS_L_p]])
            RHS = np.array([U_L_i , SS_L_i] )
        
            a0 = np.linalg.solve(LHS, RHS)
            
        elif n == 1:
            LHS1 = np.array([[U_L_s , U_T_s , -U_L_p , -U_T_p],
                            [V_L_s , V_T_s , -V_L_p , -V_T_p],
                            [SS_L_s , SS_T_s , -SS_L_p , -SS_T_p],
                            [T_L_s , T_T_s , -T_L_p , -T_T_s]])
            RHS1 = np.array([U_L_i , V_L_i , SS_L_i , T_L_i] )
            
            a1 = np.linalg.solve(LHS1, RHS1)
            
    #Save results for each frequency in array
    A0 = np.column_stack((A0,a0))
    A1 = np.column_stack((A1,a1))

#Plot results, use absolute values for complex numbers
plt.figure(figsize=(16, 14)) 
plt.scatter(f,1e4*abs(A0[0:3, 1:10001]))
plt.title('Amplitudes over frequency range')
plt.show(block=True)

