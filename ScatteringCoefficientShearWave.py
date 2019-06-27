# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:12:35 2019

@author: Jorn
"""

import math as m
import scipy.special as sp
import numpy as np
import cmath as cm
import math as m
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.close("all")
## Define spherical Hankel function
def spherical_hn1(n,z):

    return sp.spherical_jn(n,z,derivative=False)+1j*sp.spherical_yn(n,z,derivative=False)

##ambient
rho = 1100
Nf = 1000
f = np.linspace(1.0e5,3.0e6,Nf)
T_SS = np.zeros((2,Nf),dtype=complex)
T1 = np.zeros((4,Nf),dtype=complex)
T1_TL_ana = np.zeros((1,Nf),dtype=complex)
T1_TT_ana = np.zeros((1,Nf),dtype=complex)
T1_SS_ana = np.zeros((1,Nf),dtype=complex)
A1_an2_L_s = np.zeros((1,Nf),dtype=complex)
#A = np.zeros((4,Nf),dtype=complex)
KKTp = np.zeros((1,Nf),dtype=complex)
KKLp = np.zeros((1,Nf),dtype=complex)
KKTi = np.zeros((1,Nf),dtype=complex)
KKLi = np.zeros((1,Nf),dtype=complex)
F = np.zeros((1,Nf),dtype=complex)

# Main loop over frequency domain
for iterator in range(0,Nf):
    
    # Create some constants for particle(sphere) and ambient
    
    omega = 2*m.pi*f[iterator]
    lame1 = 3.6e9 + 0.1e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.3e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) +0.25e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2) + 0.3e9*((omega*8e-9)**2 - 1j*omega*8e-9)/(1+(omega*8e-9)**2)
    lame2 = 1.05e9 + 0.2e9*((omega*5000e-9)**2 - 1j*omega*5000e-9)/(1+(omega*5000e-9)**2) + 0.15e9*((omega*500e-9)**2 - 1j*omega*500e-9)/(1+(omega*500e-9)**2) - 0.1e9*((omega*14e-9)**2 - 1j*omega*14e-9)/(1+(omega*14e-9)**2) + 0.15e9*((omega*50e-9)**2 - 1j*omega*50e-9)/(1+(omega*50e-9)**2)

    c_L_i = cm.sqrt((lame1+2*lame2)/rho)
    c_T_i = cm.sqrt(lame2/rho)
    
    alphaL_f = 0.1 + 0.25/2000000*f[iterator]
    alphaS_f = 0.05 + 0.1/2000000*f[iterator]

    
    k_L_i = omega/c_L_i + 1j*alphaL_f
    k_T_i = omega/c_T_i + 1j*alphaS_f    
    lamda = lame1
    mhu = lame2
    #eta_s = mhu_i/(-1j*omega)

    R_0 = 200e-6  #"m"

    K_L = k_L_i * R_0
    K_T = k_T_i * R_0
    KKTi[:,iterator] = K_T
    KKLi[:,iterator] = K_L
    
    rho_p = 15800.0 #g/cm^3 rho_thung = 15250.0 #g/cm^3
    mhu_p = 243.0e9 #Pa mhu_thung = 243.0e9 #Pa
    lamda_p = 162.0e9#Pa Al
    
    rho_ratio = rho_p/rho
    mhu_ratio = mhu_p/mhu
    c_L_p = m.sqrt((lamda_p + 2*mhu_p)/rho_p)
    c_T_p = m.sqrt(mhu_p/rho_p)
    k_L_p = omega*m.sqrt(rho_p/(lamda_p + 2*mhu_p))
    k_T_p = omega*m.sqrt(rho_p/mhu_p)
    K_L_p = k_L_p * R_0
    K_T_p = k_T_p * R_0
#Loop over two values of n
    
    for n in range(1,2):
        
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
        
        #Build system of equations elastic sphere
        Designmatrix_LHS = np.array([[U_L , U_T , -U_L_p , -U_T_p],
                                    [V_L , V_T , -V_L_p , -V_T_p],
                                    [SS_L , SS_T , -SS_L_p , -SS_T_p],
                                    [Tau_L , Tau_T , -Tau_L_p , -Tau_T_p]])
        Designmatrix_RHS = np.array([U_I , V_I , SS_I , Tau_I])
        
        t1 = np.linalg.solve(Designmatrix_LHS, -Designmatrix_RHS)
        T1[:,iterator] = t1  
        
        Designmatrix2_LHS = np.array([[W_S , -W_S_p],
                                      [Sig_S , -Sig_S_p]])
        Designmatrix2_RHS = np.array([W_I , Sig_I ])
        t2 = np.linalg.solve(Designmatrix2_LHS, -Designmatrix2_RHS)
        T_SS[:,iterator] = t2
     
        



    #Check
    
    t1_TL_ana = 1j*(-K_L**2*K_T*cm.cos(K_T))*((2*rho_ratio - 1)*(cm.tan(K_T)+1j))/(1j*K_T**2*(2*rho_ratio + 1) - 9*K_T - 1j*9)
    T1_TL_ana[:,iterator] = t1_TL_ana
    t1_TT_ana = cm.cos(K_T)*cm.exp(1j*(-K_T))*(((2*rho_ratio+1)*K_T**2 - 9)*cm.tan(K_T) + 9*K_T)/(1j*K_T**2*(2*rho_ratio + 1) - 9*K_T - 1j*9)
    T1_TT_ana[:,iterator] = t1_TT_ana
    t1_SS_ana = -cm.cos(K_T)*cm.exp(1j*(-K_T))*(rho_ratio*K_T**3 - (5 + rho_ratio)*K_T**2*cm.tan(K_T) + 15*(cm.tan(K_T) - K_T))/(rho_ratio*K_T**3 + 1j*K_T**2*(rho_ratio + 5) - 15*K_T -1j*15)
    T1_SS_ana[:,iterator] = t1_SS_ana 
    
    
    
def func(KTres):
    return 1j*KTres**2*(2*rho_ratio + 1) - 9*KTres- 1j*9 

def derfunc(KTres):
    return 2j*KTres*(2*rho_ratio + 1) - 9


def func2(KT2res):
    return rho_ratio*KT2res**3 + 1j*KT2res**2*(rho_ratio + 5) - 15*KT2res -1j*15

def derfunc2(KT2res):
    return rho_ratio*3*KT2res**2 + 2j*KT2res*(rho_ratio + 5) - 15

F = 10000
F2 = 10000
KTres = 10000 + 40j
KT2res = 10000 + 40j

while abs(F) >0.001:
    F = func(KTres)
    DF = derfunc(KTres)
    KTres = KTres - F/DF
    print(abs(KTres))
#print(abs(KTres))

while abs(F2) >0.001:
    F2 = func2(KT2res)
    DF2 = derfunc2(KT2res)
    KT2res = KT2res - F2/DF2
print(abs(KT2res))



    
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.figure(1)
#plt.figure(figsize=(20,35))
plt.suptitle("Scattering coefficients due to shear wave (Tungsten Carbine)", fontsize=40)
plt.subplot(311)
#plt.axvline(x=abs(K_L_i_res),linewidth=3.0)
plt.axvline(x=abs(KTres))
plt.plot(abs(KKTi[0,:]) , abs(T1[0,:]), linewidth=3.0)
plt.plot(abs(KKTi[0,:]) , abs(T1_TL_ana[0,:]), linewidth=3.0)
plt.tick_params(labelsize=40)
#plt.plot(KKLi[0,:]  , abs(T1_an_L_s[0,:]), linewidth=3.0)
#plt.plot(KKLi[0,:]  , abs(T1_an2_L_s[0,:]), linewidth=3.0)
#plt.xlabel("Non-dimensional wavenumber")
plt.ylabel("$T_1^{TL}$", fontsize=60)
plt.legend(( "$K_{T-res}$" ,"Numerical" , "Analytical (rigid)"), fontsize=30,loc='best')
#plt.title("T1_TL")
plt.grid()
plt.subplot(312)
plt.tick_params(labelsize=40)
plt.axvline(x=abs(KTres))
plt.ylabel("$T_1^{TT}$", fontsize=60)
#plt.xlabel("Non-dimensional wavenumber")
plt.plot(abs(KKTi[0,:] ), abs(T1[1,:]), linewidth=3.0)
plt.plot(abs(KKTi[0,:] ) , abs(T1_TT_ana[0,:]), linewidth=3.0)
#plt.plot(KKLi[0,:]  , abs(A1_an_T_s[0,:]), linewidth=3.0)
plt.legend(("$K_{T-res}$" , "Numerical" , "Analytical (rigid)"), fontsize=30,loc='best')
#plt.title("T1_TT ")
plt.grid()
plt.subplot(313)
plt.tick_params(labelsize=40)
#plt.axvline(x=abs(K_L_i_res),linewidth=3.0)
plt.axvline(x=abs(KT2res))
plt.ylabel("$T_1^{SS}$", fontsize=60)
plt.xlabel("$K_L$", fontsize=60)
plt.plot(abs(KKTi[0,:]) , abs(T_SS[0,:]), linewidth=3.0)
plt.plot(abs(KKTi[0,:])  , abs(T1_SS_ana[0,:]), linewidth=3.0)
plt.legend(("$K_{S-res}$" ,"Numerical" , "Analytical (rigid)"), fontsize=30,loc='best')
#plt.title("T1_SS ")
plt.grid()
plt.savefig('ShearwaveScatteringCoef.jpg')





        
        
        
        
    