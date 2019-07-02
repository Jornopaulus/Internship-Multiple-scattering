# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

    return sp.spherical_jn(n,z)+1j*sp.spherical_yn(n,z)

##ambient
rho = 1100
Nf = 1000
f = np.linspace(5.0e4,5.0e6,Nf)

A0 = np.zeros((2,Nf),dtype=complex)
A1 = np.zeros((4,Nf),dtype=complex)
T_LL_ri = np.zeros((1,Nf),dtype=complex)
T_LT_ri = np.zeros((1,Nf),dtype=complex)
A0_an_L_s = np.zeros((1,Nf),dtype=complex)
A1_an_L_s = np.zeros((1,Nf),dtype=complex)
A1_an_T_s = np.zeros((1,Nf),dtype=complex)
A1_an2_L_s = np.zeros((1,Nf),dtype=complex)
#A = np.zeros((4,Nf),dtype=complex)
KKTp = np.zeros((1,Nf))
KKLp = np.zeros((1,Nf))
KKTi = np.zeros((1,Nf),dtype=complex)
KKLi = np.zeros((1,Nf),dtype=complex)
R_omega = np.zeros((1,Nf),dtype=complex)
T_omega = np.zeros((1,Nf),dtype=complex)
omegastore = np.zeros((1,Nf))
Omega_L = np.zeros((1,Nf))
Omega_T = np.zeros((1,Nf))
S_1s_store = np.zeros((1,Nf))
S_1s_phase_store = np.zeros((1,Nf))
T0_L = np.zeros((2,Nf),dtype=complex)
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
    omegastore[:,iterator] = omega
    c_0 = 1500 #m/s

    lamda = lame1
    mhu = lame2
    
    R_0 = 200e-6  #"m"

    K_L = k_L_i * R_0
    K_T = k_T_i * R_0
    KKTi[:,iterator] = K_T
    KKLi[:,iterator] = K_L
    
    lamda_Lead = 26.76923e9 #Pa  26.76923_48.54545
    rho_Lead = 11310 #kg/m^3 11310_11390
    mhu_Lead = 4.0e9 #Pa    4_6   
    
    
    
    rho_0 = 1100 #
    lamda_p = 162.0e9#Pa Al  #lamda_thung = 162.0e9#Pa Al
    lamda_Lead = 26.76923e9 #Pa  26.76923_48.54545
    rho_Lead = 11310 #kg/m^3 11310_11390
    mhu_Lead = 4.0e9 #Pa    4_6
    
    rho_p = 15800.0 #g/cm^3 rho_thung = 15250.0 #g/cm^3
    mhu_p = 243.0e9 #Pa mhu_thung = 243.0e9 #Pa
    rho_ratio = rho_p/rho_0
    #rho_ratio = 0.16
    mhu_ratio = mhu_p/mhu
    c_L_p = m.sqrt((lamda_p + 2*mhu_p)/rho_p)
    c_T_p = m.sqrt(mhu_p/rho_p)
    k_L_p = omega/c_L_p
    k_T_p = omega/c_T_p
    K_L_p = k_L_p * R_0
    K_T_p = k_T_p * R_0

#Loop over two values of n
    
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
        
        W_S   = 1j*(K_T*hnT_n)
        W_S_p = 1j*(K_T_p*jnT_np)
        W_I   = 1j*(K_T*jnT_n)
        
        
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
        
        Sig_S_p = 1j*mhu_p*K_T_p*((n-1)*jnT_np- K_T_p*jnT_1np)
        Sig_S   = 1j*mhu*K_T*((n-1)*hnT_n - K_T*hnT_1n)
        Sig_I   = 1j*mhu*K_T*((n-1)*jnT_n - K_T*jnT_1n)
        
        if n == 0:
            #elastic
            LHS = np.array([[U_L , -U_L_p],
                           [SS_L, -SS_L_p]])
            RHS = np.array([U_I , SS_I] )
        
            a0 = np.linalg.solve(LHS, -RHS)
            A0[:,iterator] = a0
            
            #moving rigid
            T0_L[:,iterator] = -U_I/U_L
            
            
        elif n == 1:
            #elastic
            LHS1 = np.array([[U_L , U_T , -U_L_p , -U_T_p],
                            [V_L , V_T , -V_L_p , -V_T_p],
                            [SS_L , SS_T , -SS_L_p , -SS_T_p],
                            [Tau_L , Tau_T , -Tau_L_p , -Tau_T_p]])
            
            RHS1 = np.array([U_I , V_I , SS_I , Tau_I] )
            
            a1 = np.linalg.solve(LHS1, -RHS1)
            
            A1[:,iterator] = a1
            
            
            #Moving rigid
            
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
            
            T_LL_ri[:,iterator] = solution[0]
            T_LT_ri[:,iterator] = solution[1]
            

    ## Analytical solution
    # Bulk modulus of particle
    
    #K_p = lamda_p + 2/3*mhu_p
    K_p = 72.0e9
    #A0_an_s_L
    complexconst = (K_L**3)/3*(lamda + 2/3*mhu - lamda_p - 2/3*mhu_p)/(lamda_p + 2/3*mhu_p + 4/3*mhu)
    a0_an_L_s = 1j*complexconst
    A0_an_L_s[:,iterator] = a0_an_L_s
    
    # A1_an_s_L
    g1 = K_T_p**3 - 6*K_T_p + 3*(2-K_T_p**2)*m.tan(K_T_p)
    g2 = K_T_p**2*m.tan(K_T_p) + 3*K_T_p - 3*m.tan(K_T_p)
    
 
    denomerator1 = 1j*((1-rho_ratio)*K_L**3)*(mhu_ratio*g1*(-3*K_T +1j*(K_T**2-3)) + g2*(K_T**3 - 6*K_T + 1j*( 3*K_T**2 - 6)))
    numerator1 = 3*mhu_ratio*g1*(1j*(K_T**2*(2*rho_ratio + 1)) - 9*K_T -  1j*9) + 3*g2*((2*rho_ratio + 1)*(K_T**3 + 1j*3*K_T**2)-18*(K_T +1j))
    
    a1_an_L_s = denomerator1/numerator1
    A1_an_L_s[:,iterator] = a1_an_L_s
    
    #A1_an_L_s different
    
    a1_an2_L_s = (complex(0,K_L**3))/9*(rho_ratio - 1)*complex(3*(K_T)**2 - 9, 9*K_T)/((2*rho_ratio + 1)*K_T**2 + complex(-9, 9*K_T))
    A1_an2_L_s[: , iterator] = a1_an2_L_s
    #B = 6*m.pi*
    numerator2 = mhu_ratio*g1*(complex(0,K_T**2*(2*rho_ratio + 1)) - 9*complex(K_T, 1)) + g2*((2*rho_ratio + 1)*complex(K_T**3, 3*K_T**2)-18*complex(K_T, 1))
    denomerator2 = K_L*K_T**2*cm.exp(1j*(-K_T))*(1-rho_ratio)*(mhu_ratio*g1 + 2*g2)
    a1_an_T_s = denomerator2/numerator2
    A1_an_T_s[:,iterator] = a1_an_T_s


# resonance frequency

f_res = (3*c_T_i*m.sqrt(8*rho_ratio))/(4*m.pi*R_0*(2*rho_ratio + 1))
omega_res = 2*m.pi*f_res
K_L_i_res = R_0*omega_res/c_L_i
K_T_i_res = R_0*omega_res/c_T_i

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.figure(1)
plt.suptitle("Scattering coefficients due to longitudinal wave (Tungsten Carbine)", fontsize=40)
plt.subplot(311)
#plt.axvline(x=abs(K_L_i_res),linewidth=3.0)
#plt.ylim(0,1.2)
plt.tick_params(labelsize=40)
#plt.axvline(x=K_L_i_res,linewidth = 3.0)
#plt.axvline(x=K_T_i_res)
plt.plot(abs(f/f_res) , abs(A0[0,:]), linewidth=3.0)
#plt.plot(abs(KKLi[0,:]) , abs(T0_L[0,:]), linewidth=3.0)
#plt.plot(abs(KKLi[0,:])  , abs(A0_an_L_s[0,:]), linewidth=3.0)
plt.legend(("Numerical" , "Analytical"), fontsize=30,loc='best')
#plt.xlabel("Non-dimensional wavenumber")
#plt.legend(("Resonance freq" , "Numerical" , "Analytical1", "Analytical2"), fontsize = 20 ,loc='best')
plt.ylabel("$T_0^{LL}$", fontsize=60)
plt.grid()
plt.subplot(312)
plt.tick_params(labelsize=40)
plt.axvline(x=abs(K_L_i_res),linewidth=3.0)
plt.ylim(0,0.4)
plt.plot(abs(KKLi[0,:]) , abs(A1[0,:]), linewidth=3.0)
#plt.plot(abs(KKLi[0,:]) , abs(T_LL_ri[0,:]), linewidth=3.0)
plt.plot(abs(KKLi[0,:]) , abs(A1_an_L_s[0,:]), linewidth=3.0)
#plt.plot(abs(KKLi[0,:]) , abs(A1_an2_L_s[0,:]), linewidth=3.0)
plt.legend(("$K_{L-res}$" , "Numerical" , "Analytical"), fontsize = 30 ,loc='best')
plt.ylabel("$T_1^{LL}$", fontsize=60)
plt.grid()
plt.subplot(313)
plt.tick_params(labelsize=40)
#plt.ylim(0,1.2)
plt.axvline(x=abs(K_L_i_res),linewidth=3.0)
#plt.axvline(x=K_T_i_res)
plt.xlabel("$K_L$", fontsize=60)
plt.plot(abs(KKLi[0,:]) , abs(A1[1,:]), linewidth=3.0)
#plt.plot(abs(KKLi[0,:]) , abs(T_LT_ri[0,:]), linewidth=3.0)
#plt.plot(abs(KKLi[0,:])  , abs(A1_an_T_s[0,:]), linewidth=3.0)
plt.legend(("$K_{L-res}$", "Numerical"), fontsize = 30 ,loc='best')
plt.ylabel("$T_1^{LT}$", fontsize=60)
plt.grid()
plt.savefig('LongitudinalwaveScatteringCoefVerifcationplot.jpg')







