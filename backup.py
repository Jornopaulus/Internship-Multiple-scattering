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
plt.close("all")

## Define spherical Hankel function
def spherical_hn1(n,z):

    return sp.spherical_jn(n,z,derivative=False)+1j*sp.spherical_yn(n,z,derivative=False)

##ambient
rho = 1200
Nf = 1000
f = np.linspace(5.0e4,5.0e6,Nf)

A0 = np.zeros((2,Nf),dtype=complex)
A1 = np.zeros((4,Nf),dtype=complex)
A0_an_L_s = np.zeros((1,Nf),dtype=complex)
A1_an_L_s = np.zeros((1,Nf),dtype=complex)
A1_an_T_s = np.zeros((1,Nf),dtype=complex)
A1_an2_L_s = np.zeros((1,Nf),dtype=complex)
#A = np.zeros((4,Nf),dtype=complex)
KKTp = np.zeros((1,Nf))
KKLp = np.zeros((1,Nf))
KKTi = np.zeros((1,Nf),dtype=complex)
KKLi = np.zeros((1,Nf),dtype=complex)

# Main loop over frequency domain
for iterator in range(0,Nf):
    
    # Create some constants for particle(sphere) and ambient
    omega = 2*m.pi*f[iterator]
    c_0 = 1500 #m/s
#    n_v = 0.024 #Pa 
#    n_s = 0.001    # Pa
#    
#    b = -omega*(n_v - 2/3*n_v)
#    c = rho*c_0**2
#    lamda_i =  complex(c,b)
#    mhu_i = complex(0,-omega*n_s)
    lamda_i = 3.5e9
    mhu_i = 1.0e9
    #eta_s = mhu_i/(-1j*omega)
    c_L_i = cm.sqrt((lamda_i + 2*mhu_i)/rho)
    c_T_i = cm.sqrt(mhu_i/rho)
    c_L_i = c_L_i.real
    c_T_i = c_T_i.real
    R_0 = 10**-4  #"m"
    k_L_i = omega/c_L_i
    k_T_i = omega/c_T_i
    K_L_i = k_L_i * R_0
    K_T_i = k_T_i * R_0
    KKTi[:,iterator] = K_T_i
    KKLi[:,iterator] = K_L_i
    
    
    
    rho_0 = 1200  #
    lamda_p = 55.0e9#Pa Al
    rho_p = 15000.0 #g/cm^3
    mhu_p = 25.0e9 #Pa
    rho_ratio = rho_p/rho_0
    mhu_ratio = mhu_p/mhu_i
    c_L_p = m.sqrt((lamda_p + 2*mhu_p)/rho_p)
    c_T_p = m.sqrt(mhu_p/rho_p)
    k_L_p = omega/c_L_p
    k_T_p = omega/c_T_p
    K_L_p = k_L_p * R_0
    K_T_p = k_T_p * R_0
    KKTi[:,iterator] = K_T_i
    KKLi[:,iterator] = K_L_i
#Loop over two values of n
    
    for n in range(0,2):
        U_L_p = n*sp.spherical_jn(n,K_L_p) - K_L_p*sp.spherical_jn((n+1),K_L_p)
        U_T_p = n*(n+1)*sp.spherical_jn(0, K_T_p)
        V_L_p = sp.spherical_jn(n,K_L_p)
        V_T_p = (1+n)*sp.spherical_jn(n,K_T_p) - K_T_p*sp.spherical_jn((n+1),K_T_p)
        SS_L_p = (2*n*(n-1)*mhu_p - (lamda_p + 2*mhu_p)*K_L_p**2)*sp.spherical_jn(n,K_L_p) + 4*mhu_p*K_L_p*sp.spherical_jn((n+1),K_L_p)
        SS_T_p = 2*n*(n**2 - 1)*mhu_p*sp.spherical_jn(n,K_T_p) - 2*n*(n+1)*mhu_p*K_T_p*sp.spherical_jn(n+1,K_T_p)
        T_L_p = 2*mhu_p*((n-1)*sp.spherical_jn(n,K_L_p) - K_L_p*sp.spherical_jn(n+1,K_L_p))
        T_T_p = mhu_p*(2*(n**2-1) - K_T_p**2)*sp.spherical_jn(n,K_T_p) + 2*mhu_p*K_T_p*sp.spherical_jn(n+1,K_T_p)
        
        
        
        U_L_i = n*sp.spherical_jn(n,K_L_i) - K_L_i*sp.spherical_jn((n+1),K_L_i)
        V_L_i = sp.spherical_jn(n,K_L_i)
        SS_L_i = (2*n*(n-1)*mhu_i - (lamda_i + 2*mhu_i)*K_L_i**2)*sp.spherical_jn(n,K_L_i) + 4*mhu_i*K_L_i*sp.spherical_jn((n+1),K_L_i)
        T_L_i = 2*mhu_i*((n-1)*sp.spherical_jn(n,K_L_i) - K_L_i*sp.spherical_jn(n+1,K_L_i))
        
        U_L_s = n*spherical_hn1(n,K_L_i) - K_L_i*spherical_hn1((n+1),K_L_i)
        U_T_s = n*(n+1)*spherical_hn1(0, K_T_i)
        V_L_s = spherical_hn1(n,K_L_i)
        V_T_s = (1+n)*spherical_hn1(n,K_T_i) - K_T_i*spherical_hn1((n+1),K_T_i)
        SS_L_s = (2*n*(n-1)*mhu_i - (lamda_i + 2*mhu_i)*K_L_i**2)*spherical_hn1(n,K_L_i) + 4*mhu_i*K_L_i*spherical_hn1((n+1),K_L_i)
        SS_T_s = 2*n*(n**2 - 1)*mhu_i*spherical_hn1(n,K_T_i) - 2*n*(n+1)*mhu_i*K_T_i*spherical_hn1(n+1,K_T_i)
        T_L_s = 2*mhu_i*((n-1)*spherical_hn1(n,K_L_i) - K_L_i*spherical_hn1(n+1,K_L_i))
        T_T_s = mhu_i*(2*(n**2-1) - K_T_i**2)*spherical_hn1(n,K_T_i) + 2*mhu_i*K_T_i*spherical_hn1(n+1,K_T_i)
        
        
        if n == 0:
            LHS = np.array([[U_L_s , -U_L_p],
                       [SS_L_s, SS_L_p]])
            RHS = np.array([U_L_i , SS_L_i] )
        
            a0 = np.linalg.solve(LHS, RHS)
            A0[:,iterator] = a0
        elif n == 1:
            LHS1 = np.array([[U_L_s , U_T_s , -U_L_p , -U_T_p],
                            [V_L_s , V_T_s , -V_L_p , -V_T_p],
                            [SS_L_s , SS_T_s , -SS_L_p , -SS_T_p],
                            [T_L_s , T_T_s , -T_L_p , -T_T_s]])
            RHS1 = np.array([U_L_i , V_L_i , SS_L_i , T_L_i] )
            
            a1 = np.linalg.solve(LHS1, RHS1)
            
            A1[:,iterator] = a1
    
    ## Analytical solution
    # Bulk modulus of particle
    
    #K_p = lamda_p + 2/3*mhu_p
    K_p = 72.0e9
    #A0_an_s_L
    complexconst = (K_L_i**3)/3*(lamda_i + 2/3*mhu_i - lamda_p - 2/3*mhu_p)/(lamda_p + 2/3*mhu_p + 4/3*mhu_i)
    a0_an_L_s = complex(0,complexconst)
    A0_an_L_s[:,iterator] = a0_an_L_s
    
    # A1_an_s_L
    g1 = K_T_p**3 - 6*K_T_p + 3*(2-K_T_p**2)*m.tan(K_T_p)
    g2 = K_T_p**2*m.tan(K_T_p) + 3*K_T_p - 3*m.tan(K_T_p)
    
 
    denomerator1 = complex(0,((1-rho_ratio)*K_L_i**3))*(mhu_ratio*g1*complex(-3*K_T_i,(K_T_i**2-3)) + g2*complex(K_T_i**3 - 6*K_T_i , 3*K_T_i**2 - 6)
   )
    numerator1 = 3*mhu_ratio*g1*(complex(0,K_T_i**2*(2*rho_ratio + 1)) - 9*complex(K_T_i, 1)) + 3*g2*((2*rho_ratio + 1)*complex(K_T_i**3, 3*K_T_i**2)-18*complex(K_T_i, 1))
    
    a1_an_L_s = denomerator1/numerator1
    A1_an_L_s[:,iterator] = a1_an_L_s
    
    #A1_an_L_s different
    
    a1_an2_L_s = (complex(0,K_L_i)**3)/9*(rho_ratio - 1)*complex(3*(K_T_i)**2 - 9, 9*K_T_i)/((2*rho_ratio + 1)*K_T_i**2 + complex(-9, 9*K_T_i))
    A1_an2_L_s[: , iterator] = a1_an2_L_s
    #B = 6*m.pi*
    numerator2 = mhu_ratio*g1*(complex(0,K_T_i**2*(2*rho_ratio + 1)) - 9*complex(K_T_i, 1)) + g2*((2*rho_ratio + 1)*complex(K_T_i**3, 3*K_T_i**2)-18*complex(K_T_i, 1))
    denomerator2 = K_L_i*K_T_i**2*cm.exp(1j*(-K_T_i))*(1-rho_ratio)*(mhu_ratio*g1 + 2*g2)
    a1_an_T_s = denomerator2/numerator2
    A1_an_T_s[:,iterator] = a1_an_T_s


# resonance frequency

f_res = (3*c_T_i*m.sqrt(8*rho_ratio))/(4*m.pi*R_0*(2*rho_ratio + 1))
omega_res = 2*m.pi*f_res
K_L_i_res = R_0*omega_res/c_L_i
K_T_i_res = R_0*omega_res/c_T_i

#Plot results, use absolute values for complex numbers
#plt.figure(1)
#plt.rcParams.update({'font.size': 22})
#plt.figure(figsize=(15,10))
#plt.subplot(211)
#plt.xscale('log')
#plt.plot(KKLi[0,:] , abs(A0[0,:]), linewidth=3.0)
#plt.plot(KKLi[0,:] , abs(A0_an_L_s[0,:]), linewidth=3.0)
#plt.legend(("Numerical" , "Analytical"),loc='upper left')
#plt.title("A0_L_s logx")
#plt.grid()
#plt.subplot(212)
#plt.xscale('log')
#plt.plot(f , abs(A0[1,:]), linewidth=3.0)
#plt.grid()
#
#
#plt.figure(2)
#plt.figure(figsize=(15,20))
#plt.subplot(411)
#plt.xscale('log')
#plt.plot(f , abs(A1[0,:]), linewidth=3.0)
#plt.plot(f , abs(A1_an_L_s[0,:]), linewidth=3.0)
#plt.legend(("Numerical" , "Analytical"),loc='upper left')
#plt.title("A1_L_s logx")
#plt.grid()
#plt.subplot(412)
#plt.xscale('log')
#plt.plot(f , abs(A1[1,:]), linewidth=3.0)
#plt.plot(f , abs(A1_an_T_s[0,:]), linewidth=3.0)
#plt.legend(("Numerical" , "Analytical"),loc='upper left')
#plt.title("A1_T_s logx")
#plt.grid()
#plt.subplot(413)
#plt.xscale('log')
#plt.plot(f , abs(A1[2,:]), linewidth=3.0)
#plt.grid()
#plt.subplot(414)
#plt.xscale('log')
#plt.plot(f , abs(A1[3,:]), linewidth=3.0)
#plt.grid()


plt.figure(3)
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(15,10))
plt.axvline(x=K_L_i_res,linewidth = 3.0)
#plt.axvline(x=K_T_i_res)
plt.plot(KKLi[0,:]  , abs(A0[0,:]), linewidth=3.0)
plt.plot(KKLi[0,:]  , abs(A0_an_L_s[0,:]), linewidth=3.0)
plt.legend(("Resonance freq" , "Numerical" , "Analytical"),loc='upper left')
plt.xlabel("Non-dimensional wavenumber")
plt.title("A0_L_s linx")
plt.grid()



plt.figure(4)
plt.figure(figsize=(15,20))
plt.subplot(211)
plt.axvline(x=K_L_i_res,linewidth=3.0)
#plt.axvline(x=K_T_i_res)
plt.plot(KKLi[0,:] , abs(A1[0,:]), linewidth=3.0)
plt.plot(KKLi[0,:]  , abs(A1_an_L_s[0,:]), linewidth=3.0)
plt.plot(KKLi[0,:]  , abs(A1_an2_L_s[0,:]), linewidth=3.0)
plt.xlabel("Non-dimensional wavenumber")
plt.legend(("Resonance freq" , "Numerical" , "Analytical1", "Analytical2"),loc='upper left')
plt.title("A1_L_s linx")
plt.grid()
plt.subplot(212)
plt.axvline(x=K_L_i_res,linewidth=3.0)
#plt.axvline(x=K_T_i_res)
plt.xlabel("Non-dimensional wavenumber")
plt.plot(KKLi[0,:] , abs(A1[1,:]), linewidth=3.0)
plt.plot(KKLi[0,:]  , abs(A1_an_T_s[0,:]), linewidth=3.0)
plt.legend(("Resonance freq" , "Numerical" , "Analytical"),loc='upper left')
plt.title("A1_T_s linx")
plt.grid()


