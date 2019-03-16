from numpy import *
import numpy as np
import time
from convolution import conv
import os
import sys
import plot1d as plt
from scipy import optimize

print('running renormalized ME')

# params
Nw = 100
Nk = 10
beta = 5.0
iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)

omega = 0.5

lamb = 0.1
alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)
idelta = 0.020j
dens = 1.0

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('g = %1.3f'%g)
print('lamb = %1.3f'%lamb)
print('lamb correct = %1.3f'%(2*g**2/(4.0*omega)))

folder = 'data1d/data_renormalized_Nk%d_lamb%1.1f_beta%1.1f/'%(Nk,lamb,beta)
if not os.path.exists(folder): os.mkdir(folder)

kxs = arange(-pi, pi, 2*pi/Nk)
def band(kxs):
    return -2.0*cos(kxs) 
ek = band(kxs)
dw = 0.001

assert Nw%2==0 and Nk%2==0

# estimate filling and dndmu at the desired filling
mu = optimize.fsolve(lambda mu : 2.0*sum(1.0/(exp(beta*(ek-mu))+1.0))/Nk-dens, 0.0)
deriv = lambda mu : 2.0*sum(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)/Nk
dndmu = deriv(mu)

print('mu optimized = %1.3f'%mu)
print('dndmu = %1.3f'%dndmu)

w = arange(-3.1, 3.1, dw)
nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

assert Nw%2==0 and Nk%2==0 and len(w)%2==0 and abs(w[len(w)//2])<1e-8

def compute_G(S, mu):
    return 1.0/(iwm[None,:] - (ek[:,None]-mu) - S)

def compute_GR(SR, mu, idelta):
    return 1.0/(w[None,:]+idelta - (ek[:,None]-mu) - SR)
    
def compute_D(PI, omega):
    return 1.0/(-((vn**2)[None,:] + omega**2)/(2.0*omega) - PI) 

DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
def compute_DR(PIR):
    return 1.0/(DRbareinv[None,:] - PIR) 

def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk) * np.sum(G).real

def compute_S(G, D):
    return -g**2/(beta * Nk) * conv(G, D, ['k-q,q','m,n-m'], [0,1], [True,False])[:,:Nw]

def compute_PI(G):
    return 2.0*g**2/(beta * Nk) * conv(G, G, ['k,k+q','m,m+n'], [0,1], [True,False])[:,:Nw+1]

def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag
    return -g**2*dw/Nk*(conv(B, Gsum, ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)] \
             -conv(B*(1+nB)[None,:], GR, ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)] \
             +conv(B, GR*nF[None,:], ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)])

def compute_PI_real_axis(GR, Gsum):
    GA = conj(GR)
    A  = -1.0/pi * GR.imag
    return 2.0*g**2*dw/Nk*(conv(A, Gsum, ['k+q,k','z,w-z'], [0,1], [True,False])[:,:len(w)] \
                    -conv(A, GA*nF[None,:], ['k+q,k','w+z,z'], [0,1], [True,False])[:,:len(w)] \
                    +conv(A*nF[None,:], GA, ['k+q,k','w+z,z'], [0,1], [True,False])[:,:len(w)])

S  = zeros([Nk,Nw], dtype=complex)
PI = zeros([Nk,Nw+1], dtype=complex)
G = compute_G(S, mu)
D = compute_D(PI, omega)

# solve Matsubara piece
print('\n Solving Matsubara piece')

#print('G', mean(abs(G.real)))
#print('G', mean(abs(G.imag)))
#print('D', mean(abs(D.real)))
#print('D', mean(abs(D.imag)))

def myp(x):
    print(mean(abs(x.real)), mean(abs(x.imag)))

change = [0, 0]
frac = 0.6
for i in range(200):
    S0  = S[:]
    PI0 = PI[:]
    
    S  = compute_S(G, D)
    PI = compute_PI(G)

    S  = frac*S + (1-frac)*S0
    PI = frac*PI + (1-frac)*PI0
    
    change[0] = mean(abs(S-S0))/mean(abs(S+S0))
    change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))

    G = compute_G(S, mu)
    D = compute_D(PI, omega)

    n = compute_fill(G)
    mu += 0.25*(n-dens)/dndmu
    
    if i%10==0: print('change = %1.3e, %1.3e, fill=%1.3f, mu=%1.3f'%(change[0], change[1], n, mu))
    
    if i>10 and change[0]<1e-15 and change[1]<1e-15: break

    
save(folder+'iwm', iwm)
save(folder+'w', w)
save(folder+'Nk', [Nk])
save(folder+'lamb', [lamb])
save(folder+'omega', [omega])
save(folder+'S', S)
save(folder+'PI', PI)
save(folder+'idelta', [abs(idelta)])

# analytic continuation
print('\nSolving real axis')

SR = zeros([Nk,len(w)], dtype=complex)
PIR = zeros([Nk,len(w)], dtype=complex)
GR = compute_GR(SR, mu, idelta)
DR = compute_DR(PIR)

del S

Gsum_plus  = zeros([Nk,len(w)], dtype=complex)
Gsum_minus = zeros([Nk,len(w)], dtype=complex)
for iw in range(len(w)):
    Gsum_plus[:,iw]  = np.sum(G/((w[iw]+iwm)[None,:]), axis=1) / beta
    Gsum_minus[:,iw] = np.sum(G/((w[iw]-iwm)[None,:]), axis=1) / beta

del G

print('finished Gsum')
    
change = [0,0]
frac = 0.6
for i in range(5):
    SR0 = SR[:]
    PIR0 = PIR[:]
        
    SR  = compute_S_real_axis(GR, DR, Gsum_minus)
    PIR = compute_PI_real_axis(GR, Gsum_plus)
    
    SR  = frac*SR  + (1.0-frac)*SR0
    PIR = frac*PIR + (1.0-frac)*PIR0
    
    change[0] = mean(abs(SR-SR0))/mean(abs(SR+SR0))
    change[1] = mean(abs(PIR-PIR0))/mean(abs(PIR+PIR0))

    GR = compute_GR(SR, mu, idelta)
    DR = compute_DR(PIR)
    
    if i%4==0: print('change = %1.3e, %1.3e'%(change[0], change[1]))
    
    if i>5 and change[0]<1e-15 and change[1]<1e-15: break
    
    if i%10==0:
        save(folder+'GR', GR)
        save(folder+'SR', SR)
        save(folder+'DR', DR)
        save(folder+'PIR', PIR)
        print(' ')





