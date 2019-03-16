from numpy import *
import numpy as np
import time
from convolution import conv
import os
import plot as plt

print('running unrenormalized ME')

def myp(x):
    print(mean(abs(x.real)), mean(abs(x.imag)))

# params
Nw = 600
Nk = 6
beta = 4.0
iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)

dw = 0.005
w = arange(-4.0, 4.0, dw)

print(abs(w[len(w)//2]))
print(len(w))

omega = 1.0

#lamb = 1.5
#lamb = 0.75
#lamb = 0.5
lamb = 3.0
alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)
idelta = 0.020j
#idelta = 0.015j

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('lamb = %1.3f'%lamb)
print('g = %1.3f'%g)
print('lamb correct = %1.3f'%(2*g**2/(8.0*omega)))

folder = 'data_unrenormalized_%db%d_lamb%1.1f_beta%1.1f/'%(Nk,Nk,lamb,beta)
if not os.path.exists(folder): os.mkdir(folder)

kys, kxs = meshgrid(arange(-pi, pi, 2*pi/Nk), arange(-pi, pi, 2*pi/Nk))

def band(kxs, kys):
    return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2

ek = band(kxs, kys)
nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

assert Nw%2==0 and Nk%2==0 and len(w)%2==0 and abs(w[len(w)//2])<1e-13


def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk**2) * np.sum(G).real

def compute_fill_R(GR):
    return -2.0 / (pi * Nk**2) * np.sum(GR.imag*nF[None,None,:]) * dw

def compute_S(G, D):
    return -g**2/(beta * Nk**2) * conv(sum(G, axis=(0,1)), D, ['m,n-m'], [0], [False])[:Nw]

def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag
    GRloc = sum(GR, axis=(0,1))
    return -g**2*dw/Nk**2*(conv(B, Gsum, ['z,w-z'], [0], [False])[:len(w)] \
             -conv(B*(1+nB), GRloc, ['z,w-z'], [0], [False])[:len(w)] \
             +conv(B, GRloc*nF, ['z,w-z'], [0], [False])[:len(w)])

# solve Matsubara piece
print('\nSolving Matsubara piece')

S  = zeros(Nw, dtype=complex)
G  = 1.0/(iwm[None,None,:] - ek[:,:,None] - S)
D  = 1.0/(-((vn**2) + omega**2)/(2.0*omega)) 
D0 = -2.0/omega

print('fill = %1.3f'%(compute_fill(G)))

change = 0
for i in range(10):
    S0  = S[:]

    S  = compute_S(G, D) 
    change = mean(abs(S-S0))/mean(abs(S+S0))
    S  = 0.3*S + 0.7*S0
    
    G = 1.0/(iwm[None,None,:] - ek[:,:,None] - S[None,None,:])
    
    if i%10==0: print('change = %1.3e, fill = %1.3f'%(change, compute_fill(G)))
    myp(S)

    if change<1e-15: break

save(folder+'iwm', iwm)
save(folder+'w', w)
save(folder+'Nk', [Nk])
save(folder+'lamb', [lamb])
save(folder+'omega', [omega])
save(folder+'G', G)
save(folder+'S', S)

# analytic continuation
print('\nSolving real axis')

SR  = zeros(len(w), dtype=complex)
GR  = 1.0/(w[None,None,:] - ek[:,:,None] + idelta - SR[None,None,:])
DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
DR  = 1.0/(DRbareinv) 

Gsum_minus = zeros(len(w), dtype=complex)
for iw in range(len(w)):
    Gsum_minus[iw] = np.sum(G/((w[iw]-iwm)[None,None,:]), axis=(0,1,2)) / beta

print('finished Gsum')
    
change = 0
for _ in range(5):
    SR0 = SR[:]
        
    SR  = compute_S_real_axis(GR, DR, Gsum_minus)
 
    SR  = 0.8*SR + 0.2*SR0
    
    change = mean(abs(SR-SR0))/mean(abs(SR+SR0))
    
    GR  = 1.0/(w[None,None,:] + idelta - ek[:,:,None] - SR[None,None,:])
    
    print('change = %1.3e'%change)

    myp(SR)

    if change<1e-15: break
    
save(folder+'GR', GR)
save(folder+'SR', SR)
save(folder+'DR', DR)

#plt.main(folder)


