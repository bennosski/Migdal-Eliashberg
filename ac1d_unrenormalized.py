from numpy import *
import numpy as np
import time
from convolution import conv
import os
import plot1d as plt
from scipy import optimize

from matplotlib.pyplot import *

print('running unrenormalized ME')

# params
Nw = 200
Nk = 50
beta = 5.0
iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)

omega = 0.5

lamb = 2.0
alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)
idelta = 0.020j
dens = 1.0

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('lamb = %1.3f'%lamb)
print('g = %1.3f'%g)
print('lamb correct = %1.3f'%(2*g**2/(8.0*omega)))

folder = 'data1d/data_unrenormalized_Nk%d_lamb%1.1f_beta%1.1f/'%(Nk,lamb,beta)
if not os.path.exists(folder): os.mkdir(folder)

kxs = arange(-pi, pi, 2*pi/Nk)

def band(kxs):
    return -2.0*cos(kxs) 

ek = band(kxs)
dw = 0.002

assert Nw%2==0 and Nk%2==0

# estimate filling and dndmu at the desired filling
mu = optimize.fsolve(lambda mu : 2.0*sum(1.0/(exp(beta*(ek-mu))+1.0))/Nk-dens, 0.0)
deriv = lambda mu : 2.0*sum(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)/Nk
dndmu = deriv(mu)

print('mu optimized = %1.3f'%mu)
print('dndmu = %1.3f'%dndmu)

figure()
plot(ek-mu)
savefig('fill')
close()

#w = arange(amin(ek-mu)-omega-0.3, amax(ek-mu)+omega+0.3, dw)
w = arange(-3.1, 3.1, dw)
nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

assert len(w)%2==0 and abs(w[len(w)//2])<1e-8


def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk) * np.sum(G).real

def compute_fill_R(GR):
    return -2.0 / (pi * Nk) * np.sum(GR.imag*nF[None,:]) * dw

def compute_S(G, D):
    return -g**2/(beta * Nk) * conv(sum(G, axis=0), D, ['m,n-m'], [0], [False])[:Nw]

def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag
    GRloc = sum(GR, axis=0)
    return -g**2*dw/Nk*(conv(B, Gsum, ['z,w-z'], [0], [False])[:len(w)] \
             -conv(B*(1+nB), GRloc, ['z,w-z'], [0], [False])[:len(w)] \
             +conv(B, GRloc*nF, ['z,w-z'], [0], [False])[:len(w)])

def compute_G(S, mu):
    return 1.0/(iwm[None,:] - (ek[:,None]-mu) - S[None,:])

def compute_GR(SR, mu, idelta):
    return 1.0/(w[None,:] - (ek[:,None]-mu) + idelta - SR[None,:])

# solve Matsubara piece
print('\nSolving Matsubara piece')

S  = zeros(Nw, dtype=complex)
G  = compute_G(S, mu)
D  = 1.0/(-((vn**2) + omega**2)/(2.0*omega)) 
D0 = -2.0/omega


#print('Re G avg ', mean(abs(G.real)))
#print('Im G avg ', mean(abs(G.imag)))
#exit()

print('fill = %1.3f'%(compute_fill(G)))

change = 0
frac = 0.6
for i in range(100):
    S0  = S[:]

    S  = compute_S(G, D) 
    change = mean(abs(S-S0))/mean(abs(S+S0))
    S  = frac*S + (1-frac)*S0

    #print('Re S avg', mean(abs(S.real)))
    #print('Im S avg', mean(abs(S.imag)))
    
    G = compute_G(S, mu)

    n = compute_fill(G)

    mu += (n-dens)/dndmu
    
    if i%10==0: print('change=%1.3e, diag=%1.3e, fill=%1.3f, mu=%1.3f'%(change, mean(abs(S.imag)), n, mu))
    
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
GR  = compute_GR(SR, mu, idelta)
DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
DR  = 1.0/(DRbareinv) 

Gsum_minus = zeros(len(w), dtype=complex)
for iw in range(len(w)):
    Gsum_minus[iw] = np.sum(G/((w[iw]-iwm)[None,:])) / beta

print('finished Gsum')
    
change = 0
frac = 0.6
for i in range(100):
    SR0 = SR[:]
        
    SR  = compute_S_real_axis(GR, DR, Gsum_minus)
 
    SR  = frac*SR + (1-frac)*SR0
    
    change = mean(abs(SR-SR0))/mean(abs(SR+SR0))
    
    GR = compute_GR(SR, mu, idelta)
    
    if i%4==0: print('change = %1.3e'%change)

    if change<1e-15: break
    
save(folder+'GR', GR)
save(folder+'SR', SR)
save(folder+'DR', DR)

plt.main(folder)


