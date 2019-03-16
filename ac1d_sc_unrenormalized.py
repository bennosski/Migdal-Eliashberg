from numpy import *
import numpy as np
import time
from convolution import conv
import os
import plot1d as plt
from scipy import optimize

print('running unrenormalized ME')

# params
Nw = 600
Nk = 50
beta = 40.0
iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)

omega = 0.5

lamb = 3.0
alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)
idelta = 0.020j
dens = 1.0
SC = 1

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('omega = %1.3f'%omega)
print('lamb = %1.3f'%lamb)
print('g = %1.3f'%g)
print('lamb correct = %1.3f'%(2*g**2/(8.0*omega)))

folder = 'data1d/data_sc_unrenormalized_%d_lamb%1.1f_beta%1.1f/'%(Nk,lamb,beta)
if not os.path.exists(folder): os.mkdir(folder)

kxs = arange(-pi, pi, 2*pi/Nk)

def band(kxs):
    return -2.0*cos(kxs) 

ek = band(kxs)
dw = 0.001

assert Nw%2==0 and Nk%2==0

tau0 = array([[1.0, 0.0], [0.0, 1.0]])
tau1 = array([[0.0, 1.0], [1.0, 0.0]])
tau3 = array([[1.0, 0.0], [0.0,-1.0]])

# estimate filling and dndmu at the desired filling
mu = optimize.fsolve(lambda mu : 2.0*sum(1.0/(exp(beta*(ek-mu))+1.0))/Nk-dens, 0.0)
dndmu = 2.0*sum(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)/Nk

print('mu optimized = %1.3f'%mu)
print('dndmu = %1.3f'%dndmu)

w = arange(-3.1, 3.1, dw)
nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

assert len(w)%2==0 and abs(w[len(w)//2])<1e-8


def compute_G(S, mu):
    return linalg.inv(iwm[None,:,None,None]*tau0[None,None,:,:] \
                      - (ek[:,None,None,None]-mu)*tau3[None,None,:,:] \
                      - S[None,:,:,:])

def compute_GR(SR, mu, idelta):
    return linalg.inv((w[None,:,None,None]+idelta)*tau0[None,None,:,:] \
                      - (ek[:,None,None,None]-mu)*tau3[None,None,:,:] \
                      - SR[None,:,:,:])

def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk) * np.sum(G[:,:,0,0]).real

def compute_fill_R(GR):
    return -2.0 / (pi * Nk) * np.sum(GR[:,:,0,0].imag*nF[None,:]) * dw

def compute_S(G, D):
    return -g**2/(beta*Nk) * conv(einsum('ab,wbc,cd->wad', tau3, sum(G, axis=0), tau3), D[:,None,None]*ones([len(vn),2,2]), ['m,n-m'], [0], [False])[:Nw]
    
def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag[:,None,None] * ones([len(w),2,2])
    GRloc = einsum('ab,wbc,cd->wad',tau3,sum(GR,axis=0),tau3)
    return -g**2*dw/Nk*(conv(B, Gsum, ['z,w-z'], [0], [False])[:len(w)] \
                       -conv(B*(1+nB[:,None,None]), GRloc, ['z,w-z'], [0], [False])[:len(w)] \
                       +conv(B, GRloc*nF[:,None,None], ['z,w-z'], [0], [False])[:len(w)])

# solve Matsubara piece
print('\nSolving Matsubara piece')

S  = -SC * 0.01 * ones([Nw,2,2], dtype=complex)*tau1[None,:,:]
G  = compute_G(S, mu)
D  = 1.0/(-((vn**2) + omega**2)/(2.0*omega)) 
D0 = -2.0/omega

#print('Re G avg', mean(abs(G[:,:,0,0].real)))
#print('Im G avg', mean(abs(G[:,:,0,0].imag)))
#exit()

print('fill = %1.3f'%(compute_fill(G)))

change = 0
frac = 0.8
for i in range(200):
    S0  = S[:]

    S  = compute_S(G, D) 
    change = mean(abs(S-S0))/mean(abs(S+S0))
    S  = frac*S + (1-frac)*S0

    #print('Re S avg', mean(abs(S[:,0,0].real)))
    #print('Im S avg', mean(abs(S[:,0,0].imag)))
    
    G = compute_G(S, mu)

    n = compute_fill(G)
    mu += 0.25*(n-dens)/dndmu
    
    if i%10==0: print('change=%1.3e, diag=%1.3e, odlro=%1.3e, fill=%1.3f, mu=%1.3f'%(change, mean(abs(S[:,0,0].imag)), mean(abs(S[:,0,1])), n, mu))
    
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

SR  = - SC * 0.01 * ones([len(w),2,2], dtype=complex)*tau1[None,:,:]
GR  = compute_GR(SR, mu, idelta)
DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
DR  = 1.0/(DRbareinv) 

print('computing Gsum')
Gsum_minus = zeros([len(w),2,2], dtype=complex)
for iw in range(len(w)):
    Gsum_minus[iw] = sum(G/((w[iw]-iwm)[None,:,None,None]), axis=(0,1)) / beta
Gsum_minus = einsum('ab,wbc,cd->wad',tau3,Gsum_minus,tau3)
print('finished Gsum')
    
change = 0
frac = 0.8
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


