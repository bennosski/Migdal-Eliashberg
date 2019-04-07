from numpy import *
import numpy as np
import time
from convolution import conv
import os
import plot1d as plt
from scipy import optimize

print('running unrenormalized ME')

def myp(x): print(mean(abs(x.real)), mean(abs(x.imag)))

# params
Nw = 1600
Nk = 100
beta = 100.0
iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)

dw = 0.02
w = arange(-5.0, 5.0, dw)

omega = 1.0

lamb = 1.0
alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)
idelta = 0.030j
dens = 0.7

SC = 1

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('omega = %1.3f'%omega)
print('lamb = %1.3f'%lamb)
print('g = %1.3f'%g)
print('lamb correct = %1.3f'%(2*g**2/(8.0*omega)))
print('Nk = %d'%Nk)
print('idelta = %1.3f'%idelta.imag)

folder = 'data2d/data_sc_unrenormalized_%db%d_lamb%1.1f_beta%1.1f_idelta%1.3f/'%(Nk,Nk,lamb,beta,idelta.imag)
if not os.path.exists(folder): os.mkdir(folder)

kys, kxs = meshgrid(arange(-pi, pi, 2*pi/Nk), arange(-pi, pi, 2*pi/Nk))
def band(kxs, kys):
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)
    return -2.0*(cos(kxs) + cos(kys))
ek = band(kxs, kys)

assert Nw%2==0 and Nk%2==0 and abs(w[len(w)//2])<1e-8

tau0 = array([[1.0, 0.0], [0.0, 1.0]])
tau1 = array([[0.0, 1.0], [1.0, 0.0]])
tau3 = array([[1.0, 0.0], [0.0,-1.0]])

# estimate filling and dndmu at the desired filling
mu = optimize.fsolve(lambda mu : 2.0*sum(1.0/(exp(beta*(ek-mu))+1.0))/Nk**2-dens, 0.0)
dndmu = 2.0*sum(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)/Nk**2

print('mu optimized = %1.3f'%mu)
print('dndmu = %1.3f'%dndmu)

nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

assert len(w)%2==0 and abs(w[len(w)//2])<1e-8

def compute_G(S, mu):
    return linalg.inv(iwm[None,None,:,None,None]*tau0[None,None,None,:,:] \
                    -(ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
                    -S[None,None,:,:,:])

def compute_GR(SR, mu, idelta):
    return linalg.inv((w[None,None,:,None,None]+idelta)*tau0[None,None,None,:,:] \
                     -(ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
                     -SR[None,None,:,:,:])

def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk**2) * np.sum(G[:,:,:,0,0]).real

def compute_fill_R(GR):
    return -2.0 / (pi * Nk**2) * np.sum(GR[:,:,:,0,0].imag*nF[None,None,:]) * dw

def compute_S(G, D):
    return -g**2/(beta*Nk**2) * conv(einsum('ab,wbc,cd->wad', tau3, sum(G, axis=(0,1)), tau3), D[:,None,None]*ones([len(vn),2,2]), ['m,n-m'], [0], [False])[:Nw]
    
def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag[:,None,None] * ones([len(w),2,2])
    GRloc = einsum('ab,wbc,cd->wad',tau3,sum(GR,axis=(0,1)),tau3)
    return -g**2*dw/Nk**2*(conv(B, Gsum, ['z,w-z'], [0], [False])[:len(w)] \
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
    
    G = compute_G(S, mu)

    n = compute_fill(G)
    mu += 0.1*(n-dens)/dndmu
    
    if i%10==0: print('change=%1.3e, diag=%1.3e, odlro=%1.3e, fill=%1.3f, mu=%1.3f'%(change, mean(abs(S[:,0,0].imag)), mean(abs(S[:,0,1])), n, mu))
    
    if change<1e-15: break

    if i%10==0:
        save(folder+'G', G)
        save(folder+'S', S)

save(folder+'iwm', iwm)
save(folder+'w', w)
save(folder+'Nk', [Nk])
save(folder+'lamb', [lamb])
save(folder+'omega', [omega])

# analytic continuation
print('\nSolving real axis')

SR  = - SC * 0.01 * ones([len(w),2,2], dtype=complex)*tau1[None,:,:]
GR  = compute_GR(SR, mu, idelta)
DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
DR  = 1.0/(DRbareinv) 

print('computing Gsum')
Gsum_minus = zeros([len(w),2,2], dtype=complex)
for iw in range(len(w)):
    Gsum_minus[iw] = sum(G/((w[iw]-iwm)[None,None,:,None,None]), axis=(0,1,2)) / beta
Gsum_minus = einsum('ab,wbc,cd->wad',tau3,Gsum_minus,tau3)
print('finished Gsum')
    
change = 0
frac = 0.8
for i in range(50):
    SR0 = SR[:]

    SR  = compute_S_real_axis(GR, DR, Gsum_minus)
    
    SR  = frac*SR + (1-frac)*SR0
    
    change = mean(abs(SR-SR0))/mean(abs(SR+SR0))

    GR = compute_GR(SR, mu, idelta)
    
    if i%4==0: print('change = %1.3e'%change)
    #myp(SR[:,0,0])

    if change<1e-15: break
    
    if i%10==0:
        save(folder+'GR', GR)
        save(folder+'SR', SR)
        save(folder+'DR', DR)



