from numpy import *
import numpy as np
import time
from convolution import conv
import os
import sys

savedir = None
if len(sys.argv)>1:
    savedir = sys.argv[1]
    print('Using data from %s\n'%savedir)

print('running renormalized ME')

def myp(x):
    print(mean(abs(x.real)), mean(abs(x.imag)))

# params
Nw = 600
Nk = 100
beta = 40.0
iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)

dw = 0.001
w = arange(-5.0, 5.0, dw)

omega = 1.0

lamb = 1.0
alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)
idelta = 0.020j

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('g = %1.3f'%g)
print('lamb = %1.3f'%lamb)
print('lamb correct = %1.3f'%(2*g**2/(8.0*omega)))

folder = 'data_renormalized_%db%d_lamb%1.1f_beta%1.1f/'%(Nk,Nk,lamb,beta)
if not os.path.exists(folder): os.mkdir(folder)

kys, kxs = meshgrid(arange(-pi, pi, 2*pi/Nk), arange(-pi, pi, 2*pi/Nk))

def band(kxs, kys):
    return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)

ek = band(kxs, kys)
nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

assert Nw%2==0 and Nk%2==0 and len(w)%2==0

def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk**2) * np.sum(G).real

def compute_S(G, D):
    return -g**2/(beta * Nk**2) * conv(G, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False])[:,:,:Nw]

def compute_PI(G):
    return 2.0*g**2/(beta * Nk**2) * conv(G, G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False])[:,:,:Nw+1]

def compute_PI_real_axis(GR, Gsum):
    GA = conj(GR)
    A  = -1.0/pi * GR.imag
    return 2.0*g**2*dw/Nk**2*(conv(A, Gsum, ['k+q,k','k+q,k','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
                    -conv(A, GA*nF[None,None,:], ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
                    +conv(A*nF[None,None,:], GA, ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:len(w)])

def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag
    return -g**2*dw/Nk**2*(conv(B, Gsum, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
             -conv(B*(1+nB)[None,None,:], GR, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
             +conv(B, GR*nF[None,None,:], ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)])

if savedir is None:
    S  = zeros([Nk,Nk,Nw], dtype=complex)
    PI = zeros([Nk,Nk,Nw+1], dtype=complex)
else:
    S = load(savedir+'S.npy')
    PI = load(savedir+'PI.npy')

G  = 1.0/(iwm[None,None,:] - ek[:,:,None] - S)
D  = 1.0/(-((vn**2)[None,None,:] + omega**2)/(2.0*omega) - PI)

# solve Matsubara piece
print('\n Solving Matsubara piece')

change = [0, 0]
frac = 0.6
for i in range(10):
    S0  = S[:]
    PI0 = PI[:]
    
    S  = compute_S(G, D)
    change[0] = mean(abs(S-S0))/mean(abs(S+S0))
    S  = frac*S + (1-frac)*S0

    PI = compute_PI(G)
    change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))
    PI = frac*PI + (1-frac)*PI0
    
    G = 1.0/(iwm[None,None,:] - ek[:,:,None] - S)
    D = 1.0/(-((vn**2)[None,None,:] + omega**2)/(2.0*omega) - PI) 
    
    if i%10==0: print('change = %1.3e, %1.3e and fill = %.13f'%(change[0], change[1], compute_fill(G)))

    if i>10 and change[0]<1e-15 and change[1]<1e-15: break

    if i%10==0:
        save(folder+'S', S)
        save(folder+'PI', PI)
        
save(folder+'iwm', iwm)
save(folder+'w', w)
save(folder+'Nk', [Nk])
save(folder+'lamb', [lamb])
save(folder+'omega', [omega])
save(folder+'idelta', [abs(idelta)])

# analytic continuation
print('\nSolving real axis')

if savedir is None:
    SR  = zeros([Nk,Nk,len(w)])
    PIR = zeros([Nk,Nk,len(w)])
else:
    SR = load(savedir+'SR.npy')
    PIR = load(savedir+'PIR.npy')

GR  = 1.0/(w[None,None,:] - ek[:,:,None] + idelta - SR)
DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
DR  = 1.0/(DRbareinv[None,None,:] - PIR) 

del S

Gsum_plus  = zeros([Nk,Nk,len(w)], dtype=complex)
Gsum_minus = zeros([Nk,Nk,len(w)], dtype=complex)
for iw in range(len(w)):
    Gsum_plus[:,:,iw]  = np.sum(G/((w[iw]+iwm)[None,None,:]), axis=2) / beta
    Gsum_minus[:,:,iw] = np.sum(G/((w[iw]-iwm)[None,None,:]), axis=2) / beta

del G

print('finished Gsum')
    
change = [0,0]
frac = 0.6
for i in range(50):
    SR0 = SR[:]
    PIR0 = PIR[:]
        
    SR  = compute_S_real_axis(GR, DR, Gsum_minus)
    PIR = compute_PI_real_axis(GR, Gsum_plus)
    
    SR  = frac*SR  + (1.0-frac)*SR0
    PIR = frac*PIR + (1.0-frac)*PIR0
    
    change[0] = mean(abs(SR-SR0))/mean(abs(SR+SR0))
    change[1] = mean(abs(PIR-PIR0))/mean(abs(PIR+PIR0))
    
    GR  = 1.0/(w[None,None,:] - ek[:,:,None] + idelta - SR)
    DR  = 1.0/(DRbareinv[None,None,:] - PIR) 
        
    if i%5==0: print('change = %1.3e, %1.3e'%(change[0], change[1]))
    
    if i>5 and change[0]<1e-15 and change[1]<1e-15: break
    
    if i%10==0:
        save(folder+'GR', GR)
        save(folder+'SR', SR)
        save(folder+'DR', DR)
        save(folder+'PIR', PIR)
        print(' ')

    #myp(SR)



