from numpy import *
import numpy as np
import time
from convolution import conv
import os
import plot1d as plt
from scipy import optimize
import sys

print('running renormalized ME')

savedir = None
if len(sys.argv)>1:
    savedir = sys.argv[1]
    print('Using data from %s\n'%savedir)

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
dens = 0.7

SC = 1

print('beta %1.3f'%beta)
print('alpha = %1.3f'%alpha)
print('omega = %1.3f'%omega)
print('lamb = %1.3f'%lamb)
print('g = %1.3f'%g)
print('lamb correct = %1.3f'%(2*g**2/(8.0*omega)))

folder = 'data2d/data_sc_renormalized_%db%d_lamb%1.1f_beta%1.1f/'%(Nk,Nk,lamb,beta)
if not os.path.exists(folder): os.mkdir(folder)

kys, kxs = meshgrid(arange(-pi, pi, 2*pi/Nk), arange(-pi, pi, 2*pi/Nk))
def band(kxs, kys):
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)
    return -2.0*(cos(kxs) + cos(kys))
ek = band(kxs, kys)

assert Nw%2==0 and Nk%2==0 and len(w)%2==0 and abs(w[len(w)//2])<1e-8

tau0 = array([[1.0, 0.0], [0.0, 1.0]])
tau1 = array([[0.0, 1.0], [1.0, 0.0]])
tau3 = array([[1.0, 0.0], [0.0,-1.0]])

# estimate filling and dndmu at the desired filling
mu = optimize.fsolve(lambda mu : 2.0*sum(1.0/(exp(beta*(ek-mu))+1.0))/Nk**2-dens, 0.0)
dndmu = 2.0*sum(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)/Nk**2

nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

print('mu optimized = %1.3f'%mu)
print('dndmu = %1.3f'%dndmu)

def compute_G(S, mu):
    return linalg.inv(\
            iwm[None,None,:,None,None]*tau0[None,None,None,:,:] \
          - (ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
          - S)

def compute_D(PI, omega):
    return 1.0/(-((vn**2)[None,None,:] + omega**2)/(2.0*omega) - PI) 

def compute_GR(SR, mu, idelta):
    return linalg.inv(\
         (w[None,None,:,None,None]+idelta)*tau0[None,None,None,:,:] \
       - (ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
       - SR)

DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)
def compute_DR(PIR):
    return 1.0/(DRbareinv[None,None,:] - PIR) 

def compute_fill(G):
    return 1.0 + 2.0/(beta * Nk**2) * np.sum(G[:,:,:,0,0]).real

def compute_fill_R(GR):
    return -2.0 / (pi * Nk**2) * np.sum(GR[:,:,0,0].imag*nF[None,:]) * dw

def compute_S(G, D):
    tau3Gtau3 = einsum('ab,...bc,cd->...ad', tau3, G, tau3)
    return -g**2/(beta*Nk**2) * conv(tau3Gtau3, D[:,:,:,None,None]*ones([Nk,Nk,len(vn),2,2]), ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False])[:,:,:Nw,:,:]

def compute_PI(G):
    tau3G = einsum('ab,...bc->...ac', tau3, G)
    return 2.0*g**2/(beta*Nk**2) * 0.5*einsum('...aa->...', conv(tau3G, tau3G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:Nw+1,:,:])

def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag[:,:,:,None,None] * ones([Nk,Nk,len(w),2,2])
    tau3GRtau3 = einsum('ab,...bc,cd->...ad',tau3,GR,tau3)
    return -g**2*dw/Nk**2*(conv(B, Gsum, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
      - conv(B*(1+nB[None,None,:,None,None]), tau3GRtau3, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
      + conv(B, tau3GRtau3*nF[None,None,:,None,None], ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)])

def compute_PI_real_axis(GR, Gsum):
    tau3GA = einsum('ab,...bc->...ac', tau3, conj(GR))
    tau3A  = einsum('ab,...bc->...ac', tau3, -1.0/pi * GR.imag)
    return 2.0*g**2*dw/Nk**2 * 0.5*einsum('...aa->...', conv(tau3A, Gsum, ['k+q,q','k+q,k','z,w-z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:len(w)] \
      - conv(tau3A, tau3GA*nF[None,None,:,None,None], ['k+q,q','k+q,k','w+z,z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:len(w)] \
      + conv(tau3A*nF[None,None,:,None,None], tau3GA, ['k+q,q','k+q,k','w+z,z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:len(w)])

# solve Matsubara piece
print('\nSolving Matsubara piece')

if savedir is None:
    S  = -SC * 0.01 * ones([Nk,Nk,Nw,2,2], dtype=complex)*tau1[None,None,None,:,:]
    PI = zeros([Nk,Nk,Nw+1], dtype=complex)
else:
    #S = zeros([Nk,Nw,2,2], dtype=complex)
    #S  = load(savedir+'S.npy')[None,:,:,:]
    #PI = zeros([Nk,Nw+1], dtype=complex)
    S  = load(savedir+'S.npy')
    PI = load(savedir+'PI.npy')

G  = compute_G(S, mu)
D  = compute_D(PI, omega)
print('fill = %1.3f'%(compute_fill(G)))

#print('G', mean(abs(G[:,:,0,0].real)))
#print('G', mean(abs(G[:,:,0,0].imag)))
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
    mu += 0.1*(n-dens)/dndmu
    
    print('change=%1.3e %1.3e, diag=%1.3e, odlro=%1.3e, fill=%1.3f, mu=%1.3f'%(change[0], change[1], mean(abs(S[:,:,:,0,0])), mean(abs(S[:,:,:,0,1])), n, mu))
    
    if i>10 and change[0]<1e-14 and change[1]<1e-14: break

    if i%10==0:
        save(folder+'S', S)
        save(folder+'PI', PI)

    
save(folder+'iwm', iwm)
save(folder+'w', w)
save(folder+'Nk', [Nk])
save(folder+'lamb', [lamb])
save(folder+'omega', [omega])
save(folder+'S', S)
save(folder+'PI', PI)

# analytic continuation
print('\nSolving real axis')

SR  = - SC * 0.01 * ones([Nk,Nk,len(w),2,2], dtype=complex)*tau1[None,None,None,:,:]
PIR = zeros([Nk,Nk,len(w)], dtype=complex)
GR  = compute_GR(SR, mu, idelta)
DR  = compute_DR(PIR)

print('compting Gsum')
Gsum_plus  = zeros([Nk,Nk,len(w),2,2], dtype=complex)
Gsum_minus = zeros([Nk,Nk,len(w),2,2], dtype=complex)
for iw in range(len(w)):
    Gsum_plus[:,:,iw,:,:]  = sum(G/((w[iw]+iwm)[None,None,:,None,None]), axis=2) / beta
    Gsum_minus[:,:,iw,:,:] = sum(G/((w[iw]-iwm)[None,None,:,None,None]), axis=2) / beta
Gsum_plus  = einsum('ab,...bc->...ac',tau3,Gsum_plus)
Gsum_minus = einsum('ab,...bc,cd->...ad',tau3,Gsum_minus,tau3)
print('finished Gsum')
    
change = [0,0]
frac = 0.6
for i in range(50):
    SR0 = SR[:]
    PIR0 = PIR[:]
    
    SR  = compute_S_real_axis(GR, DR, Gsum_minus)
    PIR = compute_PI_real_axis(GR, Gsum_plus)
    
    SR  = frac*SR  + (1-frac)*SR0
    PIR = frac*PIR + (1-frac)*PIR0

    change[0] = mean(abs(SR-SR0))/mean(abs(SR+SR0))
    change[1] = mean(abs(PIR-PIR0))/mean(abs(PIR+PIR0))

    GR = compute_GR(SR, mu, idelta)
    DR = compute_DR(PIR)
    
    if i%5==0: print('change = %1.3e %1.3e'%(change[0],change[1]))
    
    if change[0]<1e-14 and change[1]<1e-14: break
    
    if i%10==0:
        save(folder+'GR', GR)
        save(folder+'SR', SR)
        save(folder+'PIR', PIR)
        save(folder+'DR', DR)
        print(' ')

    #myp(SR[:,:,:,0,0])

#plt.main(folder)


