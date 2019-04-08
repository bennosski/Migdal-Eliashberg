import time
import os
import plot1d as plt
from scipy import optimize
import sys

print('\nRenormalized Migdal-(Eliashberg)')
from functions import *

savedir = None
if len(sys.argv)>2:
    savedir = sys.argv[2]
    print('using data from %s\n'%savedir)
    
folder = 'data/renormalized_SC%d_ndim%d_nk%d_beta%1.5f_nw%d_wmax%1.5f_dw%1.3e_omega%1.3f_lamb%1.5f_idelta%1.5f_dens%1.3f/'%(SC, ndim, nk, beta, nw, wmax, dw, omega, lamb, idelta.imag, dens)
if not os.path.exists('data/'): os.mkdir('data/')
if not os.path.exists(folder): os.mkdir(folder)

# estimate filling and dndmu at the desired filling
mu = optimize.fsolve(lambda mu : 2.0*sum(1.0/(exp(beta*(ek-mu))+1.0))/nk_total-dens, 0.0)
dndmu = 2.0*sum(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)/nk_total

print('mu optimized = %1.3f'%mu)
print('dndmu = %1.3f'%dndmu)

#---------------------------------------------
# solve Matsubara piece

print('\nSolving Matsubara piece')

if savedir is None:
    S = init_S()
    PI = init_PI()
else:
    S  = load(savedir+'S.npy')
    PI = load(savedir+'PI.npy')

print('shape S', shape(S))
print('shape PI', shape(PI))

G  = compute_G(S, mu)
D  = compute_D(PI)
print('fill = %1.3f'%(compute_fill(G)))

print('shape G', shape(G))
print('shape D', shape(D))

change = [0, 0]
frac = 0.8
for i in range(100):
    S0  = S[:]
    PI0 = PI[:]
    
    S  = compute_S(G, D) 
    PI = compute_PI(G)

    S  = frac*S + (1-frac)*S0
    PI = frac*PI + (1-frac)*PI0
    
    change[0] = mean(abs(S-S0))/mean(abs(S+S0))
    change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))
    
    G = compute_G(S, mu)
    D = compute_D(PI)
    
    n = compute_fill(G)
    mu += 0.1*(n-dens)/dndmu
    
    print('change=%1.3e %1.3e, diag=%1.3e, odlro=%1.3e, fill=%1.3f, mu=%1.3f'%(change[0], change[1], mean(abs(S[:,:,:,0,0])), mean(abs(S[:,:,:,0,1])), n, mu))
    
    if i>10 and change[0]<1e-7 and change[1]<1e-7: break

    if i%10==0:
        save(folder+'S', S)
        save(folder+'PI', PI)
        print('saved')

save(folder+'SC', [SC])
save(folder+'ndim', [ndim])
save(folder+'nk', [nk])
save(folder+'beta', [beta])
save(folder+'ek', [ek])
save(folder+'mu', [mu])
save(folder+'idelta', [idelta.imag])
save(folder+'iwm', iwm)
save(folder+'vn', vn)
save(folder+'w', w)
save(folder+'lamb', [lamb])
save(folder+'omega', [omega])
save(folder+'S', S)
save(folder+'PI', PI)

#---------------------------------------------
# analytic continuation

print('\nSolving real axis')

SR = init_SR()
PIR = init_PIR()
GR  = compute_GR(SR, mu)
DR  = compute_DR(PIR)

print('computing Gsum')
Gsum_plus, Gsum_minus = compute_Gsum(G)
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

    GR = compute_GR(SR, mu)
    DR = compute_DR(PIR)
    
    if i%5==0: print('change = %1.3e %1.3e'%(change[0],change[1]))
    
    if change[0]<1e-14 and change[1]<1e-14: break
    
    if i%10==0:
        save(folder+'GR', GR)
        save(folder+'SR', SR)
        save(folder+'PIR', PIR)
        save(folder+'DR', DR)
        print('saved')



