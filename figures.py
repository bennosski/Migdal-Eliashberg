import matplotlib
matplotlib.use('Agg')
from numpy import *
import time

folder = 'data2d/data_sc_renormalized_100b100_lamb1.0_beta40.0_idelta0.030/'

S0 = load(folder+'S.npy')
w = load(folder+'w.npy')

print('shape S', shape(S0))
Nk,Nk,Nw,_,_ = shape(S0)

S = zeros([Nk+1,Nk+1,Nw,2,2], dtype=complex)
S[:-1,:-1,:,:,:] = S0
S[-1,:-1] = S0[0,:]
S[:-1,-1] = S0[:,0]
S[-1,-1]  = S0[0,0]

print('extension done')

del S0

from scipy.interpolate import RectBivariateSpline
ks = linspace(-pi, pi, Nk+1)

Nk_cut = 3
kxs_cut = linspace(-pi, pi, Nk_cut)
kys_cut = pi/4 * ones(Nk_cut)

Sinterp = zeros([Nk_cut, Nw, 2, 2], dtype=complex)

for a in range(2):
    for b in range(2):
        for iw in range(Nw):
            print('iw = %d'iw)

            spline = RectBivariateSpline(ks, ks, S[:,:,iw,a,b])
            
            for ik in range(Nk_cut):
                kx,ky = kxs_cut[ik], kys_cut[ik]
                Sinterp[ik, iw, a, b] = spline(kx,ky)

save(folder+'Sinterp', Sinterp)
                
tau0 = array([[1.0, 0.0], [0.0, 1.0]])
tau1 = array([[0.0, 1.0], [1.0, 0.0]])
tau3 = array([[1.0, 0.0], [0.0,-1.0]])

#iwm = load(folder+'iwm.npy')

def band(kxs, kys):
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)
    return -2.0*(cos(kxs) + cos(kys))

ek = band(kxs_cut, kys_cut)

def compute_G(S, mu):
    return linalg.inv(\
            iwm[None,None,:,None,None]*tau0[None,None,None,:,:] \
          - (ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
          - S)

Ginterp = zeros([Nk_cut, Nw, 2, 2], dtype=complex)







