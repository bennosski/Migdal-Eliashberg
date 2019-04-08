from numpy import *
import sys
from convolution import conv

arg = int(sys.argv[1])

'''
# ---------------------------------------------
# edit this section only to set parameters
#
SC     = 1
betas  = [1.0, 10.0, 20.0, 30.0]
ndim   = 2
nk     = 100
beta   = betas[arg]
nw     = 600
wmax   = 5.0
dw     = 0.02
omega  = 1.0
lamb   = 1.0 # Marsiglio definition?
idelta = 0.030j
dens   = 0.7
kys, kxs = meshgrid(arange(-pi, pi, 2*pi/nk), arange(-pi, pi, 2*pi/nk))
def band(kxs, kys):
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)
    return -2.0*(cos(kxs) + cos(kys))
ek = band(kxs, kys)
# ----------------------------------------------
'''

# ---------------------------------------------
# edit this section only to set parameters
#
SC     = 1
betas  = [1.0, 10.0, 20.0, 30.0]
ndim   = 2
nk     = 10
beta   = betas[arg]
nw     = 60
wmax   = 5.0
dw     = 0.2
omega  = 1.0
lamb   = 1.0 # Marsiglio definition?
idelta = 0.030j
dens   = 0.7
kys, kxs = meshgrid(arange(-pi, pi, 2*pi/nk), arange(-pi, pi, 2*pi/nk))
def band(kxs, kys):
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)
    return -2.0*(cos(kxs) + cos(kys))
ek = band(kxs, kys)
# ----------------------------------------------

assert ndim==len(shape(ek))

w = arange(-wmax, wmax, dw)

assert nw%2==0 and nk%2==0 and len(w)%2==0 and abs(w[len(w)//2])<1e-8

alpha = sqrt(omega**2*lamb)
g = alpha/sqrt(omega)

iwm = 1j * pi/beta * (2*arange(-nw//2, nw//2) + 1)
vn = pi/beta * 2*arange(-nw//2, nw//2+1)

nF = 1.0/(exp(beta*w)+1.0)
nB = 1.0/(exp(beta*w)-1.0)

nk_total = nk**ndim

DRbareinv = ((w+idelta)**2 - omega**2)/(2.0*omega)

#params = (SC, ndim, nk, beta, nw, wmax, dw, omega, lamb, idelta, dens)
#constants = (nk_total, ek, iwm, vn, w, nF, nB, alpha, g, DRbareinv)

print('-------------------------')
print('ndim   = %d'%ndim)
print('nk     = %d'%nk)
print('beta   = %1.3f'%beta)
print('nw     = %d'%nw)
print('wmax   = %1.5f'%wmax)
print('dw     = %1.5e'%dw)
print('omega  = %1.3f'%omega)
print('alpha  = %1.3f'%alpha)
print('lamb   = %1.3f'%lamb)
print('idelta = %1.3f'%idelta.imag)
print('dens   = %1.3f'%dens)
print('SC     = %d'%SC)
print('-------------------------\n')

