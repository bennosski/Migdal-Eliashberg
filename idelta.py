import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
from numpy import *

def band(kxs, kys):
    return -2.0*(cos(kxs) + cos(kys))

Nk = 200
kys, kxs = meshgrid(arange(-pi, pi, 2*pi/Nk), arange(-pi, pi, 2*pi/Nk))
ek = band(kxs, kys)

dw = 0.002
w = arange(-2.0, 2.0, dw)

GRbare  = 1.0/(w[None,None,:] - ek[:,:,None] + 0.020j)

figure()
plot(w, -1.0/pi*sum(GRbare, axis=(0,1)).imag/Nk**2)
ylim(0, 0.5)
savefig('dos')
close()
