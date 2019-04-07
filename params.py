
from numpy import *

def get_params(i):

    '''
    if i==0:
        # params test
        Nw = 6
        Nk = 10
        beta = 1.0
        iwm = 1j * pi/beta * (2*arange(-Nw//2, Nw//2) + 1)
        vn = pi/beta * 2*arange(-Nw//2, Nw//2+1)
        dw = 0.2
        w = arange(-1.0, 1.0, dw)
        omega = 1.0
        lamb = 1.0
        alpha = sqrt(omega**2*lamb)
        g = alpha/sqrt(omega)
        idelta = 0.030j
        dens = 0.7
        SC = 1
    '''

    # for beta in betas

    betas = [1.0, 10.0, 20.0, 30.0]

    Nw = 600
    Nk = 100
    beta = betas[i]
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

    return (Nw, Nk, beta, iwm, vn, dw, w, omega, lamb, alpha, g, idelta, dens, SC)

