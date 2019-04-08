from numpy import *
from parameters import *

tau0 = array([[1.0, 0.0], [0.0, 1.0]])
tau1 = array([[0.0, 1.0], [1.0, 0.0]])
tau3 = array([[1.0, 0.0], [0.0,-1.0]])
#----------------------------------------------------
def init_S():
    if ndim==1 and not SC:
        return zeros([nk,nw], dtype=complex)

    elif ndim==2 and not SC:
        return zeros([nk,nk,nw], dtype=complex)

    elif ndim==1 and SC:
        return -SC * 0.01 * ones([nk,nw,2,2], dtype=complex)*tau1[None,None,:,:]

    elif ndim==2 and SC:
        return -SC * 0.01 * ones([nk,nk,nw,2,2], dtype=complex)*tau1[None,None,None,:,:]
#----------------------------------------------------
def init_PI():
    if ndim==1:
        return zeros([nk,nw+1], dtype=complex)

    elif ndim==2:
        return zeros([nk,nk,nw+1], dtype=complex)
#----------------------------------------------------
def init_SR():
    if ndim==1 and not SC:
        return zeros([nk,len(w)], dtype=complex)

    elif ndim==2 and not SC:
        return zeros([nk,nk,len(w)], dtype=complex)

    elif ndim==1 and SC:
        return -SC * 0.01 * ones([nk,len(w),2,2], dtype=complex)*tau1[None,None,:,:]

    elif ndim==2 and SC:
        return -SC * 0.01 * ones([nk,nk,len(w),2,2], dtype=complex)*tau1[None,None,None,:,:]
#----------------------------------------------------
def init_PIR():
    if ndim==1:
        return zeros([nk,len(w)], dtype=complex)

    elif ndim==2:
        return zeros([nk,nk,len(w)], dtype=complex)
#----------------------------------------------------
def compute_Gsum(G):
    if ndim==1 and not SC:
        Gsum_plus  = zeros([nk,len(w)], dtype=complex)
        Gsum_minus = zeros([nk,len(w)], dtype=complex)
        for iw in range(len(w)):
            Gsum_plus[:,iw]  = sum(G/((w[iw]+iwm)[None,:]), axis=1) / beta
            Gsum_minus[:,iw] = sum(G/((w[iw]-iwm)[None,:]), axis=1) / beta

    elif ndim==2 and not SC:
        Gsum_plus  = zeros([nk,nk,len(w)], dtype=complex)
        Gsum_minus = zeros([nk,nk,len(w)], dtype=complex)
        for iw in range(len(w)):
            Gsum_plus[:,:,iw]  = sum(G/((w[iw]+iwm)[None,None,:]), axis=2) / beta
            Gsum_minus[:,:,iw] = sum(G/((w[iw]-iwm)[None,None,:]), axis=2) / beta

    elif ndim==1 and SC:
        Gsum_plus  = zeros([nk,len(w),2,2], dtype=complex)
        Gsum_minus = zeros([nk,len(w),2,2], dtype=complex)
        for iw in range(len(w)):
            Gsum_plus[:,iw,:,:]  = sum(G/((w[iw]+iwm)[None,:,None,None]), axis=1) / beta
            Gsum_minus[:,iw,:,:] = sum(G/((w[iw]-iwm)[None,:,None,None]), axis=1) / beta
        Gsum_plus  = einsum('ab,kwbc->kwac',tau3,Gsum_plus)
        Gsum_minus = einsum('ab,kwbc,cd->kwad',tau3,Gsum_minus,tau3)

    elif ndim==2 and SC:
        Gsum_plus  = zeros([nk,nk,len(w),2,2], dtype=complex)
        Gsum_minus = zeros([nk,nk,len(w),2,2], dtype=complex)
        for iw in range(len(w)):
            Gsum_plus[:,:,iw,:,:]  = sum(G/((w[iw]+iwm)[None,None,:,None,None]), axis=2) / beta
            Gsum_minus[:,:,iw,:,:] = sum(G/((w[iw]-iwm)[None,None,:,None,None]), axis=2) / beta
        Gsum_plus  = einsum('ab,...bc->...ac',tau3,Gsum_plus)
        Gsum_minus = einsum('ab,...bc,cd->...ad',tau3,Gsum_minus,tau3)

    return Gsum_plus, Gsum_minus    
#----------------------------------------------------
def compute_G(S, mu):    
    if not SC and ndim==1:
        return 1.0/(iwm[None,:] - (ek[:,None]-mu) - S)

    elif not SC and ndim==2:
        return 1.0/(iwm[None,None,:] - ek[:,:,None] - S)

    elif SC and ndim==1:
        return linalg.inv(\
            iwm[None,:,None,None]*tau0[None,None,:,:] \
          - (ek[:,None,None,None]-mu)*tau3[None,None,:,:] \
          - S)

    elif SC and ndim==2:
        return linalg.inv(\
            iwm[None,None,:,None,None]*tau0[None,None,None,:,:] \
          - (ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
          - S)
#----------------------------------------------------
def compute_D(PI):
    if ndim==1:
        return 1.0/(-((vn**2)[None,:] + omega**2)/(2.0*omega) - PI) 

    elif ndim==2:
        return 1.0/(-((vn**2)[None,None,:] + omega**2)/(2.0*omega) - PI) 
#----------------------------------------------------
def compute_GR(SR, mu):
    if ndim==1 and not SC:
        return 1.0/(iwm[None,:] - (ek[:,None]-mu) + idelta - SR)

    elif ndim==2 and not SC:
        return 1.0/(iwm[None,None,:] - (ek[:,:,None]-mu) + idelta - SR)

    elif ndim==1 and SC:
        return linalg.inv(\
            (w[None,None,:,None,None]+idelta)*tau0[None,None,None,:,:] \
          - (ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
          - SR)

    elif ndim==2 and SC:
        return linalg.inv(\
            (w[None,None,:,None,None]+idelta)*tau0[None,None,None,:,:] \
          - (ek[:,:,None,None,None]-mu)*tau3[None,None,None,:,:] \
          - SR)
#----------------------------------------------------
def compute_DR(PIR):
    if ndim==1:
        return 1.0/(DRbareinv[None,:] - PIR) 

    elif ndim==2:
        return 1.0/(DRbareinv[None,None,:] - PIR) 
#----------------------------------------------------
def compute_fill(G):
    if not SC:
        return 1.0 + 2.0/(beta * nk_total) * sum(G).real

    elif ndim==1 and SC:
        return 1.0 + 2.0/(beta * nk_total) * sum(G[:,:,0,0]).real

    elif ndim==2 and SC:
        return 1.0 + 2.0/(beta * nk_total) * sum(G[:,:,:,0,0]).real
#----------------------------------------------------
def compute_S(G, D):
    if ndim==1 and not SC:
        return -g**2/(beta*nk_total) * conv(G, D, ['k-q,q','m,n-m'], [0,1], [True,False])[:,:nw]

    elif ndim==2 and not SC:
        return -g**2/(beta*nk_total) * conv(G, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False])[:,:,:nw]

    elif ndim==1 and SC:
        tau3Gtau3 = einsum('ab,...bc,cd->...ad', tau3, G, tau3)
        return -g**2/(beta*nk_total) * conv(tau3Gtau3, D[:,:,None,None]*ones([nk,len(vn),2,2]), ['k-q,q','m,n-m'], [0,1], [True,False])[:,:nw,:,:]

    elif ndim==2 and SC:
        tau3Gtau3 = einsum('ab,...bc,cd->...ad', tau3, G, tau3)
        return -g**2/(beta*nk_total) * conv(tau3Gtau3, D[:,:,:,None,None]*ones([nk,nk,len(vn),2,2]), ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False])[:,:,:nw,:,:]
#----------------------------------------------------
def compute_PI(G):
    if ndim==1 and not SC:
        return 2.0*g**2/(beta*nk_total) * conv(G, G, ['k,k+q','m,m+n'], [0,1], [True,False])[:,:nw+1]

    elif ndim==2 and not SC:
        return 2.0*g**2/(beta*nk_total) * conv(G, G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False])[:,:,:Nw+1]

    elif ndim==1 and SC:
        tau3G = einsum('ab,...bc->...ac', tau3, G)
        return 2.0*g**2/(beta*nk_total) * 0.5*einsum('...aa->...', conv(tau3G, tau3G, ['k,k+q','m,m+n'], [0,1], [True,False], op='...ab,...bc->...ac')[:,:nw+1,:,:])
        
    elif ndim==2 and SC:
        tau3G = einsum('ab,...bc->...ac', tau3, G)
        return 2.0*g**2/(beta*nk_total) * 0.5*einsum('...aa->...', conv(tau3G, tau3G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:nw+1,:,:])
#----------------------------------------------------
def compute_S_real_axis(GR, DR, Gsum):
    if ndim==1 and not SC:
        B  = -1.0/pi * DR.imag
        return -g**2*dw/nk_total*(conv(B, Gsum, ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)] \
               -conv(B*(1+nB)[None,:], GR, ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)] \
               +conv(B, GR*nF[None,:], ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)])

    elif ndim==2 and not SC:
        B  = -1.0/pi * DR.imag
        return -g**2*dw/nk_total*(conv(B, Gsum, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
             -conv(B*(1+nB)[None,None,:], GR, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
             +conv(B, GR*nF[None,None,:], ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)])

    elif ndim==1 and SC:
        B  = -1.0/pi * DR.imag[:,:,None,None] * ones([nk,len(w),2,2])
        tau3GRtau3 = einsum('ab,...bc,cd->...ad',tau3,GR,tau3)
        return -g**2*dw/nk_total*(conv(B, Gsum, ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)] \
               -conv(B*(1+nB[None,:,None,None]), tau3GRtau3, ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)] \
               +conv(B, tau3GRtau3*nF[None,:,None,None], ['k-q,q','z,w-z'], [0,1], [True,False])[:,:len(w)])

    elif ndim==2 and SC:
        B  = -1.0/pi * DR.imag[:,:,:,None,None] * ones([nk,nk,len(w),2,2])
        tau3GRtau3 = einsum('ab,...bc,cd->...ad',tau3,GR,tau3)
        return -g**2*dw/nk_total*(conv(B, Gsum, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
               - conv(B*(1+nB[None,None,:,None,None]), tau3GRtau3, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
               + conv(B, tau3GRtau3*nF[None,None,:,None,None], ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)])
#----------------------------------------------------
def compute_PI_real_axis(GR, Gsum):
    if ndim==1 and not SC:
        GA = conj(GR)
        A  = -1.0/pi * GR.imag
        return 2.0*g**2*dw/nk_total * (conv(A, Gsum, ['k+q,k','z,w-z'], [0,1], [True,False])[:,:len(w)] \
              -conv(A, GA*nF[None,:], ['k+q,k','w+z,z'], [0,1], [True,False])[:,:len(w)] \
              +conv(A*nF[None,:], GA, ['k+q,k','w+z,z'], [0,1], [True,False])[:,:len(w)])

    elif ndim==2 and not SC:
        GA = conj(GR)
        A  = -1.0/pi * GR.imag
        return 2.0*g**2*dw/nk_total * (conv(A, Gsum, ['k+q,k','k+q,k','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
              -conv(A, GA*nF[None,None,:], ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
              +conv(A*nF[None,None,:], GA, ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:len(w)])

    elif ndim==1 and SC:
        tau3GA = einsum('ab,kwbc->kwac', tau3, conj(GR))
        tau3A  = einsum('ab,kwbc->kwac', tau3, -1.0/pi * GR.imag)
        return 2.0*g**2*dw/nk_total * 0.5*einsum('kwaa->kw', conv(tau3A, Gsum, ['k+q,k','z,w-z'], [0,1], [True,False], op='kwab,kwbc->kwac')[:,:len(w)] \
              -conv(tau3A, tau3GA*nF[None,:,None,None], ['k+q,k','w+z,z'], [0,1], [True,False], op='kwab,kwbc->kwac')[:,:len(w)] \
              +conv(tau3A*nF[None,:,None,None], tau3GA, ['k+q,k','w+z,z'], [0,1], [True,False], op='kwab,kwbc->kwac')[:,:len(w)])

    elif ndim==2 and SC:
        tau3GA = einsum('ab,...bc->...ac', tau3, conj(GR))
        tau3A  = einsum('ab,...bc->...ac', tau3, -1.0/pi * GR.imag)
        return 2.0*g**2*dw/nk_total * 0.5*einsum('...aa->...', conv(tau3A, Gsum, ['k+q,q','k+q,k','z,w-z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:len(w)] \
             -conv(tau3A, tau3GA*nF[None,None,:,None,None], ['k+q,q','k+q,k','w+z,z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:len(w)] \
             +conv(tau3A*nF[None,None,:,None,None], tau3GA, ['k+q,q','k+q,k','w+z,z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:len(w)])


