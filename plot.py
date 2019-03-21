import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
from numpy import *
import sys
import os

def main(folder):
    print('folder = %s'%folder)

    w = load(folder+'w.npy')

    print('shape w', shape(w))

    SC = True

    GR = load(folder+'GR.npy')
    Nk = load(folder+'Nk.npy')[0]
    SR = load(folder+'SR.npy')
    PIR = None
    if os.path.exists(folder+'PIR.npy'): PIR = load(folder+'PIR.npy')
    DR = load(folder+'DR.npy')

    A = -1.0/pi*GR.imag
    B = -1.0/pi*DR.imag

    if len(shape(A))==5: A = A[:,:,:,0,0]

    figure()
    plot(w, A[Nk//4,Nk//4,:])
    title('$A(k_F,\omega)$')
    xlim(-1.8, 1.8)
    ylim(0, 2.0)
    savefig(folder+'Akfw')

    z = array([A[i, Nk//4] for i in range(Nk)])
    figure()
    imshow(z.T, origin='lower', aspect='auto', extent=[0,len(z),w[0],w[-1]], interpolation='bilinear', vmax=1.0)
    ylim(-2.0, 2.0)
    colorbar()
    savefig(folder+'Akw_cut1')


    ilist = list(reversed(range(Nk//2))) + [0]*(Nk//2-1) + list(range(1,Nk//2)) 
    jlist = [Nk//2]*(Nk//2) + list(reversed(range(Nk//2-1))) + list(range(1,Nk//2)) 
    z = array([A[i,j] for i,j in zip(ilist, jlist)])
    figure()
    imshow(z.T, origin='lower', aspect='auto', extent=[0,len(z),w[0],w[-1]], interpolation='bilinear', vmax=1.0)
    ylim(-3.0, 3.0)
    colorbar()
    savefig(folder+'Akw')

    print(shape(B))
    if len(shape(B))>1:
        z = array([B[i,j] for i,j in zip(ilist, jlist)])
    else:
        z = array([B for _ in range(len(ilist))])

    figure()
    imshow(z.T, origin='lower', aspect='auto', extent=[0,len(z),w[0],w[-1]], interpolation='bilinear', vmin=0.0)
    ylim(0.0, 1.0)
    colorbar()
    savefig(folder+'Bkw')

    DOS =  sum(A, axis=(0,1)) / Nk**2
    figure()
    plot(w, DOS)
    title('DOS')
    xlim(-2.0, 2.0)
    savefig(folder+'DOS')

    if len(shape(SR))==5:
        figure()
        plot(w, SR[Nk//4,Nk//4,:,0,0].real)
        plot(w, SR[Nk//4,Nk//4,:,0,0].imag)
        title('$S(k_F,\omega)$')
        savefig(folder+'Skfw')  

        figure()
        plot(w, SR[Nk//4,Nk//4,:,0,1].real)
        plot(w, SR[Nk//4,Nk//4,:,0,1].imag)
        title('$S01(k_F,\omega)$')
        savefig(folder+'S01kfw')        
    elif len(shape(SR))==3 and SC:
        print('shape SR', shape(SR))
        figure()
        plot(w, SR[:,0,0].real)
        plot(w, SR[:,0,0].imag)
        title('$S(k_F,\omega)$')
        savefig(folder+'Skfw')

        figure()
        plot(w, SR[:,0,1].real)
        plot(w, SR[:,0,1].imag)
        title('$S01(k_F,\omega)$')
        savefig(folder+'S01kfw')
    else:
        figure()
        plot(w, SR.real)
        plot(w, SR.imag)
        title('$S(k_F,\omega)$')
        xlim(-2.0, 2.0)
        savefig(folder+'Skfw')

    if PIR is not None:

        figure()
        plot(w, PIR[Nk//4,Nk//4,:].real)
        plot(w, PIR[Nk//4,Nk//4,:].imag)
        title('$PI(k_F,\omega)$')
        savefig(folder+'PIkfw')


if __name__ == '__main__':
    print('in main')
    main(sys.argv[1])
