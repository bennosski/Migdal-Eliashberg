#from matplotlib.pyplot import *
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
from numpy import *
import sys
import os

def main(folder):

    Nk = load(folder+'Nk.npy')[0]

    if os.path.exists(folder+'iwm.npy'):
        iwm = load(folder+'iwm.npy')
        S = load(folder+'S.npy')

        if len(shape(S))==1:
            figure()
            plot(iwm.imag, S.real)
            plot(iwm.imag, S.imag)
            xlabel('wm')
            savefig(folder+'SM')
            close()
        elif len(shape(S))==3:
            figure()
            plot(iwm.imag, S[:,0,0].real)
            plot(iwm.imag, S[:,0,0].imag)
            xlabel('wm')
            savefig(folder+'SM')
            close()

            figure()
            plot(iwm.imag, S[:,0,1].real)
            plot(iwm.imag, S[:,0,1].imag)
            xlabel('wm')
            savefig(folder+'SM01')
            close()
        else:
            figure()
            plot(iwm.imag, S[Nk//4,:,0,0].real)
            plot(iwm.imag, S[Nk//4,:,0,0].imag)
            xlabel('wm')
            savefig(folder+'SM')
            close()

            figure()
            plot(iwm.imag, S[Nk//4,:,0,1].real)
            plot(iwm.imag, S[Nk//4,:,0,1].imag)
            xlabel('wm')
            savefig(folder+'SM01')
            close()

            
    GR = load(folder+'GR.npy')
    w  = load(folder+'w.npy')

    if os.path.exists(folder+'DR.npy'):
        DR = load(folder+'DR.npy')
        B = -1.0/pi*DR.imag

        figure()
        imshow(B.T, aspect='auto', origin='lower', extent=[-pi,pi,w[0],w[-1]], vmin=0)
        ylim(0.0, 1.0)
        colorbar()
        savefig(folder+'Bkw')
        close()

    if os.path.exists(folder+'PIR.npy'):
        PIR = load(folder+'PIR.npy')
        figure()
        plot(w,PIR[Nk//4,:].real)
        plot(w,PIR[Nk//4,:].imag)
        title('PIkf')
        savefig(folder+'PIkf')


    if len(shape(GR))>2:
        A = -1.0/pi * imag(GR[:,:,0,0])
    else:
        A = -1.0/pi * imag(GR)

    print(shape(A))
    
    figure()
    imshow(A.T, aspect='auto', origin='lower', extent=[-pi,pi,w[0],w[-1]])
    ylim(-1.0, 1.0)
    colorbar()
    savefig(folder+'Akw')
    close()

    SR = load(folder+'SR.npy')
    if len(shape(SR))==4:
        x = SR[Nk//4,:,0,0]
    elif len(shape(SR))==3:
        x = SR[:,0,0]
        
    figure()
    plot(w, x.real)
    plot(w, x.imag)
    savefig(folder+'Skf')
    close()

    if len(shape(SR))==3:
        figure()
        plot(w, SR[:,0,1].real)
        plot(w, SR[:,0,1].imag)
        savefig(folder+'Skf01')
        close()
    elif len(shape(SR))==4:
        figure()
        plot(w, SR[Nk//4,:,0,1].real)
        plot(w, SR[Nk//4,:,0,1].imag)
        savefig(folder+'Skf01')
        close()

    
if __name__=='__main__':
    folder = sys.argv[1]
    main(folder)
