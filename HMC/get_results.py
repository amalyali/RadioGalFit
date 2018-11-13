#######################################################################
# Radio Galaxy shape measurement from HMC chain  (2018)               #
# 1. Gelman-Rubin test for chain convergence                          #
# 2. Measured values computation (mean and std of the posterior)      #
# 3. Comparison to original values (bestfit line)                     #
# Author: Marzia Rivi                                                 #
#                                                                     #
# Input: filename - chain file                                        #
#       ns - number of sersic sources                                 #
#       N  - HMC chain length                                         #
#                                                                     #
#                                                                     #
#######################################################################

import sys
import argparse
import math
import numpy as np
import GR   
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='data_values')
parser.add_argument('filename', help='Input chain filename')
parser.add_argument('-ns',dest='nssrc', type=int, default=1, help='Number of Sersic Galaxies')

args = parser.parse_args(sys.argv[1:])
nsrc = args.nssrc

# Read chain values (first column of the file returns the likelihood)
start_col=1
burn = 250
c=np.loadtxt(args.filename+'.chain')[burn:,start_col:]
print len(c)

ind=len(c[:, 0])/3
c1 = c[:ind, :]
c2 = c[ind:2*ind, :]
c3 = c[2*ind:3*ind, :]


### Test convergence with gelman rubin ###
step, results = GR.convergence_test([c1,c2,c3],jump=100)
if step!=-1:
    print 'Chain converged within',step,'steps'
else:
    print 'Chain unconverged'

# Load original source catalog
#orig_values = np.loadtxt('galaxy_catalog_20.0_10.0-200.0_uJy.txt')[:nsrc,:]

RAD2ARCSEC = 648000.0/np.pi
mvalues = np.empty((nsrc,6))
true_vals = np.empty((nsrc,6))
 
j=0
for i in range(nsrc):
    e1v = c[:,i]
    e2v = c[:,nsrc+i]
    scalev = c[:,2*nsrc+i]
    std1 = np.std(e1v)
    std2 = np.std(e2v)
    stdscale = np.std(scalev)
    if std1 > 1e-4 and std2 > 1e-4 and stdscale > 1e-8:
        mvalues[j] = [np.mean(e1v),std1,np.mean(e2v),std2,np.mean(scalev)*RAD2ARCSEC,stdscale*RAD2ARCSEC]
        true_vals[j,:]=orig_values[j,:]
        j = j+1
    else:
        print orig_values[i,2],std1,std2,stdscale,stdscale*RAD2ARCSEC

print "bad: ",nsrc-j
nsrc = j

mvalues.resize((j,6))
true_vals.resize((j,6))

np.savetxt(args.filename+'.results',mvalues,fmt='%.3e')



def best_fit(x,y,erry):
    
    sigma2=np.multiply(erry,erry)
    s=np.sum(1/sigma2)
    sx=np.sum(np.divide(x,sigma2))
    sy=np.sum(np.divide(y,sigma2))
    sxy=np.sum(np.divide(np.multiply(x,y),sigma2))
    sx2=np.sum(np.divide(np.multiply(x,x),sigma2))
    
    det = sx2*s-sx*sx
    m=(s*sxy-sx*sy)/det
    errm=np.sqrt(s/det)
    c=(-sx*sxy+sx2*sy)/det
    errc=np.sqrt(sx2/det)
    return (m,errm,c,errc)

def func(x,a,b,N):
    m = np.zeros(N) + a
    y=m*x+b
    return y


def plot_data(str,x,y,erry):
    plt.figure()
    plt.errorbar(x,y,yerr=erry,fmt='.')
    plt.axis([-0.8,0.8,-0.8,0.8])
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.xlabel('input',fontsize=20)
    plt.ylabel('measured',fontsize=20)
    plt.title(str,fontsize=22)
    m,errm,c,errc = best_fit(x,y,erry)
    plt.plot(x,func(x,m,c,len(x)))
    #plt.plot([-0.8,0.8],[-0.8,0.8])
    print str+": m = ",m,"+/-",errm
    print "     c = ",c,"+/-",errc
    return


def plot_binned_data(str,x,y,erry):
    bins = np.arange(-0.8, 0.8, 0.1)
    dbin = 0.1
    x_means = []
    y_means = []
    ybin_err = []
    data = np.array([x,y,erry])
    for bin in bins:
        mask = (data[0,:] >= bin) & (data[0,:] < bin + dbin)
        if mask.any():
            binned_x = data[0,:][mask]
            binned_y = data[1,:][mask]
            binned_erry = data[2,:][mask]
            binned_erry = np.divide(1.,np.multiply(binned_erry,binned_erry))
            binxmean = np.mean(binned_x)
            binymean = np.average(binned_y,weights=binned_erry)
            binyerr = np.sqrt(1./np.sum(binned_erry))
            x_means.append(binxmean)
            y_means.append(binymean)
            ybin_err.append(binyerr)
    
    plot_data(str,x_means,y_means,ybin_err)
    plt.hold
    plt.title('data binned')
    return


# Fitting function for relation between source flux and shape parameters std
def func_fit(x,a,b,k):
    return a*np.exp(-b*np.log(x*0.1))+k


x = true_vals[:,0]*RAD2ARCSEC
y = true_vals[:,1]*RAD2ARCSEC
flux = true_vals[:,2]
plt.scatter(x,y,marker='.')
plt.xlabel('l [arcsec]',fontsize=18)
plt.ylabel('m [arcsec]',fontsize=18)
plt.title('Sources positions')


e1 = true_vals[:,4]
me1 = mvalues[:,0]
err1 = mvalues[:,1]
plot_data('$e_1$',e1,me1,err1)

plt.figure()
plt.scatter(flux,err1)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('S [muJy]',fontsize=20)
plt.ylabel('std',fontsize=20)
plt.title('$e_1$',fontsize=22)
#plt.axis([0.,50000.,0.,0.08])
#fitParams,fitCovariances =curve_fit(func_fit,flux,err1)

#print "fitting err1: ",fitParams[0],fitParams[1],fitParams[2]
#plt.hold
#plt.plot(flux,func_fit(flux,fitParams[0],fitParams[1],fitParams[2]), color='black')
#plt.hold
#plt.plot(flux,func_fit(flux,7.92,1.23,0.003), color='black')



e2 = true_vals[:,5]
me2 = mvalues[:,2]
err2 = mvalues[:,3]
plot_data('$e_2$',e2,me2,err2)

plt.figure()
plt.scatter(flux,err2)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('S [muJy]',fontsize=20)
plt.ylabel('std',fontsize=20)
plt.title('$e_2$',fontsize=22)
#plt.axis([0.,50000.,0.,0.08])
#fitParams,fitCovariances =curve_fit(func_fit,flux,err2)
#plt.hold
#plt.plot(flux,func_fit(flux,fitParams[0],fitParams[1],fitParams[2]), color='black')
#print "fitting err2: ",fitParams[0],fitParams[1],fitParams[2]
#plt.plot(flux,func_fit(flux,7.92,1.23,0.003), color='black')

plot_binned_data('$e_1$ bin',e1,me1,err1)
plot_binned_data('$e_2$ bin',e2,me2,err2)

scale = true_vals[:,3]
mscale = mvalues[:,4]
errscale = mvalues[:,5]
plt.figure()
plt.scatter(scale,mscale,marker='.')
plt.errorbar(scale,mscale,yerr=errscale,fmt='.')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=10)
plt.title('scalelength [arcsec]',fontsize=22)
plt.xlabel('input',fontsize=20)
plt.ylabel('measured',fontsize=20)

m,errm,c,errc = best_fit(scale,mscale,errscale)
plt.plot(scale,func(scale,m,c,len(scale)))
print "scale: m = ",m,"+/-",errm
print "       c = ",c,"+/-",errc

plt.figure()
plt.scatter(flux,errscale)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('S [muJy]',fontsize=20)
plt.ylabel('std [arcsec]',fontsize=20)
plt.title('scalelength',fontsize=22)
#plt.axis([0.,50000.,0.,0.3])
#fitParams,fitCovariances =curve_fit(func_fit,flux,errscale)
#plt.hold
#plt.plot(flux,func_fit(flux,fitParams[0],fitParams[1],fitParams[2]), color='black')
#print "fitting errscale: ",fitParams[0],fitParams[1],fitParams[2]

#plt.plot(flux,func_fit(flux,1.16,0.45,0.007), color='black')


plt.show()

