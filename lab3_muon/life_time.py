import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 14})
from IPython import embed

'''
reading time data measured in nanoseconds between successive signals that triggered the readout electronics
'''
time=[]
data_excel1 = pd.read_excel('raw_data.xlsx', sheet_name='trial1')
data_excel2 = pd.read_excel('raw_data.xlsx', sheet_name='trial2')
time= data_excel1['data'].tolist()+data_excel2['data'].tolist()
#time= data_excel2['data'].tolist()
time = np.array(time).flatten()

print(len(time))
filter_time=time[np.where(time<40000)]/1000
print(len(filter_time))
print(filter_time.min())
for i in range(len(filter_time)):
    if filter_time[i] < 0.5:
        np.delete(filter_time,i)
print(len(filter_time))
def calc_lifetime(t, tau, A):
    return A*np.exp(-t/tau)


n, bins, _ = plt.hist(filter_time,bins=200)
bins = (bins + np.roll(bins, -1))[:-1] / 2.0
print(bins[:2])
popt, pcov = curve_fit(calc_lifetime, bins, n , p0=[2,10])
loc, scale = expon.fit(filter_time, floc=0)
err = np.sqrt(np.diag(pcov)[0])
print(scale,popt[0])
print(f'error of scale: {err}')
y = expon.pdf(bins, loc, scale)
plt.plot(bins,calc_lifetime(bins, *popt), 'r--', linewidth=2, label=r'fit: decay time =%5.3f $\pm$ %5.3f $\mu s$' % (scale,err))
#plt.plot(bins, calc_lifetime(bins, *popt), 'g--', linewidth=2)
plt.xlabel(r'time ($\mu$s)')
plt.ylabel('events')
plt.legend()
plt.show()

'''
Flux directionality
'''
def zenith_flux(theta,A):
    return A*np.cos(theta/180*np.pi)**2

angle_data = pd.read_excel('raw_data.xlsx', sheet_name='angle_data')
angles = np.array(angle_data['angle'].tolist()[:5])
flux = np.array(angle_data['average'].tolist()[:5])
err = angle_data['std'].tolist()[:5]
print(flux)
popt, pcov = curve_fit(zenith_flux, angles, flux, p0=[1])
plt.scatter(angles,flux)
plt.errorbar(angles,flux,yerr=err,fmt='o', markersize=8,ecolor = 'b',label = 'error bar in flux',capsize=5)
plt.plot(np.arange(0,90),zenith_flux(np.arange(0,90),*popt),'r--',label = r'fit: %5.1f $\cos^2(\theta)$'%(popt[0]),linewidth=2)
plt.xlabel(r'Zenith angle $\theta$')
plt.ylabel(r'Flux [$m^{-2}s^{-1}sr^{-1}$]')
plt.legend()
plt.show()