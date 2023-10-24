import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from IPython import embed
def calc_eta(ohm):
    temps = np.arange(10,40,1)
    ohms = np.array([3.239,3.118,3.004,2.897,2.795,2.700,2.610,2.526,2.446,2.371,
                     2.3,2.233,2.169,2.110,2.053,2,1.95,1.902,1.857,1.815,
                     1.774,1.736,1.7,1.666,1.634,1.603,1.574,1.547,1.521,1.496]) # 10^6 ohm

    f = interpolate.interp1d(ohms, temps)
    temp_new = f(ohm)
    a, b = np.polyfit([20,26], [1.8240,1.8520], 1)
    eta = (a*temp_new+b)*1e-5
    return eta

def calc_radius(p, eta, v_f):
    '''
    p: barometric pressure [Pa]
    eta: viscosity of air in [N• s / m2] (See Appendix A)
    v_f: velocity of fall [m/s]
    '''
    b = 8.20e-3 #[Pa • m]
    g = 9.8 #[m/s2]
    rho = 886  # kg/m3

    a = np.sqrt((b/(2*p))**2 + 9*eta*v_f/(2*g*rho)) - b/(2*p)
    return a, rho


def calc_mass(p, eta, v_f):
    a, rho = calc_radius(p, eta, v_f)
    m = 3/4*np.pi*a**3*rho

    return m

def calc_charge(p, eta, v_f, v_r, V):
    '''
    p: barometric pressure [Pa]
    eta: viscosity of air in [N• s / m2] (See Appendix A)
    v_f: velocity of fall [m/s]
    v_r: velocity of rise [m/s]
    V: potential difference across the plates [V]
    '''
    g = 9.8  # [m/s2]
    d = 31.2e-3  #m
    m = calc_mass(p, eta, v_f)
    q = m*g*d*(v_r+v_f)/(V*v_f)

    return q

print(calc_charge(1e5, 1.84e-6, 0.00001, 0.00005, 510))

def compile_data(file_name):
    data_excel = pd.read_excel(file_name,sheet_name="data_python")
    v_f_data = np.array(data_excel['v_f'].tolist())
    v_r_data = np.array(data_excel['v_r'].tolist())
    V_data = np.array(data_excel['V'].tolist())
    eta_data = calc_eta(np.array(data_excel['resistance'].tolist()))
    names = np.array(data_excel['id'].tolist())
    charge_err = np.array(data_excel['q_err'].tolist())
    radius_err = np.array(data_excel['a_err'].tolist())
    charges = []
    radius = []

    for id, v_f in enumerate(v_f_data):
        charges.append(calc_charge(1e5, eta_data[id], v_f, v_r_data[id], V_data[id]))
        radius.append(calc_radius(1e5, eta_data[id], v_f)[0])
    charges = np.asarray(charges)
    radius = np.asarray(radius)

    return charges, radius, names, charge_err, radius_err

def plot_hisogram(charges):
    plt.hist(charges,bins = np.linspace(charges.min(),charges.max(),100))
    plt.xlabel('Charge/drop [Coulombs]')
    plt.title('Charge distribution of droplets')
    plt.show()
def linearFunc(x,slope):
    y = slope * x
    return y
def plot_discrete_e(charges,names, charge_err):
    charges_sort = np.asarray(sorted(charges))
    x = np.arange(1, len(charges_sort)+1)
    plt.scatter(x,charges_sort,s=30)
    for i, txt in enumerate(charges_sort):
        plt.annotate(names[np.where(charges==txt)], (x[i], charges_sort[i]),fontsize=8)
    ns_interp = np.linspace(1, len(charges_sort)+1, 50)
    a, cov = curve_fit(linearFunc, x, charges_sort, sigma=charge_err, absolute_sigma=True)
    d_slope = np.sqrt(cov[0][0])
    #a,_,_,_ = np.linalg.lstsq(x[:,np.newaxis],charges_sort)
    qs_interp = a[0]*ns_interp
    plt.plot(ns_interp,qs_interp,'-',label = fr'slope:{a[0]:.2E} $\pm$ {d_slope:.2E}',color='r')
    plt.errorbar(x,charges_sort,yerr= charge_err,fmt='o', markersize=8,ecolor = 'b',label = 'error bar in charge',capsize=5)
    plt.xlabel('# of electrons (assumption)')
    plt.ylabel('Charge/drop [Coulombs]')
    plt.grid()
    plt.legend()
    plt.show()

charges, radius, names, charge_err, radius_err = compile_data("lab2_data.xlsx")
plot_hisogram(charges)
plot_discrete_e(charges,names, charge_err)
embed()