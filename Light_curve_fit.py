#!/usr/bin/env python
# coding: utf-8

# In[27]:


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from astropy.io import fits
from astropy.time import Time, TimeDelta


# In[28]:


lc_data = pd.read_csv("Table.csv")


# In[29]:


pd.set_option('display.max_columns', None)
lc_data


# In[30]:


time = lc_data["HJD_UTC"].values.astype(np.float64)
flux = lc_data["rel_flux_T1"].values.astype(np.float64)
flux_error = lc_data["rel_flux_err_T1"].values.astype(np.float64)


# In[31]:


fits_dir = "/Users/swapnaneeldey/Desktop/R_filter_26f23/"
unique_time = []
for file in sorted(os.listdir(fits_dir)):
    if file[0:4] == "wasp":
        hdu = fits.open(fits_dir + file)
        head = hdu[0].header
        t = head["UTC-OBS"]
        print(t)
        hours, minutes, seconds = map(float, t.split(':'))
        time_minutes = hours*60*60 + minutes*60 + seconds
        if time_minutes == 327.578:
            print(file)
        unique_time.append(time_minutes/60)
unique_time = np.array(unique_time)
time = unique_time - unique_time[0]


# In[ ]:


fig = plt.subplot()
plt.scatter(time, flux)
plt.errorbar(time, flux, yerr = flux_error, fmt = ".", color = "red")
plt.title("Light Curve data points")


# ## Fitting

# In[32]:


n = 50
new_time = (time - time[0])/(24*60)
def fourier_series(phi, *params):
    m0 = params[0]
    a = params[1:n+1]
    b = params[n+1:]
    terms = [a[i] * np.sin(2 * np.pi * (i+1) * phi) + b[i] * np.cos(2 * np.pi * (i+1) * phi) for i in range(n)]
    return m0 + sum(terms)

def straight(x,a,b):
    return a*x + b

# Define the phase
phi = new_time

time_array = time - time[0]
# Initial guess for Fourier coefficients (m0, a1-a20, b1-b20)
initial_guess = [np.mean(flux)] + [0.1] * n*2  # Initial guess for m0 and all a and b coefficients

# Fit Fourier series model to data
params, covariance = curve_fit(fourier_series, phi, flux, p0=initial_guess, sigma = flux_error)

# Compute Fourier series values
fourier_fit = fourier_series(phi, *params)

popt, pcov = curve_fit(straight, phi[185:335], fourier_fit[185:335])
p1,pcov1 = curve_fit(straight, phi[0:160], flux[0:160])
p2,pcov2 = curve_fit(straight, phi[156:189], flux[156:189])
p3,pcov3 = curve_fit(straight, phi[331:365], flux[331:365])
p4,pcov4 = curve_fit(straight, phi[360:482], flux[360:482])


straight_fit = straight(phi[185:335], *popt)
straight_fit1 = straight(phi[0:160], *p1)
straight_fit2 = straight(phi[156:189], *p2)
straight_fit3 = straight(phi[331:365], *p3)
straight_fit4 = straight(phi[360:482], *p4)



# Plot original data and Fourier fit
plt.figure()
plt.scatter(time_array, flux, label='Original Data', color='blue', marker = ".")
plt.errorbar(time_array, flux, yerr = flux_error, c = "gray", alpha = 0.3)
plt.plot(time_array[0:160], straight_fit1, color = "red")
plt.plot(time_array[159:189], fourier_fit[159:189], color = "Cyan", label = "fourier fit")
plt.plot(time_array[185:335], straight_fit, color = "red")
plt.plot(time_array[335:363], fourier_fit[335:363], color = "Cyan")
plt.plot(time_array[360:482], straight_fit4, color = "red", label = "straight fit")
plt.xlabel('Time')
plt.ylabel('Relative Flux')
plt.title('Light Curve and Fourier Fit')
plt.legend()
plt.grid(True)
plt.savefig("finalline")
plt.show()


# In[33]:


new_time = (time - time[0])/(24*60)
n = 5
def poly(x, *params):
    y = sum([params[i] * x**i for i in range(n)])
    return y
phi_mask = new_time.copy()
phi_mask[161:363] = np.nan
valid_indices = ~np.isnan(phi_mask)
guess = [1]*n
popt, pcov = curve_fit(poly, phi_mask[valid_indices], flux[valid_indices], p0 = guess, sigma = flux_error[valid_indices])
straight_15_fit = poly(new_time, *popt)
plt.scatter(new_time * 24 *60, flux, label='Original Data', color='blue', marker = ".")
plt.errorbar(new_time * 24 *60, flux, yerr = flux_error, c = "gray", alpha = 0.3)
plt.plot(new_time * 24 *60, straight_15_fit, color = "lightgreen")


# In[34]:


plt.scatter(new_time * 24 *60, flux, label='Original Data', color='blue', marker = ".")
plt.errorbar(new_time * 24 *60, flux, yerr = flux_error, c = "gray", alpha = 0.3)
plt.xlabel('Time [mins]')
plt.ylabel('Relative Flux')
plt.title('WASP0845+53 Primary eclipse Light Curve')
plt.legend()
plt.grid(True)
plt.savefig("og LC")


# In[35]:


"""offset1 = flux[0:161] - straight_15_fit[0:161]
offset2 = flux[363 : 482] - straight_15_fit[363 : 482]

#new_data = np.concatenate((straight_15_fit[0:161], flux[161:363], straight_15_fit[363 : 482]))
new_data = np.concatenate((offset1+0.19, flux[161:363], offset2+0.19))"""

new_data = flux - straight_15_fit + 0.19

plt.scatter(new_time * 24 *60, new_data, label='adjusted Data', color='blue', marker = ".")
plt.errorbar(new_time * 24 *60, new_data, yerr = flux_error, c = "gray", alpha = 0.3)
plt.xlabel('Time[mins]')
plt.ylabel('Relative Flux')
plt.title('WASP0845+53 subtracted P.E. LC')
plt.legend()
plt.grid(True)
plt.axvline(109, linestyle = "--")
plt.axvline(125, linestyle = "--")
plt.axvline(227, linestyle = "--")
plt.axvline(245, linestyle = "--")
plt.savefig("subtracted_light_curve")


# In[36]:


n = 50
new_time = (time - time[0])/(24*60)
def fourier_series(phi, *params):
    m0 = params[0]
    a = params[1:n+1]
    b = params[n+1:]
    terms = [a[i] * np.sin(2 * np.pi * (i+1) * phi) + b[i] * np.cos(2 * np.pi * (i+1) * phi) for i in range(n)]
    return m0 + sum(terms)

def straight(x,a,b):
    return a*x + b

# Define the phase
phi = new_time

time_array = (time - time[0])
# Initial guess for Fourier coefficients (m0, a1-a20, b1-b20)
initial_guess = [np.mean(new_data)] + [0.1] * n*2  # Initial guess for m0 and all a and b coefficients

# Fit Fourier series model to data
params, covariance = curve_fit(fourier_series, phi, new_data, p0=initial_guess, sigma = flux_error)

# Compute Fourier series values
new_fourier_fit = fourier_series(phi, *params)

popt, pcov = curve_fit(straight, phi[185:335], new_fourier_fit[185:335])
p1,pcov1 = curve_fit(straight, phi[0:160], new_data[0:160])
p2,pcov2 = curve_fit(straight, phi[156:189], new_data[156:189])
p3,pcov3 = curve_fit(straight, phi[331:365], new_data[331:365])
p4,pcov4 = curve_fit(straight, phi[360:482], new_data[360:482])


new_straight_fit_e = straight(phi[185:335], *popt)
new_straight_fit1 = straight(phi[0:160], *p1)
new_straight_fit2 = straight(phi[156:189], *p2)
new_straight_fit3 = straight(phi[331:365], *p3)
new_straight_fit4 = straight(phi[360:482], *p4)



# Plot original data and Fourier fit
plt.figure()
plt.scatter(time_array, new_data, label='Subtracted Data', color='blue', marker = ".")
plt.errorbar(time_array, new_data, yerr = flux_error, c = "gray", alpha = 0.3)
plt.plot(time_array[0:160], new_straight_fit1, color = "red")
plt.plot(time_array[160:187], new_fourier_fit[160:187], color = "Cyan", label = "Fourier decomposition fit")
plt.plot(time_array[185:335], new_straight_fit_e, color = "red")
plt.plot(time_array[336:363], new_fourier_fit[336:363], color = "Cyan")
plt.plot(time_array[360:482], new_straight_fit4, color = "red", label = "straight fit")
plt.axvline(109, linestyle = "--")
plt.axvline(125, linestyle = "--")
plt.axvline(227, linestyle = "--")
plt.axvline(245, linestyle = "--")

plt.xlabel('Time')
plt.ylabel('Relative Flux')
plt.title('WASP0845+53 subtracted P.E. LC')
plt.legend(fontsize=7)
plt.grid(True)
plt.savefig("Final_light_curve")
plt.show()



# In[ ]:


residual1 = new_data[0:160] - new_straight_fit1
residual2 = new_data[161:185] - new_fourier_fit[161:185]
residual3 = new_data[186:336] - new_straight_fit_e
residual4 = new_data[336:363] - new_fourier_fit[336:363]
residual5 = new_data[360:482] - new_straight_fit4
residual = np.concatenate((residual1, residual2, residual3, residual4, residual5))
print(len(residual))
plt.figure(figsize=(10, 6))
plt.scatter(time_array, residual[0:482], color='red', marker='o', label='Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8) 
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.savefig("residuals")


# In[38]:


T = 101 
tau = 16 
G = 6.6743 * 10**(-11) #m^3/kg/s^2
#period calculation
P = np.pi*(T+tau)/0.303
P = P*60
P/(24*3600)


# In[95]:


ma = np.linspace(0.8, 1.5, 15)
mb = np.linspace(0.08, 0.5, 15)
mt = np.array([])
for i in range(len(ma)):
    for j in range(len(mb)):
        sum_mass = ma[i] + mb[j]
        mt = np.append(mt, sum_mass)
ma, mb, mt


# In[96]:


a = (G*mt*(1.9891 * 10**30)*(P**2)/(4*(np.pi**2)))**(1/3)
a = a/(7*10**8)
a


# In[71]:


x = np.array([])
for i in range(len(ma)):
    for j in range(len(mb)):
        ratio = mb[j]/ma[i]
        eta = 0.38 + 0.2*np.log10(ratio)
        x = np.append(x, eta)
x


# In[119]:


x = np.array([])
for i in range(len(ma)):
    for j in range(len(mb)):
        ratio = ma[i]/mb[j]
        eta = (0.49*ratio**(2/3))/((0.6*ratio**(2/3))+np.log(1+ratio**(1/3)))
        x = np.append(x, eta)
x


# In[143]:


roche_lobe = x * a
roche_lobe


# In[144]:


#x = x.reshape(len(ma), len(mb))
#limit = np.abs(roche_lobe - rb)
roche_lobe= roche_lobe.reshape(len(ma), len(mb))
#limit = limit.reshape(len(ma), len(mb))
plt.pcolormesh(mb, ma, roche_lobe, cmap='viridis')
plt.colorbar(label='Value of roche lobe ($R_{\odot}$)')
contour_levels = np.linspace(np.min(roche_lobe), np.max(roche_lobe), 15)
contour = plt.contour(mb, ma, roche_lobe, levels=contour_levels, colors='white')
plt.clabel(contour, inline=True, fontsize=8, fmt= '%.2f $R_{\odot}$')
plt.axvline(0.19, c = "r")
plt.axhline(1.45, c = "r")

plt.xlabel(f'$M_b$')
plt.ylabel(f'$M_a$')
plt.title("Roche lobe size of $M_a$ for different mass values")
plt.savefig("roche Ma")


# In[104]:


1.45/0.19


# In[142]:


ratio = 1/0.45
(0.49*ratio**(2/3))/((0.6*ratio**(2/3))+np.log(1+ratio**(1/3)))*4.37
#0.38 + 0.2*np.log10(ratio)


# In[134]:


2.431286779525567 + 0.9768309757290186


# In[79]:


ra = 0.303 * a
rb = ra * tau/(T+tau)
ra,rb


# In[1]:


lb_la = ((15000/8000)**4)*(rb/ra)**2
lb_la


# In[ ]:




