import src.data_tools as dt
import src.gw_eqns as gwe

import math
import matplotlib.pyplot as plt
import metpy.plots as metplt
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.interpolate
import scipy.optimize
import scipy.signal
import statistics as stats

saveGraphs = False

# Extract data from files
temperature = dt.extract_data("data/high_resolution/Tint18251_31_nw4sg4nFnC") 
north_wind = dt.extract_data("data/high_resolution/Vint18251_29_nw3sg3nFnC")
east_wind = dt.extract_data("data/high_resolution/Vint18251_31_nw4sg4nFnC")

# Find number of data sets collected
data_sets = dt.get_num_data_sets_t(temperature)
assert dt.get_num_data_sets_t(temperature) == dt.get_num_data_sets_w(north_wind) == dt.get_num_data_sets_w(east_wind)

# Manually enter time interval (in hours)
time_interval = 0.25

# Generate time axis
time_axis = dt.generate_time_axis(time_interval, data_sets)

# Generate list of altitudes of interest
min, max, step = 85.0, 100.0, 1.0   
altitudes = dt.generate_altitudes_list(min, max, step)

# Initialize data set size trackers
temperature_data_sets = [data_sets] * len(temperature)
north_wind_data_sets = [data_sets] * len(north_wind)
east_wind_data_sets = [data_sets] * len(east_wind)

# Remove data which exceeds maximum percent error
max_error = 0.05 # Maximum percent error
dt.remove_bad_data(temperature, max_error, data_sets)
dt.remove_bad_data(north_wind, max_error, data_sets)
dt.remove_bad_data(east_wind, max_error, data_sets)

# Convert LOS wind to horizontal wind
dt.scale_data(north_wind, math.sin(2 * math.pi / 9))
dt.scale_data(east_wind, math.sin(math.pi / 3))

# Apply polynomial fit to data to approximate background
deg = 3 # Highest degree of polynomial fit
temperature_bg = dt.bg_fit(temperature, data_sets, deg)
north_wind_bg = dt.bg_fit(north_wind, data_sets, deg)
east_wind_bg = dt.bg_fit(east_wind, data_sets, deg)

# Subtract polynomial fit at each altitude from data sets
dt.bg_filter(temperature, data_sets, temperature_bg)
dt.bg_filter(north_wind, data_sets, north_wind_bg)
dt.bg_filter(east_wind, data_sets, east_wind_bg)

# TODO: Move data reconstruction ahead of background fit and filter, adjust data_tools functions as needed
# Interpolate data at given altitudes
temperature_interpolation = dt.linear_interpolation(temperature, data_sets)
north_wind_interpolation = dt.linear_interpolation(north_wind, data_sets)
east_wind_interpolation = dt.linear_interpolation(east_wind, data_sets)

# Reconstruct data using interpolation functions
temperature = dt.reconstruct_data(altitudes, temperature_interpolation, data_sets)
north_wind = dt.reconstruct_data(altitudes, north_wind_interpolation, data_sets)
east_wind = dt.reconstruct_data(altitudes, east_wind_interpolation, data_sets)

# Find dT / dz
dTdz = gwe.get_dTdz(altitudes, temperature, data_sets)

# Apply median filter to data to remove outliers
dt.median_filter(temperature, 3)
dt.median_filter(north_wind, 3)
dt.median_filter(east_wind, 3)

if saveGraphs:
    # Save data graphs
    dt.save_data(altitudes, temperature, time_axis, title="Temperature Perturbation", xlabel="Time (hours)", ylabel="Temperature (K)", folder="figures/TempPerturb/", filename="TempPerturb")
    dt.save_data(altitudes, north_wind, time_axis, title="North Wind Perturbation", xlabel="Time (hours)", ylabel="Speed (m/s)", folder="figures/NorthWindPerturb/", filename="NorthWindPerturb")
    dt.save_data(altitudes, east_wind, time_axis, title="East Wind Perturbation", xlabel="Time (hours)", ylabel="Speed (m/s)", folder="figures/EastWindPerturb/", filename="EastWindPerturb")

# Apply FFT to data
temperature_fft = dt.fast_fourier_transform(temperature)
north_wind_fft = dt.fast_fourier_transform(north_wind)
east_wind_fft = dt.fast_fourier_transform(east_wind)

# Graph FFTs
freq = 1/900
#dt.display_fft(temperature_fft, altitudes, data_sets, freq, title="Temperature)
#dt.display_fft(north_wind_fft, altitudes, data_sets, freq, title="North Wind")
#dt.display_fft(east_wind_fft, altitudes, data_sets, freq, title="East Wind")

if saveGraphs:
    ## Save FFT graphs
    dt.save_fft(temperature_fft, altitudes, data_sets, freq, title="Temperature", folder="figures/TempFFT/", filename="TempFFT1")
    dt.save_fft(north_wind_fft, altitudes, data_sets, freq, title="North Wind", folder="figures/NorthWindFFT/", filename="NorthWindFFT1")
    dt.save_fft(east_wind_fft, altitudes, data_sets, freq, title="East Wind", folder="figures/EastWindFFT/", filename="EastWindFFT1")

# Apply sin/cos residual fit to find first residual
temperature_resid_1 = dt.first_residual_fit(temperature, time_axis, data_sets)
north_wind_resid_1 = dt.first_residual_fit(north_wind, time_axis, data_sets)
east_wind_resid_1 = dt.first_residual_fit(east_wind, time_axis, data_sets)

if saveGraphs:
    # Save data graphs
    temp_fit = []
    north_fit = []
    east_fit = []
    for i in range(len(altitudes)):
        tf = []
        nf = []
        ef = []
        for j in range(int(data_sets * time_interval)):
            tf.append(dt.first_residual_function(j, temperature_resid_1[i][0][0], temperature_resid_1[i][0][1]))
            nf.append(dt.first_residual_function(j, north_wind_resid_1[i][0][0], north_wind_resid_1[i][0][1]))
            ef.append(dt.first_residual_function(j, east_wind_resid_1[i][0][0], east_wind_resid_1[i][0][1]))
        temp_fit.append(tf)
        north_fit.append(nf)
        east_fit.append(ef)
        
    dt.save_data(altitudes, temperature, time_axis, fitted_line=temp_fit, title="Temperature Perturbation", xlabel="Time (hours)", ylabel="Temperature (K)", folder="figures/TempPerturb/", filename="TempPerturb")
    dt.save_data(altitudes, north_wind, time_axis, fitted_line=north_fit, title="North Wind Perturbation", xlabel="Time (hours)", ylabel="Speed (m/s)", folder="figures/NorthWindPerturb/", filename="NorthWindPerturb")
    dt.save_data(altitudes, east_wind, time_axis, fitted_line=east_fit, title="East Wind Perturbation", xlabel="Time (hours)", ylabel="Speed (m/s)", folder="figures/EastWindPerturb/", filename="EastWindPerturb")

# Subtract residual fit from data at each altitude
dt.first_residual_filter(temperature, temperature_resid_1, data_sets)
dt.first_residual_filter(north_wind, north_wind_resid_1, data_sets)
dt.first_residual_filter(east_wind, east_wind_resid_1, data_sets)

# Apply sin/cos residual fit to find second residual
temperature_resid_2 = dt.second_residual_fit(temperature, time_axis, data_sets)
north_wind_resid_2 = dt.second_residual_fit(north_wind, time_axis, data_sets)
east_wind_resid_2 = dt.second_residual_fit(east_wind, time_axis, data_sets)

# Apply FFT to data
temperature_fft = dt.fast_fourier_transform(temperature)
north_wind_fft = dt.fast_fourier_transform(north_wind)
east_wind_fft = dt.fast_fourier_transform(east_wind)

# Graph FFTs
#dt.display_fft(temperature_fft, altitudes, data_sets, freq, title="Temperature")
#dt.display_fft(north_wind_fft, altitudes, data_sets, freq, title="North Wind")
#dt.display_fft(east_wind_fft, altitudes, data_sets, freq, title="East Wind")

if saveGraphs:
    # Save FFT graphs
    dt.save_fft(temperature_fft, altitudes, data_sets, freq, title="Temperature", folder="figures/TempFFT/", filename="TempFFT2")
    dt.save_fft(north_wind_fft, altitudes, data_sets, freq, title="North Wind", folder="figures/NorthWindFFT/", filename="NorthWindFFT2")
    dt.save_fft(east_wind_fft, altitudes, data_sets, freq, title="East Wind", folder="figures/EastWindFFT/", filename="EastWindFFT2")

# Graph wind hodographs
# dt.display_hodograph(east_wind, north_wind, altitudes)

if saveGraphs:
    # Save wind hodographs
    dt.save_hodograph(altitudes, east_wind, north_wind, title="Wind", folder="figures/HodographRaw/", filename="RawWindData")

# Find Planar Wind
planar_wind_magnitude, planar_wind_direction = dt.get_planar_wind(altitudes, north_wind_interpolation, east_wind_interpolation, data_sets)

#for i in range(len(planar_wind_magnitude)):
#    for j in range(len(planar_wind_magnitude[i])):
#        print(planar_wind_magnitude[i][j], end="")
#        print(" m/s at ", end="")
#        print(planar_wind_direction[i][j], end="")
#        print("degrees E of N")

new_north = []
new_east = []
for i in range(len(altitudes)):
    nn = []
    ne = []
    for j in range(data_sets):
        nn.append(dt.second_residual_function(j, north_wind_resid_2[i][0][0], north_wind_resid_2[i][0][1]))
        ne.append(dt.second_residual_function(j, east_wind_resid_2[i][0][0], east_wind_resid_2[i][0][1]))
    new_north.append(nn)
    new_east.append(ne)

if saveGraphs:
    # Save fitted wind hodographs
    dt.save_hodograph(altitudes, new_east, new_north, title="Wind", folder="figures/HodographFitted/", filename="FittedWindData")

dTdz = -0.96526
# Crunch the numbers
f = 0.055
m = 1 / 7.0
phase_diffs = gwe.get_phase_differences(altitudes, north_wind_resid_1, east_wind_resid_1, data_sets)
phi_uv = gwe.get_phi_uv(altitudes, new_north, new_east, phase_diffs, data_sets)
xi_list = gwe.get_xi(altitudes, new_north, new_east, phi_uv, data_sets)
u_para2 = gwe.get_2u_parallel2(altitudes, new_north, new_east, phi_uv, data_sets)
u_perp2 = gwe.get_2u_perpendicular2(altitudes, new_north, new_east, phi_uv, data_sets)
w2 = gwe.get_w2(altitudes, u_para2, u_perp2, data_sets, f)
N2 = gwe.get_N2(altitudes, temperature, dTdz, data_sets)
k2 = gwe.get_k2(altitudes, w2, N2, data_sets, f, m)

xi_avg = dt.get_2d_list_average(xi_list)
u_para2_avg = dt.get_2d_list_average(u_para2) / 2
u_perp2_avg = dt.get_2d_list_average(u_perp2) / 2
w2_avg = dt.get_2d_list_average(w2)
N2_avg = dt.get_2d_list_average(N2)
k2_avg = dt.get_2d_list_average(k2)

dTdz_km = (1/1000) * dTdz
xi_rad = math.sqrt(xi_avg)
xi_deg = (xi_rad * 180) / math.pi
u_para = math.sqrt(abs(u_para2_avg))
u_perp = math.sqrt(abs(u_perp2_avg))
N = math.sqrt(abs(N2_avg))
k = math.sqrt(abs(k2_avg))

w = math.sqrt(abs(w2_avg))
L = 1 / k

c_parallel = w / k
c_z = w / m

print(
    "\ndTdz:\t\t\t" + str(dTdz) + " K/m\t\t\t" + str(dTdz_km) + " K/km" + 
    "\nHorizontal direction:\t" + str(xi_rad) + " radians\t" + str(xi_deg) + " degrees" +
    "\nu_parallel:\t\t" + str(u_para) + " km/h" +
    "\nu_perpendicular:\t" + str(u_perp) + " km/h" +
    "\nBuoyancy frequency:\t" + str(N) + " /h" +
    "\nWave number:\t\t" + str(k) + " /km" +
    "\nIntrinsic frequency:\t" + str(w) + " /h" +
    "\nHorizontal wavelength:\t" + str(L) + " km" +
    "\nHorizontal phase speed:\t" + str(c_parallel) + " km/h" +
    "\nVertical phase speed:\t" + str(c_z) + " km/h"
    )

horizontal_wind = dt.get_2d_list_average(planar_wind_magnitude)
print(horizontal_wind)

zonal = dt.get_2d_list_average(east_wind)
print(zonal)

meridional = dt.get_2d_list_average(north_wind)
print(meridional)