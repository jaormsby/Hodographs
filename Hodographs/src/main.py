import src.data_tools as dt

import math
import matplotlib.pyplot as plt
import metpy.plots as metplt
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.interpolate
import scipy.optimize
import scipy.signal

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
print(len(time_axis))
print(time_axis)

# Generate list of altitudes of interest
min, max, step = 85.0, 105.0, 1.0   
altitudes = dt.generate_altitudes_list(min, max, step)

# Initialize data set size trackers
#temperature_data_sets = [data_sets] * len(temperature)
#north_wind_data_sets = [data_sets] * len(north_wind)
#east_wind_data_sets = [data_sets] * len(east_wind)

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

# Apply median filter to data to remove outliers
dt.median_filter(temperature, 3)
dt.median_filter(north_wind, 3)
dt.median_filter(east_wind, 3)

## Apply FFT to data
#temperature_fft = dt.fast_fourier_transform(temperature)
#north_wind_fft = dt.fast_fourier_transform(north_wind)
#east_wind_fft = dt.fast_fourier_transform(east_wind)

## Graph FFTs
#freq = 1/900
##dt.graph_fft(temperature_fft, data_sets, freq)
##dt.graph_fft(north_wind_fft, data_sets, freq)
##dt.graph_fft(east_wind_fft, data_sets, freq)

## Save FFT graphs
#dt.save_fft(temperature_fft, altitudes, data_sets, freq, type="Temperature", time="_before")
#dt.save_fft(north_wind_fft, altitudes, data_sets, freq, type="North_Wind", time="_before")
#dt.save_fft(east_wind_fft, altitudes, data_sets, freq, type="East_Wind", time="_before")

# Apply sin/cos residual fit to find first residual
temperature_resid_1 = dt.residual_fit(temperature, data_sets)
north_wind_resid_1 = dt.residual_fit(north_wind, data_sets)
east_wind_resid_1 = dt.residual_fit(east_wind, data_sets)

#for i in north_wind_resid_1:
#    print(i)

#fit1 = []
#for j in range(data_sets):
#    fit1.append(dt.func(j, north_wind_resid_1[0][0][0], north_wind_resid_1[0][0][1]))

#time = list(range(data_sets))
#for i in range(len(time)):
#    time[i] *= time_interval
#for i in range(len(altitudes)):
#    fig, ax = plt.subplots()
#    plt.scatter(time, north_wind[i])
#    plt.plot(time, fit1)
#    plt.show()

# Subtract residual fit from data at each altitude
dt.residual_filter(temperature, temperature_resid_1, data_sets)
dt.residual_filter(north_wind, north_wind_resid_1, data_sets)
dt.residual_filter(east_wind, east_wind_resid_1, data_sets)

# Apply sin/cos residual fit to find second residual
temperature_resid_2 = dt.residual_fit(temperature, data_sets)
north_wind_resid_2 = dt.residual_fit(north_wind, data_sets)
east_wind_resid_2 = dt.residual_fit(east_wind, data_sets)

# Apply FFT to data
temperature_fft = dt.fast_fourier_transform(temperature)
north_wind_fft = dt.fast_fourier_transform(north_wind)
east_wind_fft = dt.fast_fourier_transform(east_wind)

for i in range(len(temperature_fft)):
    print(abs(temperature_fft[i][0:int(data_sets/2)]))
    print(temperature_fft[i][0:int(data_sets/2)])


# Graph FFTs
freq = 1/900
#dt.graph_fft(temperature_fft, data_sets, freq)
#dt.graph_fft(north_wind_fft, data_sets, freq)
#dt.graph_fft(east_wind_fft, data_sets, freq)

# Save FFT graphs
# dt.save_fft(temperature_fft, altitudes, data_sets, freq, type="Temperature", time="_after")
# dt.save_fft(north_wind_fft, altitudes, data_sets, freq, type="North_Wind", time="_after")
# dt.save_fft(east_wind_fft, altitudes, data_sets, freq, type="East_Wind", time="_after")

# Graph wind hodographs
# dt.display_hodograph(east_wind, north_wind, altitudes)

# Save wind hodographs
# dt.save_hodograph(east_wind, north_wind, altitudes, type="Wind")

# Find Planar Wind
planar_wind_magnitude, planar_wind_direction = dt.get_planar_wind(altitudes, north_wind_interpolation, east_wind_interpolation, data_sets)

#for i in range(len(planar_wind_magnitude)):
#    for j in range(len(planar_wind_magnitude[i])):
#        print(planar_wind_magnitude[i][j], end="")
#        print(" m/s at ", end="")
#        print(planar_wind_direction[i][j], end="")
#        print("degrees E of N")


#        #fit1.append(dt.func(j, temperature_resid[i][0][0], temperature_resid[i][0][1]))

new_north = []
new_east = []

for i in range(len(altitudes)):
    nn = []
    ne = []
    for j in range(data_sets):
        nn.append(dt.func(j, north_wind_resid_1[i][0][0], north_wind_resid_1[i][0][1]))
        ne.append(dt.func(j, east_wind_resid_1[i][0][0], east_wind_resid_1[i][0][1]))
    new_north.append(nn)
    new_east.append(ne)

for i in range(len(new_north)):
    print(new_north[i])
    print(north_wind[i])
for i in range(len(new_east)):
    print(new_east[i])
    [print(east_wind[i])]
dt.save_hodograph(new_east, new_north, altitudes, type="Wind2")

time = list(range(data_sets))
for i in range(len(time)):
    time[i] *= time_interval
for i in range(len(altitudes)):
    fig, ax = plt.subplots()
    plt.scatter(time, north_wind[i])
    plt.plot(time, new_north)
    plt.show()
    plt.close()
    fig, ax = plt.subplots()
    plt.scatter(time, east_wind[i])
    plt.plot(time, new_east)
    plt.show()

#for i in range(len(altitudes)):
#    hodograph = metplt.Hodograph(component_range=15)
#    hodograph.add_grid(increment=1.5)
#    print("Hodograph for wind at " + str(altitudes[i]) + " km")
#    plt.show(hodograph.plot(new_east[i], new_north[i]))