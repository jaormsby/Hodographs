import src.data_tools as dt
import math
import matplotlib.pyplot as plt
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

# Initialize data set size trackers
#temperature_data_sets = [data_sets] * len(temperature)
#north_wind_data_sets = [data_sets] * len(north_wind)
#east_wind_data_sets = [data_sets] * len(east_wind)

# Convert LOS wind to horizontal wind
dt.scale_data(north_wind, math.sin(2 * math.pi / 9))
dt.scale_data(east_wind, math.sin(math.pi / 3))

# Remove data which exceeds maximum percent error
max_error = 0.05 # Maximum percent error
dt.remove_bad_data(temperature, max_error, data_sets)
dt.remove_bad_data(north_wind, max_error, data_sets)
dt.remove_bad_data(east_wind, max_error, data_sets)

# Apply polynomial fit to data to approximate background
deg = 3 # Highest degree of polynomial fit
temperature_bg = dt.polynomial_bg_fit(temperature, data_sets, deg)
north_wind_bg = dt.polynomial_bg_fit(north_wind, data_sets, deg)
east_wind_bg = dt.polynomial_bg_fit(east_wind, data_sets, deg)

# Subtract polynomial fit at each altitude from data sets
dt.polynomial_bg_filter(temperature, data_sets, temperature_bg)
dt.polynomial_bg_filter(north_wind, data_sets, north_wind_bg)
dt.polynomial_bg_filter(east_wind, data_sets, east_wind_bg)

# Interpolate data at given altitudes
temperature_interpolation = dt.create_interpolation_function(temperature, data_sets)
north_wind_interpolation = dt.create_interpolation_function(north_wind, data_sets)
east_wind_interpolation = dt.create_interpolation_function(east_wind, data_sets)

# Generate list of altitudes of interest
min, max, step = 85.0, 85.0, 1.0   
altitudes = dt.generate_altitudes_list(min, max, step)

# Reconstruct data using interpolation functions
temperature = dt.reconstruct_data(altitudes, temperature_interpolation, data_sets)
north_wind = dt.reconstruct_data(altitudes, north_wind_interpolation, data_sets)
east_wind = dt.reconstruct_data(altitudes, east_wind_interpolation, data_sets)

# Apply median filter to data to remove outliers
dt.median_filter(temperature, 3)
dt.median_filter(north_wind, 3)
dt.median_filter(east_wind, 3)

# Apply sin/cos residual fit
temperature_resid = dt.residual_fit(temperature, data_sets)
north_wind_resid = dt.residual_fit(north_wind, data_sets)
east_wind_resid = dt.residual_fit(east_wind, data_sets)

temp_transform = []
for i in temperature:
    temp_transform.append(np.fft.fft(i))

freq = 1/900
fr = (freq / 2) * np.linspace(0, 1, data_sets / 2)

#test1 = []
#for i in range(100):
#    test1.append(dt.func(i, 10, 12, 25))

#test2 = dt.residual_fit(test1, 100)
#print(test2[0][0])

#test3 = []
#for i in range(100):
#    test3.append(dt.func(i, test2[0][0][0], test2[0][0][1], test2[0][0][2]))

#plt.scatter(list(range(100)), test1, 25)
#plt.plot(list(range(100)), test3)
#plt.show()

for i in range(len(temp_transform)):
    normalized = abs(temp_transform[i][0:int(data_sets/2)])

    plt.plot(1/fr, normalized)
    plt.show()

temp2 = temperature.copy()
for i in temperature:
    for j in temperature[0]:
        temp2[i][j] = temperature[i][j]

# Subtract residual fit from data at each altitude
dt.residual_filter(temperature, data_sets, temperature_resid)

temp_transform = []
for i in temperature:
    temp_transform.append(np.fft.fft(i))

freq = 1/900
fr = (freq / 2) * np.linspace(0, 1, data_sets / 2)

for i in range(len(temp_transform)):
    normalized = abs(temp_transform[i][0:int(data_sets/2)])

    plt.plot(1/fr, normalized)
    plt.show()

# Find Planar Wind
planar_wind_magnitude, planar_wind_direction = dt.get_planar_wind(altitudes, north_wind_interpolation, east_wind_interpolation, data_sets)

#print(planar_wind_magnitude)

#for i in range(len(temperature_interpolation)):
#    print(temperature_interpolation[i](95.0))

# Reconstruct temperature data
#temperature = dt.reconstruct_data(altitudes, temperature_interpolation, data_sets)

# Filter temperature data to mitigate outliers
#temperature = dt.median_filter(temperature, 3)

#for i in range(len(planar_wind_magnitude)):
#    for j in range(len(planar_wind_magnitude[i])):
#        print(planar_wind_magnitude[i][j], end="")
#        print(" m/s at ", end="")
#        print(planar_wind_direction[i][j], end="")
#        print("degrees E of N")

def cos_fit(x, a, b):
    return a * np.cos(b * x)

for i in range(len(temp2)):
    temp = []
    time = []
    fit1 = []
    for j in range(data_sets):
        temp.append(temp2[i][j])
        time.append(j)
        fit1.append(dt.func(j, temperature_resid[i][0][0], temperature_resid[i][0][1]))

    #params, param_covariance = sp.optimize.curve_fit(cos_fit, temp, time)
    #print(params)

    #plt.plot(time, cos_fit(temp, params[0], params[1]))
    #newtemp = []
    #for k in temp:
    #    newtemp.append(cos_fit(k, params[0], params[1]))
    #plt.plot((temp, newtemp),
    #     label='Fitted function')
#    time = range(1, 1 + len(temp))

    fig, ax = plt.subplots()
    plt.plot(time, fit1)
    plt.scatter(time, temp, 25)
    plt.title("Temperature at " + str(altitudes[i]) + "km")
    plt.xlabel("Time Interval")
    plt.ylabel("Temperature Perturbation (K)")
    #plt.gca().invert_yaxis()
    every_nth = 1
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.show()