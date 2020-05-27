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

def extract_data(filename):
    with open(filename) as infile:
        data, recording = [], False
        for line in infile:
            if line.find("altit ") != -1:
                recording = True
            if recording:
                values = list(filter(None, line.split(" ")))
                for i in range(0, len(values)):
                    values[i].strip('\n')
                    values[i].strip('\t')
                data.append(values)
    del data[0]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    return data

def get_num_data_sets_t(data):
    return int((len(data[0]) - 5) / 3)

def get_num_data_sets_w(data):
    return int((len(data[0]) - 4) / 2)

def generate_time_axis(time_interval, data_sets):
    time = list(range(data_sets))
    for i in range(len(time)):
        time[i] *= time_interval
    return time

def scale_data(data, scale):
    for i in range(0, len(data)):
        for j in range(1, len(data[i])):
            if data[i][j] is not None:
                data[i][j] *= scale

def remove_bad_data(data, error, data_sets):
    for i in range(len(data) - 1, -1, -1):
        for j in range(3, data_sets + 3):
            if data[i][j] == 0 or (data[i][j + data_sets] / data[i][j]) >= error:
                data[i][j + data_sets] = None
                data[i][j] = None
        if row_is_empty(data, i, data_sets):
            del data[i]

# Polynomial background fit
def bg_fit(data, data_sets, deg):
    fit = []
    for i in range(3, data_sets + 3):
        try:
            fit.append(np.polyfit(get_altitude_column(data, i), get_column(data, i), deg))
        except:
            fit.append(None)
    return fit

def bg_filter(data, data_sets, fit):
    for i in range(len(data)):
        for j in range(3, data_sets + 3):
            if data[i][j] is not None and fit[j - 3] is not None:
                data[i][j] -= np.poly1d(fit[j - 3])(data[i][0])

def median_filter(data, k):
    for i in range(len(data)):
        data[i] = sp.signal.medfilt(data[i], k)

def first_residual_fit(data, time, data_sets):
    fit = []
    for i in range(len(data)):
        fit.append(sp.optimize.curve_fit(first_residual_function, time, data[i]))
    return fit
            
# Combine residual functions
def first_residual_function(t, A, B):
    w = 2 * np.pi / 12
    return A * np.sin(w * t) + B * np.cos(w * t)

def second_residual_fit(data, time, data_sets): 
    fit = []
    for i in range(len(data)):
        fit.append(sp.optimize.curve_fit(second_residual_function, time, data[i]))
    return fit

def second_residual_function(t, A, B):
    w = 2 * np.pi / 3
    return A * np.sin(w * t) + B * np.cos(w * t)

def first_residual_filter(data, fit, data_sets):
    for i in range(len(data)):
        for j in range(data_sets):
            data[i][j] -= first_residual_function(j, fit[i][0][0], fit[i][0][1])

def second_residual_filter(data, fit, data_sets):
    for i in range(len(data)):
        for j in range(data_sets):
            data[i][j] -= first_residual_function(j, fit[i][0][0], fit[i][0][1])

def linear_interpolation(data, data_sets):
    interpolation = []
    for i in range(3, data_sets + 3):
        try:
            interpolation.append(sp.interpolate.interp1d(get_altitude_column(data, i), get_column(data, i), kind = 'linear', fill_value = 'extrapolate'))
        except:
            #TODO: Fix in case first case fails
            interpolation.append(sp.interpolate.interp1d(get_altitude_column(data, i - 1), get_column(data, i - 1), kind = 'linear', fill_value = 'extrapolate'))
    return interpolation

def reconstruct_data(altitudes, data_interpolation, data_sets):
    reconstructed_data = []
    for i in range(len(altitudes)):
        row = []
        for j in range(data_sets):
            row.append(float(data_interpolation[j](altitudes[i])))
        reconstructed_data.append(row)
    return reconstructed_data

def fast_fourier_transform(data):
    fft = []
    for i in data:
        fft.append(np.fft.fft(i))
    return fft

def get_planar_wind(altitudes, north_wind_interpolation, east_wind_interpolation, data_sets):
    magnitude = []
    direction = []
    for i in range(len(altitudes)):
        temp_magnitude = []
        temp_direction = []
        temp_magnitude.append(altitudes[i])
        temp_direction.append(altitudes[i])
        for j in range(data_sets):
            north = north_wind_interpolation[j](altitudes[i])
            east = east_wind_interpolation[j](altitudes[i])
            if north is None or east is None:
                continue
            temp_magnitude.append(math.sqrt(math.pow(north, 2) + math.pow(east, 2)))
            angle = math.atan(east / north) * (180 / math.pi)
            angle += angle_correction(north, east)
            temp_direction.append(angle)
        magnitude.append(temp_magnitude)
        direction.append(temp_direction)
    return magnitude, direction

def display_data(altitudes, data, time, fitted_line=None, title=None, xlabel=None, ylabel=None):
    for i in range(len(altitudes)):
        fig, ax = plt.subplots()
        plt.scatter(time, data[i], 25)
        if fitted_line is not None:
            plt.plot(time, fitted_line[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        every_nth = 1
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        plt.show()

def save_data(altitudes, data, time, fitted_line=None, title="", xlabel="", ylabel="", folder="", filename=""):
    for i in range(len(altitudes)):
        fig, ax = plt.subplots()
        plt.scatter(time, data[i], 25)
        if fitted_line is not None:
            plt.plot(list(range(len(fitted_line[i]))), fitted_line[i])
        plt.title(title + " at " + str(altitudes[i]) + "km")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        every_nth = 1
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        plt.savefig(folder + str(altitudes[i]) + "km_" + filename + ".png")
        plt.close()

def display_fft(fft, altitudes, data_sets, freq, title=""):
    fr = (freq / 2) * np.linspace(0, 1, data_sets / 2)
    for i in range(len(fr)):
        if fr[i] is not 0:
            fr[i] = 1 / fr[i]
    for i in range(len(fft)):
        normalized = abs(fft[i][0:int(data_sets/2)])
        plt.title(title + " at " + str(altitudes[i]) + "km")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Intensity")
        plt.plot(fr, normalized)
        plt.show()

def save_fft(fft, altitudes, data_sets, freq, title="", folder="", filename=""):
    fr = (freq / 2) * np.linspace(0, 1, data_sets / 2)
    for i in range(len(fr)):
        if fr[i] is not 0:
            fr[i] = 1 / fr[i]
    for i in range(len(fft)):
        normalized = abs(fft[i][0:int(data_sets/2)])
        plt.plot(fr, normalized)
        plt.title(title + " at " + str(altitudes[i]) + "km")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Intensity")
        plt.savefig(folder + str(altitudes[i]) + "km_" + filename + ".png")
        plt.close()

def display_hodograph(altitudes, east_wind, north_wind, title=""):
    for i in range(len(altitudes)):
        hodograph = metplt.Hodograph(component_range=25)
        hodograph.add_grid(increment=1.5)
        hodograph.plot(east_wind[i], north_wind[i])
        plt.title(title + " at " + str(altitudes[i]) + "km")
        plt.show()

def save_hodograph(altitudes, east_wind, north_wind, title="", folder="", filename=""):
    for i in range(len(altitudes)):
        hodograph = metplt.Hodograph(component_range=40)
        hodograph.add_grid(increment=5)
        hodograph.plot(east_wind[i], north_wind[i])
        plt.title(title + " at " + str(altitudes[i]) + "km")
        plt.savefig(folder + str(altitudes[i]) + "km_" + filename + ".png")
        plt.close()

# Returns column at index with None values omitted
def get_column(data, index):
    column = []
    for row in range(len(data)):
        if data[row][index] is not None:
            column.append(data[row][index])
    return column

# Returns altitude column with values omitted that have a None value in the corresponding index
def get_altitude_column(data, index):
    altitudes = []
    for row in range(len(data)):
        if data[row][index] is not None:
            altitudes.append(data[row][0])
    return altitudes

def generate_altitudes_list(min, max, step):
    altitudes = []
    for i in range(int(((max - min) / step) + 1)):
        altitudes.append(min + i * step)
    return altitudes

def row_is_empty(data, row, data_sets):
    for j in range(3, data_sets + 3):
        if data[row][j] is not None:
            return False
    return True

def get_2d_list_average(data):
    avgs = []
    for i in range(len(data)):
        avgs.append(stats.mean(data[i]))
    return stats.mean(avgs)

def angle_correction(north, east):
    if (north >= 0 and east >= 0):
        return 0
    elif (north < 0 and east >= 0 or north < 0 and east < 0):
        return 180
    elif (north >= 0 and east < 0):
        return 360