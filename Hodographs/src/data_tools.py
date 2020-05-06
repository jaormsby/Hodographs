import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.interpolate
import scipy.optimize
import scipy.signal

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

def scale_data(data, scale):
    for i in range(0, len(data)):
        for j in range(1, len(data[i])):
            data[i][j] *= scale

def remove_bad_data(data, error, data_sets):
    for i in range(len(data) - 1, -1, -1):
        for j in range(3, data_sets + 3):
            if data[i][j] == 0 or (data[i][j + data_sets] / data[i][j]) >= error:
                data[i][j + data_sets] = None
                data[i][j] = None
        if row_is_empty(data, i, data_sets):
            del data[i]

def row_is_empty(data, row, data_sets):
    for j in range(3, data_sets + 3):
        if data[row][j] is not None:
            return False
    return True

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

def residual_fit(data, data_sets):
    fit = []
    x_vals = list(range(data_sets))
    for i in range(len(data)):
        fit.append(sp.optimize.curve_fit(func, x_vals, data[i]))
    return fit

def residual_filter(data, data_sets, fit):
    for i in range(len(data)):
        for j in range(data_sets):
            data[i][j] -= func(j, fit[i][0][0], fit[i][0][1])
            
def func(t, A, B):
    w = 2 * np.pi / (12 * 4)
    return A * np.sin(w * t) + B * np.cos(w * t)

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
        fft.append(np.fft.ftt(i))
    return i

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

def generate_altitudes_list(min, max, step):
    altitudes = []
    for i in range(int(((max - min) / step) + 1)):
        altitudes.append(min + i * step)
    return altitudes

# This function is not currently being used
def linear_bg_filter(data, data_sets):
	for i in range(0, len(data)):
		l, r = 0.0
		for j in range(3, data_sets / 2):
			l += data[i][j]
		for j in range (3 + data_sets / 2, data_sets):
			r += data[i][j]
		l /= (data_sets / 2)
		r /= (data_sets / 2)
			
def get_num_data_sets_t(data):
    return int((len(data[0]) - 5) / 3)

def get_num_data_sets_w(data):
    return int((len(data[0]) - 4) / 2)

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

def angle_correction(north, east):
    if (north >= 0 and east >= 0):
        return 0
    elif (north < 0 and east >= 0 or north < 0 and east < 0):
        return 180
    elif (north >= 0 and east < 0):
        return 360