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

def get_dTdz(altitudes, data, data_sets):
    diffs = []
    for i in range(len(altitudes) - 1):
        temp = []
        for j in range(data_sets):
            temp.append(data[i + 1][j] - data[i][j])
        diffs.append(temp)
    avgs = []
    for i in range(len(diffs)):
        total = 0
        for j in range(len(diffs[i])):
            total += diffs[i][j]
        avgs.append(total / len(diffs[i]))
    return stats.mean(avgs)

def get_phase_differences(altitudes, north_wind, east_wind, data_sets):
    diffs = []
    for i in range(len(altitudes)):
        north_phase = math.atan(north_wind[i][0][0] / north_wind[i][0][1])
        east_phase = math.atan(east_wind[i][0][0] / east_wind[i][0][1])
        diffs.append(east_phase - north_phase)
    return diffs

def get_phi_uv(altitudes, north_wind, east_wind, phase_diffs, data_sets):
    phi = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            u = east_wind[i][j]
            v = north_wind[i][j]
            temp.append(u * v * math.cos(phase_diffs[i]))
        phi.append(temp)
    return phi

def get_xi(altitudes, north_wind, east_wind, phi, data_sets, n=1):
    xi = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            u = east_wind[i][j]
            v = north_wind[i][j]
            temp.append(0.5 * (math.pi * n + math.atan((2 * phi[i][j]) / (v**2 - u**2))))
        xi.append(temp)
    return xi

def get_u_parallel(altitudes, north_wind, east_wind, phi_uv, data_sets):
    u_parallel = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            u = east_wind[i][j]
            v = north_wind[i][j]
            temp.append(math.sqrt(0.5 * (u**2 + v**2 + math.sqrt((u**2 + v**2)**2 + 4 * phi_uv[i][j]**2))))
        u_parallel.append(temp)
    return u_parallel

def get_u_perpendicular(altitudes, north_wind, east_wind, phi_uv, data_sets):
    u_perpendicular = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            u = east_wind[i][j]
            v = north_wind[i][j]
            temp.append(math.sqrt(abs(0.5 * (u**2 + v**2 - math.sqrt((u**2 + v**2)**2 + 4 * phi_uv[i][j]**2)))))
        u_perpendicular.append(temp)
    return u_perpendicular

def get_omega(altitudes, u_parallel, u_perpendicular, data_sets, f=0.34):
    omega = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            temp.append((u_parallel[i][j] / u_perpendicular[i][j]) * f)
        omega.append(temp)
    return omega

def get_N2(altitudes, temperature, dTdz, data_sets):
    N2 = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            T = temperature[i][j]
            # TODO: convert from km/s to km/h
            temp.append(((9.5 * 10**-3 * (1/3600)) / T) * (dTdz - 9.5))
        N2.append(temp)
    return N2

def get_k(altitudes, omega, N2, data_sets, f=0.34, m=7.0):
    k = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            temp.append(math.sqrt(abs(m**2 * (omega[i][j]**2 - f**2)/(N2[i][j] - omega[i][j]**2))))
        k.append(temp)
    return k