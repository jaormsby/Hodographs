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

def get_2u_parallel2(altitudes, north_wind, east_wind, phi_uv, data_sets):
    u_para = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            u = east_wind[i][j]
            v = north_wind[i][j]
            temp.append(u**2 + v**2 + math.sqrt((u**2 - v**2)**2 + 4 * phi_uv[i][j]**2))
        u_para.append(temp)
    return u_para

def get_2u_perpendicular2(altitudes, north_wind, east_wind, phi_uv, data_sets):
    u_perp = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            u = east_wind[i][j]
            v = north_wind[i][j]
            temp.append(u**2 + v**2 - math.sqrt((u**2 - v**2)**2 + 4 * phi_uv[i][j]**2))
        u_perp.append(temp)
    return u_perp

def get_w2(altitudes, u_para, u_perp, data_sets, f=0.34):
    w = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            temp.append(f**2 * (u_para[i][j] / u_perp[i][j]))
        w.append(temp)
    return w

def get_N2(altitudes, temperature, dTdz, data_sets):
    N2 = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            T = temperature[i][j]
            # TODO: fix unit conversion
            #(m/s^2) => (km/h^2) => 12960
            temp.append((12960 * (9.5 * 10**-3) / T) * ((1 / 1000) * dTdz - 9.5))
        N2.append(temp)
    return N2

def get_k2(altitudes, w2, N2, data_sets, f=0.34, m=1/7.0):
    k = []
    for i in range(len(altitudes)):
        temp = []
        for j in range(data_sets):
            temp.append(m**2 * ((w2[i][j] - f**2)/(N2[i][j] - w2[i][j])))
        k.append(temp)
    return k