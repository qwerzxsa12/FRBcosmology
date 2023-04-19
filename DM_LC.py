from cProfile import label
import numpy as np
import h5py
import matplotlib.pyplot as plt
import astropy
from astropy.io import ascii
import astropy.units as u
from matplotlib import units
from numpy import histogram2d
import matplotlib.colors as colors
import bisect
import math
from math import atan
import time
import warnings
import sys
import os
from numpy import array, sqrt, sin, cos, tan
from mpl_toolkits.mplot3d import Axes3D

from astropy.cosmology import Planck18 as cosmo
H0 = cosmo.H0
from astropy.constants import G, c, m_e, m_p
#print(H_mass_frac, He_mass_frac)
f_IGM =  0.65

warnings.simplefilter('always', RuntimeWarning)
#########
me = 9.1*10**-28
mH = 1.67*10**-24
solartog = 2.0*10**33
def M6(r,r0, h, D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    d = np.sqrt((r-r0)**2)
    q = d/h
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  h**(-D)*sigma[D-1]*w
    return W


def W(qz,qxy,D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    q = np.sqrt(qxy**2 + qz**2)
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  sigma[D-1]*w
    return W

    

def Gauss(r, r0, h, D):
    from numpy import pi, sqrt, exp
    d = sqrt((r-r0)**2)
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = [1/sqrt(pi), 1/pi, 1.0/(pi*sqrt(pi))]
    W = h**(-D)*sigma[D-1]*exp(-(r-r0)**2/h**2)
    W = W*V[D-1]
    return W



def F(qxy):
    from scipy import integrate
    v ,err = integrate.quad(W,  -np.sqrt(1-qxy**2), np.sqrt(1-qxy**2), args = (qxy,3))
    return v
qxy =np.linspace(0, 1, 1000)
Fxy = np.array(list(map(F, qxy)))

def F_xy(q):
    qxy =np.linspace(0, 1, 1000)
    q_xy = bisect.bisect(qxy, q) - 1 
    W = Fxy[q_xy]
    return W
plt.plot(qxy,Fxy)

def handle_boundary_conditions(pos, L):
    """
    处理周期性边界条件
    """
    # 检查每个维度是否超出边界
    for i in range(3):
        idx = pos[:, i] < 0
        pos[idx, i] += L[i]
        idx = pos[:, i] >= L[i]
        pos[idx, i] -= L[i]
    return pos
from scipy.spatial import cKDTree

def find_particles_within_RV(pos, halo_center, RV, Lbox):
    # Apply periodic boundary conditions to halo center
    halo_center = np.mod(halo_center, Lbox)

    # Build KDTree with periodic boundary conditions
    tree = cKDTree(pos, boxsize=Lbox)

    # Find all particles within RV of halo center
    idx = tree.query_ball_point(halo_center, RV)

    # Calculate distances of all particles within RV
    dist = np.sqrt(np.sum((pos[idx] - halo_center)**2, axis=1))

    # Check for particles that cross periodic boundaries
    crossings = pos[idx] - halo_center
    crossings = np.mod(crossings + Lbox/2, Lbox) - Lbox/2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))
    mask = crossing_dist < dist
    idx = np.array(idx)
    idx[mask] = idx[mask] + len(pos)
    dist[mask] = crossing_dist[mask]

    return idx, dist
from scipy.spatial.transform import Rotation
from math import atan2, degrees
def rotation_matrix(angle, axis):
    """计算绕着某个坐标轴旋转某个角度的旋转矩阵"""
    sina = np.sin(angle)
    cosa = np.cos(angle)
    if axis == 0:
        # 绕X轴旋转
        R = np.array([[1, 0, 0], [0, cosa, -sina], [0, sina, cosa]])
    elif axis == 1:
        # 绕Y轴旋转
        R = np.array([[cosa, 0, -sina], [0, 1, 0], [sina, 0, cosa]])
    elif axis == 2:
        # 绕Z轴旋转
        R = np.array([[cosa, sina, 0], [-sina, cosa, 0], [0, 0, 1]])
    return R

def find_los_through_sphere(Halo_c_pos, los_vec, smooth_len):

    center = Halo_c_pos
    
    # 球面半径为平滑长度
    radius = smooth_len
    
    # 存储穿过球面的LOS的索引
    through_indices = []
    
    # 遍历所有LOS
    for i in range(len(los_vec)):
        # 获取当前LOS的方向向量
        direction_vec = los_vec[i]
        
        # 将方向向量单位化
        direction_vec /= np.linalg.norm(direction_vec)
        
        # 沿着LOS的方向向量延长LOS
        # 延长长度应该取一个足够大的值，比如1e10，以确保延长后的LOS一定会穿过球面
        end_point = direction_vec * 1e10
        
        # 判断当前LOS是否与球面有交点
        center = Halo_c_pos
        center_to_origin = center - [0, 0, 0]
        
        # 计算LOS与球的交点
        a = np.dot(direction_vec, direction_vec)
        b = 2 * np.dot(direction_vec, center_to_origin)
        c = np.dot(center_to_origin, center_to_origin) - radius**2
        
        # 计算判别式
        delta = b**2 - 4*a*c
        
        if delta >= 0:
            # 如果判别式大于等于0，则说明当前LOS穿过了球面
            through_indices.append(i)
            
    return through_indices


import numpy as np
from scipy.integrate import quad
# 宇宙学参数
Omega_m = 0.31  # 物质密度参数
Omega_Lambda = 0.69  # 暗能量密度参数
H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
c = 299792.458  # 光速，单位为 km/s

# 计算共动距离和尺度因子的函数
def comoving_distance(z):
    integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(integrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)

# 红移范围
z_arr = np.linspace(0, 2, 101)

# 计算共动距离和尺度因子
dC_arr = np.array([comoving_distance(z) for z in z_arr])
a_arr = np.array([scale_factor(z) for z in z_arr])

# 打印表格
print("z\t a\t        dC (Mpc)")
for i in range(len(z_arr)):
    print("{:.2f}\t{:.6f}\t{:.6f}".format(z_arr[i], a_arr[i], dC_arr[i]))
    # 设置参数

import numpy as np
from scipy.optimize import root_scalar

# 宇宙学参数
Omega_m = 0.31  # 物质密度参数
Omega_Lambda = 0.69  # 暗能量密度参数
H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
c = 299792.458  # 光速，单位为 km/s

# 计算共动距离和尺度因子的函数
def comoving_distance(z):
    integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(integrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)

# 已知共动距离，求红移和尺度因子的函数
def find_redshift_and_scale_factor(dC):
    # 定义被求解的方程
    equation = lambda z: comoving_distance(z) - dC

    # 使用root_scalar函数求解方程
    sol = root_scalar(equation, bracket=[0, 10])

    # 检查解是否收敛，如果不收敛则返回None
    if not sol.converged:
        return None, None

    # 返回红移和尺度因子
    z = sol.root
    a = scale_factor(z)
    return z, a

# 假设共动距离为5330 Mpc
dC = 5330

# 求解红移和尺度因子
z, a = find_redshift_and_scale_factor(dC)

# 打印结果
if z is None:
    print("无法求解红移和尺度因子。")
else:
    print("共动距离 dC = {:.2f} Mpc 对应的红移为 z = {:.4f}，尺度因子为 a = {:.4f}".format(dC, z, a))
###############################L50
# sim name, snapshot file, FoF file.
print("Start the data-reading part")
snapfile = [["/home/zhaozhang/local/work/L50N256/snapshot_020/snapshot_020","/home/zhaozhang/local/work/L50N256/snapshot_020/groups_020/sub_020"],["/home/zhaozhang/local/work/L50N256/snapshot_020/snapshot_020","/home/zhaozhang/local/work/L50N256/snapshot_020/groups_020/sub_020"]]
# set alias name
alias_snap = {
    "rho": "PartType0/Density",
    "u": "PartType0/InternalEnergy",
    "sphpos": "PartType0/Coordinates",
    "dmpos": "PartType1/Coordinates",
    "starpos": "PartType4/Coordinates",
    "smoothlen": "PartType0/SmoothingLength"
}

alias_frac = {    
    "n_HI": "PartType0/HI",
    "n_HII": "PartType0/HII",
    'n_DI': "PartType0/DI",
    'n_DII': "PartType0/DII",
    "n_e": "PartType0/ElectronAbundance",
    "n_HeI": "PartType0/HeI",
    "n_HeII": "PartType0/HeII",
    "n_HeIII": "PartType0/HeIII",
    "n_H2I": "PartType0/H2I",
    "n_H2II": "PartType0/H2II",
    
 }

alias_T = {
     "T": "PartType0/Temperature"
}  

alias_mass = {
    "sphmass": 0,
    "dmmass": 1,
    "starmass": 4
}
alias_fof = {
    "halomass": "Group/GroupMass",
    "halopos": "Group/GroupPos",
    "halonum": "Group/GroupNsubs",
    "halo0": "Group/GroupFirstSub",
    "halolen": "Group/GroupLen",
    "haloRV" : "Group/Group_R_Crit200",
    "haloMV": "Group/Group_M_Crit200"
}


alias_subfind = {
    "subhalopos": "Subhalo//SubhaloPos",
    "subhalomass": "Subhalo/SubhaloMass",
    "subhaloGrNr": "Subhalo/SubhaloGrNr",
    "subhaloparent": "Subhalo/SubhaloParent",
    "subhalolen" : "Subhalo/SubhaloLen"
    
}

# radial bin of log(r[Mpc])
Xbins = np.linspace(-2, 1, num=20)
Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
# ----------------------------------------#
# initialization
data = [{} for i in range(len(snapfile))]
for i in range(len(snapfile)):
    for key in alias_snap.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_mass.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_frac.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_T.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_fof.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_subfind.keys():
        data[i][key] = np.empty((0, 3))


#data_sub = 
for i in range(len(snapfile)):
    print("\nstart reading data of {}...".format(snapfile[i][0]))
    try:
        fname = "{}.hdf5".format(snapfile[i][0])
        f = h5py.File(fname, "r")
    except:
        fname = "{}.0.hdf5".format(snapfile[i][0])
        f = h5py.File(fname, "r")
    # load header
    a = f["Header"].attrs["Time"]
    z0 = f["Header"].attrs["Redshift"]
    h = f["Parameters"].attrs["HubbleParam"]
    Omega_b = f["Parameters"].attrs["OmegaBaryon"]
    Omega_0 = f["Parameters"].attrs["Omega0"]
    Omega_Lamda = f["Parameters"].attrs["OmegaLambda"]
    comoving = f["Parameters"].attrs["ComovingIntegrationOn"]
    if comoving == 1:
        print("This is cosmological run. Scaling code values to physical value.")
        print("a = {}, h = {}".format(a, h))
    else:
        print("This is non-cosmological run. Setting a = h = 1.")
        a = h = 1
    MassTable = f["Header"].attrs["MassTable"]
    print("Mass table: {}".format(MassTable))
    NumPart_Total = f["Header"].attrs["NumPart_Total"]
    Boxsize = f["Header"].attrs["BoxSize"]
    print("Total number of particles: {}".format(NumPart_Total))
    nfile = f["Header"].attrs["NumFilesPerSnapshot"]
    SPHdensitytocgs = f['PartType0']['Density'].attrs['to_cgs']
    SPHmasstocgs = f['PartType0']['Masses'].attrs['to_cgs']
    f.close()
    # load particle data
    for j in range(nfile):
        if nfile == 1:
            fname = "{}.hdf5".format(snapfile[i][0])
        else:
            fname = "{}.{}.hdf5".format(snapfile[i][0], j)
        f = h5py.File(fname, "r")
        #d[i] = f
        NumPart_ThisFile = f["Header"].attrs["NumPart_ThisFile"]
        print("\nreading {}th file.".format(j))
        print("Number of particles in this file: {}".format(NumPart_ThisFile))
        for key in alias_snap.keys():
            print("loading {}.\n\tmultipling by a^{} h^{} to get physical value".format(key,f[alias_snap[key]].attrs["a_scaling"], f[alias_snap[key]].attrs["h_scaling"]))
            array = np.array(f[alias_snap[key]])
            array *= a**f[alias_snap[key]].attrs["a_scaling"]
            array *= h**f[alias_snap[key]].attrs["h_scaling"]
            if len(np.shape(array)) == 1:
                # scalar value
                data[i][key] = np.append(data[i][key], array)
            else:
                # vector value
                data[i][key] = np.vstack((data[i][key], array))
                print("loading {}".format(key))
        for key in alias_frac.keys():
            array = np.array(f[alias_frac[key]])
            if len(np.shape(array)) == 1:
                array = np.array(f[alias_frac[key]])
                # scalar value
                data[i][key] = np.append(data[i][key], array)
            else:
                # vector value
                array = np.array(f[alias_frac[key]])
                data[i][key] = np.vstack((data[i][key], array))
        for key in alias_T.keys():
            print("loading {}.\n\tmultipling by a^0 h^-0 to get physical value".format(key))
                # load particle mass from particle data
            array = np.array(f["{}".format(alias_T[key])])
            data[i][key] = np.append(data[i][key], array)

        for key in alias_mass.keys():
            print(
                "loading {}.\n\tmultipling by a^0 h^-1 to get physical value".format(key))
            if MassTable[alias_mass[key]] == 0:
                # load particle mass from particle data
                array = np.array(f["PartType{}/Masses".format(alias_mass[key])])
                array *= h**-1
                data[i][key] = np.append(data[i][key], array)
            else:
                array = np.full(
                    NumPart_ThisFile[alias_mass[key]], MassTable[alias_mass[key]])
                data[i][key] = np.append(data[i][key], array)
        f.close()
    try:
        fname = "{}.hdf5".format(snapfile[i][1])
        f = h5py.File(fname, "r")
    except:
        fname = "{}.0.hdf5".format(snapfile[i][1])
        f = h5py.File(fname, "r")
    nfile = f["Header"].attrs["NumFilesPerSnapshot"]
    f.close()
    for j in range(nfile):
        if nfile == 1:
            fname = "{}.hdf5".format(snapfile[i][1])
        else:
            fname = "{}.{}.hdf5".format(snapfile[i][1], j)
        f = h5py.File(fname, "r")
        for key in alias_fof.keys():
            array = np.array(f[alias_fof[key]])
            if len(np.shape(array)) == 1:
                # scalar value
                data[i][key] = np.append(data[i][key], array)
            else:
                # vector value
                data[i][key] = np.vstack((data[i][key], array))
        for key in alias_subfind.keys():
            array = np.array(f[alias_subfind[key]])
            if len(np.shape(array)) == 1:
                # scalar value
                data[i][key] = np.append(data[i][key], array)
            else:
                # vector value
                data[i][key] = np.vstack((data[i][key], array))
    f.close()
    data[i]["halomass"] *= h**-1
    data[i]["halopos"] *= a * h**-1
    data[i]["halolen"] *= a * h**-1
    data[i]["subhalomass"] *= h**-1
    data[i]["subhalopos"] *= a * h**-1
    data[i]["subhalolen"] *= a * h**-1
# computing radial profiles
# log radius
for i in range(len(snapfile)):
    r2 \
        = (data[i]["sphpos"].T[0] - data[i]["halopos"][0][0])**2 \
        + (data[i]["sphpos"].T[1] - data[i]["halopos"][0][1])**2\
        + (data[i]["sphpos"].T[2] - data[i]["halopos"][0][2])**2
    data[i]["sphlogr"] = np.log10(np.sqrt(r2))
    data[i]["sphX"] = data[i]["sphpos"].T[0] 
    data[i]["sphY"] = data[i]["sphpos"].T[1] 
    data[i]["sphZ"] = data[i]["sphpos"].T[2] 
    r2 \
        = (data[i]["dmpos"].T[0] - data[i]["halopos"][0][0])**2 \
        + (data[i]["dmpos"].T[1] - data[i]["halopos"][0][1])**2\
        + (data[i]["dmpos"].T[2] - data[i]["halopos"][0][2])**2
    data[i]["dmlogr"] = np.log10(np.sqrt(r2))
    data[i]["starlogr"] = np.log10(np.sqrt(r2))
    data[i]["starX"] = data[i]["starpos"].T[0] 
    data[i]["starY"] = data[i]["starpos"].T[1] 
    data[i]["starZ"] = data[i]["starpos"].T[2] 
print("Finish the data-reading part")



# initialize profile dictionary
print("Start the plotting of profile curve")
profile = [{} for i in range(len(snapfile))]
# density profile
for i in range(len(snapfile)):
    # mass in shell
    mshell = np.histogram(data[i]["sphlogr"], bins=Xbins,
                          weights=1e10 * data[i]["sphmass"])[0]
    # volume of shell
    vshell = 4/3 * np.pi * (np.power(10, Xbins))**3
    vshell = (np.roll(vshell, -1) - vshell)[0:-1]
    profile[i]["sphrho"] = mshell / vshell
    # mass in shell
    mshell = np.histogram(data[i]["dmlogr"], bins=Xbins,
                          weights=1e10 * data[i]["dmmass"])[0]
    # volume of shell
    vshell = 4/3 * np.pi * (np.power(10, Xbins))**3
    vshell = (np.roll(vshell, -1) - vshell)[0:-1]
    profile[i]["dmrho"] = mshell / vshell
# temperature profile
# conversion factor from u[km^2/s^2] to T[K]
# T[K] = (gamma - 1) * mu * mp / kB * 1e10 * u[km^2/s^2]
u2t = 2/3*0.588*1.67e-24/1.38e-16 * 1e10
#for i in range(len(snapfile)):
    #NT = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      #weights=data[i]["u"] * u2t)[0]
    #N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    #profile[i]["temperature"] = NT / N

for i in range(len(snapfile)):
    NT = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=data[i]["T"])[0]
    N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    profile[i]["temperature"] = NT / N
# pressure profile
# P[Msun km^2/s^2 / Mpc^3] = (gamma - 1) * 1e10 * rho[10^10 Msun/Mpc^3] * u[km^2/s^2]
for i in range(len(snapfile)):
    NP = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=2/3 * 1e10 * data[i]["rho"] * data[i]["u"])[0]
    N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    profile[i]["pressure"] = NP / N
# entropy profile
# S = (gamma - 1) * rho^(1 - gamma) * u
for i in range(len(snapfile)):
    NS = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=2/3 * data[i]["rho"]**(-2/3) * data[i]["u"])[0]
    N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    profile[i]["entropy"] = NS / N
#ionization fraction profile
for i in range(len(snapfile)):
    NII = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights= data[i]["n_HII"])[0]
    NI = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=data[i]["n_HI"])[0]
    profile[i]["ionization"] = NII / (NI+NII)  # %%
# plot density profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["sphrho"], color=cmap(
        i), label="{}, gas".format(snapfile[i][0]))
    plt.plot(Xpoints, profile[i]["dmrho"], linestyle="dashed", color=cmap(
        i), label="{}, dark matter".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"$\rho$ [M$_\odot$ Mpc$^{-3}$]")
plt.legend()
plt.savefig("density.png", bbox_inches="tight")
plt.close()
# plot temperature profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["temperature"],
             color=cmap(i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"T [K]")
plt.legend()
plt.savefig("temperature.png", bbox_inches="tight")
plt.close()
# plot pressure profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["pressure"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"P [M$_\odot$ km$^2$ s$^{-2}$ Mpc$^{-3}$]")
plt.legend()
plt.savefig("pressure.png", bbox_inches="tight")
plt.close()
# plot entropy profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["entropy"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"S [(10$^{10}$ M$_\odot$ Mpc$^{-3}$)$^{-2/3}$ km$^2$ s$^{-2}$]")
plt.legend()
plt.savefig("entropy.png", bbox_inches="tight")
plt.close()
# plot temperature profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(profile[i]["temperature"], profile[i]["sphrho"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"T [K]")
plt.ylabel(r"$\rho$ [M$_\odot$ Mpc$^{-3}$]")
plt.legend()
plt.savefig("temperature_density.png", bbox_inches="tight")
plt.close()
# plot ionization fraction profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["ionization"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"N_{HII}/(N_{HI}+N_{HII)")
plt.legend()
plt.savefig("temperature_density.png", bbox_inches="tight")
plt.close()
#make Star coordinate bin
sphX = data[0]["sphX"]*h
sphY = data[0]["sphX"]*h
sphZ = data[0]["sphZ"]*h
sphX_min = np.min(data[0]["sphX"])
sphX_max = np.max(data[0]["sphX"])
sphZ_min = np.min(data[0]["sphZ"])
sphZ_max = np.max(data[0]["sphZ"])
Bin_num = 200
sphX_bins = np.linspace(sphX_min, sphX_max, Bin_num)
sphZ_bins = np.linspace(sphZ_min, sphZ_max, Bin_num)

#made Star coordinate bin
starX = data[i]["starX"]
starZ = data[i]["starZ"]
starX_min = np.min(data[0]["starX"])
starX_max = np.max(data[0]["starX"])
starZ_min = np.min(data[0]["starZ"])
starZ_max = np.max(data[0]["starZ"])

starX_bins = np.linspace(starX_min, starX_max, Bin_num)
starZ_bins = np.linspace(starZ_min, starZ_max, Bin_num)

SPHMasses, xedges, zedges = histogram2d(sphX, sphZ, bins = [sphX_bins, sphZ_bins], weights = (data[0]["sphmass"]))
X, Z = np.meshgrid(xedges, zedges)
hh = plt.pcolormesh(X, Z, SPHMasses, cmap='rainbow')
plt.xlabel(r"X [Mpc]")
plt.ylabel(r"Y [Mpc]")
cb = plt.colorbar(hh)
cb.set_label('SPH Mass [M$\odot$]')  #设置colorbar的标签字体
plt.savefig("SPHMass-Cordinates_20.png", bbox_inches="tight")
plt.close()

StarMasses, xedges, zedges = histogram2d(starX, starZ, bins = [starX_bins, starZ_bins], weights = (data[0]["starmass"]))

X, Z = np.meshgrid(xedges, zedges)
plt.pcolormesh(X, Z, StarMasses, cmap='rainbow')
plt.xlabel(r"X [Mpc]")
plt.ylabel(r"Y [Mpc]")
hh = plt.pcolormesh(X, Z, StarMasses, cmap='rainbow')
cb = plt.colorbar(hh)
cb.set_label('Star Mass [M$\odot$]')#设置colorbar的标签字体
plt.savefig("StarMass-Cordinates_20.png", bbox_inches="tight")
plt.close()

#Temperature Bin
print("Finish the plotting of profile curve")
Box_num = len(snapfile)
MpcTocm = 3.08567758*10**24
#############################DM_HG Calculate################

print('Start the DM_HG part')
D = 3###################Dimension
Data_HG = data[0]
sphpos_HG = Data_HG["sphpos"]*h
Smoothlen_HG = Data_HG['smoothlen']
SPHmass_o_HG = Data_HG["sphmass"]
SPHmass_cgs_HG = SPHmass_o_HG*SPHmasstocgs
SPHrho_HG = Data_HG["rho"]*SPHdensitytocgs###density
f_e = Data_HG['n_e']
LINE = 20  # 视线数量
dtheta = np.pi / LINE
dpsi = 2 * np.pi / LINE
los_vecs = np.zeros((LINE**2, 3))
for n in range(LINE):
    for m in range(LINE):
        theta = (0.5 + n) * dtheta
        psi = (0.5 + m) * dpsi
        los_vecs[n*LINE+m, 0] = np.sin(theta) * np.cos(psi)
        los_vecs[n*LINE+m, 1] = np.sin(theta) * np.sin(psi)
        los_vecs[n*LINE+m, 2] = np.cos(theta)

Lbox = Boxsize
halo_center = np.array([Data_HG["halopos"][0][0]*h, Data_HG["halopos"][0][1]*h, Data_HG["halopos"][0][2]*h])
los_length = 1
nlos = LINE**2
n_bins = 100  ####Bin number of LC in 1 box
Rd = 2
mp = m_p.to(u.g).value
bin_edges = np.linspace(0, los_length, n_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_volumes = 4.0/3*np.pi*((bin_edges[1:]*MpcTocm)**3 - (bin_edges[:-1]*MpcTocm)**3)/nlos
idx, dis = find_particles_within_RV(sphpos_HG, halo_center, Rd, Lbox)

new_sphpos = np.zeros_like(sphpos_HG)
bin_edges = np.linspace(0, los_length, n_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
d_bin = bin_edges[1:] - bin_edges[:-1] 
for i in range(len(idx)):
    if idx[i]<=len(sphpos_HG):
        new_sphpos[idx[i]] = sphpos_HG[idx[i]]
    if idx[i]>len(sphpos_HG):
        new_sphpos[idx[i]-len(sphpos_HG)] = np.mod(sphpos_HG[idx[i] -len(sphpos_HG)] - halo_center + Lbox/2, Lbox) - Lbox/2 + halo_center
        idx[i] = idx[i] - len(sphpos_HG)
    ###############以Halo中心为原点的坐标系，得到Rd内的所有粒子的坐标位置###################
Halo_c_pos = new_sphpos[idx] - halo_center

Smoothlen_halo_c = Smoothlen_HG[idx]
Num_Par_withinRd = len(Halo_c_pos)
sphmass_g = SPHmass_o_HG*SPHmasstocgs
sphmass_halo_within = sphmass_g[idx]
f_e_within = f_e[idx]
SPHrho_within = SPHrho_HG[idx]
n_e_arr = np.zeros((nlos, n_bins))
n_e_arr1 = np.zeros((nlos, n_bins))
bin_edges = np.linspace(0, los_length, n_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#thetas = np.degrees(np.arctan2(los_centers[:, 1], los_centers[:,
start_time = time.time()  # 记录程序开始时间
for i in range(Num_Par_withinRd): #Num_Par_withinRd
    ################绕X轴旋转################
    center = Halo_c_pos[i]
    h_smooth= Smoothlen_halo_c[i]
    line_idx = find_los_through_sphere(center, los_vecs, h_smooth)
    #print("i = ", i)
    #print("line_idx = ",  line_idx)
    #print("h = ", h_smooth)
    for x in line_idx:
        phi = np.arctan2(los_vecs[x][1], los_vecs[x][0])

        #print("x= ",  x)
        Rz = rotation_matrix(-phi, 2)
        line_xz = los_vecs[x]@Rz

        ################绕Y轴旋转################
        theta  = np.arctan2(line_xz[2], line_xz[0])

        Ry = rotation_matrix(theta, 1)

        line_x = line_xz@Ry
        ################根据LOS与粒子的相对位置将粒子旋转################
        sphere_center_rot = Halo_c_pos[i]@Rz@Ry
        ################计算旋转后的粒子与X轴的交点################
        xc, yc, zc = sphere_center_rot 
        x1 = xc - sqrt(Smoothlen_halo_c[i]**2-yc**2-zc**2)
        x2 = xc + sqrt(Smoothlen_halo_c[i]**2-yc**2-zc**2)
        bin_idx1 = 0
        bin_idx2 = 0
        bin_idx = []
        if (x1 > 0) & (los_length > x2 > 0): ############两个交点全在LOS长度内#########
            bin_idx1 = np.digitize(x1, bin_centers) 
            bin_idx2 = np.digitize(x2, bin_centers)
        elif (los_length > x1 > 0) & (x2>= los_length):##############x1在LOS长度内##########
            bin_idx1 = np.digitize(x1, bin_centers)
            bin_idx2 = np.digitize(los_length, bin_centers)
        elif (x1<= 0) & (los_length>x2>= 0):#############只有x2在LOS长度内##########
            bin_idx1 = np.digitize(0, bin_centers)
            bin_idx2 = np.digitize(x2, bin_centers)
        elif (x1<= 0) & (x2>= los_length):#############粒子平滑长度包含整个LOS长度##########
            bin_idx1 = np.digitize(0, bin_edges)
            bin_idx2 = np.digitize(los_length, bin_edges)
        #############交点包含的bin索引以及粒子距离bin中心的距离##########
        bin_idx = [m for m in range(bin_idx1, bin_idx2)]
        ##################将包含的中心点转化为（x, 0, 0）的三维向量
        if not bin_idx:
            continue
        else:
            bin_centers_within = np.expand_dims(bin_centers[bin_idx], axis=1)
            y_z = np.zeros([2,len(bin_centers_within)])
            bin_centers_within = np.insert(bin_centers_within, 1, y_z, axis = 1)
            binc_partc_dist = np.linalg.norm(bin_centers_within - sphere_center_rot, axis = 1)
            weight_within=  np.array(list(map(lambda x:M6(x*MpcTocm,0, h_smooth*MpcTocm,3), binc_partc_dist)))
            n_e_arr[x][bin_idx] += sphmass_halo_within[i]*sphmass_halo_within[i]*f_e_within[i]*weight_within/mp/bin_volumes[bin_idx]
            n_e_arr1[x][bin_idx] += sphmass_halo_within[i]*f_e_within[i]*weight_within/mp

    if i%1000 == 0:
        print("it's  the %d step"  %i )
        end_time = time.time()  # 记录程序结束时间
        run_time = end_time - start_time  # 计算程序运行时间，单位为秒
        print(f"DM_HG程序已运行时间为：{run_time:.2f}秒")
        print(f"DM_HG程序剩余时间约为：{((Num_Par_withinRd/(i+1e-30)*run_time-run_time)):.2f}秒")
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(LINE**2):
    ax.plot(bin_edges[:-1] + d_bin/2,  n_e_arr1[i], alpha=0.5)
#ax.plot(bin_edges[:-1] + d_bin/2,  np.sum(n_e_arr, axis = 0)/len(n_e_arr), alpha=0.5)
#ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.01,1)
#ax.set_ylim(1e-8,1e1)
ax.set_xlabel('Distance along LOS (Mpc/h)')
ax.set_ylabel('Electron Density (cm$^{-3}$)')
plt.savefig('ne_arr.png', dpi = 400)
D_C_halo = bin_edges[1:] - bin_edges[0]

DM_HG = np.zeros_like(n_e_arr)

dl = bin_edges[1:] - bin_edges[:-1]
####################Host Galaxy#################
for j in range(len(n_e_arr)):
    for i in range(len(D_C_halo)-1):
        #zz[i], aa[i] = find_redshift_and_scale_factor(D_C_halo[i])
        #zz[i+1], aa[i+1] = find_redshift_and_scale_factor(D_C_halo[i+1])
        #dz[i] = zz[i+1] - zz[i]
        DM_HG[j][0] = n_e_arr[j][0]*dl[0]*MpcTocm
        DM_HG[j][i+1] = DM_HG[j][i] + n_e_arr[j][i+1]*dl[i+1]*MpcTocm
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(LINE**2):
    ax.plot(bin_edges[:-1] + d_bin/2,  DM_HG[i], alpha=0.5)
#ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.01,1)
#ax.set_ylim(1e-6,1e0)
ax.set_xlabel('Distance along LOS [Mpc/h]')
ax.set_ylabel('DM(<R)[pc cm$^{-3}$]')
print('Finish the DM_HG part')

Lbox =  Boxsize
#L = Lbox-1.0/10*Boxsize
L = Lbox
m = 3           #Light cone的方向
n = 4
LOS = np.array([L/m, L/n, L])

nm = n * m
image = np.array([n*L, m*L, nm*L])

# Calculate unit vectors in new coordinate system
u3 = (np.array([n, m, nm]) / (n**2 + m**2 + nm**2)**0.5)
axis = np.argmin([n, m])
u1 = np.zeros(3)
u1[axis] = 1
u1 = np.cross(u1, u3)
u1 /= np.linalg.norm(u1)
u2 = np.cross(u3, u1)
u2 /= np.linalg.norm(u2)

delta_alpha = 1/m
delta_delta = 1/n
# Calculate field limits
alpha_max = np.arctan(delta_alpha /2)
delta_max = np.arctan(delta_alpha/2)




LC_length = sqrt(sum(LOS**2))


print("Start the DM_IGM calucalte part")
#The shifted coordinates when light cone passes through 1 BOX
X_Bound_shift = 0
Y_Bound_shift = 0
Z_Bound_shift = 0
X_num = 25
Y_num = 40
LC_num = 1000

ne_LC_shifted = np.zeros([Box_num, LC_num, n_bins])
ne_LC_Ion_shifted = np.zeros([Box_num, LC_num, n_bins])
z_list_tot =  []
a_list_tot = []
dz_tot = []
DM_IGM_num = n_bins
DM_IGM_LC = np.zeros([Box_num, LC_num, DM_IGM_num])
Bin_centers_shifted_tot = []
start_time = time.time() 
for ii in range(Box_num):
    print("Start the DM calucalte part of{}".format(snapfile[ii][0]))
    sphX_bins = np.linspace(0, Boxsize, Bin_num)
    sphY_bins = np.linspace(0, Boxsize, Bin_num)
    sphZ_bins = np.linspace(0, Boxsize, Bin_num)
    sphpos = data[ii]["sphpos"]*h
    Smoothlen = data[ii]['smoothlen']
    bin_size = Boxsize / Bin_num
    dx = sphX_bins[1] - sphX_bins[0]
    dy = sphY_bins[1] - sphY_bins[0]
    dz = sphZ_bins[1] - sphZ_bins[0]
    diag = np.sqrt(2)/2*dx
    for i in range(len(Smoothlen)):
        Smoothlen[i] = max(Smoothlen[i], diag)   
    #h_L_max = np.ceil(np.max(data[0]['smoothlen']/diag)) ######向上取整
    #gird_num = h_L_max + 1
    #dS = dx*dz
    #dV = dx*dy*dz
    sphpos = data[ii]['sphpos']*h
    sphX = data[ii]["sphX"]*h
    sphY = data[ii]["sphX"]*h
    sphZ = data[ii]["sphZ"]*h
    F_e = data[ii]["n_e"]
    F_HI = data[ii]["n_HI"]
    F_HII = data[ii]["n_HII"]
    F_HeI = data[ii]["n_HeI"]
    F_HeII = data[ii]["n_HeII"]
    F_HeIII = data[ii]["n_HeIII"]
    dS = bin_size**2
    dV = (bin_size*MpcTocm)**3
    sphX_bin_mid = (sphX_bins + np.roll(sphX_bins, -1))[0:-1] / 2      ###
    sphY_bin_mid = (sphY_bins + np.roll(sphY_bins, -1))[0:-1] / 2   
    sphZ_bin_mid = (sphZ_bins + np.roll(sphZ_bins, -1))[0:-1] / 2
    sph_pos_mid = [sphX_bin_mid[0:-1], sphZ_bin_mid[0:-1]]

    SPHmass_o = data[ii]["sphmass"] #################The SPHmass not considering the Smoothing length###########
    SPHmass_cgs = SPHmass_o*SPHmasstocgs
    SPHrho = data[ii]["rho"]*SPHdensitytocgs###density
    SPHrho_column0 =  np.zeros([len(sphZ_bin_mid), len(sphZ_bin_mid) ])###########Non-normalization
    SPHrho_column =  np.zeros([len(sphZ_bin_mid), len(sphZ_bin_mid) ])###########Normalization
    #Calculate the new system 
    tan_alpha = np.sum(sphpos * u1, axis=1) / np.sum(sphpos * u3, axis=1)
    tan_delta = np.sum(sphpos * u2, axis=1) / np.sum(sphpos * u3, axis=1)
    # Select light cone within field limits
    in_field = np.abs(tan_alpha) <= alpha_max
    in_field &= np.abs(tan_delta) <= delta_max

   
    SPHmasstocgs = SPHmass_o*SPHmasstocgs 
    SPHrho = data[ii]['rho']*SPHdensitytocgs
    f_e = data[ii]['n_e']
    f_nHI = data[ii]['n_HI']
    f_nHII = data[ii]['n_HII']
    f_nHeI = data[ii]['n_HeI']
    f_nHeII = data[ii]['n_HeII']
    f_nHeIII = data[ii]['n_HeIII']
    Lbox = Boxsize


    Bin_edges = np.linspace(0, LC_length, n_bins+1)
    Bin_edges_shifted = np.linspace(ii*LC_length, (ii+1)*LC_length, n_bins+1)
    Bin_centers = (Bin_edges[:-1] + Bin_edges[1:]) / 2
    Bin_centers_shifted = (Bin_edges_shifted[:-1] + Bin_edges_shifted[1:]) / 2
    Bin_centers_shifted_tot.append(Bin_centers_shifted)
    Bin_centers_pos = np.array([u3*x for x in Bin_centers])
    h_smooth_max = np.max(Smoothlen)
    X_shift = np.linspace(0, Lbox, X_num+1)
    Y_shift = np.linspace(0, Lbox, Y_num+1)
    LC_origin = np.zeros([(X_num)*(Y_num),3])
    for i in range(X_num):
        for j in range(Y_num):
            LC_origin[i*Y_num+j, 0] = X_shift[i] + X_Bound_shift
            LC_origin[i*Y_num+j, 1] = Y_shift[j] + Y_Bound_shift
            LC_origin[i*Y_num+j, 2] = 0
    LC_shifted = np.array([Bin_centers_pos+x for x in LC_origin])
    # Apply periodic boundary conditions to the x and y coordinates
    LC_shifted[:, :, 0] %= Lbox
    LC_shifted[:, :, 1] %= Lbox
    LC_1000_1 = LC_shifted
    # Apply wrap-around for x and y coordinates that are less than 0
    LC_shifted[:, :, 0] += Lbox * (LC_shifted[:, :, 0] < 0)
    LC_shifted[:, :, 1] += Lbox * (LC_shifted[:, :, 1] < 0)

    #################### 大尺度结构#################
    z_list = np.array([output1 for output1, output2 in map(find_redshift_and_scale_factor, Bin_edges_shifted)]) #redshift
    a_list = np.array([output2 for output1, output2 in map(find_redshift_and_scale_factor, Bin_edges_shifted)]) #scale factor
    dz = np.zeros(len(z_list))
    dz =  z_list[1:] - z_list[:-1]
    z_list_tot.append(z_list)
    a_list_tot.append(a_list)
    dz_tot.append(z_list[1:] - z_list[:-1])
    def calc_ne_LC_shifted(k, i, LC_shifted, sphpos, h_smooth_max, Lbox, SPHmasstocgs, f_e, mp, Smoothlen):
        ne_LC_shifted_k_i = 0
        ne_LC_Ion_shifted_k_i = 0
        idx_LC, dist_LC = find_particles_within_RV(sphpos, LC_shifted[k][i], h_smooth_max, Lbox)
        for j in range(len(idx_LC)):
            if idx_LC[j] >= len(sphpos):
                idx_LC[j] = idx_LC[j] - len(sphpos)
            if dist_LC[j] >= Smoothlen[idx_LC[j]]:
                continue
            else:
                weight = M6(dist_LC[j]*MpcTocm, 0, Smoothlen[idx_LC[j]]*MpcTocm, 3)
                ne_LC_shifted_k_i += SPHmasstocgs[idx_LC[j]]*f_e[idx_LC[j]]/mp * weight
                ne_LC_Ion_shifted_k_i += SPHmasstocgs[idx_LC[j]]*f_nHII[idx_LC[j]] * weight/mp + SPHmasstocgs[idx_LC[j]]*f_nHeII[idx_LC[j]]/mp * weight + SPHmasstocgs[idx_LC[j]]/2*f_nHeIII[idx_LC[j]]/mp * weight
        if (k*n_bins + i)%400 == 0:
            print("This is the %d steps" %(k*n_bins + i))
            end_time = time.time()  # 记录程序结束时间
            run_time = end_time - start_time  # 计算程序运行时间，单位为秒
            print(f"程序已运行时间为：{run_time:.2f}秒 " + "\nfor {}".format(snapfile[ii][0]))
            print(f"程序剩余时间约为：{((Box_num*LC_num*n_bins/(k*n_bins + i +1e-30)*run_time-run_time)/3600):.2f}小时"+"\nfor {}".format(snapfile[ii][0]))
            sys.stdout.flush()
        return ne_LC_shifted_k_i, ne_LC_Ion_shifted_k_i

    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()  # 获取CPU核数
    results = Parallel(n_jobs=num_cores)(delayed(calc_ne_LC_shifted)(k, i, LC_shifted, sphpos, h_smooth_max, Lbox, SPHmasstocgs, f_e, mp, Smoothlen) for k in range(LC_num) for i in range(n_bins))
    X_shift = np.linspace(0, Lbox, X_num+1)
    Y_shift = np.linspace(0, Lbox, Y_num+1)
    LC_origin = np.zeros([(X_num)*(Y_num),3])

    for k in range(LC_num):
        for i in range(n_bins):
            ne_LC_shifted[ii][k][i], ne_LC_Ion_shifted[ii][k][i] = results[k*n_bins+i]


    for i in range(LC_num):
        for j in range(DM_IGM_num-1):
            if ii == 0 :
                DM_IGM_LC[ii][i][0] = c/H_0*ne_LC_shifted[ii][i][0]*(1+z_list[0])*dz[0]/np.sqrt(Omega_0*(1+z_list[0])**3+Omega_Lambda)*1e6
            else:
                DM_IGM_LC[ii][i][0] = DM_IGM_LC[ii-1][i][-1] +  c/H_0*ne_LC_shifted[ii][i][0]*(1+z_list[0])*dz[0]/np.sqrt(Omega_0*(1+z_list[0])**3+Omega_Lambda)*1e6
            DM_IGM_LC[ii][i][j+1] = DM_IGM_LC[ii][i][j] + c/H_0*ne_LC_shifted[ii][i][j+1]*(1+z_list[j+1])*dz[j+1]/np.sqrt(Omega_0*(1+z_list[j+1])**3+Omega_Lambda)*1e6
    
    X_Bound_shift += L/m
    Y_Bound_shift += L/n
    Z_Bound_shift += L
    print("Fnished the DM calucalte part of {}".format(snapfile[ii][0]))

z_list_tot = np.array(z_list_tot)
a_list_tot = np.array(a_list_tot)
print("Finish the DM_IGM calucalte part")

print("#################################################")
print("Start storing results")
filename = 'DM_calculate.hdf5'
if os.path.exists(filename):
    os.remove(filename)
f = h5py.File(filename, "w")
# 在文件中创建数据集、组等等
f.create_dataset('Distance', (Box_num, n_bins), data = Bin_centers_shifted_tot)
f.create_dataset('Redshift', (Box_num, n_bins+1), data = z_list_tot)
f.create_dataset('dz', (Box_num, n_bins), data = dz_tot)
f.create_dataset('Scale_factor', (Box_num, n_bins+1), data = a_list_tot)
f.create_dataset('ne_HG', (nlos, n_bins), data = n_e_arr1)
f.create_dataset('DM_HG', (nlos, n_bins), data = DM_HG)
f.create_dataset('ne_IGM', (Box_num, LC_num, n_bins), data = ne_LC_shifted)
dset = f.create_dataset('ne_IGM_Ion', (Box_num, LC_num, n_bins), data = ne_LC_Ion_shifted)
# 给数据集添加一个属性
dset.attrs['Light_cone_number'] = str(LC_num)
dset.attrs['Box_number'] = str(Box_num)
dset.attrs['LOS_bins'] = str(n_bins)
dset1 = f.create_dataset('DM_IGM', (Box_num, LC_num, n_bins), data = DM_IGM_LC)
dset1.attrs['Light_cone_number'] = str(LC_num)
dset1.attrs['Box_number'] = str(Box_num)
dset1.attrs['LOS_bins'] = str(n_bins)
f.close()
run_time = end_time - start_time  # 计算程序运行时间，单位为秒
print(f"程序总运行时间为：{run_time/3600:.2f}小时 ")
print("Fnished all")