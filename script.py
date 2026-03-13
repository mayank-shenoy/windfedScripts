#IMPORTS
import numpy as np
import glob
import os
import sys
import matplotlib.gridspec as grd
from scipy.ndimage import zoom, map_coordinates, label, find_objects
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks, peak_widths, peak_prominences
import matplotlib.pyplot as plt
from skan import Skeleton, summarize
from skimage.morphology import skeletonize
import networkx as netx


#FUNCTIONS
def slice(field, vx, vy, L, axis):
    v = np.array([vx, vy])
    v = v / np.linalg.norm(v)

    n = 256
    r_val = np.linspace(0, L, n)
    z_val = np.linspace(-L/2, L/2, n)
    r_grid, z_grid = np.meshgrid(r_val, z_val, indexing='ij')

    if axis == 'z':
        x = r_grid * v[0]
        y = r_grid * v[1]
        z = z_grid
    elif axis == 'y':
        x = r_grid * v[0]
        y = z_grid
        z = r_grid * v[1]
    elif axis == 'x':
        x = z_grid
        y = r_grid * v[0]
        z = r_grid * v[1]

    def to_index(coord):
        return (coord + L) * (n - 1) / (2 * L)

    x_idx = to_index(x)
    y_idx = to_index(y)
    z_idx = to_index(z)

    coords = np.array([x_idx, y_idx, z_idx])
    slice_data = map_coordinates(field, coords, order=1, mode='nearest')

    return slice_data, coords, r_grid, z_grid

def project(Fx, Fy, Fz, coords, vx, vy):
    v = np.array([vx, vy])
    v = v / np.linalg.norm(v)
    F_e1 = Fx * v[0] + Fy * v[1]
    F1_slice = map_coordinates(F_e1, coords, order=1, mode='nearest')
    F2_slice = map_coordinates(Fz, coords, order=1, mode='nearest')
    return F1_slice, F2_slice

def profile(field, dx, dz, l, x0, z0, x_axis, z_axis):
    if m==0:
        vx = 0
        vz = 1
    else:
        dx = 1.0
        dz = -1.0 / m
        vx = dx / np.sqrt(dx**2 + dz**2)
        vz = dz / np.sqrt(dx**2 + dz**2)
    s = np.linspace(-l/2, l/2, 100)
    x_l = x0 + s*vx
    z_l = z0 + s*vz
    points = np.stack([z_l, x_l], axis=-1)
    interp = RegularGridInterpolator((z_axis, x_axis), field.T, bounds_error=False, fill_value=np.nan)
    return interp(points), x_l, z_l, (x_l-x0)*vx + (z_l-z0)*vz

def der(f,n,x,y,z):
    if (n==0):
        return 0
    elif (n==1):
        return np.gradient(f,x[:,0,0],axis=n-1, edge_order=2)
    elif (n==2):
        return np.gradient(f,y[0,:,0],axis=n-1, edge_order=2)
    elif (n==3):
        return np.gradient(f,z[0,0,:],axis=n-1, edge_order=2)

def symbol(i,j,k,l):
    if ((i==j) or (i==k) or (l==i) or (k==j) or (l==j) or (l==k)):
        return 0
    else:
        return (j - i) * (k - i) * (l - i) * (k - j) * (l - j) * (l - k) / (np.abs(j - i) * np.abs(k - i) * np.abs(l - i) * np.abs(k - j) * np.abs(l - j) * np.abs(l - k))

def F(g,ud,bd):
    det_g = np.empty((128,128,128))
    for i in range(128):
        for j in range(128):
            for k in range(128):
                det_g[i,j,k] = np.linalg.det(g[:,:,i,j,k])
    far = np.zeros(g.shape)
    for i in range(4):
        for j in range(4):
            sum0 = 0
            for k in range(4):
                for l in range(4):
                    temp = symbol(i,j,k,l) * ud[k,:,:,:] * bd[l,:,:,:]
                    sum0 = temp + sum0
            far[i,j,:,:,:] = -((-det_g[:,:,:])**(-0.5)) * sum0
    return far

def J(gd,gu,F,F_post,x,y,z):
    chris = np.empty((4,4,4,128,128,128))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                sum = 0
                for l in range(4):
                    temp = 0.5 * gu[i,l,:,:,:] * ( der(gd[l,k,:,:,:],j,x,y,z) + der(gd[j,l,:,:,:],k,x,y,z) + der(gd[j,k,:,:,:],l,x,y,z) )
                    sum = sum + temp
                chris[i,j,k,:,:,:] = sum
    cur = np.empty((4,128,128,128))
    for i in range(4):
        sum0 = 0
        for j in range(4):
            sum1 = 0
            for k in range(4):
                temp = chris[j,j,k] * F[k,i,:,:,:] + chris[i,j,k] * F[j,k,:,:,:]
                sum1 = sum1 + temp
            if (j==0):
                temp0 = (F_post[j,i,:,:,:] - F[j,i,:,:,:]) / 10
            else:
                temp0 = der(F[j,i,:,:,:],j,x,y,z)
            sum0 = temp0 + sum1
        cur[i,:,:,:] = sum0
    return cur, np.sqrt(np.einsum('mxyz,nxyz,mnxyz->xyz', cur[1:,:,:,:], cur[1:,:,:,:], gd[1:,1:,:,:,:]))

def tangent(x, y, b):
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    for i in range(x.size-2*b):
        ii = b+i
        delx = 0
        dely = 0
        for bb in range(-b,b+1):
            delx = delx + np.abs(x[ii+bb]-x[ii])
            dely = dely + np.abs(y[ii+bb]-y[ii])
        dx[ii] = np.sign(x[ii+1]-x[ii])*delx/np.sqrt(delx**2+dely**2)
        dy[ii] = np.sign(y[ii+1]-y[ii])*dely/np.sqrt(delx**2+dely**2)
    for i in range(1,b):
        delx = 0
        delx2 = 0
        dely = 0
        dely2 = 0
        for ii in range(-i,i+1):
            delx = delx + np.abs(x[i+ii]-x[i])
            dely = dely + np.abs(y[i+ii]-y[i])
            delx2 = delx2 + np.abs(x[-1-i+ii]-x[-1-i])
            dely2 = dely2 + np.abs(y[-1-i+ii]-y[-1-i])
        dx[i] = np.sign(x[i+1]-x[i])*delx/np.sqrt(delx**2+dely**2)
        dy[i] = np.sign(y[i+1]-y[i])*dely/np.sqrt(delx**2+dely**2)
        dx[-1-i] = np.sign(x[-i]-x[-1-i])*delx2/np.sqrt(delx2**2+dely2**2)
        dy[-1-i] = np.sign(y[-i]-y[-1-i])*dely2/np.sqrt(delx2**2+dely2**2)
    delx = x[1]-x[0]
    dely = y[1]-y[0]
    dx[0] = delx/np.sqrt(delx**2+dely**2)
    dy[0] = dely/np.sqrt(delx**2+dely**2)
    delx2 = x[-1]-x[-2]
    dely2 = y[-1]-y[-2]
    dx[-1] = delx2/np.sqrt(delx2**2+dely2**2)
    dy[-1] = dely2/np.sqrt(delx2**2+dely2**2)
    return dx,dy

def cleanup(sig, da, rr, rplus):
    dbp[rr < rplus] = dbp.min()
    sig[rr < rplus] = sig.max()
    da[rr < rplus] = da.min()
    mask = (sig < 2) & (np.log10(da)>-1)
    labeled, num = label(mask)
    for k in range(1, num + 1):
        if rr[labeled == k].min() > 1.025*rplus:
            mask[labeled == k] = 0
    labeled2, num2 = label(mask)
    sizes = np.bincount(labeled2.ravel())
    if num2 > 0:
        [s] =np.where(sizes == sizes[1:].max())
        if len(s) > 1:
            check = []
            for i in range(len(s)):
                check.append(rr[labeled == s[i]].max())
            s = s[np.argmax(check)]
            mask  = labeled2 == s
        else:
            mask  = labeled2 == s
    return mask

def sheet(mask, sigma_slice, r_slice, z_slice):
    skel_img = skeletonize(mask).astype(bool)
    skel = Skeleton(skel_img)

    adj = skel.graph
    G = netx.from_scipy_sparse_array(adj)

    if G.number_of_nodes() == 0:
        return np.array([]), np.array([]), np.array([])

    coords = np.array([skel.coordinates[n] for n in range(G.number_of_nodes())])
    r_vals = r_slice[coords[:, 0].astype(int), coords[:, 1].astype(int)]
    z_vals = z_slice[coords[:, 0].astype(int), coords[:, 1].astype(int)]
    sigma_vals = sigma_slice[coords[:, 0].astype(int), coords[:, 1].astype(int)]
    rr = np.sqrt(r_vals**2 + z_vals**2)

    node_start = np.argmin(rr)

    def edge_weight(u, v, d):
        return 0.5 * (sigma_vals[u] + sigma_vals[v])

    degree_dict = dict(G.degree())
    terminal_nodes = [n for n, deg in degree_dict.items() if deg == 1 and n != node_start]
    if not terminal_nodes:
        return np.array([]), np.array([]), np.array([])
    
    best_path, best_avg_sigma = None, np.inf
    
    for n_end in terminal_nodes:
        try:
            path_nodes = netx.shortest_path(G, node_start, n_end, weight=edge_weight)
            avg_sigma = np.mean(sigma_vals[path_nodes])
            if avg_sigma < best_avg_sigma:
                best_avg_sigma = avg_sigma
                best_path = path_nodes
        except netx.NetworkXNoPath:
            continue

    if best_path is None:
        return np.array([]), np.array([]), np.array([]), branch_stats

    path_r = r_vals[best_path]
    path_z = z_vals[best_path]
    diffs = np.sqrt(np.diff(path_r)**2 + np.diff(path_z)**2)
    l = np.concatenate([[0], np.cumsum(diffs)])

    return path_r, path_z, l

def polar(field, r0, z_axis, L):
    n_th = np.round(512*np.pi*r0/15)
    th_val = np.linspace(0,2*np.pi,int(n_th))
    theta_grid, z_grid = np.meshgrid(th_val, z_axis, indexing='ij')
    x = r0 * np.cos(theta_grid)
    y = r0 * np.sin(theta_grid)
    z = z_grid
    def to_index(coord):
        return (coord + L) * (128 - 1) / (2 * L)
    x_idx = to_index(x)
    y_idx = to_index(y)
    z_idx = to_index(z)

    coords = np.array([x_idx, y_idx, z_idx])
    slice_data = map_coordinates(field, coords, order=1, mode='nearest')
    return slice_data, coords, theta_grid, z_grid

def eruption_data():
    data = {
        'standard': {
            'short': {}
            },
        'mad': {
            'beta6_a0': {},
            'beta6_a9': {}
            },
        'nonmad': {
            'beta6_a0': {},
            'beta6_a9': {},
            'beta2_a9': {},
            'beta2_a0': {}
            }
        }
    global vars
    vars = ['rflux', 'dt', 'jetmax', 'njetmax', 'phimax', 'del_t']
    for state in data.keys():
        for run in data[state].keys():
            for var in vars:
                data[state][run][var] = []
    os.chdir("/fs/lustre/scratch/mshenoy/sim_runs/grmhd_restart_beta_1e6_cooling_121_a0_electrons")
    rd_1d_avg()
    ir2 = r_to_ir(2)
    ir5 = r_to_ir(5)   
    global peaks60, troughs60  
    peaks60 = [85, 135, 266, 294, 523, 557, 810, 938, 1070, 1125, 1357, 1449, 1616, 2012, 2047, 2170, 2534, 2615, 2894, 2930, 3158, 3269, 3464, 3534, 3628, 3795, 4014, 4184, 4443, 4529, 4556, 4575, 4725, 4786, 4852, 4892, 4977, 5019, 5092, 5157, 5245, 5351, 5432, 5681, 5716, 5762, 5807, 5879, 5949, 6035, 6089]
    troughs60 = [129, 151, 282, 315, 536, 572, 837, 964, 1081, 1142, 1370, 1469, 1641, 2033, 2060, 2193, 2551, 2624, 2902, 2937, 3169, 3279, 3474, 3549, 3646, 3808, 4035, 4194, 4455, 4536, 4568, 4589, 4747, 4830, 4875, 4902, 4983, 5052, 5107, 5170, 5287, 5378, 5469, 5692, 5722, 5777, 5827, 5888, 5968, 6076, 6108]
    njet60 = 100*Edot[:,ir5]/np.abs(mdot[:,ir2])
    for i in range(len(peaks60)):
        state = 'nonmad'
        if peaks60[i] >= 4720:
            state = 'mad'
        data[state]['beta6_a0']['rflux'].append((Phibh[:,ir2][peaks60[i]]-Phibh[:,ir2][troughs60[i]])/Phibh[:,ir2][peaks60[i]])
        data[state]['beta6_a0']['dt'].append(t[troughs60[i]]-t[peaks60[i]])
        data[state]['beta6_a0']['jetmax'].append(((rjet_max_p[peaks60[i]:troughs60[i]+1]+rjet_max_m[peaks60[i]:troughs60[i]+1])/2).max())
        data[state]['beta6_a0']['njetmax'].append(njet60[peaks60[i]:troughs60[i]+1].max())
        data[state]['beta6_a0']['phimax'].append(Phibh[:,ir2][peaks60[i]])
        if i==0:
            continue
        else:
            data[state]['beta6_a0']['del_t'].append(t[peaks60[i]]-t[troughs60[i-1]])
    os.chdir("/fs/lustre/scratch/mshenoy/sim_runs/grmhd_restart_beta_1e6_cooling_121_a9_electrons")
    rd_1d_avg()
    ir2 = r_to_ir(2)
    ir5 = r_to_ir(5)
    global peaks69, troughs69
    peaks69 = [87, 121, 176, 214, 234, 299, 361, 381, 425, 633, 967, 1101, 1299, 1535, 1552, 1712, 1733, 1805, 1986, 2076, 2203, 2324, 2623, 2682, 2852, 2892, 3072, 3101, 3141, 3358, 3467, 3617, 3710, 3722, 3734, 3766, 3817, 3883, 3994, 4039, 4472, 4499, 4991, 5106, 5143, 5256, 5413, 5568, 5633, 5727, 5770, 5886, 5903, 5965, 6008, 6033, 6228, 6358]
    troughs69 = [98, 131, 192, 224, 247, 309, 371, 406, 435, 642, 977, 1117, 1309, 1546, 1563, 1720, 1760, 1812, 2000, 2090, 2213, 2334, 2633, 2697, 2860, 2902, 3085, 3111, 3159, 3373, 3486, 3646, 3718, 3732, 3745, 3775, 3830, 3893, 4020, 4050, 4485, 4505, 4999, 5113, 5181, 5269, 5435, 5600, 5643, 5751, 5805, 5892, 5926, 5981, 6019, 6066, 6247, 6374]
    njet69 = 100*Edot[:,ir5]/np.abs(mdot[:,ir2])
    for i in range(len(peaks69)):
        state = 'nonmad'
        if peaks69[i] >= 4578:
            state = 'mad'
        data[state]['beta6_a9']['rflux'].append((Phibh[:,ir2][peaks69[i]]-Phibh[:,ir2][troughs69[i]])/Phibh[:,ir2][peaks69[i]])
        data[state]['beta6_a9']['dt'].append(t[troughs69[i]]-t[peaks69[i]])
        data[state]['beta6_a9']['jetmax'].append(((rjet_max_p[peaks69[i]:troughs69[i]+1]+rjet_max_m[peaks69[i]:troughs69[i]+1])/2).max())
        data[state]['beta6_a9']['njetmax'].append(njet69[peaks69[i]:troughs69[i]+1].max())
        data[state]['beta6_a9']['phimax'].append(Phibh[:,ir2][peaks69[i]])
        if i==0:
            continue
        else:
            data[state]['beta6_a9']['del_t'].append(t[peaks69[i]]-t[troughs69[i-1]])
    os.chdir("/fs/lustre/scratch/mshenoy/sim_runs/grmhd_restart_beta_1e2_cooling_121_a9_electrons")
    rd_1d_avg()
    ir2 = r_to_ir(2)
    ir5 = r_to_ir(5)
    global peaks29, troughs29
    peaks29 = [127, 183, 281, 715, 971, 1104, 1195, 1289, 1324, 1490, 1614, 1631, 1698, 1771, 1838, 1972, 1998, 2060, 2102, 2178, 2820, 2856, 3031, 3085, 3103, 3380, 3596, 3674, 3787, 4268, 4402, 4467, 4915, 5018, 5738, 5762, 5798, 5886]
    troughs29 = [137, 195, 295, 751, 998, 1145, 1213, 1307, 1350, 1508, 1623, 1651, 1709, 1799, 1855, 1983, 2009, 2098, 2119, 2196, 2841, 2869, 3055, 3100, 3118, 3413, 3615, 3702, 3804, 4291, 4439, 4489, 4947, 5063, 5749, 5780, 5811, 5901]
    njet29 = 100*Edot[:,ir5]/np.abs(mdot[:,ir2])
    for i in range(len(peaks29)):
        data['nonmad']['beta2_a9']['rflux'].append((Phibh[:,ir2][peaks29[i]]-Phibh[:,ir2][troughs29[i]])/Phibh[:,ir2][peaks29[i]])
        data['nonmad']['beta2_a9']['dt'].append(t[troughs29[i]]-t[peaks29[i]])
        data['nonmad']['beta2_a9']['jetmax'].append(((rjet_max_p[peaks29[i]:troughs29[i]+1]+rjet_max_m[peaks29[i]:troughs29[i]+1])/2).max())
        data['nonmad']['beta2_a9']['njetmax'].append(njet29[peaks29[i]:troughs29[i]+1].max())
        data['nonmad']['beta2_a9']['phimax'].append(Phibh[:,ir2][peaks29[i]])
        if i==0:
            continue
        else:
            data['nonmad']['beta2_a9']['del_t'].append(t[peaks29[i]]-t[troughs29[i-1]])
    os.chdir("/fs/lustre/scratch/mshenoy/sim_runs/grmhd_restart_beta_1e2_cooling_121_a0_electrons")
    rd_1d_avg()
    ir2 = r_to_ir(2)
    ir5 = r_to_ir(5)
    global peaks20, troughs20
    peaks20 = [178, 286, 728, 795, 922, 1359, 1423, 1527, 1704, 1757, 1985, 2090, 2362, 2721, 2872, 2937, 3056, 3094, 3231, 3416, 3605, 3732, 3908, 3963, 4062, 4123, 4316, 4542, 4647, 4690, 4740, 4777, 4859, 4914, 4951, 5178, 5276, 5380, 5405, 5442, 5497, 5545, 5588, 5611, 5745, 5944, 5964, 6426, 6643]
    troughs20 = [215, 338, 744, 811, 967, 1371, 1455, 1558, 1752, 1788, 1993, 2125, 2418, 2761, 2920, 2984, 3067, 3124, 3259, 3429, 3623, 3765, 3952, 3987, 4072, 4132, 4344, 4561, 4662, 4724, 4763, 4804, 4890, 4929, 4964, 5210, 5296, 5389, 5420, 5481, 5509, 5576, 5607, 5654, 5780, 5956, 5990, 6441, 6676]
    njet20 = 100*Edot[:,ir5]/np.abs(mdot[:,ir2])
    for i in range(len(peaks20)):
        data['nonmad']['beta2_a0']['rflux'].append((Phibh[:,ir2][peaks20[i]]-Phibh[:,ir2][troughs20[i]])/Phibh[:,ir2][peaks20[i]])
        data['nonmad']['beta2_a0']['dt'].append(t[troughs20[i]]-t[peaks20[i]])
        data['nonmad']['beta2_a0']['jetmax'].append(((rjet_max_p[peaks20[i]:troughs20[i]+1]+rjet_max_m[peaks20[i]:troughs20[i]+1])/2).max())
        data['nonmad']['beta2_a0']['njetmax'].append(njet20[peaks20[i]:troughs20[i]+1].max())
        data['nonmad']['beta2_a0']['phimax'].append(Phibh[:,ir2][peaks20[i]])
        if i==0:
            continue
        else:
            data['nonmad']['beta2_a0']['del_t'].append(t[peaks20[i]]-t[troughs20[i-1]])
    os.chdir("/fs/lustre/scratch/mshenoy/sim_runs/mad_case_a_0.9_128_ppm")
    rd_1d_avg()
    ir2 = r_to_ir(2)
    ir5 = r_to_ir(5)
    global peaksm, troughsm
    peaksm = [212, 237, 255, 285, 290, 301, 317, 343, 353, 357, 375, 378, 384, 398, 405, 416, 438, 446, 451, 463, 466, 472, 479, 494]
    troughsm = [220, 240, 268, 289, 299, 303, 318, 346, 356, 364, 377, 381, 395, 404, 408, 420, 441, 449, 453, 464, 467, 477, 484, 496]
    njetm = 100*Edot[:,ir5]/np.abs(mdot[:,ir2])
    for i in range(len(peaksm)):
        data['standard']['short']['rflux'].append((Phibh[:,ir2][peaksm[i]]-Phibh[:,ir2][troughsm[i]])/Phibh[:,ir2][peaksm[i]])
        data['standard']['short']['dt'].append(t[troughsm[i]]-t[peaksm[i]])
        data['standard']['short']['jetmax'].append(((rjet_max_p[peaksm[i]:troughsm[i]+1]+rjet_max_m[peaksm[i]:troughsm[i]+1])/2).max())
        data['standard']['short']['njetmax'].append(njetm[peaksm[i]:troughsm[i]+1].max())
        data['standard']['short']['phimax'].append(Phibh[:,ir2][peaksm[i]])
        if i==0:
            continue
        else:
            data['standard']['short']['del_t'].append(t[peaksm[i]]-t[troughsm[i-1]])
    return data

def hist(data, xdata, ydata, state):
  labels = {
      'rflux': 'Relative Flux Drop',
      'dt': 'Eruption Duration (in M)',
      'jetmax': 'Max Jet Radius (in $r_G$)',
      'njetmax': 'Max Jet Efficiency (in %)',
      'phimax': 'Peak Flux'
  }
  bins = {
      'rflux': np.logspace(np.log10(0.02), np.log10(0.6), 21),
      'dt': np.linspace(0, 599, 21),
      'jetmax': np.logspace(np.log10(2), np.log10(1500), 21),
      'njetmax': np.logspace(-1, np.log10(1200), 21),
      'phimax': np.logspace(np.log10(40), np.log10(300), 21),
  }
  name = {
      'beta6_a0': r'$a=0$',
      'beta6_a9': r'$a=0.9375$',
      'beta2_a9': r'$a=0.9375$',
      'beta2_a0': r'$a=0$',
      'a9': r'$a=0.9375$',
      'mad': 'Windfed MAD',
      'nonmad': 'Windfed Non-MAD',
      'standard': 'Standard MAD'
  }
  title = {
      'mad': 'Windfed MAD Flux Eruptions',
      'nonmad': 'Windfed Non-MAD Flux Eruptions',
      'standard': 'Standard MAD Flux Eruptions',
      'all': 'Windfed vs Standard Flux Eruptions',
      'detail': 'Windfed vs Standard Flux Eruptions'
  }
  mark = {
      'beta6_a0': '.',
      'beta6_a9': '*',
      'beta2_a9': '*',
      'beta2_a0': '.',
      'a9': '*',
  }
  color = {
      'mad': '#0072B2',
      'nonmad': '#E69F00',
      'standard': '#009E73'
  }
  clf()
  fig, ax = plt.subplots(2,2, dpi=200, figsize=(6,5), gridspec_kw=dict(height_ratios=[1,3], width_ratios=[3,1]))
  ax[0,1].axis('off')
  if state in ['standard','mad','nonmad']:
      for run in data[state].keys():
          ax[1,0].scatter(data[state][run][xdata], data[state][run][ydata], label=name[run], )
          ax[0,0].hist(data[state][run][xdata], bins=bins[xdata], alpha=0.4)
          ax[1,1].hist(data[state][run][ydata], bins=bins[ydata], orientation='horizontal', alpha=0.4)
  elif state=='all':
      for stat in data.keys():
          dataset_x = []
          dataset_y = []
          for run in data[stat].keys():
              dataset_x.extend(data[stat][run][xdata])
              dataset_y.extend(data[stat][run][ydata])
          ax[1,0].scatter(dataset_x, dataset_y, label=name[stat])
          ax[0,0].hist(dataset_x, bins=bins[xdata], alpha=0.4)
          ax[1,1].hist(dataset_y, bins=bins[ydata], orientation='horizontal', alpha=0.4)
  else:
      for stat in data.keys():
          dataset_x = []
          dataset_y = []
          for run in data[stat].keys():
              dataset_x.extend(data[stat][run][xdata])
              dataset_y.extend(data[stat][run][ydata])
              ax[1,0].scatter(data[stat][run][xdata], data[stat][run][ydata], color=color[stat], marker=mark[run])
          ax[0,0].hist(dataset_x, bins=bins[xdata], alpha=0.4, color=color[stat])
          ax[1,1].hist(dataset_y, bins=bins[ydata], orientation='horizontal', alpha=0.4, color=color[stat])
      spin_handles = [Line2D([0], [0], marker='.', linestyle='', color='black', label=r'$a=0$'),
                      Line2D([0], [0], marker='*', linestyle='', color='black', label=r'$a=0.9375$')]
      state_handles = [Line2D([0], [0], marker='s', linestyle='', markerfacecolor=c, markeredgecolor=c, label=name[key]) for key, c in color.items()]
      handle = spin_handles + state_handles
      ax[0,1].legend(handles=handle, loc='center', fontsize=8, frameon=False, ncol=1)
  ax[1,0].set_xscale('log')
  ax[0,0].set_xscale('log')
  ax[1,0].set_yscale('log')
  ax[1,1].set_yscale('log')
  if xdata == 'phimax':
    k = 0.044
    a = 0.9375
    rplus = 1.+ np.sqrt(1.-a**2)
    omega = a/(2*rplus)
    f = 1 + 1.38 * (omega)**2 - 9.2 * (omega)**4
    BZ = k * omega**2 * (bins[xdata]/2)**2 * f
    ax[1,0].plot(bins[xdata], BZ, color='black', ls=':', alpha=0.5, label=r'$P_\text{BZ}$ ($a=0.9375$)')
    ax[1,0].legend(frameon=False)
  elif xdata == 'dt':
    ax[1,0].set_xscale('linear')
    ax[0,0].set_xscale('linear')
    dataset_x = []
    dataset_y = []
    for stat in data.keys():
      for run in data[stat].keys():
        dataset_x.extend(data[stat][run][xdata])
        dataset_y.extend(data[stat][run][ydata])
    def model(x, m, b):
      return m * x + b
    opt, cov = curve_fit(model, dataset_x, np.log10(dataset_y))
    ax[1,0].plot(bins[xdata], 10**opt[1] * 10**(bins[xdata]*opt[0]), color='white', ls=':', alpha=0.5, label=r'$|(\Delta\Phi)_{rel}| \propto e^{t/%0.2f}$'%(np.log(np.e)/opt[0]))
    ax[1,0].legend(frameon=False)
  ax[1,0].set_xlabel(labels[xdata])
  ax[1,0].set_ylabel(labels[ydata])
  ax[1,0].set_xlim(bins[xdata][0], bins[xdata][-1])
  ax[1,0].set_ylim(bins[ydata][0], bins[ydata][-1])
  ax[0,0].set_xlim(bins[xdata][0], bins[xdata][-1])
  ax[1,1].set_ylim(bins[ydata][0], bins[ydata][-1])
  for a in ax.flat:
      a.tick_params(which='both', direction='in', top=True, right=True)
  ax[0,0].tick_params(axis='x', which='both', labelbottom=False)
  ax[1,1].tick_params(axis='y', which='both', labelleft=False)
  plt.tight_layout()
  fig.suptitle(title[state])
  fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.975, top=0.925)
  plt.savefig('/fs/lustre/scratch/mshenoy/distribution_plots/test_%s_%s_%s.png'%(state, xdata, ydata), dpi=400)

#START
%run -i ./athena_script.py
a = 0.9375
rplus = 1.+ np.sqrt(1.-a**2)
th = np.linspace(0.,2*np.pi,1000)
xh = rplus*np.sin(th)
yh = rplus*np.cos(th)
rerg = 1.+np.sqrt(1.-a**2*np.cos(th)**2)
xe = rerg*np.sin(th)
ye = rerg*np.cos(th)
plt.rcParams.update({"font.size": 14})


#ALL PLOT
fig, ax = plt.subplots(2,3,figsize=(24,12), dpi=400)
f1 = ax[0,0].pcolormesh(x[:,64,:], z[:,64,:], np.log10(rho[:,64,:]), cmap='afmhot')
fig.colorbar(f1, ax=ax[0,0], label=r'$\log_{10}(\rho)$')
ax[0,0].fill(xh,yh,'k')
ax[0,0].plot(xe,ye,'w')
ax[0,0].set_xlabel(r'$x$ ($r_G$)')
ax[0,0].set_ylabel(r'$y$ ($r_G$)')
ax[0,0].set_title(r'$t = 12400M$' )
f2 = ax[0,1].pcolormesh(x[:,64,:], z[:,64,:], np.log10(T[:,64,:]), cmap='afmhot')
fig.colorbar(f2, ax=ax[0,1], label=r'$\log_{10}(T)$')
ax[0,1].fill(xh,yh,'k')
ax[0,1].plot(xe,ye,'w')
ax[0,1].set_xlabel(r'$x$ ($r_G$)')
ax[0,1].set_ylabel(r'$z$ ($r_G$)')
ax[0,1].set_title(r'$t = 12400M$' )
f3 = ax[0,2].pcolormesh(x[:,64,:], z[:,64,:], np.log10(beta[:,64,:]), cmap='afmhot')
fig.colorbar(f3, ax=ax[0,2], label=r'$\log_{10}(\beta)$')
ax[0,2].fill(xh,yh,'k')
ax[0,2].plot(xe,ye,'w')
ax[0,2].set_xlabel(r'$x$ ($r_G$)')
ax[0,2].set_ylabel(r'$z$ ($r_G$)')
ax[0,2].set_title(r'$t = 12400M$' )
f4 = ax[1,0].pcolormesh(x[:,64,:], z[:,64,:], np.log10(sigma[:,64,:]), cmap='bwr')
fig.colorbar(f4, ax=ax[1,0], label=r'$\log_{10}(\sigma)$')
ax[1,0].streamplot(x[:,64,:].transpose(), z[:,64,:].transpose(), Bcc1[:,64,:].transpose(), Bcc3[:,64,:].transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
ax[1,0].fill(xh,yh,'k')
ax[1,0].plot(xe,ye,'w')
ax[1,0].set_xlabel(r'$x$ ($r_G$)')
ax[1,0].set_ylabel(r'$z$ ($r_G$)')
ax[1,0].set_title(r'$t = 12400M$' )
f5 = ax[1,1].pcolormesh(x[:,64,:], z[:,64,:], np.log10(B_r[:,64,:]), vmin=-15, vmax=4, cmap='afmhot')
fig.colorbar(f5, ax=ax[1,1], label=r'$\log_{10}(\frac{B_x^2+B_z^2}{B_y^2})$')
ax[1,1].fill(xh,yh,'k')
ax[1,1].plot(xe,ye,'w')
ax[1,1].set_xlabel(r'$x$ ($r_G$)')
ax[1,1].set_ylabel(r'$z$ ($r_G$)')
ax[1,1].set_title(r'$t = 12400M$' )
f6 = ax[1,2].pcolormesh(x[:,64,:], z[:,64,:], np.log10(cur_mag[:,64,:]), cmap='afmhot')
fig.colorbar(f6, ax=ax[1,2], label=r'$\log_{10}|J|$')
ax[1,2].fill(xh,yh,'k')
ax[1,2].plot(xe,ye,'w')
ax[1,2].set_xlabel(r'$x$ ($r_G$)')
ax[1,2].set_ylabel(r'$z$ ($r_G$)')
ax[1,2].set_title(r'$t = 12400M$' )
plt.savefig('all.png')


#REPORT PLOT
fig, ax = plt.subplots(3,figsize=(5.5,12), dpi=400, constrained_layout = True, sharex=True)
for i in range(3):
    yt_extract_box(i_dump=1241, box_radius=15/2**i, mhd=True, gr=True, a=0.9375)
    far_post = F(g, Lower(uu,g), bd)
    yt_extract_box(i_dump=1240, box_radius=15/2**i, mhd=True, gr=True, a=0.9375)
    far = F(g, Lower(uu,g), bd)
    cur = J(g,gi,far,far_post,x,y,z)
    cur_mag = np.sqrt(cur[1, :, :, :] ** 2 + cur[2, :, :, :] ** 2 + cur[3, :, :, :] ** 2)
    T = press / rho
    gam = 5/3
    sigma = bsq / (rho + gam / (gam - 1) * press)
    f2 = ax[0].pcolormesh(x[2:125,64,2:125], z[2:125,64,2:125], np.log10(T[2:125,64,2:125]), cmap='afmhot',vmin=-1.5,vmax=1.5)
    f4 = ax[1].pcolormesh(x[2:125,64,2:125], z[2:125,64,2:125], np.log10(sigma[2:125,64,2:125]), cmap='bwr',vmin=-3,vmax=2)
    f6 = ax[2].pcolormesh(x[2:125,64,2:125], z[2:125,64,2:125], np.log10(cur_mag[2:125,64,2:125]), cmap='afmhot',vmin=-3,vmax=2)
[a],[b] = np.where(T[:,64,:]==np.max(T[:,64,:]))
ax[0].axvline(x=x[a,64,0], color='#008040', ls='--', linewidth=1)
ax[1].axvline(x=x[a,64,0], color='#008040', ls='--', linewidth=1)
ax[2].axvline(x=x[a,64,0], color='#008040', ls='--', linewidth=1)
fig.colorbar(f2, ax=ax[0], label=r'$\log_{10}(T)$')
ax[0].fill(xh,yh,'k')
ax[0].plot(xe,ye,'w')
ax[2].set_xlabel(r'$x$ ($r_G$)')
ax[0].set_ylabel(r'$z$ ($r_G$)')
fig.colorbar(f4, ax=ax[1], label=r'$\log_{10}(\sigma_{hot})$')
yt_extract_box(i_dump=1240, box_radius=15, mhd=True, gr=True, a=0.9375)
ax[1].streamplot(x[2:125,64,2:125].transpose(), z[2:125,64,2:125].transpose(), Bcc1[2:125,64,2:125].transpose(), Bcc3[2:125,64,2:125].transpose(), color='black', linewidth=0.5, density=5, arrowsize=0.5)
ax[1].fill(xh,yh,'k')
ax[1].plot(xe,ye,'w')
ax[1].set_ylabel(r'$z$ ($r_G$)')
fig.colorbar(f6, ax=ax[2], label=r'$\log_{10}|J|$')
ax[2].fill(xh,yh,'k')
ax[2].plot(xe,ye,'w')
ax[2].set_ylabel(r'$z$ ($r_G$)')
ax[0].set_title(r'$t = 12400M$' )
plt.savefig('plot_1.png')


#TOTAL FLUX TUBE PLOTS
fig, ax = plt.subplots(2,3,figsize=(12,6.75), dpi=400, constrained_layout = True, sharex=True,sharey=True)
for j in range(3):
    name = 1207 + j*14
    time = name * 10
    for i in range(3):
        yt_extract_box(i_dump=name, box_radius=15/2**i, mhd=True, gr=True, a=0.9375)
        gam = 5/3
        sigma = bsq / (rho + gam/(gam-1) * press)
        f1 = ax[0,j].pcolormesh(x[:,:,64], y[:,:,64], np.log10(rho[:,:,64]), cmap='afmhot',vmax=0.5,vmin=-1)
        f2 = ax[1,j].pcolormesh(x[:,:,64], y[:,:,64], np.log10(sigma[:,:,64]), cmap='bwr',vmin=-4,vmax=0.5)
    ax[0,j].set_title(r'$t = %i M$'%time )
    ax[0,j].fill(xh,yh,'k')
    ax[1,j].fill(xh,yh,'k')
    ax[1,j].set_xlabel(r'$x$ ($r_G$)')
fig.colorbar(f1, ax=ax[0,2], label=r'$\log_{10}(\rho)$')
fig.colorbar(f2, ax=ax[1,2], label=r'$\log_{10}(\sigma_{hot})$')
ax[0,0].set_ylabel(r'$y$ ($r_G$)')
ax[1,0].set_ylabel(r'$y$ ($r_G$)')
plt.savefig('flux_tubes.png')


#JET ANALYSIS
%run -i ./athena_script.py
xbox = [-25,-45,-78.125,-112.5]
h = []
rad = []
xcent = []
ycent = []
angle = []
for i in range(3, -1,-1):
    yt_extract_box(i_dump=1240, box_radius=50, mhd=True, gr=True, a=0.9375, center_x=xbox[i], center_y=25,center_z=-(100*i+50))
    gam = 5 / 3
    sigma = bsq / (rho + gam / (gam - 1) * press)
    for k in range(128):
        a,b = np.where(np.log10(sigma[:,:,k])>=0)
        m1 = a.min()
        m2 = a.max()
        n1 = b.min()
        n2 = b.max()
        numx = 0
        numy = 0
        denom = 0
    
        for i in range(m1,m2):
            for j in range(n1,n2):
                numx = numx + sigma[i,j,k] * i
                numy = numy + sigma[i,j,k] * j
                denom = denom + sigma[i,j,k]
    
        mc = int(np.round(numx/denom))
        nc = int(np.round(numy/denom))
        xc = x[mc,nc,k]
        yc = y[mc,nc,k]
        zc = z[0,0,k]
        rc = 100/128 * min(a.max() - mc, mc - a.min(), b.max() - nc, nc - b.min())
        h.append(zc)
        rad.append(rc)
        angle.append( np.arctan( np.sqrt(xc**2+yc**2) / (-zc)) )
        xcent.append(xc)
        ycent.append(yc)

data = np.array([h,xcent,ycent,r,angle])

plt.figure(dpi=200)
plt.plot(t, model(t, opt[0], opt[1]), color="red", label=r"$2.75|z|^{-0.52}$")
plt.plot(-np.array(h), -np.array(rad)/np.array(h), color="black", label="Radius")
plt.legend()
plt.xlabel(r"$|z|(r_g)$")
plt.ylabel(r"$\frac{R_J}{|z|}$")
plt.tight_layout()
plt.savefig("Rj_z_fit.png")


#TOROIDAL/POLOIDAL FIELDS
yt_extract_box(i_dump=1240, box_radius=400, mhd=True, gr=True, a=0.9375)
Bz = bu[3,:,:,:]
get_mdot(mhd=True, gr=True, ax=0, ay=0, az=0.9375)
Bphi = bu[3,:,:,:]
Bp = np.sqrt(bu[1,:,:,:]**2 + bu[2,:,:,:]**2)
rat = np.abs(Bphi/Bp)

i=16
hmin = np.abs(h - z[0,0,i])
[n] = np.where(hmin == np.min(hmin))
R = np.mean(np.array(rad)[n])
xc = np.mean(np.array(xcent)[n])
yc = np.mean(np.array(ycent)[n])
lim = 2*np.pi*R/np.abs(z[0,0,i])
th = np.linspace(0,2*np.pi,1000)
xr = xc+R*np.sin(th)
yr = yc+R*np.cos(th)
a, b = np.where(rat[:,:,i] < lim)

fig,ax = plt.subplots(dpi=400,constrained_layout = True)
f = ax.pcolormesh(x[:,:, i], y[:,:, i], np.log10(ratio[:,:, i]), cmap="plasma")
fig.colorbar(f, ax=ax, label=r"$log_{10}(|\frac{B_\phi}{B_p}|)$")
ax.scatter(x[a, b, i],y[a, b, i],color='white',marker='s')
ax.scatter(xc,yc,color='black',marker='x',s=4)
ax.plot(xr,yr,color='black',linewidth=2)
ax.set_xlabel(r"$x$ ($r_G$)")
ax.set_ylabel(r"$y$ ($r_G$)")
ax.set_title(r"$t = 12400M$")
#props = dict(boxstyle='round', facecolor='white', alpha=0.75)
#ax.text(0.04, 0.92, r'$z\approx%i r_g$'%z[0,0,i], transform=ax.transAxes, fontsize=14, bbox=props, verticalalignment='center')
plt.savefig("ratio.png")


#INSTABILITY PLOT
yt_extract_box(i_dump=1240, box_radius=400, mhd=True, gr=True, a=0.9375)
Bz = bu[3,:,:,:]
get_mdot(mhd=True, gr=True, ax=0, ay=0, az=0.9375)
Bphi = bu[3,:,:,:]
Bp = np.sqrt(bu[1,:,:,:]**2 + bu[2,:,:,:]**2)
rat = np.abs(Bphi/Bp)
th = np.linspace(0,2*np.pi,1000)
fig,ax = plt.subplots(2,3,figsize=(12,6.75), dpi=400, constrained_layout = True, sharex=True,sharey=True)
for k in range(3):
    i = [16,32,48][k]
    zval = [-296.875,-196.875,-96.875][k]
    hmin = np.abs(np.array(h) - zval)
    [n] = np.where(hmin == np.min(hmin))
    R = np.mean(np.array(rad)[n])
    xc = np.mean(np.array(xcent)[n])
    yc = np.mean(np.array(ycent)[n])
    lim = 2*np.pi*R/np.abs(zval)
    xr = xc+R*np.sin(th)
    yr = yc+R*np.cos(th)
    a, b = np.where(rat[:,:,i] < lim)
    f = ax[0,k].pcolormesh(x[:,:, i], y[:,:, i], np.log10(ratio[:,:, i]), cmap="plasma",vmin=-5,vmax=2)
    g = ax[1,k].pcolormesh(x[:,:, i], y[:,:, i], Bz[:,:, i], cmap="bwr",vmin=-0.07,vmax=0.01)
    ax[0,k].set_xlim(-200,200)
    ax[0,k].set_ylim(-200,200)
    ax[1,k].set_xlim(-200,200)
    ax[1,k].set_ylim(-200,200)
    ax[0,k].scatter(x[a, b, i],y[a, b, i],color='black',marker='s',s=6)
    ax[0,k].scatter(xc,yc,color='white',marker='x',s=4)
    ax[0,k].plot(xr,yr,color='white',linewidth=2)
    ax[1,k].scatter(xc,yc,color='black',marker='x',s=4)
    ax[1,k].plot(xr,yr,color='black',linewidth=2)
    ax[1,k].set_xlabel(r"$x$ ($r_G$)")
    ax[0,k].set_title(r"$z \approx %i r_g$"%zval)
fig.colorbar(f, ax=ax[0,2], label=r"$log_{10}(|\frac{B_\phi}{B_p}|)$")
fig.colorbar(g, ax=ax[1,2], label=r"$B_z$")
ax[0,0].set_ylabel(r"$y$ ($r_G$)")
ax[1,0].set_ylabel(r"$y$ ($r_G$)")
plt.savefig('instab.png')








#=======================================================================================================================================================================================
#=======================================================================================================================================================================================








#Flux Effieciency Plot
mad = np.where(Phibh[:,ir2]/np.sqrt(np.abs(mdot[:,ir2])) > 50)[0]
breaks = np.diff(mad) != 1
labelled = np.cumsum(np.concatenate(([0], breaks)))
fig,ax = plt.subplots(3,figsize=(20,8), constrained_layout = True, sharex=True)
for i in range(labelled.max()+1):
    if mad[labelled == i].size == 1:
        ax[0].axvline(t[mad[labelled == i][0]], color='green', alpha=0.25)
        ax[1].axvline(t[mad[labelled == i][0]], color='green', alpha=0.25)
        ax[2].axvline(t[mad[labelled == i][0]], color='green', alpha=0.25)
    else:
        ax[0].axvspan(t[mad[labelled == i][0]], t[mad[labelled == i][-1]], color='green', alpha=0.25)
        ax[1].axvspan(t[mad[labelled == i][0]], t[mad[labelled == i][-1]], color='green', alpha=0.25)
        ax[2].axvspan(t[mad[labelled == i][0]], t[mad[labelled == i][-1]], color='green', alpha=0.25)       
ax[0].plot(t,Phibh[:,ir2], color='black')
ax[0].set_ylabel(r'$\Phi_\text{BH}$')
ax[1].plot(t,Phibh[:,ir2]/np.sqrt(np.abs(mdot[:,ir2])), color='black')
ax[1].set_ylabel(r'$\phi_\text{BH}$')
ax[2].plot(t,100*Edot[:,ir5]/np.abs(mdot[:,ir2]), color='black')
ax[2].set_yscale('log')
ax[2].set_ylabel(r'$\eta$ [%]')
ax[2].set_xlabel('Time t (in M)')
fig.suptitle(r'Standard MAD')
# ax[0].axvline(t[2651], color='black', ls='--', alpha=0.4)
# ax[1].axvline(t[2651], color='black', ls='--', alpha=0.4)
# ax[2].axvline(t[2651], color='black', ls='--', alpha=0.4)
# ax[1].axhline(np.mean(Phibh[2651:,ir2]/np.sqrt(np.abs(mdot[2651:,ir2]))), color='black', ls='--', alpha=0.4)
# ax[0].scatter(t[2651:][peaks29], Phibh[2651:,ir2][peaks29], color='blue', marker='x')
# ax[0].scatter(t[2651:][troughs29], Phibh[2651:,ir2][troughs29], color='red', marker='x')
plt.savefig('plots_mayank/test.png')


#Radial Magnetic Field Polar Plot Video
r_eq = np.sqrt(zoom_x**2+xzoom_y**2)
B_r = (zoom_x*zoom_bx+zoom_y*zoom_by)/r_eq
r_val=np.arange(1.5,7.5,0.05)
for i in range(r_val.size):
    print(r'Plotting %0.2f'%r_val[i])
    br_p, coords, theta_p, z_p = polar(B_r, r_val[i], zoom_z[0,0,:], 7.5)
    clf()
    plt.figure(figsize=(12,5))
    fig, ax = plt.subplots(2, figsize=(8,5), sharex=True)
    f = ax[0].pcolormesh(180*theta_p/np.pi, z_p, br_p, cmap='seismic', vmin=-0.25, vmax=0.25)
    plt.colorbar(label=r'$v_r$')
    plt.xlabel(r'$\theta$ (in deg)')
    plt.ylabel(r'z (in $r_G$)')
    plt.title(r'$r0=%0.2fr_G$'%r_val[i])
    plt.savefig('plots_mayank/polar_%03d.png'%i)[l]




%run -i ./athena_script.py
a = 0.9375
rplus = 1.+ np.sqrt(1.-a**2)
th = np.linspace(0.,2*np.pi,1000)
xh = rplus*np.sin(th)
yh = rplus*np.cos(th)
rerg = 1.+np.sqrt(1.-a**2*np.cos(th)**2)
xe = rerg*np.sin(th)
ye = rerg*np.cos(th)
rd_1d_avg()
ir2 = r_to_ir(2)
ir5 = r_to_ir(5)

def delete():
    !rm -rf ./plots_mayank/perpendicular_*.png
    !rm -rf ./plots_mayank/profiles_*.png
    !rm -rf /home/mshenoy/perpendicular_*.png
    !rm -rf /home/mshenoy/profiles_*.png

i=1121
yt_extract_box(i_dump=i+1, box_radius=7.5, mhd=True, gr=True, a=0.9375)
far_post = F(g, Lower(uu,g), bd)
yt_extract_box(i_dump=i, box_radius=7.5, mhd=True, gr=True, a=0.9375)
far = F(g, Lower(uu,g), bd)
cur, cur_mag = J(g,gi,far,far_post,x,y,z)
gamma=5/3
sigma_max=100
beta_min=1e-3
set_constants()
get_Te_Tg(ke_ent2,rho,press,gr=True,mue=2.0,mu_tot = None)
uu_ks = cks_vec_to_ks(uu,x,y,z,0,0,a)
bu_ks = cks_vec_to_ks(bu,x,y,z,0,0,a)
Bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
j=180
rho_slice, coords, r_slice, z_slice = slice(rho, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
press_slice, coords, r_slice, z_slice = slice(press, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
bphi_slice, coords, r_slice, z_slice = slice(Bphi, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
temp_e_slice, coords, r_slice, z_slice = slice(1836*Te, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
cur_slice, coords, r_slice, z_slice = slice(cur_mag, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
bux_slice, coords, r_slice, z_slice = slice(bu[1,:,:,:], np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
buy_slice, coords, r_slice, z_slice = slice(bu[2,:,:,:], np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
buz_slice, coords, r_slice, z_slice = slice(bu[3,:,:,:], np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
bsq_slice, coords, r_slice, z_slice = slice(bsq, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
bu1_slice, bu2_slice = project(bu[1,:,:,:], bu[2,:,:,:], bu[3,:,:,:], coords, np.cos(np.pi*j/180), np.sin(np.pi*j/180))
sigma_slice = bsq_slice / (rho_slice + gamma/(gamma-1) * press_slice)
temp_slice = press_slice/rho_slice
beta_slice = 2*press_slice/bsq_slice
ang = np.arctan2(bu2_slice,bu1_slice)
da_x, da_y = np.gradient(ang)
a_gradmag = np.sqrt(da_x**2+da_y**2)
rr = np.sqrt(r_slice**2 + z_slice**2)
maskbh = rr < rplus
mask = cleanup(sigma_slice, a_gradmag, rr, rplus)
sk_r, sk_z, l = sheet(mask, sigma_slice, r_slice, z_slice)
dr, dz = tangent(sk_r, sk_z, 2)

for i in range(sk_r.size):
    print('Plotting %i of %i'%(i,sk_r.size))
    sigma_prof, r_l, z_l, d = profile(sigma_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    clf()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    f=ax.pcolormesh(r_slice, z_slice, np.log10(sigma_slice), cmap='plasma', vmin=-2, vmax=1)
    fig.colorbar(f, ax=ax, label=r'$\log_{10}(\sigma)$')
    ax.scatter(sk_r, sk_z, color='black', s=5)
    ax.plot(r_l, z_l, color='black', linewidth=1)
    sigma_slice2 = sigma_slice.copy()
    temp_slice2 = temp_slice.copy()
    sigma_slice2[maskbh] = sigma_slice.max()
    temp_slice2[maskbh] = temp_slice.min()
    s=ax.contour(r_slice, z_slice, sigma_slice2, levels=[10], colors='cyan', linewidths=1)
    ax.clabel(s, [10], fontsize=10)
    ax.set_xlabel(r'$r$ (in $r_G$)')
    ax.set_ylabel(r'$z$ (in $r_G$)')
    ax.set_xlim(0,7.5)
    ax.set_ylim(-3.75,3.75)
    ax.fill(xh,yh,'k')
    fig.legend()
    fig.suptitle(r'$l=%0.2f r_G$'%(l[i]+rplus))
    plt.savefig('plots_mayank/perpendicular_%03d.png'%i)

for i in range(sk_r.size):
    print('Plotting %i of %i'%(i,sk_r.size))
    temp_prof, r_l, z_l, d = profile(temp_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    tempe_prof, r_l, z_l, d = profile(temp_e_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    sigma_prof, r_l, z_l, d = profile(sigma_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    cur_prof, r_l, z_l, d = profile(cur_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    beta_prof, r_l, z_l, d = profile(beta_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    bx_prof, r_l, z_l, d = profile(bux_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    by_prof, r_l, z_l, d = profile(buy_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    bz_prof, r_l, z_l, d = profile(buz_slice, dr[i], dz[i], 1, sk_r[i], sk_z[i], r_slice[:,0], z_slice[0,:])
    clf()
    fig, ax = plt.subplots(1,2, figsize=(12,5), tight_layout=True)
    ax[0].plot(d, temp_prof, label=r'T')
    ax[0].plot(d, tempe_prof, label=r'$T_e$')
    ax[0].plot(d, sigma_prof, label=r'$\sigma$')
    ax[0].plot(d, cur_prof, label=r'|J|')
    ax[0].plot(d, beta_prof, label=r'$\beta$')
    ax[0].axvline(0, color='black', ls='--', alpha=0.5)
    ax[0].axhline(sigma_max*beta_min/2, color='black', ls='dotted', alpha=0.5, label=r'$\sigma_\text{flr}\beta_\text{flr}/2$')
    ax[0].set_xlabel(r'Perpendicular Distance (in $r_G$)')
    ax[0].set_ylabel(r'Values')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[1].plot(d, bx_prof, label=r'$B_x$')
    ax[1].plot(d, by_prof, label=r'$B_y$')
    ax[1].plot(d, bz_prof, label=r'$B_z$')
    ax[1].axvline(0, color='black', ls='--', alpha=0.5)
    ax[1].axhline(0, color='black', ls='--', alpha=0.5)
    ax[1].set_xlabel(r'Perpendicular Distance (in $r_G$)')
    ax[1].set_ylabel(r'Field Values')
    ax[1].legend()
    fig.suptitle(r'$l=%0.2f r_G$'%(l[i]+rplus))
    plt.savefig('plots_mayank/profiles_%03d.png'%i)

mask_s = sigma_slice<10
sk = skeletonize(mask_s).astype(bool)
mp_r, mp_z = r_slice[sk], z_slice[sk]
theta = np.arctan2(mp_z, mp_r)
m = np.mean(theta[:len(theta)//8])
rr_prof, r_l, z_l, d = profile(rr, np.cos(m+np.pi/2), np.sin(m+np.pi/2), 3, 3.5*np.cos(m), 3.5*np.sin(m), r_slice[:,0],z_slice[0,:])
bsq_prof, r_l, z_l, d = profile(bsq_slice, np.cos(m+np.pi/2), np.sin(m+np.pi/2), 3, 3.5*np.cos(m), 3.5*np.sin(m), r_slice[:,0],z_slice[0,:])
bsq_prof = bsq_prof/bsq_prof[0]
rho_prof, r_l, z_l, d = profile(rho_slice, np.cos(m+np.pi/2), np.sin(m+np.pi/2), 3, 3.5*np.cos(m), 3.5*np.sin(m), r_slice[:,0],z_slice[0,:])
rho_prof = rho_prof/rho_prof[0]
press_prof, r_l, z_l, d = profile(press_slice, np.cos(m+np.pi/2), np.sin(m+np.pi/2), 3, 3.5*np.cos(m), 3.5*np.sin(m), r_slice[:,0],z_slice[0,:])
press_prof = press_prof/press_prof[0]
clf()
plt.plot(rr_prof, bsq_prof, label=r'$B^2/B_0^2$')
plt.plot(rr_prof, rho_prof, label=r'$\rho^2/\rho_0^2$')
plt.plot(rr_prof, press_prof, label=r'$P^2/P_0^2$')
plt.plot(rr_prof, 2/rr_prof, color='black', ls='--', alpha=0.5, label=r'$\propto 1/r$')
plt.plot(rr_prof, 8/rr_prof**3, color='black', ls='-.', alpha=0.5, label=r'$\propto 1/r^3$')
plt.plot(rr_prof, 32/rr_prof**5, color='black', ls=':', alpha=0.5, label=r'$\propto 1/r^5$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Radius $r[r_G]$')
plt.ylabel(r'Dimensionless Values')
plt.legend()
plt.savefig('plots_mayank/parallel.png')


fig, ax = plt.subplots(2,2, figsize=(12,5), tight_layout=True, sharey=True, sharex=True)
ax[0,0].hist(data['nonmad']['beta2_a0']['dt'], bins=bins, color='black')
ax[0,0].set_title(r'$\beta=10^2,a=0$')
ax[0,1].hist(data['nonmad']['beta2_a9']['dt'], bins=bins, color='black')
ax[0,1].set_title(r'$\beta=10^2,a=0.9375$')
ax[1,0].hist(data['nonmad']['beta6_a0']['dt'], bins=bins, color='black')
ax[1,0].set_title(r'$\beta=10^6,a=0$')
ax[1,1].hist(data['nonmad']['beta6_a9']['dt'], bins=bins, color='black')
ax[1,1].set_title(r'$\beta=10^6,a=0.9375$')
for i in range(2):
    ax[i,0].set_ylabel('Number of Eruptions')
    ax[1,i].set_xlabel('Flare Durations (in M)')
    for j in range(2):
        ax[i,j].set_xscale('log')
fig.suptitle('Bipolar Flux Eruption Durations')
plt.savefig('plots_mayank/test.png')

fig, ax = plt.subplots(2, figsize=(7,5), tight_layout=True, sharex=True)
ax[0].hist(data['mad']['beta6_a0']['del_t'], bins=bins, color='black')
ax[0].set_title(r'$\beta=10^6,a=0$')
ax[1].hist(data['mad']['beta6_a9']['del_t'], bins=bins, color='black')
ax[1].set_title(r'$\beta=10^6,a=0.9375$')
ax[1].set_xlabel('Inter Flare Durations (in M)')
for i in range(2):
    ax[i].set_xscale('log')
    ax[i].set_ylabel('Number of Eruptions')
fig.suptitle('MAD Flux Eruption Intervals')
plt.savefig('plots_mayank/test.png')


from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

mad_delt = []
mad_dt = []
for run in data['mad'].keys():
    mad_delt.extend(data['mad'][run]['del_t'])
    mad_dt.extend(data['mad'][run]['dt'])

kde9t = gaussian_kde(data['mad']['beta6_a9']['dt'])
kde0t = gaussian_kde(data['mad']['beta6_a0']['dt'])
kde0T = gaussian_kde(data['mad']['beta6_a0']['del_t'])
kde9T = gaussian_kde(data['mad']['beta6_a9']['del_t'])
bins_t = np.linspace(min(mad_dt), max(mad_dt), 100)
bins_T = np.linspace(min(mad_delt), max(mad_delt), 100)

fig, ax = plt.subplots(2, figsize=(7,5), tight_layout=True, sharex=False)
ax[0].plot(bins_t, kde0t(bins_t), color='#0072B2', label=r'$a=0$')
ax[0].axvline(bins_t[np.where(kde0t(bins_t) == np.max(kde0t(bins_t)))], color='#0072B2', ls='--', alpha=0.5)
ax[0].plot(bins_t, kde9t(bins_t), color='#E69F00', label=r'$a=0.9375$')
ax[0].axvline(bins_t[np.where(kde9t(bins_t) == np.max(kde9t(bins_t)))], color='#E69F00', ls='--', alpha=0.5)
ax[0].set_title('MAD Flare Durations')
ax[0].set_xlabel('Flare Durations (in M)')
ax[0].set_ylabel('Probability Density')
ax[0].legend()
ax[1].plot(bins_T, kde0T(bins_T), color='#0072B2', label=r'$a=0$')
ax[1].axvline(bins_T[np.where(kde0T(bins_T) == np.max(kde0T(bins_T)))], color='#0072B2', ls='--', alpha=0.5)
ax[1].plot(bins_T, kde9T(bins_T), color='#E69F00', label=r'$a=0.9375$')
ax[1].axvline(bins_T[np.where(kde9T(bins_T) == np.max(kde9T(bins_T)))], color='#E69F00', ls='--', alpha=0.5)
ax[1].set_title('MAD Inter Flare Durations')
ax[1].set_xlabel('Inter Flare Durations (in M)')
ax[1].set_ylabel('Probability Density')
ax[1].legend()
plt.savefig('plots_mayank/test.png')

fig, ax = plt.subplots(2, figsize=(7,5), tight_layout=True, sharex=False)
ax[0].hist(mad_delt, bins=bins, color='black')
ax[0].set_title('MAD')
ax[1].hist(nonmad_delt, bins=bins, color='black')
ax[1].set_title('Bipolar')
ax[1].set_xlabel('Inter Flare Durations (in min)')
for i in range(2):
    # ax[i].set_xscale('log')
    ax[i].set_ylabel('Number of Eruptions')
fig.suptitle('MAD vs Bipolar Flux Eruption Intervals')
plt.savefig('plots_mayank/test.png')



gamma=5/3
sigma_max=100
beta_min=1e-3
uu_ks = cks_vec_to_ks(uu,x,y,z,0,0,a)
bu_ks = cks_vec_to_ks(bu,x,y,z,0,0,a)
Bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
beta = 2 * press / bsq
beta_slice = np.mean(beta[:,63:65,:], axis=1)
for j in range(360):
    print(r'Plotting %0.2f deg'%j)
    rho_slice, coords, r_slice, z_slice = slice(rho, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    press_slice, coords, r_slice, z_slice = slice(press, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bsq_slice, coords, r_slice, z_slice = slice(bsq, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bphi_slice, coords, r_slice, z_slice = slice(Bphi, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bu1_slice, bu2_slice = project(bu[1,:,:,:], bu[2,:,:,:], bu[3,:,:,:], coords, np.cos(np.pi*j/180), np.sin(np.pi*j/180))
    sigma_slice = bsq_slice / (rho_slice + gamma/(gamma-1) * press_slice)
    clf()
    fig = plt.figure(figsize=(13.65,6), tight_layout=True)
    gs1 = grd.GridSpec(2,3,height_ratios=[2,1])
    gs1.update(wspace=0.4, right=0.92, hspace=0.5, left=0.05)
    ax0 = fig.add_subplot(gs1[0,0])
    ax1 = fig.add_subplot(gs1[0,1])
    ax2 = fig.add_subplot(gs1[0,2])
    gs2 = grd.GridSpec(2,1,height_ratios=[2,1])
    ax3 = fig.add_subplot(gs2[1,0])
    ax0.set_aspect('equal', adjustable='box')
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax0.set_xlim(x.min(), x.max())
    ax0.set_ylim(z.min(), z.max())
    ax1.set_xlim(r_slice.min(), r_slice.max())
    ax1.set_ylim(z_slice.min(), z_slice.max())
    ax2.set_xlim(r_slice.min(), r_slice.max())
    ax2.set_ylim(z_slice.min(), z_slice.max())
    h = ax0.pcolormesh(x[:,64,:], z[:,64,:], np.log10(beta_slice), cmap='plasma', vmin=-1, vmax=1)
    fig.colorbar(h,ax=ax0,label=r'$\log_{10}(\beta)$',fraction=0.046, pad=0.04)
    ax0.streamplot(x[:,64,:].transpose(), z[:,64,:].transpose(), np.mean(bu[1,:,63:65,:], axis=1).transpose(), np.mean(bu[2,:,63:65,:], axis=1).transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
    ax0.fill(xh,yh,'k')
    ax0.set_xlabel(r'$x$ ($r_G$)')
    ax0.set_ylabel(r'$z$ ($r_G$)')
    f = ax1.pcolormesh(r_slice, z_slice, bphi_slice, cmap='bwr', vmin=-0.02, vmax=0.02)
    fig.colorbar(f,ax=ax1,label=r'$B_\phi$',fraction=0.046, pad=0.04)
    ax1.streamplot(r_slice.transpose(), z_slice.transpose(), bu1_slice.transpose(), bu2_slice.transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
    ax1.contour(r_slice, z_slice, sigma_slice<1, levels=[0.5], colors='lime', linewidths=1.5)
    ax1.fill(xh,yh,'k')
    ax1.set_xlabel(r'$r$ ($r_G$)')
    ax1.set_ylabel(r'$y$ ($r_G$)')
    g = ax2.pcolormesh(r_slice, z_slice, np.log10(sigma_slice), cmap='plasma', vmin=-2, vmax=1)
    fig.colorbar(g,ax=ax2, label=r'$\log_{10}(\sigma)$',fraction=0.046, pad=0.04)
    ax2.contour(r_slice, z_slice, sigma_slice<1, levels=[0.5], colors='black', linewidths=1.5)
    ax2.fill(xh,yh,'k')
    ax2.set_xlabel(r'$r$ ($r_G$)')
    ax2.set_ylabel(r'$y$ ($r_G$)')
    fig.suptitle('t=%iM'%t[i])
    ax3.plot(t[2931:2991], Phibh[2931:2991,ir2], color='black')
    ax3.axvline(t[i], color='black', ls='--')
    ax3.scatter(t[i], Phibh[i,ir2], color='red',alpha=0.5)
    ax3.set_xlabel('Time t (in M)')
    ax3.set_ylabel(r'$\Phi_\text{BH}$')
    fig.suptitle(r'Slice at %d$^\circ$'% j)
    plt.savefig('plots_mayank/slice_beta_phi_%03d_%03d.png'%(j,i))

   
a=0.9375
rplus = 1 + np.sqrt(1 - a**2) 
theta = np.linspace(0, 2*np.pi, 1000)
xh = rplus * np.sin(theta)
yh = rplus * np.cos(theta)  
d = []
i=1104
yt_extract_box(i_dump=i, box_radius=7.5, mhd=True, gr=True, a=a)
gamma = 5/3
uu_ks = cks_vec_to_ks(uu,x,y,z,0,0,a)
bu_ks = cks_vec_to_ks(bu,x,y,z,0,0,a)
br = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])
bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
bth = (bu_ks[2] * uu_ks[0] - bu_ks[0] * uu_ks[2])
vr = uu_ks[1]/uu_ks[0]
for j in range(360):
    print(r'Plotting %d deg'%j)
    br_slice, coords, r_slice, z_slice = slice(br, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bphi_slice, coords, r_slice, z_slice = slice(bphi, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bth_slice, coords, r_slice, z_slice = slice(bth, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    vr_slice, coords, r_slice, z_slice = slice(vr, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    b_slice, coords, r_slice, z_slice = slice(Bcc3**2/(Bcc1**2 + Bcc2**2), np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    rho_slice, coords, r_slice, z_slice = slice(rho, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    press_slice, coords, r_slice, z_slice = slice(press, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bsq_slice, coords, r_slice, z_slice = slice(bsq, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
    bu1_slice, bu2_slice = project(bu[1,:,:,:], bu[2,:,:,:], bu[3,:,:,:], coords, np.cos(np.pi*j/180), np.sin(np.pi*j/180))
    sigma_slice = bsq_slice / (rho_slice + gamma/(gamma-1) * press_slice)
    ang = np.arctan2(bu2_slice,bu1_slice)
    da_x, da_y = np.gradient(ang)
    a_gradmag = np.sqrt(da_x**2+da_y**2)
    b_slice = np.sign(bphi_slice) * np.sign(bth_slice) * np.sign(br_slice)
    clf()
    plt.figure()
    plt.pcolormesh(r_slice, z_slice, b_slice, cmap='bwr', vmin=-3, vmax=3)
    plt.colorbar(label=r'$B^r$')
    plt.streamplot(r_slice.transpose(), z_slice.transpose(), bu1_slice.transpose(), bu2_slice.transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
    plt.fill(xh,yh,'k')
    sigma_slice2 = np.copy(sigma_slice)
    maskbh = (r_slice**2 + z_slice**2) < rplus**2
    sigma_slice2[maskbh] = sigma_slice2.max()
    o = plt.contour(r_slice, z_slice, sigma_slice2, levels=[1], colors='cyan', linewidths=1)
    plt.clabel(o, [1], fmt={1: 'σ=1'}, inline=True, fontsize=8)
    t = plt.contour(r_slice, z_slice, sigma_slice2, levels=[10], colors='green', linewidths=1)
    plt.clabel(t, [10], fmt={10: 'σ=10'}, inline=True, fontsize=8)
    plt.xlim(0,7.5)
    plt.xlabel(r'$x/r_G$')
    plt.ylabel(r'$z/r_G$')
    plt.title(r'$\phi=%d^\circ$'% j)
    plt.savefig('plots_mayank/poloidal_%03d.png'% j)
    # diff = np.sum(sigma_slice2<=10) - np.sum(sigma_slice2<=1)
    # d.append(diff)
    





##########################################################
################# PAPER 4 PANEL PLOT######################
##########################################################
a=0
rplus = 1 + np.sqrt(1 - a**2)
theta = np.linspace(0, 2*np.pi, 1000)
xh = rplus * np.sin(theta)
yh = rplus * np.cos(theta)
i=2952
# for i in range(1104,1146):
#     print('Dump %d'%i)
yt_extract_box(i_dump=i, box_radius=10, mhd=True, gr=True, a=a)
uu_ks = cks_vec_to_ks(uu,x,y,z,0,0,a)
bu_ks = cks_vec_to_ks(bu,x,y,z,0,0,a)
gamma=5/3
# sigma = bsq / (rho + gamma/(gamma-1) * press)
# temp = press / rho
# sigma_slice = np.mean(sigma[:,63:65,:], axis=1)
# t_slice = np.mean(temp[:,63:65,:], axis=1)
# bz_slice = np.mean((Bcc3**2/(Bcc1**2 + Bcc2**2))[:,63:65,:], axis=1)
# vr_slice = np.mean((uu_ks[1]/uu_ks[0])[:,63:65,:], axis=1)
# r_slice = x[:,64,:]
# z_slice = z[:,64,:]
# bu1_slice = np.mean(bu[1,:,63:65,:], axis=1)
# bu2_slice = np.mean(bu[3,:,63:65,:], axis=1)
for j in range(360):
    print(r'Plotting %d deg'%j)
    rho_slice, coords, r_slice, z_slice = slice(rho, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 10)
    press_slice = slice(press, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 10)[0]
    bsq_slice = slice(bsq, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 10)[0]
    bu1_slice, bu2_slice = project(bu[1,:,:,:], bu[2,:,:,:], bu[3,:,:,:], coords, np.cos(np.pi*j/180), np.sin(np.pi*j/180))
    sigma_slice = bsq_slice / (rho_slice + gamma/(gamma-1) * press_slice)
    bz_slice = slice(Bcc3**2/(Bcc1**2 + Bcc2**2), np.cos(np.pi*j/180), np.sin(np.pi*j/180), 10)[0]
    br_slice = slice((bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1]), np.cos(np.pi*j/180), np.sin(np.pi*j/180), 10)[0]
    vr_slice = slice(uu_ks[1]/uu_ks[0], np.cos(np.pi*j/180), np.sin(np.pi*j/180), 10)[0]
    clf()
    fig, ax = plt.subplots(2,2, figsize=(8,6), tight_layout=True)
    f = ax[0,0].pcolormesh(r_slice, z_slice, np.log10(sigma_slice), cmap='plasma', vmin=-1, vmax=2)
    fig.colorbar(f,ax=ax[0,0],label=r'$\log_{10}(\sigma)$')
    maskbh = (r_slice**2 + z_slice**2) < rplus**2
    sigma_slice2 = np.copy(sigma_slice)
    sigma_slice2[maskbh] = sigma_slice2.max()
    ax[0,0].contour(r_slice, z_slice, sigma_slice2, levels=[1], colors='cyan', linewidths=1.5)
    ax[0,0].contour(r_slice, z_slice, sigma_slice2, levels=[10], colors='green', linewidths=1.5)
    g = ax[0,1].pcolormesh(r_slice, z_slice, br_slice, cmap='PRGn', vmin=-0.2, vmax=0.2)
    fig.colorbar(g,ax=ax[0,1],label=r'$B^r$')
    h = ax[1,0].pcolormesh(r_slice, z_slice, np.log10(bz_slice), cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(h,ax=ax[1,0],label=r'$\log_{10}(\frac{B_z^2}{B_x^2+B_y^2})$')
    k = ax[1,1].pcolormesh(r_slice, z_slice, vr_slice, cmap='PuOr', vmin=-0.2, vmax=0.2)
    fig.colorbar(k,ax=ax[1,1],label=r'$v^r$')
    fig.suptitle(r'$\phi=%d^\circ$'% j)
    # fig.suptitle(r'$t=%dM$'% i)
    for l in range(2):
        for m in range(2):
            ax[l,m].set_xlabel(r'$x/r_G$')
            ax[l,m].set_ylabel(r'$z/r_G$')
            ax[l,m].set_aspect('equal', adjustable='box')
            ax[l,m].fill(xh,yh,'k')
            ax[l,m].set_xlim(r_slice.min(), r_slice.max())
            ax[l,m].set_ylim(z_slice.min(), z_slice.max())
            ax[l,m].streamplot(r_slice.transpose(), z_slice.transpose(), bu1_slice.transpose(), bu2_slice.transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
    plt.savefig('plots_mayank/poloidal_%03d.png'% j)  
