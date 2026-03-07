"""
Read Athena++ output data files.
"""

# Python modules

import numpy as np
from numpy import *
import glob
import os
import sys
import matplotlib.gridspec as grd
from scipy.ndimage import zoom, map_coordinates, label, find_objects
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage.morphology import skeletonize
import networkx as netx
from skan import Skeleton, summarize
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.stats import gaussian_kde

#=========================================================================================


def tab(filename, headings=None, dimensions=1):
  """Read .tab files and return dict or array."""

  # Check for valid number of dimensions
  if dimensions != 1 and dimensions !=2 and dimensions != 3:
    raise AthenaError('Improper number of dimensions')

  # Read raw data
  with open(filename, 'r') as data_file:
    raw_data = data_file.readlines()

  # Organize data into array of numbers
  data_array = []
  first_line = True
  last_line_number = len(raw_data)
  line_number = 0
  for line in raw_data:
    line_number += 1
    if line.split()[0][0] == '#':  # comment line
      continue
    row = []
    col = 0
    for val in line.split():
      col += 1
      if col == 1:
        if first_line:
          i_min = int(val)
        if line_number == last_line_number:
          i_max = int(val)
      elif col == 3 and dimensions >= 2:
        if first_line:
          j_min = int(val)
        if line_number == last_line_number:
          j_max = int(val)
      elif col == 5 and dimensions == 3:
        if first_line:
          k_min = int(val)
        if line_number == last_line_number:
          j_max = int(val)
      else:
        row.append(float(val))
    first_line = False
    data_array.append(row)

  # Reshape array based on number of dimensions
  if dimensions == 1:
    j_min = j_max = 0
  if dimensions <= 2:
    k_min = k_max = 0
  array_shape = (k_max-k_min+1,j_max-j_min+1,i_max-i_min+1,len(row))
  data_array = np.reshape(data_array, array_shape)

  # Store separate variables as dictionary entries if headings given
  if headings is not None:
    data_dict = {}
    for n in range(len(headings)):
      data_dict[headings[n]] = data_array[:,:,:,n]
    return data_dict
  else:
    return data_array

#=========================================================================================

def vtk(filename):
  """Read .vtk files and return dict of arrays of data."""

  # Python module
  import struct

  # Read raw data
  with open(filename, 'r') as data_file:
    raw_data = data_file.read()

  # Skip header
  current_index = 0
  current_char = raw_data[current_index]
  while current_char == '#':
    while current_char != '\n':
      current_index += 1
      current_char = raw_data[current_index]
    current_index += 1
    current_char = raw_data[current_index]

  # Function for skipping though the file
  def skip_string(expected_string):
    expected_string_len = len(expected_string)
    if raw_data[current_index:current_index+expected_string_len] != expected_string:
      raise AthenaError('File not formatted as expected')
    return current_index+expected_string_len

  # Read metadata
  current_index = skip_string('BINARY\nDATASET RECTILINEAR_GRID\nDIMENSIONS ')
  end_of_line_index = current_index + 1
  while raw_data[end_of_line_index] != '\n':
    end_of_line_index += 1
  face_dimensions = map(int, raw_data[current_index:end_of_line_index].split(' '))
  current_index = end_of_line_index + 1

  # Function for reading interface locations
  def read_faces(letter, num_faces):
    identifier_string = '{0}_COORDINATES {1} float\n'.format(letter,num_faces)
    begin_index = skip_string(identifier_string)
    format_string = '>' + 'f'*num_faces
    end_index = begin_index + 4*num_faces
    vals = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    return vals,end_index+1

  # Read interface locations
  x_faces,current_index = read_faces('X', face_dimensions[0])
  y_faces,current_index = read_faces('Y', face_dimensions[1])
  z_faces,current_index = read_faces('Z', face_dimensions[2])

  # Prepare to read quantities defined on grid
  cell_dimensions = np.array([max(dim-1,1)
      for dim in face_dimensions])
  num_cells = cell_dimensions.prod()
  current_index = skip_string('CELL_DATA {0}\n'.format(num_cells))
  if raw_data[current_index:current_index+1] == '\n':
    current_index = skip_string('\n')  # extra newline inserted by join script
  data = {}

  # Function for reading scalar data
  def read_cell_scalars():
    begin_index = skip_string('SCALARS ')
    end_of_word_index = begin_index + 1
    while raw_data[end_of_word_index] != ' ':
      end_of_word_index += 1
    array_name = raw_data[begin_index:end_of_word_index]
    string_to_skip = 'SCALARS {0} float\nLOOKUP_TABLE default\n'.format(array_name)
    begin_index = skip_string(string_to_skip)
    format_string = '>' + 'f'*num_cells
    end_index = begin_index + 4*num_cells
    data[array_name] = struct.unpack(format_string, raw_data[begin_index:end_index])
    dimensions = tuple(cell_dimensions[::-1])
    data[array_name] = np.array(data[array_name]).reshape(dimensions)
    return end_index+1

  # Function for reading vector data
  def read_cell_vectors():
    begin_index = skip_string('VECTORS ')
    end_of_word_index = begin_index + 1
    while raw_data[end_of_word_index] != '\n':
      end_of_word_index += 1
    array_name = raw_data[begin_index:end_of_word_index]
    string_to_skip = 'VECTORS {0}\n'.format(array_name)
    array_name = array_name[:-6]  # remove ' float'
    begin_index = skip_string(string_to_skip)
    format_string = '>' + 'f'*num_cells*3
    end_index = begin_index + 4*num_cells*3
    data[array_name] = struct.unpack(format_string, raw_data[begin_index:end_index])
    dimensions = tuple(np.append(cell_dimensions[::-1],3))
    data[array_name] = np.array(data[array_name]).reshape(dimensions)
    return end_index+1

  # Read quantities defined on grid
  while current_index < len(raw_data):
    expected_string = 'SCALARS'
    expected_string_len = len(expected_string)
    if raw_data[current_index:current_index+expected_string_len] == expected_string:
      current_index = read_cell_scalars()
      continue
    expected_string = 'VECTORS'
    expected_string_len = len(expected_string)
    if raw_data[current_index:current_index+expected_string_len] == expected_string:
      current_index = read_cell_vectors()
      continue
    raise AthenaError('File not formatted as expected')
  return x_faces,y_faces,z_faces,data

#=========================================================================================

def athdf(filename, data=None, quantities=None, dtype=np.float32, level=None,
    subsample=False, fast_restrict=False, x1_min=None, x1_max=None, x2_min=None,
    x2_max=None, x3_min=None, x3_max=None, vol_func=None, vol_params=None,
    face_func_1=None, face_func_2=None, face_func_3=None,center_func_1=None, center_func_2=None, center_func_3=None):
  """Read .athdf files and populate dict of arrays of data."""

  # Python modules
  import sys
  import warnings
  import h5py
  global t 

  # Open file
  with h5py.File(filename, 'r') as f:

    # Extract size information
    t = f.attrs['Time']
    max_level = f.attrs['MaxLevel']
    if level is None:
      level = max_level
    block_size = f.attrs['MeshBlockSize']
    root_grid_size = f.attrs['RootGridSize']
    levels = f['Levels'][:]
    logical_locations = f['LogicalLocations'][:]
    nx_vals = []
    for d in range(3):
      if block_size[d] == 1 and root_grid_size[d] > 1:  # sum or slice
        other_locations = [location for location in \
            zip(levels, logical_locations[:,(d+1)%3], logical_locations[:,(d+2)%3])]
        if len(set(other_locations)) == len(other_locations):  # effective slice
          nx_vals.append(1)
        else:  # nontrivial sum
          nx_vals.append(2**level)
      elif block_size[d] == 1:  # singleton dimension
        nx_vals.append(1)
      else:  # normal case
        nx_vals.append(root_grid_size[d] * 2**level)
    nx1 = nx_vals[0]
    nx2 = nx_vals[1]
    nx3 = nx_vals[2]
    lx1 = nx1 // block_size[0]
    lx2 = nx2 // block_size[1]
    lx3 = nx3 // block_size[2]
    num_extended_dims = 0
    for nx in nx_vals:
      if nx > 1:
        num_extended_dims += 1

    # Set volume function for preset coordinates if needed
    coord = f.attrs['Coordinates']
    if level < max_level and not subsample and not fast_restrict and vol_func is None:
      x1_rat = f.attrs['RootGridX1'][2]
      x2_rat = f.attrs['RootGridX2'][2]
      x3_rat = f.attrs['RootGridX3'][2]
      if coord == b'cartesian' or coord == b'minkowski' or coord == b'tilted' \
          or coord == b'sinusoidal':
        if (nx1 == 1 or x1_rat == 1.0) and (nx2 == 1 or x2_rat == 1.0) and \
            (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda xm,xp,ym,yp,zm,zp: (xp-xm) * (yp-ym) * (zp-zm)
      elif coord == b'cylindrical':
        if nx1 == 1 and (nx2 == 1 or x2_rat == 1.0) and (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda rm,rp,phim,phip,zm,zp: (rp**2-rm**2) * (phip-phim) * (zp-zm)
      elif coord == b'spherical_polar' or coord == b'schwarzschild':
        if nx1 == 1 and nx2 == 1 and (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda rm,rp,thetam,thetap,phim,phip: \
              (rp**3-rm**3) * abs(np.cos(thetam)-np.cos(thetap)) * (phip-phim)
      elif coord == b'kerr-schild':
        if nx1 == 1 and nx2 == 1 and (nx3 == 3 or x3_rat == 1.0):
          fast_restrict = True
        else:
          a = vol_params[0]
          def vol_func(rm, rp, thetam, thetap, phim, phip):
            cosm = np.cos(thetam)
            cosp = np.cos(thetap)
            return \
                ((rp**3-rm**3) * abs(cosm-cosp) + a**2 * (rp-rm) * abs(cosm**3-cosp**3)) \
                * (phip-phim)
      elif coord ==b'gr_user':
        if (nx1 == 1 or x1_rat == 1.0) and (nx2 == 1 or x2_rat == 1.0) and \
            (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda xm,xp,ym,yp,zm,zp: (xp-xm) * (yp-ym) * (zp-zm)
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)

    # Set cell center functions for preset coordinates
    if center_func_1 is None:
      if coord == b'cartesian' or coord == b'minkowski' or coord == b'tilted' \
          or coord == b'sinusoidal' or coord == b'kerr-schild':
        center_func_1 = lambda xm,xp : 0.5*(xm+xp)
      elif coord == b'cylindrical':
        center_func_1 = lambda xm,xp : 2.0/3.0 * (xp**3-xm**3) / (xp**2-xm**2)
      elif coord == b'spherical_polar':
        center_func_1 = lambda xm,xp : 3.0/4.0 * (xp**4-xm**4) / (xp**3-xm**3)
      elif coord == b'schwarzschild':
        center_func_1 = lambda xm,xp : (0.5*(xm**3+xp**3)) ** 1.0/3.0
      elif coord ==b'gr_user':
        center_func_1 = lambda xm,xp : 0.5*(xm+xp)
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)
    if center_func_2 is None:
      if coord == b'cartesian' or coord == b'cylindrical' or coord == b'minkowski' \
          or coord == b'tilted' or coord == b'sinusoidal' or coord == b'kerr-schild' or coord==b'gr_user':
        center_func_2 = lambda xm,xp : 0.5*(xm+xp)
      elif coord == b'spherical_polar':
        def center_func_2(xm, xp):
          sm = np.sin(xm)
          cm = np.cos(xm)
          sp = np.sin(xp)
          cp = np.cos(xp)
          return (sp-xp*cp - sm+xm*cm) / (cm-cp)
      elif coord == b'schwarzschild':
        center_func_2 = lambda xm,xp : np.arccos(0.5*(np.cos(xm)+np.cos(xp)))
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)
    if center_func_3 is None:
      if coord == b'cartesian' or coord == b'cylindrical' or coord == b'spherical_polar' \
          or coord == b'minkowski' or coord == b'tilted' or coord == b'sinusoidal' \
          or coord == b'schwarzschild' or coord == b'kerr-schild' or coord==b'gr_user':
        center_func_3 = lambda xm,xp : 0.5*(xm+xp)
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)

    # Check output level compared to max level in file
    if level < max_level and not subsample and not fast_restrict:
      warnings.warn('Exact restriction being used: performance severely affected; see' \
          + ' documentation', AthenaWarning)
      sys.stderr.flush()
    if level > max_level:
      warnings.warn('Requested refinement level higher than maximum level in file: all' \
          + ' cells will be prolongated', AthenaWarning)
      sys.stderr.flush()

    # Check that subsampling and/or fast restriction will work if needed
    if level < max_level and (subsample or fast_restrict):
      max_restrict_factor = 2**(max_level-level)
      for current_block_size in block_size:
        if current_block_size != 1 and current_block_size%max_restrict_factor != 0:
          raise AthenaError('Block boundaries at finest level must be cell boundaries' \
              + ' at desired level for subsampling or fast restriction to work')

    # Create list of all quantities if none given
    file_quantities = f.attrs['VariableNames'][:]
    coord_quantities = ('x1f', 'x2f', 'x3f', 'x1v', 'x2v', 'x3v')
    if data is not None:
      quantities = data.values()
    elif quantities is None:
      quantities = file_quantities
    else:
      for q in quantities:
        if q not in file_quantities and q not in coord_quantities:
          possibilities = '", "'.join(file_quantities)
          possibilities = '"' + possibilities + '"'
          error_string = 'Quantity not recognized: file does not include "{0}" but does' \
              + ' include {1}'
          raise AthenaError(error_string.format(q, possibilities))
    quantities = [str(q.decode('utf-8')) for q in quantities if q not in coord_quantities]

    # Get metadata describing file layout
    num_blocks = f.attrs['NumMeshBlocks']
    #dataset_names = f.attrs['DatasetNames'][:]
    dataset_names = []
    for dset_name in f.attrs['DatasetNames'][:]:
      dataset_names.append(dset_name.decode('utf-8'))
    dataset_names = np.array(dataset_names)
    dataset_sizes = f.attrs['NumVariables'][:]
    dataset_sizes_cumulative = np.cumsum(dataset_sizes)
    variable_names = []
    for var_name in f.attrs['VariableNames'][:]:
      variable_names.append(var_name.decode('utf-8'))
    variable_names = np.array(variable_names)
    #variable_names = f.attrs['VariableNames'][:]

    quantity_datasets = []
    quantity_indices = []
    for q in quantities:
      var_num = np.where(variable_names == q)[0][0]
      dataset_num = np.where(dataset_sizes_cumulative > var_num)[0][0]
      if dataset_num == 0:
        dataset_index = var_num
      else:
        dataset_index = var_num - dataset_sizes_cumulative[dataset_num-1]
      quantity_datasets.append(dataset_names[dataset_num])
      quantity_indices.append(dataset_index)

    # Locate fine block for coordinates in case of slice
    fine_block = np.where(levels == max_level)[0][0]
    x1m = f['x1f'][fine_block,0]
    x1p = f['x1f'][fine_block,1]
    x2m = f['x2f'][fine_block,0]
    x2p = f['x2f'][fine_block,1]
    x3m = f['x3f'][fine_block,0]
    x3p = f['x3f'][fine_block,1]

    # Prepare dictionary for results
    if data is None:
      data = {}
      new_data = True

    # Populate coordinate arrays
    face_funcs = (face_func_1, face_func_2, face_func_3)
    center_funcs = (center_func_1, center_func_2, center_func_3)
    for d,nx,face_func,center_func in zip(range(1, 4), nx_vals, face_funcs,center_funcs):
      if nx == 1:
        xm = (x1m, x2m, x3m)[d-1]
        xp = (x1p, x2p, x3p)[d-1]
        data['x'+repr(d)+'f'] = np.array([xm, xp])
      else:
        xmin = f.attrs['RootGridX'+repr(d)][0]
        xmax = f.attrs['RootGridX'+repr(d)][1]
        xrat_root = f.attrs['RootGridX'+repr(d)][2]
        if face_func is not None:
          data['x'+repr(d)+'f'] = face_func(xmin, xmax, xrat_root, nx+1)
        elif (xrat_root == 1.0):
          data['x'+repr(d)+'f'] = np.linspace(xmin, xmax, nx+1)
        else:
          xrat = xrat_root ** (1.0 / 2**level)
          data['x'+repr(d)+'f'] = \
              xmin + (1.0-xrat**np.arange(nx+1)) / (1.0-xrat**nx) * (xmax-xmin)
      data['x'+repr(d)+'v'] = np.empty(nx)
      for i in range(nx):
        data['x'+repr(d)+'v'][i] = \
            center_func(data['x'+repr(d)+'f'][i], data['x'+repr(d)+'f'][i+1])

    # Account for selection
    x1_select = False
    x2_select = False
    x3_select = False
    i_min = j_min = k_min = 0
    i_max = nx1
    j_max = nx2
    k_max = nx3
    error_string = '{0} must be {1} than {2} in order to intersect data range'
    if x1_min is not None and x1_min >= data['x1f'][1]:
      if x1_min >= data['x1f'][-1]:
        raise AthenaError(error_string.format('x1_min', 'less', data['x1f'][-1]))
      x1_select = True
      i_min = np.where(data['x1f'] <= x1_min)[0][-1]
    if x1_max is not None and x1_max <= data['x1f'][-2]:
      if x1_max <= data['x1f'][0]:
        raise AthenaError(error_string.format('x1_max', 'greater', data['x1f'][0]))
      x1_select = True
      i_max = np.where(data['x1f'] >= x1_max)[0][0]
    if x2_min is not None and x2_min >= data['x2f'][1]:
      if x2_min >= data['x2f'][-1]:
        raise AthenaError(error_string.format('x2_min', 'less', data['x2f'][-1]))
      x2_select = True
      j_min = np.where(data['x2f'] <= x2_min)[0][-1]
    if x2_max is not None and x2_max <= data['x2f'][-2]:
      if x2_max <= data['x2f'][0]:
        raise AthenaError(error_string.format('x2_max', 'greater', data['x2f'][0]))
      x2_select = True
      j_max = np.where(data['x2f'] >= x2_max)[0][0]
    if x3_min is not None and x3_min >= data['x3f'][1]:
      if x3_min >= data['x3f'][-1]:
        raise AthenaError(error_string.format('x3_min', 'less', data['x3f'][-1]))
      x3_select = True
      k_min = np.where(data['x3f'] <= x3_min)[0][-1]
    if x3_max is not None and x3_max <= data['x3f'][-2]:
      if x3_max <= data['x3f'][0]:
        raise AthenaError(error_string.format('x3_max', 'greater', data['x3f'][0]))
      x3_select = True
      k_max = np.where(data['x3f'] >= x3_max)[0][0]

    # Adjust coordinates if selection made
    if x1_select:
      data['x1f'] = data['x1f'][i_min:i_max+1]
      data['x1v'] = data['x1v'][i_min:i_max]
    if x2_select:
      data['x2f'] = data['x2f'][j_min:j_max+1]
      data['x2v'] = data['x2v'][j_min:j_max]
    if x3_select:
      data['x3f'] = data['x3f'][k_min:k_max+1]
      data['x3v'] = data['x3v'][k_min:k_max]

    # Prepare arrays for data and bookkeeping
    if new_data:
      for q in quantities:
        data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
    else:
      for q in quantities:
        data[q].fill(0.0)
    if not subsample and not fast_restrict and max_level > level:
      restricted_data = np.zeros((lx3, lx2, lx1), dtype=bool)

    # Go through blocks in data file
    for block_num in range(num_blocks):

      # Extract location information
      block_level = levels[block_num]
      block_location = logical_locations[block_num,:]

      # Prolongate coarse data and copy same-level data
      if block_level <= level:

        # Calculate scale (number of copies per dimension)
        s = 2 ** (level - block_level)

        # Calculate destination indices, without selection
        il_d = block_location[0] * block_size[0] * s if nx1 > 1 else 0
        jl_d = block_location[1] * block_size[1] * s if nx2 > 1 else 0
        kl_d = block_location[2] * block_size[2] * s if nx3 > 1 else 0
        iu_d = il_d + block_size[0] * s if nx1 > 1 else 1
        ju_d = jl_d + block_size[1] * s if nx2 > 1 else 1
        ku_d = kl_d + block_size[2] * s if nx3 > 1 else 1

        # Calculate (prolongated) source indices, with selection
        il_s = max(il_d, i_min) - il_d
        jl_s = max(jl_d, j_min) - jl_d
        kl_s = max(kl_d, k_min) - kl_d
        iu_s = min(iu_d, i_max) - il_d
        ju_s = min(ju_d, j_max) - jl_d
        ku_s = min(ku_d, k_max) - kl_d
        if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
          continue

        # Account for selection in destination indices
        il_d = max(il_d, i_min) - i_min
        jl_d = max(jl_d, j_min) - j_min
        kl_d = max(kl_d, k_min) - k_min
        iu_d = min(iu_d, i_max) - i_min
        ju_d = min(ju_d, j_max) - j_min
        ku_d = min(ku_d, k_max) - k_min

        # Assign values
        for q,dataset,index in zip(quantities, quantity_datasets, quantity_indices):
          block_data = f[dataset][index,block_num,:]
          if s > 1:
            if nx1 > 1:
              block_data = np.repeat(block_data, s, axis=2)
            if nx2 > 1:
              block_data = np.repeat(block_data, s, axis=1)
            if nx3 > 1:
              block_data = np.repeat(block_data, s, axis=0)
          data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] = \
              block_data[kl_s:ku_s,jl_s:ju_s,il_s:iu_s]

      # Restrict fine data
      else:

        # Calculate scale
        s = 2 ** (block_level - level)

        # Calculate destination indices, without selection
        il_d = block_location[0] * block_size[0] // s if nx1 > 1 else 0
        jl_d = block_location[1] * block_size[1] // s if nx2 > 1 else 0
        kl_d = block_location[2] * block_size[2] // s if nx3 > 1 else 0
        iu_d = il_d + block_size[0] // s if nx1 > 1 else 1
        ju_d = jl_d + block_size[1] // s if nx2 > 1 else 1
        ku_d = kl_d + block_size[2] // s if nx3 > 1 else 1

        # Calculate (restricted) source indices, with selection
        il_s = max(il_d, i_min) - il_d
        jl_s = max(jl_d, j_min) - jl_d
        kl_s = max(kl_d, k_min) - kl_d
        iu_s = min(iu_d, i_max) - il_d
        ju_s = min(ju_d, j_max) - jl_d
        ku_s = min(ku_d, k_max) - kl_d
        if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
          continue

        # Account for selection in destination indices
        il_d = max(il_d, i_min) - i_min
        jl_d = max(jl_d, j_min) - j_min
        kl_d = max(kl_d, k_min) - k_min
        iu_d = min(iu_d, i_max) - i_min
        ju_d = min(ju_d, j_max) - j_min
        ku_d = min(ku_d, k_max) - k_min

        # Account for restriction in source indices
        if nx1 > 1:
          il_s *= s
          iu_s *= s
        if nx2 > 1:
          jl_s *= s
          ju_s *= s
        if nx3 > 1:
          kl_s *= s
          ku_s *= s

        # Apply subsampling
        if subsample:

          # Calculate fine-level offsets (nearest cell at or below center)
          o1 = s//2 - 1 if nx1 > 1 else 0
          o2 = s//2 - 1 if nx2 > 1 else 0
          o3 = s//2 - 1 if nx3 > 1 else 0

          # Assign values
          for q,dataset,index in zip(quantities, quantity_datasets, quantity_indices):
            data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] = \
                f[dataset][index,block_num,kl_s+o3:ku_s:s,jl_s+o2:ju_s:s,il_s+o1:iu_s:s]

        # Apply fast (uniform Cartesian) restriction
        elif fast_restrict:

          # Calculate fine-level offsets
          io_vals = range(s) if nx1 > 1 else (0,)
          jo_vals = range(s) if nx2 > 1 else (0,)
          ko_vals = range(s) if nx3 > 1 else (0,)

          # Assign values
          for q,dataset,index in zip(quantities, quantity_datasets, quantity_indices):
            for ko in ko_vals:
              for jo in jo_vals:
                for io in io_vals:
                  data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] += \
                      f[dataset]\
                      [index,block_num,kl_s+ko:ku_s:s,jl_s+jo:ju_s:s,il_s+io:iu_s:s]
            data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] /= s ** num_extended_dims

        # Apply exact (volume-weighted) restriction
        else:

          # Calculate sets of indices
          i_s_vals = range(il_s, iu_s)
          j_s_vals = range(jl_s, ju_s)
          k_s_vals = range(kl_s, ku_s)
          i_d_vals = range(il_d, iu_d)
          j_d_vals = range(jl_d, ju_d)
          k_d_vals = range(kl_d, ku_d)
          if nx1 > 1:
            i_d_vals = np.repeat(i_d_vals, s)
          if nx2 > 1:
            j_d_vals = np.repeat(j_d_vals, s)
          if nx3 > 1:
            k_d_vals = np.repeat(k_d_vals, s)

          # Accumulate values
          for k_s,k_d in zip(k_s_vals, k_d_vals):
            if nx3 > 1:
              x3m = f['x3f'][block_num,k_s]
              x3p = f['x3f'][block_num,k_s+1]
            for j_s,j_d in zip(j_s_vals, j_d_vals):
              if nx2 > 1:
                x2m = f['x2f'][block_num,j_s]
                x2p = f['x2f'][block_num,j_s+1]
              for i_s,i_d in zip(i_s_vals, i_d_vals):
                if nx1 > 1:
                  x1m = f['x1f'][block_num,i_s]
                  x1p = f['x1f'][block_num,i_s+1]
                vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                for q,dataset,index in \
                    zip(quantities, quantity_datasets, quantity_indices):
                  data[q][k_d,j_d,i_d] += f[dataset][index,block_num,k_s,j_s,i_s] * vol
          loc1 = (nx1 > 1 ) * block_location[0] // s
          loc2 = (nx2 > 1 ) * block_location[1] // s
          loc3 = (nx3 > 1 ) * block_location[2] // s
          restricted_data[loc3,loc2,loc1] = True

    # Remove volume factors from restricted data
    if level < max_level and not subsample and not fast_restrict:
      for loc3 in range(lx3):
        for loc2 in range(lx2):
          for loc1 in range(lx1):
            if restricted_data[loc3,loc2,loc1]:
              il = loc1 * block_size[0]
              jl = loc2 * block_size[1]
              kl = loc3 * block_size[2]
              iu = il + block_size[0]
              ju = jl + block_size[1]
              ku = kl + block_size[2]
              il = max(il, i_min) - i_min
              jl = max(jl, j_min) - j_min
              kl = max(kl, k_min) - k_min
              iu = min(iu, i_max) - i_min
              ju = min(ju, j_max) - j_min
              ku = min(ku, k_max) - k_min
              for k in range(kl, ku):
                if nx3 > 1:
                  x3m = data['x3f'][k]
                  x3p = data['x3f'][k+1]
                for j in range(jl, ju):
                  if nx2 > 1:
                    x2m = data['x2f'][j]
                    x2p = data['x2f'][j+1]
                  for i in range(il, iu):
                    if nx1 > 1:
                      x1m = data['x1f'][i]
                      x1p = data['x1f'][i+1]
                    vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                    for q in quantities:
                      data[q][k,j,i] /= vol

  # Return dictionary containing requested data arrays
  return data

#=========================================================================================

def rdhdf5(ifile, ndim = 2,coord ='xy',block_level = 0,x1min=None, x1max=None, x2min=None,
    x2max=None, x3min=None, x3max=None,box_radius = None,user_x1 = False, user_x2=False,gr = False,a = 0,rbr = 200.0,
    x1_harm_max=2.0,npow2=4.0,cpow2=1.0,h =0.2,theta_min=0.075,fast_restrict=False,vol_params=None,uov=False):
  global x,y,z,nx,ny,nz,r,th,ph
  global dic


  def theta_func(xmin, xmax, xrat, nf):
    x_vals = np.linspace(0,1.0,nf)
    t_vals = 2.0*x_vals -1.0
    w_vals = 0.25*(t_vals*(t_vals*t_vals+1.0))+0.5
    return w_vals*xmax + (1.0-w_vals)*xmin

  def mks_theta_func(xmin,xmax,xrat,nf):
    x_vals = np.linspace(0,1.0,nf)
    th_max = xmax
    th_min = xmin
    x2 = (th_max-th_min) * x_vals + (th_min) + 0.5*(1.0-h)*np.sin(2.0*((th_max-th_min) * x_vals + (th_min)))
    x2[0] = xmin
    x2[-1] = xmax
    return x2
    ##return np.pi*x_vals + 0.5*(1.0-h) * np.sin(2.0*np.pi*x_vals)

  def mks_pole_fix(xmin,xmax,xrat,nf):
    # def func(x_lower):
    #   return np.pi * x_lower + (1.0-h)/2.0 * np.sin(np.pi * 2.0 * x_lower) -theta_min ;
    # x_l = scipy.optimize.bisect(func,0,0.5)
    # x_p = 1.0-x_l
    y = np.linspace(0,1,nf)
    dy = 1.0/(nf-1)

    #map (dy,1-dy) to x_l,x_p

    # (x_p-x_l)*(y-dy) + x_l

    # x_l = m*dy + b 
    # x_p = m*(1-dy) + b 
    # m = (x_l -b)/dy
    # x_p = (x_l-b)/dy * (1-dy) + b

    # x_p = x_l/dy *(1-dy) - b*(1-dy)/dy + b = x_l/dy *(1-dy)  + b(1-(1-dy)/dy)
    b =1.0 / (2*dy-1) * (dy*(x_p+x_l) - x_l)
    m = (x_p-x_l)/(1.0-2.0*dy)

    var = m*y+b 

    th_f = np.pi*var + (1-h)/2.0*sin(2*np.pi*var)
    th_f[0] = xmin
    th_f[-1] = xmax
    return th_f
  def hyper_exponetial_r_func(xmin,xmax,xrat,nf):
    logrmin = np.log(xmin) 
    xbr = log(rbr)
    x_scaled = logrmin + x* (x1_harm_max - logrmin); 

    return np.exp(x_scaled + (cpow2*(x_scaled-xbr))**npow2 * (x_scaled>xbr));

  global x_l,x_p
  face_func_1 = None
  face_func_2 = None
  if (user_x2) ==True : face_func_2 = theta_func
  if (user_x2  =="mks" or user_x2 =="MKS"): face_func_2 = mks_theta_func 
  if (user_x2 =="mks_pole_fix" or user_x2=="pole_fix"): 
    def func(x_lower):
      return np.pi * x_lower + (1.0-h)/2.0 * np.sin(np.pi * 2.0 * x_lower) -theta_min ;
    x_l = scipy.optimize.bisect(func,0,0.5)
    x_p = 1.0-x_l
    face_func_2 = mks_pole_fix
  if (user_x1  =="hyper_exp" or user_x1==True): face_func_1 = hyper_exponetial_r_func
  if (box_radius is not None):
    x1min = -box_radius
    x1max = box_radius
    x2min = -box_radius
    x2max = box_radius
    x3min = -box_radius  
    x3max = box_radius

  #file_prefix = glob.glob("*.athdf")[0][:-11]
  #if (gr==True): 
  file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  dic = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
    x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_1=face_func_1, face_func_2 = face_func_2,fast_restrict=fast_restrict,
    vol_params=vol_params)

  global x1f,x2f,x3f,x1v,x3v,x2v,rho, vel1, vel2, vel3, press, Bcc1,Bcc2,Bcc3,k_ent,ke_ent,ke_ent2,ke_ent3
  global user_out3, user_out4, user_out5

  x1f = dic['x1f']
  x2f = dic['x2f']
  x3f = dic['x3f']
  x1v = dic['x1v']
  x2v = dic['x2v']
  x3v = dic['x3v']
  rho = dic['rho'].transpose()
  vel1 = dic['vel1'].transpose()
  vel2 = dic['vel2'].transpose()
  vel3 = dic['vel3'].transpose()
  if ('press' in dic.keys()): press = dic['press'].transpose()

  if ('Bcc1' in dic.keys()):
    Bcc1 = dic['Bcc1'].transpose()
    Bcc2 = dic['Bcc2'].transpose()
    Bcc3 = dic['Bcc3'].transpose()
  if ('r0' in dic.keys()): k_ent = dic['r0'].transpose()
  if ('r1' in dic.keys()): ke_ent = dic['r1'].transpose()
  if ('r2' in dic.keys()): ke_ent2 = dic['r2'].transpose()
  if ('r3' in dic.keys()): ke_ent3 = dic['r3'].transpose()
  # for key in dic.keys():
  #   exec("globals()['%s'] = dic['%s']" % (key,key),globals(),locals())
  #   exec("if type(globals()['%s']) is np.ndarray: globals()['%s'] = globals()['%s'].transpose()" %(key,key,key),globals(),locals())

  if (ndim ==2 and coord =="cyl"):
    x = x1v[:,None] * np.cos(x2v)[None,:]
    y = x1v[:,None] * np.sin(x2v)[None,:]
    r = x1v[:,None] * np.cos(x2v*0)[None,:]
    ph =x2v[None,:] * np.cos(x1v*0)[:,None]
  # elif (ndim==1 and coord=="spherical"):
  #   r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   ph = x2v
  #   x = r
  #   y = r*0
  elif ((ndim==3 or ndim==1) and coord == "spherical"):
    x = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.cos(x3v)[None,None,:]
    y = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.sin(x3v)[None,None,:]
    z = x1v[:,None,None] * np.cos(x2v)[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
    th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
    ph = (x1v/x1v)[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * x3v[None,None,:]
  elif (coord=="xy" and ndim==2):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  elif (coord =="xy" and ndim ==3):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  elif (coord=='xy' and ndim ==1):
    x = x1v
  else: 
    print ("coordinates not determined in ", ndim, "dimensions")
  nx = x.shape[0]
  if (ndim==1): ny = 1
  else: ny = y.shape[1]
  if (ndim ==2 or ndim==1):
    nz = 1
  else:
    nz = z.shape[2]

  global bsq
  if ("Bcc1" in dic.keys() and gr==False): 
    bsq = Bcc1**2 + Bcc2**2 + Bcc3**2
  
  global uu,gamma,gdet,bu
  if (gr==True and coord=="spherical"):
    gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
    ks_metric(r,th,a)
    tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
    gamma = np.sqrt(1.0 + tmp);

    # Calculate 4-velocity
    ks_inverse_metric(r,th,a)
    alpha = np.sqrt(-1.0/gi[0,0]);
    uu = np.zeros((4,nx,ny,nz))
    uu[0] = gamma/alpha;
    uu[1] = vel1 - alpha * gamma * gi[0,1];
    uu[2] = vel2 - alpha * gamma * gi[0,2];
    uu[3] = vel3 - alpha * gamma * gi[0,3];
    if ("Bcc1" in dic.keys()):
      B_vec = np.zeros(uu.shape)
      bu = np.zeros(uu.shape)
      B_vec[1] = Bcc1 
      B_vec[2] = Bcc2 
      B_vec[3] = Bcc3      
      for i in range(1,4):
        for mu in range(0,4):
          bu[0] += g[i,mu]*uu[mu]*B_vec[i]
      bu[1] = 1.0/uu[0] * (Bcc1 + bu[0]*uu[1])
      bu[2] = 1.0/uu[0] * (Bcc2 + bu[0]*uu[2])
      bu[3] = 1.0/uu[0] * (Bcc3 + bu[0]*uu[3])
      bu = np.array(bu)
      bu_tmp = bu* 1.0

      bsq = 0
      for i in range(4):
        for j in range(4):
          bsq += g[i,j] * bu[i] * bu[j]

  if (gr==True and glob.glob("*out3*athdf") != []):
    uu = np.zeros((4,nx,ny,nz))
    file_prefix = glob.glob("*out3*athdf")[0][:-11]
    dic2 = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
      x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_2 = face_func_2)
    gamma = dic2['user_out_var0'].transpose()
    uu[0] = dic2['user_out_var1'].transpose()
    uu[1] = dic2['user_out_var2'].transpose()
    uu[2] = dic2['user_out_var3'].transpose()
    uu[3] = dic2['user_out_var4'].transpose()
    if (coord=="spherical"): gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
    else: gdet = x*0+1
    if (coord=="spherical"): ks_metric(r,th,a)
    else: cks_metric(x,y,z,0,0,a)
    #global ud,bu,bd
   
    ud = Lower(uu,g)
    if ('user_out_var5' in dic2.keys() and "Bcc1" in dic.keys()):
      bsq = dic2['user_out_var5'].transpose()*2.0
      B_vec = np.zeros(uu.shape)
      bu = np.zeros(uu.shape)
      B_vec[1] = Bcc1 
      B_vec[2] = Bcc2 
      B_vec[3] = Bcc3
      for i in range(1,4):
        for mu in range(0,4):
          bu[0] += g[i,mu]*uu[mu]*B_vec[i]
      bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
      bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
      bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
      bd = Lower(bu,g)

    global A1,A2,A3,divb_array
    if ('user_out_var6') in dic2.keys():
      A1 = dic2['user_out_var6'].transpose()
      A2 = dic2['user_out_var7'].transpose()
      A3 = dic2['user_out_var8'].transpose()
    if ('user_out_var9') in dic2.keys():
      divb_array = dic2['user_out_var9'].transpose()

  global fluxr,fluxth,dM_floor, dr_grid,dth_grid,dphi_grid,r_grid,th_grid

  if (glob.glob("*user*athdf") != []):
    file_prefix = glob.glob("*user*athdf")[0][:-11]
    dic2 = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
      x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_2 = face_func_2)
    fluxr= dic2['user_out_var0'].transpose()
    fluxth = dic2['user_out_var1'].transpose()
    dM_floor = dic2['user_out_var2'].transpose()
    if ('user_out_var3' in dic2.keys() ): dr_grid = dic2['user_out_var3'].transpose()
    if ('user_out_var4' in dic2.keys() ): dth_grid = dic2['user_out_var4'].transpose()
    if ('user_out_var5' in dic2.keys() ): dphi_grid = dic2['user_out_var5'].transpose()
    if ('user_out_var6' in dic2.keys() ): r_grid = dic2['user_out_var6'].transpose()
    if ('user_out_var7' in dic2.keys() ): th_grid = dic2['user_out_var7'].transpose()

  if (uov==True):
    dic2 = athdf("fm_torus.out3." + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
      x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_2 = face_func_2)    
    global uov0,uov1,uov2,uov3,uov4,uov5,uov6,uov7,uov8,uov9,uov10
    if ('user_out_var0' in dic2.keys() ): uov0 = dic2['user_out_var0'].transpose()
    if ('user_out_var1' in dic2.keys() ): uov1 = dic2['user_out_var1'].transpose()
    if ('user_out_var2' in dic2.keys() ): uov2 = dic2['user_out_var2'].transpose()
    if ('user_out_var3' in dic2.keys() ): uov3 = dic2['user_out_var3'].transpose()
    if ('user_out_var4' in dic2.keys() ): uov4 = dic2['user_out_var4'].transpose()
    if ('user_out_var5' in dic2.keys() ): uov5 = dic2['user_out_var5'].transpose()
    if ('user_out_var6' in dic2.keys() ): uov6 = dic2['user_out_var6'].transpose()
    if ('user_out_var7' in dic2.keys() ): uov7 = dic2['user_out_var7'].transpose()
    if ('user_out_var8' in dic2.keys() ): uov8 = dic2['user_out_var8'].transpose()
    if ('user_out_var9' in dic2.keys() ): uov9 = dic2['user_out_var9'].transpose()
    if ('user_out_var10' in dic2.keys() ): uov10 = dic2['user_out_var10'].transpose()


def rdhdf5_chris(ifile, ndim = 2,coord ='xy',block_level = 0,x1min=None, x1max=None, x2min=None,
    x2max=None, x3min=None, x3max=None,box_radius = None,user_x1 = False, user_x2=False,gr = False,a = 0,rbr = 200.0,
    x1_harm_max=2.0,npow2=4.0,cpow2=1.0):
  global x,y,z,nx,ny,nz,r,th,ph
  global dic


  def theta_func(xmin, xmax, xrat, nf):
    x_vals = np.linspace(0,1.0,nf)
    t_vals = 2.0*x_vals -1.0
    w_vals = 0.25*(t_vals*(t_vals*t_vals+1.0))+0.5
    return w_vals*xmax + (1.0-w_vals)*xmin

  def mks_theta_func(xmin,xmax,xrat,nf):
    x_vals = np.linspace(0,1.0,nf)
    return np.pi*x_vals + 0.5*(1.0-h) * np.sin(2.0*np.pi*x_vals)

  def hyper_exponetial_r_func(xmin,xmax,xrat,nf):
    logrmin = np.log(xmin) 
    xbr = log(rbr)
    x_scaled = logrmin + x* (x1_harm_max - logrmin); 

    return np.exp(x_scaled + (cpow2*(x_scaled-xbr))**npow2 * (x_scaled>xbr));


  face_func_1 = None
  face_func_2 = None
  if (user_x2) ==True : face_func_2 = theta_func
  if (user_x2  =="mks" or user_x2 =="MKS"): face_func_2 = mks_theta_func 
  if (user_x1  =="hyper_exp" or user_x1==True): face_func_1 = hyper_exponetial_r_func
  if (box_radius is not None):
    x1min = -box_radius
    x1max = box_radius
    x2min = -box_radius
    x2max = box_radius
    x3min = -box_radius  
    x3max = box_radius

  file_prefix = glob.glob("*.athdf")[0][:-11]
  if (gr==True): file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  dic = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
    x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_1=face_func_1, face_func_2 = face_func_2, fast_restrict=fast_restrict)

  global x1f,x2f,x3f,x1v,x3v,x2v,rho, vel1, vel2, vel3, press, Bcc1,Bcc2,Bcc3

  x1f = dic['x1f']
  x2f = dic['x2f']
  x3f = dic['x3f']
  x1v = dic['x1v']
  x2v = dic['x2v']
  x3v = dic['x3v']
  rho = dic['rho'].transpose()
  vel1 = dic['vel1'].transpose()
  vel2 = dic['vel2'].transpose()
  vel3 = dic['vel3'].transpose()
  if ('press' in dic.keys()): press = dic['press'].transpose()

  if ('Bcc1' in dic.keys()):
    Bcc1 = dic['Bcc1'].transpose()
    Bcc2 = dic['Bcc2'].transpose()
    Bcc3 = dic['Bcc3'].transpose()
  # for key in dic.keys():
  #   exec("globals()['%s'] = dic['%s']" % (key,key),globals(),locals())
  #   exec("if type(globals()['%s']) is np.ndarray: globals()['%s'] = globals()['%s'].transpose()" %(key,key,key),globals(),locals())

  if (ndim ==2 and coord =="cyl"):
    x = x1v[:,None] * np.cos(x2v)[None,:]
    y = x1v[:,None] * np.sin(x2v)[None,:]
  # elif (ndim==1 and coord=="spherical"):
  #   r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   ph = x2v
  #   x = r
  #   y = r*0
  elif ((ndim==3 or ndim==1) and coord == "spherical"):
    x = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.cos(x3v)[None,None,:]
    y = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.sin(x3v)[None,None,:]
    z = x1v[:,None,None] * np.cos(x2v)[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
    th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
    ph = (x1v/x1v)[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * x3v[None,None,:]

  elif (coord=="xy" and ndim==2):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  elif (coord =="xy" and ndim ==3):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  else: 
    print ("coordinates not determined in ", ndim, "dimensions")
  nx = x.shape[0]
  if (ndim==1): ny = 1
  else: ny = y.shape[1]
  if (ndim ==2 or ndim==1):
    nz = 1
  else:
    nz = z.shape[2]

  global bsq
  if ("Bcc1" in dic.keys() ): 
    bsq = Bcc1**2 + Bcc2**2 + Bcc3**2
  
  global uu,gamma,gdet
  if (gr==True and glob.glob("*out3*athdf") != []):
    uu = np.zeros((4,nx,ny,nz))
    file_prefix = glob.glob("*out3*athdf")[0][:-11]
    dic2 = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
      x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_2 = face_func_2)
    gamma = dic2['user_out_var0'].transpose()
    # uu[0] = dic2['user_out_var1'].transpose()
    # uu[1] = dic2['user_out_var2'].transpose()
    # uu[2] = dic2['user_out_var3'].transpose()
    # uu[3] = dic2['user_out_var4'].transpose()
    if (coord=="spherical"): gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
    else: gdet = x*0+1
    if (coord=="spherical"): ks_metric(r,th,a)
    else: cks_metric(x,y,z,0,0,a)
    global ud,bu,bd
   
    ud = Lower(uu,g)
    if True: ##('user_out_var5' in dic2.keys() and "Bcc1" in dic.keys()):
      bsq = dic2['user_out_var5'].transpose()*2.0




# def plot_projection():
#   #do something
def rd_hst_entropy(file,is_magnetic=False,entropy = True):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E,Ktot,Ke,Ue
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME2 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  if (entropy==True):
    Ktot = hst[:,n]; n+=1
    Ke = hst[:,n]; n+=1
    if (n<hst.shape[1]): Ue = hst[:,n]; n+=1


def rd_hst_basic(file,is_magnetic=False):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E
  global ME1,ME2,ME3,dM1,dM2,dM3,dM_floor
  global dM1_bound,dM2_bound,dM1_user,dM2_user, divb
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME2 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM_floor =hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM1 = hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM2 = hst[:,n]; n+=1
  
  if (n<hst.shape[1] and is_magnetic==True): divb = hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM_floor = hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM1_bound =hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM2_bound = hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM1_user =hst[:,n]; n+=1
  # if (n<hst.shape[1]): dM2_user = hst[:,n]; n+=1
def rd_hst(file,is_magnetic=False):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E
  global rho_avg,v1_avg,v2_avg,v3_avg,p_avg,r,n
  global mdot_avg, vr_avg,Lx_avg,Ly_avg,Lz_avg,L_avg,ME1,ME2,ME3
  global Phi
  N_r = 128
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME2 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  rho_avg = hst[:,n:n+N_r]; n+= N_r
  v1_avg = hst[:,n:n+N_r]; n+= N_r
  v2_avg = hst[:,n:n+N_r]; n+= N_r
  v3_avg = hst[:,n:n+N_r]; n+= N_r
  p_avg = hst[:,n:n+N_r]; n+= N_r
  r = hst[:,n:n+N_r]; n+= N_r
  mdot_avg = hst[:,n:n+N_r]; n+= N_r
  vr_avg = hst[:,n:n+N_r]; n+= N_r
  Lx_avg = hst[:,n:n+N_r]; n+= N_r
  Ly_avg = hst[:,n:n+N_r]; n+= N_r
  Lz_avg = hst[:,n:n+N_r]; n+= N_r
  L_avg = np.sqrt(Lx_avg**2. + Ly_avg**2. + Lz_avg**2.)
  set_constants()
  global v_kep, l_kep,l_theta,l_phi
  v_kep = np.sqrt(gm_/r)
  l_kep = v_kep * r
  l_theta  = np.arccos(Lz_avg/L_avg)
  l_phi  = np.arctan2(Ly_avg,Lx_avg)

  global mdot_out,mdot_in,M_boundary, M_removed, Edot_avg,Edot_in,Edot_out
  global Lxdot_avg,Lydot_avg,Lzdot_avg,Ldot_avg,kappa_dot_avg
  global Kappa_avg, v_phi_x_avg,v_phi_y_avg,v_phi_z_avg
  global Lxdot_in,Lydot_in,Lzdot_in,Lxdot_out,Lydot_out,Lzdot_out
  global Q_cool,t_cool, bsq_avg,Bx_avg,By_avg,Bz_avg,Br_avg,Bphi_avg,divb, rho_added,Br_abs_avg
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    mdot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    mdot_in = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Edot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Edot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Edot_in = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lxdot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lydot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lzdot_avg = hst[:,n:n+N_r]; n+= N_r
    Ldot_avg = np.sqrt(Lxdot_avg**2. + Lydot_avg**2. + Lzdot_avg**2.)
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lxdot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lydot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lzdot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    rho_added = hst[:,n:n+N_r]; n+= N_r #Lxdot_in = hst[:,n:n+N_r]; n+= N_r
    tmp= np.gradient(rho_added,axis=0)
    tmp[tmp<0] = rho_added[tmp<0]
    rho_added = np.cumsum(tmp,axis=0)
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    if (is_magnetic == False): Lydot_in = hst[:,n:n+N_r]; n+= N_r
    else: Br_abs_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lzdot_in = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    v_phi_x_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    v_phi_y_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    v_phi_z_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    kappa_dot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Kappa_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Q_cool = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    t_cool = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Bx_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    By_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Bz_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    bsq_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Br_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Bphi_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1]):
    M_removed = hst[:,n]; n+=1
  if (n<hst.shape[1]):
    M_boundary = hst[:,n]; n+=1
  if (n<hst.shape[1]):
    divb = hst[:,n]; n+=1

  if (is_magnetic==True):
    v_ff = np.sqrt(2.0*gm_/r)
    Phi = Br_abs_avg * 2.0 * np.pi * r /np.sqrt(np.fabs(mdot_avg) * v_ff)
def rd_gr_hst(file,is_magnetic=False):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E,ME1,ME2,ME3
  global mdot,edot,jdot,phibh,vol,divb
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  mdot = hst[:,n]; n+=1
  jdot = hst[:,n]; n+=1
  edot = hst[:,n]; n+=1
  area = hst[:,n]; n+=1
  area = 4
  if (is_magnetic==True):
    phibh = hst[:,n]; n+=1
    #phibh = phibh/vol
    divb = hst[:,n]; n+=1
  # mdot = mdot/vol
  # jdot = jdot/vol
  # edot = edot/vol


def yt_extract_box(i_dump,box_radius = 0.5,mhd=True,gr=False,a=0.0,res=128,center_x=0.0,center_y=0.0,center_z=0.0,uov=False):
  global region,x,y,z,rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3,bsq,vsq,uu,bsq,bu,bd
  global gamma
  global nx,ny,nz
  global uov1,uov2,uov3,uov4, uov0
  yt_load(i_dump,gr=gr)
  resolution = complex("%dj" %res)
  region = ds.r[(-box_radius+center_x,'pc'):(box_radius+center_x,'pc'):resolution,(-box_radius+center_y,'pc'):(box_radius+center_y,'pc'):resolution,
      (-box_radius+center_z,'pc'):(box_radius+center_z,'pc'):resolution]

  if (uov==True): region2 = ds2.r[(-box_radius+center_x,'pc'):(box_radius+center_x,'pc'):resolution,(-box_radius+center_y,'pc'):(box_radius+center_y,'pc'):resolution, (-box_radius+center_z,'pc'):(box_radius+center_z,'pc'):resolution]
  x = np.array(region['x'])
  y = np.array(region['y'])
  z = np.array(region['z'])

  rho = np.array(region['rho'])
  press = np.array(region['press'])
  vel1 = np.array(region['vel1'])
  vel2 = np.array(region['vel2'])
  vel3 = np.array(region['vel3'])

  vsq = vel1**2 + vel2**2 + vel3**2

  if (uov==True):
    uov0 = np.array(region2['user_out_var0'])
    uov1 = np.array(region2['user_out_var1'])
    uov2 = np.array(region2['user_out_var2'])
    uov3 = np.array(region2['user_out_var3'])




  if (mhd==True):
    Bcc1 = np.array(region['Bcc1'])
    Bcc2 = np.array(region['Bcc2'])
    Bcc3 = np.array(region['Bcc3'])
    if (gr==False): bsq = Bcc1**2 + Bcc2**2 + Bcc3**2 


  # if (gr==True): region = ds2.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
  #     (-box_radius,'pc'):(box_radius,'pc'):128j]

  if (gr==True):
      uu = [0,0,0,0]
      bu = [0,0,0,0]
      cks_metric(x,y,z,0,0,a)
      tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
      gamma = np.sqrt(1.0 + tmp);

      # Calculate 4-velocity
      cks_inverse_metric(x,y,z,0,0,a)
      alpha = np.sqrt(-1.0/gi[0,0]);
      uu[0] = gamma/alpha;
      uu[1] = vel1 - alpha * gamma * gi[0,1];
      uu[2] = vel2 - alpha * gamma * gi[0,2];
      uu[3] = vel3 - alpha * gamma * gi[0,3];

      uu = np.array(uu)
      if (mhd==True): 
          B_vec = np.zeros(uu.shape)
          B_vec[1] = Bcc1 
          B_vec[2] = Bcc2 
          B_vec[3] = Bcc3
          cks_metric(x,y,z,0,0,a)
          for i in range(1,4):
            for mu in range(0,4):
              bu[0] += g[i,mu]*uu[mu]*B_vec[i]
          bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
          bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
          bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
          bu = np.array(bu)
          bu_tmp = bu* 1.0

          bsq = 0
          for i in range(4):
            for j in range(4):
              bsq += g[i,j] * bu[i] * bu[j]

          bd = Lower(bu,g)

  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]

  global ke_ent,k_ent,ke_ent2,ke_ent3
  if (('athena_pp','r0') in ds.field_list ): 
    k_ent = np.array(region['r0'])
  if (('athena_pp','r1') in ds.field_list ): 
    ke_ent = np.array(region['r1'])
  if (('athena_pp','r2') in ds.field_list ): 
    ke_ent2 = np.array(region['r2'])
  if (('athena_pp','r3') in ds.field_list ): 
    ke_ent3 = np.array(region['r3'])

def yt_extract_box_rotated(i_dump,box_radius = 0.5,mhd=True,gr=False,a=0.0,res=128,center_x=0.0,center_y=0.0,center_z=0.0,uov=False,orbit_file=None,t0=0,xbox_radius=None,ybox_radius=None,zbox_radius=None,slice=False,th_tilt=0,phi_tilt=0,resx=None,resy=None,resz=None):
  global region,x,y,z,rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3,bsq,vsq,uu,bsq,bu,bd
  global gamma
  global nx,ny,nz
  global uov1,uov2,uov3,uov4, uov0,uov5,uov6
  yt_load(i_dump,gr=gr,a=a)
  if (resx is None): resx = res
  if (resy is None): resy = res 
  if (resz is None): resz = res
 
  # yt_extract_box(i_dump,box_radius = box_radius,mhd=True,gr=gr,a=a,res=res,center_x=center_x,center_y=center_y,center_z=center_z,uov=uov,orbit_file=orbit_file,t0=t0,xbox_radius=xbox_radius,ybox_radius=ybox_radius,zbox_radius=zbox_radius)
  t = np.array(ds.current_time)
  #rd_binary_orbits(orbit_file)
  #get_binary_quantities(t,t0)
  ph = phi_tilt
  th = th_tilt
  #unit vectors for the new coordinate system in the frame of the old coordinates
  z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])   #r
  x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  #theta
  y_hat = np.array([-sin(ph),cos(ph),0])                        #phi
  #unit vectors for original coordinate system in the frame of the new coordinates
  #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
  x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #in terms of [theta_hat, phi_hat, r_hat] 
  y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
  z_hat_prime = [-np.sin(th),0,np.cos(th)]
  centerx_tmp,centery_tmp,centerz_tmp = 1.0*center_x, 1.0*center_y, 1.0*center_z
  center_x = centerx_tmp * x_hat_prime[0] + centery_tmp * y_hat_prime[0] + centerz_tmp * z_hat_prime[0]
  center_y = centerx_tmp * x_hat_prime[1] + centery_tmp * y_hat_prime[1] + centerz_tmp * z_hat_prime[1]
  center_z = centerx_tmp * x_hat_prime[2] + centery_tmp * y_hat_prime[2] + centerz_tmp * z_hat_prime[2]
  index = np.arange(ds.r['rho'].shape[0])
  #faces
  new_x = np.linspace(-box_radius+center_x,box_radius+center_x,resx+1)
  if (slice==False): new_y = np.linspace(-box_radius+center_y,box_radius+center_y,resy+1)
  else: new_y = np.array([center_y])
  new_z = np.linspace(-box_radius+center_z,box_radius+center_z,resz+1)
  dx = np.diff(new_x)[0]
  if (slice==False):
    dy = np.diff(new_y)[0]
    new_y = (new_y + dy/2.0)[:-1]
  dz = np.diff(new_z)[0]
  new_x = (new_x + dx/2.0)[:-1]
  new_z = (new_z + dz/2.0)[:-1]
  new_x,new_y,new_z = np.meshgrid(new_x,new_y,new_z,indexing='ij')
  xi = new_x * x_hat[0] + new_y * y_hat[0] + new_z * z_hat[0]
  yi = new_x * x_hat[1] + new_y * y_hat[1] + new_z * z_hat[1]
  zi = new_x * x_hat[2] + new_y * y_hat[2] + new_z * z_hat[2]
  # if (slice==False):
  new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = "nearest",fill_value = 0).astype(np.int64)
  # else:   new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = "linear",fill_value = 0)
  x = new_x * 1.0
  y = new_y * 1.0
  z = new_z * 1.0
  # rho = ds.['rho'][new_index]*1.0 
  # press = ds.['press'].flatten()[new_index] * 1.0
  # vel1 = ds.['vel1'].flatten()[new_index] * 1.0
  # vel2 = ds.['vel2'].flatten()[new_index] * 1.0
  # vel3 = ds.['vel3'].flatten()[new_index] * 1.0
  # uu[0] = uu[0].flatten()[new_index] * 1.0
  # uu[1] = uu[1].flatten()[new_index] * 1.0
  # uu[2] = uu[2].flatten()[new_index] * 1.0
  # uu[3] = uu[3].flatten()[new_index] * 1.0
  # bu[0] = bu[0].flatten()[new_index] * 1.0
  # bu[1] = bu[1].flatten()[new_index] * 1.0
  # bu[2] = bu[2].flatten()[new_index] * 1.0
  # bu[3] = bu[3].flatten()[new_index] * 1.0
  # bsq = bsq.flatten()[new_index] * 1.0
  from yt.units import pc, msun,kyr
  rho = np.array(ds.r['density'][(new_index)] * pc**3/msun)
  press = np.array(ds.r['press'][(new_index)] * pc**3/msun * kyr**2/pc**2)
  vel1 = np.array(ds.r['vel1'][(new_index)] * kyr/pc)
  vel2 = np.array(ds.r['vel2'][(new_index)] * kyr/pc)
  vel3 = np.array(ds.r['vel3'][(new_index)] * kyr/pc)
  global uu,bu,bsq
  uu = [0,0,0,0]
  bu = [0,0,0,0]
  cks_metric(xi,yi,zi,0,0,a)
  tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
  gamma = np.sqrt(1.0 + tmp);
  # Calculate 4-velocity
  #invert_metric(g)
  invert_metric(g)
  alpha = np.sqrt(-1.0/gi[0,0]);
  uu[0] = gamma/alpha;
  uu[1] = vel1 - alpha * gamma * gi[0,1];
  uu[2] = vel2 - alpha * gamma * gi[0,2];
  uu[3] = vel3 - alpha * gamma * gi[0,3];
# uu[0] = ds2.r['user_out_var1'][new_index]
# uu[1] = ds2.r['user_out_var2'][new_index]
# uu[2] = ds2.r['user_out_var3'][new_index]
# uu[3] = ds2.r['user_out_var4'][new_index]
  uu = np.array(uu)
  uu_tmp = uu*1.0
  
  B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
  Bcc1 = np.array(ds.r['Bcc1'][new_index]/B_unit)
  Bcc2 = np.array(ds.r['Bcc2'][new_index]/B_unit)
  Bcc3 = np.array(ds.r['Bcc3'][new_index]/B_unit)
  B_vec = np.zeros(uu.shape)
  B_vec[1] = Bcc1 
  B_vec[2] = Bcc2 
  B_vec[3] = Bcc3
  # cks_metric(xi,yi,zi,ax,ay,az)  // No need to call again
  for i in range(1,4):
    for mu in range(0,4):
      bu[0] += g[i,mu]*uu[mu]*B_vec[i]
  bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
  bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
  bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
  bu = np.array(bu)
  bu_tmp = bu* 1.0
  bsq = 0
  for i in range(4):
    for j in range(4):
      bsq += g[i,j] * bu[i] * bu[j]
  uu_tmp = uu * 1.0
  bu_tmp = bu * 1.0
  uu[1] = uu_tmp[1] * x_hat_prime[0] + uu_tmp[2] * y_hat_prime[0]  + uu_tmp[3] * z_hat_prime[0]
  uu[2] = uu_tmp[1] * x_hat_prime[1] + uu_tmp[2] * y_hat_prime[1]  + uu_tmp[3] * z_hat_prime[1]
  uu[3] = uu_tmp[1] * x_hat_prime[2] + uu_tmp[2] * y_hat_prime[2]  + uu_tmp[3] * z_hat_prime[2]
  bu[1] = bu_tmp[1] * x_hat_prime[0] + bu_tmp[2] * y_hat_prime[0]  + bu_tmp[3] * z_hat_prime[0]
  bu[2] = bu_tmp[1] * x_hat_prime[1] + bu_tmp[2] * y_hat_prime[1]  + bu_tmp[3] * z_hat_prime[1]
  bu[3] = bu_tmp[1] * x_hat_prime[2] + bu_tmp[2] * y_hat_prime[2]  + bu_tmp[3] * z_hat_prime[2]
  Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
  Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
  Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])
  global nx,ny,nz 
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]
def yt_extract_box_corotating_old(i_dump,box_radius = 0.5,mhd=True,gr=False,a=0.0,res=128,center_x=0.0,center_y=0.0,center_z=0.0,uov=False,orbit_file=None,t0=0,xbox_radius=None,ybox_radius=None,zbox_radius=None,just_rho=False):
  global region,x,y,z,rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3,bsq,vsq,uu,bsq,bu,bd
  global gamma
  global nx,ny,nz
  global uov1,uov2,uov3,uov4, uov0,uov5,uov6
 
  yt_extract_box(i_dump,box_radius = box_radius,mhd=True,gr=gr,a=a,res=res,center_x=center_x,center_y=center_y,center_z=center_z,uov=uov,orbit_file=orbit_file,t0=t0,xbox_radius=xbox_radius,ybox_radius=ybox_radius,zbox_radius=zbox_radius)
  t = np.array(ds.current_time)
  rd_binary_orbits(orbit_file)
  get_binary_quantities(t,t0)
  ph = arctan2(y2,x2)
  th = 0;
  #unit vectors for the new coordinate system in the frame of the old coordinates
  z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])   #r
  x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  #theta
  y_hat = np.array([-sin(ph),cos(ph),0])                        #phi
  #unit vectors for original coordinate system in the frame of the new coordinates
  #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
  x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #in terms of [theta_hat, phi_hat, r_hat] 
  y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
  z_hat_prime = [-np.sin(th),0,np.cos(th)]
  centerx_tmp,centery_tmp,centerz_tmp = 1.0*center_x, 1.0*center_y, 1.0*center_z
  center_x = centerx_tmp * x_hat_prime[0] + centery_tmp * y_hat_prime[0] + centerz_tmp * z_hat_prime[0]
  center_y = centerx_tmp * x_hat_prime[1] + centery_tmp * y_hat_prime[1] + centerz_tmp * z_hat_prime[1]
  center_z = centerx_tmp * x_hat_prime[2] + centery_tmp * y_hat_prime[2] + centerz_tmp * z_hat_prime[2] 
  index = np.arange(rho.flatten().shape[0])
  #faces
  new_x = np.linspace(-box_radius+center_x,box_radius+center_x,nx+1)
  new_y = np.linspace(-box_radius+center_y,box_radius+center_y,ny+1)
  new_z = np.linspace(-box_radius+center_z,box_radius+center_z,nz+1)
  dx = np.diff(new_x)[0]
  dy = np.diff(new_y)[0]
  dz = np.diff(new_z)[0]
  new_x = (new_x + dx/2.0)[:-1]
  new_y = (new_y + dy/2.0)[:-1]
  new_z = (new_z + dz/2.0)[:-1]
  new_x,new_y,new_z = np.meshgrid(new_x,new_y,new_z,indexing='ij')
  xi = new_x * x_hat[0] + new_y * y_hat[0] + new_z * z_hat[0]
  yi = new_x * x_hat[1] + new_y * y_hat[1] + new_z * z_hat[1]
  zi = new_x * x_hat[2] + new_y * y_hat[2] + new_z * z_hat[2]
  from scipy.interpolate import RegularGridInterpolator
  points_to_interpolate = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()]).T  # Shape (N, 3)
  interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), rho,bounds_error=False, fill_value=None,method='linear')
  rho = interp(points_to_interpolate).reshape(xi.shape)
  if (just_rho==False):
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), press,bounds_error=False, fill_value=None,method='linear')
    press = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), vel1,bounds_error=False, fill_value=None,method='linear')
    vel1 = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), vel2,bounds_error=False, fill_value=None,method='linear')
    vel2 = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), vel3,bounds_error=False, fill_value=None,method='linear')
    vel3 = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), uu[0],bounds_error=False, fill_value=None,method='linear')
    uu[0] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), uu[1],bounds_error=False, fill_value=None,method='linear')
    uu[1] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), uu[2],bounds_error=False, fill_value=None,method='linear')
    uu[2] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), uu[3],bounds_error=False, fill_value=None,method='linear')
    uu[3] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), bu[0],bounds_error=False, fill_value=None,method='linear')
    bu[0] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), bu[1],bounds_error=False, fill_value=None,method='linear')
    bu[1] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), bu[2],bounds_error=False, fill_value=None,method='linear')
    bu[2] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), bu[3],bounds_error=False, fill_value=None,method='linear')
    bu[3] = interp(points_to_interpolate).reshape(xi.shape)
    interp = RegularGridInterpolator((x[:,0,0], y[0,:,0],z[0,0,:]), bsq,bounds_error=False, fill_value=None,method='linear')
    bsq = interp(points_to_interpolate).reshape(xi.shape)
    # new_index = scipy.interpolate.griddata((x.flatten(),y.flatten(),z.flatten()),index,(xi,yi,zi),method = "nearest",fill_value = 0)
    
    # x = new_x * 1.0
    # y = new_y * 1.0
    # z = new_z * 1.0
    # rho = rho.flatten()[new_index]*1.0 
    # press = press.flatten()[new_index] * 1.0
    # vel1 = vel1.flatten()[new_index] * 1.0
    # vel2 = vel2.flatten()[new_index] * 1.0
    # vel3 = vel3.flatten()[new_index] * 1.0
    # uu[0] = uu[0].flatten()[new_index] * 1.0
    # uu[1] = uu[1].flatten()[new_index] * 1.0
    # uu[2] = uu[2].flatten()[new_index] * 1.0
    # uu[3] = uu[3].flatten()[new_index] * 1.0
    # bu[0] = bu[0].flatten()[new_index] * 1.0
    # bu[1] = bu[1].flatten()[new_index] * 1.0
    # bu[2] = bu[2].flatten()[new_index] * 1.0
    # bu[3] = bu[3].flatten()[new_index] * 1.0
    # bsq = bsq.flatten()[new_index] * 1.0
    uu_tmp = uu * 1.0
    bu_tmp = bu * 1.0
    uu[1] = uu_tmp[1] * x_hat_prime[0] + uu_tmp[2] * y_hat_prime[0]  + uu_tmp[3] * z_hat_prime[0]
    uu[2] = uu_tmp[1] * x_hat_prime[1] + uu_tmp[2] * y_hat_prime[1]  + uu_tmp[3] * z_hat_prime[1]
    uu[3] = uu_tmp[1] * x_hat_prime[2] + uu_tmp[2] * y_hat_prime[2]  + uu_tmp[3] * z_hat_prime[2]
    bu[1] = bu_tmp[1] * x_hat_prime[0] + bu_tmp[2] * y_hat_prime[0]  + bu_tmp[3] * z_hat_prime[0]
    bu[2] = bu_tmp[1] * x_hat_prime[1] + bu_tmp[2] * y_hat_prime[1]  + bu_tmp[3] * z_hat_prime[1]
    bu[3] = bu_tmp[1] * x_hat_prime[2] + bu_tmp[2] * y_hat_prime[2]  + bu_tmp[3] * z_hat_prime[2]
    Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
    Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
    Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])
  x = 1.0*new_x 
  y = 1.0*new_y 
  z = 1.0*new_z

def yt_extract_box_chris(i_dump,box_radius = 0.5,mhd=True,gr=False,a=0.0):
  global region,x,y,z,rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3,bsq,vsq,uu,bsq,bu,bd
  global nx,ny,nz
  yt_load(i_dump,gr=gr)
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
      (-box_radius,'pc'):(box_radius,'pc'):128j]

  x = np.array(region['x'])
  y = np.array(region['y'])
  z = np.array(region['z'])

  rho = np.array(region['rho'])
  press = np.array(region['press'])
  vel1 = np.array(region['vel1'])
  vel2 = np.array(region['vel2'])
  vel3 = np.array(region['vel3'])

  vsq = vel1**2 + vel2**2 + vel3**2
  
  if (mhd==True):
    Bcc1 = np.array(region['Bcc1'])
    Bcc2 = np.array(region['Bcc2'])
    Bcc3 = np.array(region['Bcc3'])
    if (gr==False): bsq = Bcc1**2 + Bcc2**2 + Bcc3**2 


  if (gr==True): region = ds2.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
      (-box_radius,'pc'):(box_radius,'pc'):128j]

  if (gr==True):
    u0 = np.array(region['user_out_var1'])
    # u1 = np.array(region['user_out_var2'])
    # u2 = np.array(region['user_out_var3'])
    # u3 = np.array(region['user_out_var4'])
    # uu = np.array([u0,u1,u2,u3])
    if (mhd==True): 
      bsq = np.array(region['user_out_var2'])*2.0
      # B_vec = np.zeros(uu.shape)
      # bu = np.zeros(uu.shape)
      # B_vec[1] = Bcc1 
      # B_vec[2] = Bcc2 
      # B_vec[3] = Bcc3
      # for i in range(1,4):
      #   for mu in range(0,4):
      #     bu[0] += g[i,mu]*uu[mu]*B_vec[i]
      # bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
      # bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
      # bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
      # bd = Lower(bu,g)

  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]

def phibh():
  dx3 = np.diff(x3f)
  dx2 = np.diff(x2f)
  dOmega = (gdet * dx2[None,:,None]) * dx3[None,None,:]
  return 0.5*np.abs(Bcc1*dOmega).sum(-1).sum(-1)

def r_to_ir(r_input):
  dlog10r = np.diff(np.log10(r[-1,:]))[0]
  r_min = r[-1,0]
  r_out = r[-1,-1]
  #r = r_min * 10**(ir*dlog10r)
  return int(np.round(np.log10(r_input/r_min)/dlog10r))

# def r_to_ir_array(r_input):
#   dlog10r = np.diff(np.log10(r[:,:]),axis=-1)[:,-1]
#   r_min = r[:,0]
#   r_out = r[:,-1]
#   #r = r_min * 10**(ir*dlog10r)
#   return np.int(np.round(np.log10(r_input/r_min)/dlog10r))

def get_mdot(mhd=False,gr=False,ax=0,ay=0,az=0):
  global mdot, v,vr,bernoulli,mdot_in,mdot_out, Lx_dot,Ly_dot,Lz_dot
  global l_x,l_y,l_z
  global vth,vphi,Br,Bth,Bphi,Bz,r
  if (nz ==1):
    r = np.sqrt(x**2. + y**2.)
    v = np.sqrt(vel1**2. + vel2**2.)
    vr = vel1 * (x/r)[:,:,None] + vel2 * (y/r)[:,:,None]
    vphi = vel1* (-y/(r+1e-15))[:,:,None]  + vel2 * (x/(r+1e-15))[:,:,None]
    phi = np.arctan2(y,x)
    mdot = 2.*np.pi * r[:,:,None] * rho * vr
    if (mhd==True):
      Br = Bcc1 * (x/r)[:,:,None] + Bcc2 * (y/r)[:,:,None]
      Bphi = Bcc1* (-y/(r+1e-15))[:,:,None]  + Bcc2 * (x/(r+1e-15)) [:,:,None]
      Bz = Bcc3
  elif (gr==False):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    vr = vel1 * (x/r) + vel2 * (y/r) + vel3 * (z/r)
    vth = vel1 * (x*z)/(r*s+1e-15)+ vel2 * (y*z)/(r*s+1e-15) + vel3 * (-s/(r+1e-15))
    vphi = vel1* (-y/(s+1e-15))  + vel2 * (x/(s+1e-15)) 
    if (mhd ==True):
      Br = Bcc1 * (x/r) + Bcc2 * (y/r) + Bcc3 * (z/r)
      Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
      Bphi = Bcc1* (-y/(s+1e-15))  + Bcc2 * (x/(s+1e-15))     
    mdot = 4.*np.pi * r**2. * rho * vr
    mdot_out = mdot * (mdot>0)
    mdot_in = mdot * (mdot<0)
    gam = 5./3.
    gm1 = gam-1.
    gm_ =  0.019264
    bernoulli = (vel1**2.+vel2**2.+vel3**2)/2. + gam*press/rho/gm1 - gm_/r
    l_x = y*vel3 - z*vel2
    l_y = z*vel1 - x*vel3
    l_z = x*vel2 - y*vel1
    Lx_dot = mdot * l_x
    Ly_dot = mdot * l_y 
    Lz_dot = mdot * l_z
  else: #GR
    a = np.sqrt(ax**2.0 + ay**2.0 + az**2.0)
    R = np.sqrt(x**2+y**2+z**2)
    adotx = ax*x+ay*y+az*z
    r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*(adotx)**2.0 ) )/np.sqrt(2.0)
    global uu_ks,bu_ks,Bcc1_ks,Bcc2_ks,Bcc3_ks
    uu_ks = cks_vec_to_ks(uu,x,y,z,ax,ay,az)

    mdot = 4.0*np.pi/3.0 * (3.0*r**2+a**2) * rho * uu_ks[1]

    if (mhd==True):
      bu_ks = cks_vec_to_ks(bu,x,y,z,ax,ay,az)
      Bcc1_ks = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])
      Bcc2_ks = (bu_ks[2] * uu_ks[0] - bu_ks[0] * uu_ks[2])
      Bcc3_ks = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])

def cartesian_vector_to_spherical(vx,vy,vz):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    vr = vx* (x/r) + vy * (y/r) + vz * (z/r)
    vth = vx * (x*z)/(r*s+1e-15)+ vy * (y*z)/(r*s+1e-15) + vz * (-s/(r+1e-15))
    vphi = vx* (-y/(s+1e-15))  + vy * (x/(s+1e-15))
    return (vr,vth,vphi)
class AthenaError(RuntimeError):
  """General exception class for Athena++ read functions."""
  pass

class AthenaWarning(RuntimeWarning):
  """General warning class for Athena++ read functions."""
  pass

def bhole(ax=0.0,ay=0.0,az=0.0,facecolor='white'):
    a = np.sqrt(ax**2.0+ay**2.0+az**2.0)
    rhor = ( 1.0 + np.sqrt(1.0-a**2) )
    ax = plt.gca()
    el = Ellipse((0,0), 2*rhor, 2*rhor, facecolor=facecolor, alpha=1)
    art=ax.add_artist(el)
    art.set_zorder(20)
    plt.draw()
def bhole2(x_,y_,q=0.0,aprime=0.0,facecolor='white'):
    rhor = ( q + np.sqrt(q**2.0-aprime**2) )
    ax = plt.gca()
    el = Ellipse((x_,y_), 2*rhor, 2*rhor, facecolor=facecolor, alpha=1)
    art=ax.add_artist(el)
    art.set_zorder(20)
    plt.draw()
def set_constants(n_levels = 8):
  global arc_secs,km_per_s,gm_,rho_to_n_cgs,mp_over_kev
  global v_kep_norm,l_norm, mue,mu_highT,keV_to_Kelvin
  global e_charge,me,cl,mp,pc,kyr,msun,Bunit
  D_BH = 8.3e3 #in parsecs
  #tan(theta) ~ theta ~ x_pc /D_BH
  arc_secs = 4.84814e-6 * D_BH
  km_per_s = 0.001022  # in parsec/k year
  gm_ = 0.019264
  rho_to_n_cgs = 40.46336
  mp_over_kev = 9.994827
  v_kep_norm = np.sqrt(gm_/arc_secs)
  l_norm = v_kep_norm * arc_secs
  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = 3. * Z_solar
  X = 0.
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  keV_to_Kelvin = 1.16045e7

  e_charge = 4.803e-10
  me = 9.109e-28
  cl = 2.997924e10
  mp = 1.6726e-24
  pc = 3.086e18
  kyr = 3.154e10
  msun = 1.989e33

  Bunit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
  global k_green, silver,midnight_green,charcoal,dark_gray,light_blue
  k_green = '#4cbb17'
  silver = '#8D9093'
  midnight_green = '#004851'
  charcoal = '#54585A'
  dark_gray = '#A9A9A9'
  light_blue = "#ADD8E6"


def run_stellar_wind_test(suffix = ""):
  for i in range(1,5):
    N = 2.**i
    os.system("rm *.hst *.rst *.athdf*")
    os.system("sed -i 's_^star\_radius.*_star\_radius = %g_' athinput.star\_cartesian\_smr" %N)
    os.system("mpirun -n 16 ./athena -i athinput.star_cartesian_smr")
    os.system("mkdir N_%g%s -p" %(N,suffix))
    os.system("cp star_wind.out2.00500* *.hst N_%g%s" %(N,suffix))
    os.system("rm *.hst *.rst *.athdf*")

def run_bondi_test():
    for i in range(5,10):
        N = 2**i
        os.system("rm *.hst *.rst *.athdf*")
        os.system("sed -i 's_^nx1   .*_nx1      = %d_' athinput.bondi\_cyl" %(N))
        n_processors = np.amin(np.array([16,N/32]))
        os.system("mpirun -n %d ./athena -i athinput.bondi_cyl" %(n_processors))
        os.system("mkdir N_%g -p" %(N))
        dump_files = glob.glob("*.athdf*")
        dump_files.sort()
        last_dump = dump_files[-1]
        os.system("cp star_wind.out2.00501* *.hst N_%g" %(N))
        os.system("cp star_wind.out2.00000* *.hst N_%g" %(N))
        os.system("rm *.hst *.rst *.athdf*")

def plot_bondi_errors():
    global N_dirs,rho_err_arr,v_err_arr,press_err_arr
    N_dirs = [64,128,256,512]
    rho_err_arr = []
    v_err_arr = []
    press_err_arr = []
    for N in N_dirs:
        os.chdir("N_%g" %N)
        rdhdf5(0,block_level = 0, ndim=2)
        rho_init = rho
        v_init = vel1
        press_init = press
        rdhdf5(501,block_level = 0,ndim = 2)
        rho_err = np.sum(np.abs(rho-rho_init))/(1.*N)
        v_err =np.sum(np.abs(vel1-v_init))/(1.*N)
        press_err = np.sum(np.abs(press-press_init))/(1.*N)
        rho_err_arr.append(rho_err)
        v_err_arr.append(v_err)
        press_err_arr.append(press_err)
        os.chdir("..")



def plot_stellar_wind_test(suffix = ""):
  clf()
  for i in range(1,5):
    N = 2.**i
    os.chdir("N_%g%s" %(N,suffix))
    rdhdf5(500)
    ny = rho.shape[1]
    get_mdot()
    mdot_new = convert_to_polar(mdot).mean(-1)
    #plot(r[:,ny/2],mdot[:,ny/2,0],label = r'$N=%g$' %N)
    plot(r,mdot_new,label = r'$N=%g$' %N)
    os.chdir("..")
  plot(r,0.01*r/r,label = r'$\dot M_\ast$')
  plt.legend(loc = 'best')
  plt.xlabel(r'$r$',fontsize = 30)
  plt.ylabel(r'$\dot M$',fontsize = 30)
  plt.ylim(0.0099,0.0101)
  figure()
  for i in range(1,5):
    N = 2.**i
    os.chdir("N_%g%s" %(N,suffix))
    rdhdf5(500)
    ny = rho.shape[1]
    get_mdot()
    v_new = convert_to_polar(vr).mean(-1)
    #plot(r[:,ny/2],mdot[:,ny/2,0],label = r'$N=%g$' %N)
    plot(r,v_new,label = r'$N=%g$' %N)
    os.chdir("..")
  plot(r,1*r/r,label = r'$v_{wind}$')
  plt.legend(loc = 'best')
  plt.xlabel(r'$r$',fontsize = 30)
  plt.ylabel(r'$v$',fontsize = 30)
  plt.ylim(.95,1.05)


def submit_stellar_wind_test(suffix = ""):
  for i in range(1,5):
      N = 2.**i
      os.system("sed -i 's_^star\_radius.*_star\_radius = %g_' athinput.star\_cartesian\_smr" %N)
      os.system("mkdir N_%g%s -p" %(N,suffix))
      os.system("cp athena athinput.star_cartesian_smr qsub_mpi run_athena.py stellar_wind_test.txt N_%g%s" %(N,suffix))
      os.chdir("N_%g%s" %(N,suffix))
      os.system("sbatch qsub_mpi")
      os.chdir("..")

import scipy
from scipy import interpolate

def convert_to_polar(arr):
    global r,phi,xi,yi
    rmin = np.amin(np.sqrt(x**2.+y**2.))
    rmax = np.amax(np.sqrt(x**2.+y**2.))
    r = np.logspace(log10(rmin),log10(rmax),128)
    phi = np.linspace(0,2.*pi,128)
    xi = r[:,None]*cos(phi[None,:])
    yi = r[:,None]*sin(phi[None,:])
    result = scipy.interpolate.griddata((x.flatten(),y.flatten()),arr[:,:,0].flatten(),(xi,yi),method='nearest')
    return result

def get_conversion_array_2d():
    global r,phi,xi,yi
    global igrid_polar,jgrid_polar
    rmin = np.amin(np.sqrt(x**2.+y**2.))
    rmax = np.amax(np.sqrt(x**2.+y**2.))
    r = np.logspace(log10(rmin),log10(rmax),128)
    phi = np.linspace(0,2.*pi,128)
    xi = r[:,None]*cos(phi[None,:])
    yi = r[:,None]*sin(phi[None,:])
    jgrid,igrid = meshgrid(np.arange(0,ny),np.arange(0,nx))
    mgrid = igrid + jgrid*nx 
    mnew = scipy.interpolate.griddata((x.flatten(),y.flatten()),mgrid[:,:].flatten(),(xi,yi),method='nearest')

    igrid_polar= mod(mnew,nx)
    jgrid_polar = mnew//nx

def get_conversion_array_3d():
    global r,phi,theta,xi,yi,zi,igrid_spherical,jgrid_spherical,kgrid_spherical
    r_tmp = np.sqrt(x**2.+y**2.+z**2.)
    rmin =np.amin(r_tmp)
    rmax = np.amax(r_tmp)
    r = np.logspace(log10(rmin),log10(rmax),128)
    theta = np.linspace(0.,np.pi,128)
    phi = np.linspace(0,2.*np.pi,128)
    xi = r[:,None,None]*np.cos(phi[None,None,:])*np.sin(theta[None,:,None])
    yi = r[:,None,None]*np.sin(phi[None,None,:])*np.sin(theta[None,:,None])
    zi = r[:,None,None]*np.cos(theta[None,:,None]) * ((phi + 4.*np.pi)/(phi + 4.*np.pi))[None,None,:]
    kgrid,jgrid,igrid = meshgrid(np.arange(0,nz),np.arange(0,ny),np.arange(0,nx))
    mgrid = igrid + jgrid*nx  + kgrid*nx*ny
    mnew = scipy.interpolate.griddata((x.flatten(),y.flatten(),z.flatten()),mgrid[:,:,:].flatten(),(xi,yi,zi),method='nearest')

    igrid_spherical= mod(mod(mnew,ny*nx),nx)
    jgrid_spherical = mod(mnew,ny*nx)/nx
    kgrid_spherical = mnew/(ny*nx)
def convert_to_spherical(arr,th = 0,ph = 0):
    global r,phi,theta,xi,yi,zi
    z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])  #aligned with the angular momentum vector
    x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  # theta direction at this angle
    y_hat = np.array([-sin(ph),cos(ph),0])    #phi direction at this angle

    x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #xhat vector of original coordinates in terms of rotated coords
    y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    z_hat_prime = [-np.sin(th),0,np.cos(th)]
    r_tmp = np.sqrt(x**2.+y**2.+z**2.)
    rmin =np.amin(r_tmp)
    rmax = np.amax(r_tmp)
    r = np.logspace(log10(rmin),log10(rmax),128)
    theta = np.linspace(0.,np.pi,128)
    phi = np.linspace(0,2.*np.pi,128)
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')
    xi_prime = r*np.cos(phi)*np.sin(theta)
    yi_prime = r*np.sin(phi)*np.sin(theta)
    zi_prime = r*np.cos(theta)

    xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
    yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
    zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2]
    result = scipy.interpolate.griddata((x.flatten(),y.flatten(),z.flatten()),arr[:,:,:].flatten(),(xi,yi,zi),method='nearest')
    return result
def get_radial_profiles():
  global vr_avg,rho_avg,mdot_avg,p_avg,T_avg
  get_mdot()
  vr_avg = (convert_to_spherical(vr)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  mdot_avg = (convert_to_spherical(mdot)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  rho_avg = (convert_to_spherical(rho)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  p_avg =(convert_to_spherical(press)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  T_avg = p_avg/rho_avg

def get_radial_profile(num,denom=None):

  if (denom is None):
    return (convert_to_spherical(num)*np.sin(theta)).mean(-1).mean(-1)/(np.sin(theta)).mean(-1).mean(-1)

  else:
    return (convert_to_spherical(num)*np.sin(theta)).mean(-1).mean(-1)/(convert_to_spherical(denom)*np.sin(theta)).mean(-1).mean(-1)

def get_radial_profiles_2d(mhd=False):
  global vr_avg,vphi_avg,rho_avg,mdot_avg,p_avg,T_avg,Br_avg,Bphi_avg,bsq_avg
  get_mdot(mhd = mhd)
  # vr_avg = convert_to_polar(vr).mean(-1)
  # mdot_avg = convert_to_polar(mdot).mean(-1)
  # rho_avg = convert_to_polar(rho).mean(-1)
  # p_avg =convert_to_polar(press).mean(-1)

  vr_avg = vr[igrid_polar,jgrid_polar,0].mean(-1)
  vphi_avg = vphi[igrid_polar,jgrid_polar,0].mean(-1)
  mdot_avg = mdot[igrid_polar,jgrid_polar,0].mean(-1)
  rho_avg = rho[igrid_polar,jgrid_polar,0].mean(-1)
  p_avg = press[igrid_polar,jgrid_polar,0].mean(-1)
  T_avg = p_avg/rho_avg

  if (mhd==True):
    Br_avg = Br[igrid_polar,jgrid_polar,0].mean(-1)
    Bphi_avg = Bphi[igrid_polar,jgrid_polar,0].mean(-1)
    bsq_avg = bsq[igrid_polar,jgrid_polar,0].mean(-1)


from mpl_toolkits.mplot3d import Axes3D
def plot_orbits():
  hst_list = glob.glob("*.hst")
  hst_file = hst_list[0]
  hst = np.loadtxt(hst_file)
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  for i_star in range(30):
    x = hst[:,10+i_star]
    y = hst[:,11+i_star]
    z = hst[:,12+i_star]
    ax.plot(x,y,zs=z)

def mk1davg(blocklevel = 2,n_dim = 3):
    dump_files = glob.glob("*.athdf")
    dump_files.sort()
    n_dump = 0
    for dump_file in dump_files:
        rdhdf5(n_dump,block_level = blocklevel,ndim = n_dim)
        get_radial_profiles()
        dic = {"r": r, "vr_avg": vr_avg, "rho_avg": rho_avg,"p_avg": p_avg,"T_avg":T_avg,"mdot_avg":mdot_avg}
        np.savez("dump_%05d.npz" %n_dump,**dic)
        n_dump = n_dump + 1

def rd_1d_avg(binary=False):
  fname = "1d_avg.npz"
  if (binary==True): fname="1d_binary_avg.npz"
  if (os.path.isfile(fname)): 
    rdnpz(fname)
    return
  if (binary==False): dump_list = glob.glob("1d_dump_*")
  else: dump_list = glob.glob("1d_binary_dump_*")


  def extract_number(filename):
      return int(''.join(filter(str.isdigit, filename)))

  dump_list = sorted(dump_list, key=extract_number)

  # dump_list= sorted(dump_list, key=lambda x: (int(re.search(r'\d+', x).group()), x))
  #dump_list.sort()
  i_dump_max = len(dump_list)


  array_names = ["r", "th","mdot","mdot_out","Edot","Jdot","t","Phibh", "EdotEM", "Lx", "Ly", 
  "Lz", "Bx", "By", "Bz","A_jet_p", "A_jet_m", "x_jet_p","x_jet_m", "y_jet_p","y_jet_m", 
  "z_jet_p","z_jet_m","rjet_max_p", "rjet_max_m", "gamma_jet_m", "gamma_jet_p","vr_jet","rho",
  "rho_fft","phi_fft","mdot_fft","Bphi","mdot_mid","Jdot_mid","BrBphi_mid", "rho_Bcc3_dr_los_1",
  "rho_Bcc3_dr_los_sum_1","rho_Bcc3_dr_los_2","rho_Bcc3_dr_los_sum_2", "rho_Bcc3_dr_los_3",
  "rho_Bcc3_dr_los_sum_3","mdot_th_bound", "mdot_th_bound_tot","M","M_cum","mdot_th_bound_cum",
  "mdot_r_bound_tot", "mdot_r_bound_tot","mdot_r_bound_out_tot","Mnet_r_inner_bound","Mnet_th_bound", 
  "Mnet_r","Mnet_r_inner_bound_pos","Mnet_r_outer_bound","dMfloor_r","Mnet_r_inner_bound_v_th","dMfloor","M50",
  "rho_Bcc3_dr_los_cumsum_1", "rho_Bcc3_dr_los_scumum_2", "rho_Bcc3_dr_los_cumsum_3",
  "EdotUint","EdotKE_grav","Edotgrav", "x_cavity_p","y_cavity_p","z_cavity_p", "Phibh_net", 
  "Phibh_x", "Phibh_y", "Phibh_z","mdot_out_up_1c","mdot_out_up_2c", "mdot_out_up_3c", 
  "mdot_out_up_4c","mdot_out_up_5c", "mdot_out_down_1c","mdot_out_down_2c", "mdot_out_down_3c", 
  "mdot_out_down_4c","mdot_out_down_5c","Pdotx","Pdoty","Pdotz","Jdotx","Jdoxy","mdot_out_unbound"]

  dic_key_list = ["r", "th","mdot","mdot_out","Edot","Jdot","t","Phibh", "EdotEM", "Lx", "Ly", 
  "Lz", "Bx", "By", "Bz","A_jet_p", "A_jet_m", "x_jet_p","x_jet_m", "y_jet_p","y_jet_m", 
  "z_jet_p","z_jet_m","rjet_max_p", "rjet_max_m", "gamma_jet_m", "gamma_jet_p","vr_jet","rho",
  "rho_fourier","phi_fourier","mdot_fourier","Bphi_r_5","mdot_mid","Jdot_mid","BrBphi_mid", "rho_Bcc3_dr_los_1",
  "rho_Bcc3_dr_los_sum_1","rho_Bcc3_dr_los_2","rho_Bcc3_dr_los_sum_2", "rho_Bcc3_dr_los_3",
  "rho_Bcc3_dr_los_sum_3","mdot_th_bound", "mdot_th_bound_tot","M","M_cum","mdot_th_bound_cum",
  "mdot_r_bound_tot", "mdot_r_bound_tot","mdot_r_bound_out_tot","Mnet_r_inner_bound","Mnet_th_bound", 
  "Mnet_r","Mnet_r_inner_bound_pos","Mnet_r_outer_bound","dMfloor_r","Mnet_r_inner_bound_v_th","dMfloor","M50",
  "rho_Bcc3_dr_los_cumsum_1", "rho_Bcc3_dr_los_scumum_2", "rho_Bcc3_dr_los_cumsum_3",
  "EdotUint","EdotKE_grav","Edotgrav", "x_cavity_p","y_cavity_p","z_cavity_p", "Phibh_net", 
  "Phibh_x", "Phibh_y", "Phibh_z","mdot_out_up_1c","mdot_out_up_2c", "mdot_out_up_3c", 
  "mdot_out_up_4c","mdot_out_up_5c", "mdot_out_down_1c","mdot_out_down_2c", "mdot_out_down_3c", 
  "mdot_out_down_4c","mdot_out_down_5c","Pdotx","Pdoty","Pdotz","Jdotx","Jdoxy","mdot_out_unbound"]
  
  for arr_name in array_names:
    exec("%s = []" %arr_name)


  for dump in dump_list: ## in arange(i_dump_max):
    dic = np.load(dump) #"1d_dump_%04d.npz" %i)

    for i_ in arange(len(array_names)):
      arr_name  = array_names[i_]
      key_name  = dic_key_list[i_]

      if (key_name in dic.keys()): exec("%s.append(dic[\"%s\"])" %(arr_name,key_name))

  for arr_name in array_names:
    exec("%s = np.array(%s)" % (arr_name,arr_name) )

  dic = {}

  for arr_name in array_names:
    exec("dic[\"%s\"]= %s" % (arr_name, arr_name) )

  np.savez(fname,**dic)

  for arr_name in array_names:
    exec("globals()['%s'] = %s" % (arr_name,arr_name) )


def rd_1d_torus_avg(mk_time_avg=False,tmin=0,tmax=1000):
  fname = "1d_torus_avg.npz"

  array_names = ["r", "t","rho","press","beta_inv","pmag","uu_phi","H_r", "PTOT_jet", "PEN_jet", "PEM_jet", 
    "Mdot_jet", "PPAKE_jet"]

  dic_key_list = ["r", "t","rho","press","beta_inv","pmag","uu_phi","H_r", "PTOT_jet", "PEN_jet", "PEM_jet", 
    "Mdot_jet", "PPAKE_jet"]
    
  if (os.path.isfile(fname)): 
    rdnpz(fname)
  else:
    dump_list = glob.glob("1d_torus_dump_*")
    def extract_number(filename):
      return int(''.join(filter(str.isdigit, filename)))

    dump_list = sorted(dump_list, key=extract_number)
    # dump_list= sorted(dump_list, key=lambda x: (int(re.search(r'\d+', x).group()), x))
    # dump_list.sort()
    i_dump_max = len(dump_list)


    for arr_name in array_names:
      exec("%s = []" %arr_name)


    for dump in dump_list: ## in arange(i_dump_max):
      dic = np.load(dump) #"1d_dump_%04d.npz" %i)

      for i_ in arange(len(array_names)):
        arr_name  = array_names[i_]
        key_name  = dic_key_list[i_]

        if (key_name in dic.keys()): exec("%s.append(dic[\"%s\"])" %(arr_name,key_name))

    for arr_name in array_names:
      exec("%s = np.array(%s)" % (arr_name,arr_name) )

    dic = {}

    for arr_name in array_names:
      exec("dic[\"%s\"]= %s" % (arr_name, arr_name) )

    np.savez(fname,**dic)

    for arr_name in array_names:
      exec("globals()['%s'] = %s" % (arr_name,arr_name) )

  if (mk_time_avg):
    dic_time = {}
    itmin = t_to_it(tmin)
    itmax = t_to_it(tmax)

    for arr_name in array_names:
      if (arr_name=="t"): continue
      else:
        exec("%s= %s[itmin:itmax,:].mean(0)" % (arr_name+"_avg", arr_name) )
        exec("%s= %s[itmin:itmax,:].std(0)" % (arr_name+"_std", arr_name) )
        exec("dic_time[\"%s\"]= %s" % (arr_name+"_avg", arr_name+"_avg") )
        exec("dic_time[\"%s\"]= %s" % (arr_name+"_std", arr_name+"_std") )
    
    np.savez(fname[:-4]+"_%s_%s_tavg.npz" %(tmin,tmax),**dic_time)


    return


# def rd_1d_torus_avg():
#   fname = "1d_torus_avg.npz"
#   if (os.path.isfile(fname)): 
#     rdnpz(fname)
#     return
#   dump_list = glob.glob("1d_torus_dump_*")
#   dump_list.sort()
#   i_dump_max = len(dump_list)
#   global r, rho,press,beta_inv,pmag,t
#   r = []
#   rho = []
#   press = []
#   beta_inv = []
#   t = []
#   pmag = []
#   for dump in dump_list: ## in arange(i_dump_max):
#     dic = np.load(dump) #"1d_dump_%04d.npz" %i)
#     r.append(dic['r'])
#     rho.append(dic['rho'])
#     press.append(dic['press'])
#     beta_inv.append(dic['beta_inv'])
#     pmag.append(dic['pmag'])

#     t.append(dic['t'])




#   r = np.array(r)
#   rho = np.array(rho)
#   press = np.array(press)
#   beta_inv = np.array(beta_inv)
#   t = np.array(t)
#   pmag = np.array(pmag)


#   dic = {"r":r,"rho":rho,"press":press,"pmag":pmag,"t":t,"beta_inv":beta_inv}
#   np.savez(fname,**dic)
if __name__ == "__main__":
    if len(sys.argv)>1:
        if sys.argv[1] == "mk1davg":
            mk1davg(blocklevel = np.int(sys.argv[2]),n_dim = np.int(sys.argv[3]))
        else:
            print( "Unknown command %s" % sys.argv[1] )



def run_scaling_test():
  n_cores = [1,2,4,16,28,56,112]

  for n in n_cores:
    os.system("mkdir n_%d" %n)
    os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_mpi" %n)
    os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.04000\.rst time\/tlim=40.02_' qsub_mpi" %n)
    os.system("cp qsub_mpi athena rand_stars_test_2D.txt star_wind.04000.rst ./n_%d" %n)
    os.chdir("n_%d" %n)
    os.system("sbatch qsub_mpi")
    os.chdir("..")

def run_scaling_test_3D():
    import os
    n_cores = [1,2,4,8,16,23,46,92]
    n_cores = [113,226,452]
    n_cores =  [2,4,8,16,32,64,142,284,568,1136]
    n_cores = [64,142,284,568,1136]
    n_cores = [16,64,128,256,362,512,724,1024]
    n_cores = [16, 32, 64, 128, 208, 384, 624, 832, 1248]
    #1,2,4,8,16,32,64,128,256,512,1024,2048,4096

    for n in n_cores:
        os.system("mkdir n_%d" %n)
        os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_strong_scaling" %n)
        #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
        os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-i athinput\.strong\_scaling_' qsub_strong_scaling" %n)
        os.system("cp qsub_strong_scaling  athena orbits_spin_v3.in.txt athinput.strong_scaling ./n_%d" %n)
        os.chdir("n_%d" %n)
        os.system("sbatch qsub_strong_scaling")
        os.chdir("..")

def run_scaling_test_3D_stampede():
    import os
    #n_cores = [136, 272, 340, 680, 1360, 2720, 5440]
    #n_cores = [142, 284, 568, 1136, 2272, 4544]

    n_cores = [32,64, 142, 284, 568, 1136, 2272, 4544,9088]
    N_nodes = np.array(n_cores)/68 +1
    #1,2,4,8,16,32,64,128,256,512,1024,2048,4096
#SBATCH -N 56
#SBATCH -n 3600#
    for i in range(len(n_cores)):
        n_core = n_cores[i]
        n_node = N_nodes[i]
        os.system("mkdir n_%d" %n_core)
        os.system("sed -i 's_^\#SBATCH \-N.*_\#SBATCH \-N %g_' qsub_strong_scaling" %n_node)
        os.system("sed -i 's_^\#SBATCH \-n.*_\#SBATCH \-n %g_' qsub_strong_scaling" %n_core)

        #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
        os.system("cp qsub_strong_scaling  athena athinput.* ./n_%d" %n_core)

        os.chdir("n_%d" %n_core)
        os.system("sbatch qsub_strong_scaling")
        os.chdir("..")

def run_weak_scaling():
  n_cores =  [2,4,8,16,32,64,128,256,512,512*2]
  n_dims = [64,128,256,512,576,640,704,768]
  n_dims = [64,128,256,512,640,768]
  n_dims = np.array(n_dims)//2
  #n_dims = [640,704]
  for n_dim in n_dims:
    n_cores = (n_dim/32)**3
    os.system("mkdir n_%d" %n_cores)
    os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_weak_scaling" %n_cores)
    #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
    os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-i athinput\.weak\_scaling_' qsub_weak_scaling" %n_cores)
    os.system("sed -i 's_^nx1     .*_nx1           \= %d_' athinput.weak_scaling" %(n_dim))
    os.system("sed -i 's_^nx2     .*_nx2           \= %d_' athinput.weak_scaling" %(n_dim))
    os.system("sed -i 's_^nx3     .*_nx3           \= %d_' athinput.weak_scaling" %(n_dim))
    os.system("cp qsub_weak_scaling  athena orbits.in.txt athinput.weak_scaling ./n_%d" %n_cores)

    os.chdir("n_%d" %n_cores)
    os.system("sbatch qsub_weak_scaling")
    os.chdir("..")

  # n_dims = np.array(n_dims)/2
  # nx_block = 32
  # ny_block = 32
  # nz_block = 16
  # n_cores = [1,8,64,512,729, 1000,1331]
  # #n_dims = [640,704]
  # n_mults = [1,2,4,8,9,10,11]
  # for n_mult in n_mults:
  #   #n_cores = (n_dim/32)**3
  #   n_cores = n_mult**3
  #   os.system("mkdir n_%d" %n_cores)
  #   os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_weak_scaling" %n_cores)
  #   #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
  #   os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-i athinput\.weak\_scaling_' qsub_weak_scaling" %n_cores)
  #   nx = nx_block*n_mult
  #   ny = ny_block*n_mult 
  #   nz = nz_block*n_mult
  #   os.system("sed -i 's_^nx1     .*_nx1           \= %d_' athinput.weak_scaling" %(nx))
  #   os.system("sed -i 's_^nx2     .*_nx2           \= %d_' athinput.weak_scaling" %(ny))
  #   os.system("sed -i 's_^nx3     .*_nx3           \= %d_' athinput.weak_scaling" %(nz))
  #   os.system("cp qsub_weak_scaling  athena orbits.in.txt athinput.weak_scaling ./n_%d" %n_cores)

  #   os.chdir("n_%d" %n_cores)
  #   os.system("sbatch qsub_weak_scaling")
  #   os.chdir("..")

import re

def get_scaling_results(t_end = 2e4):
  global end_time_arr,cpu_time_arr, zone_cycles_per_second_arr
  global projected_time,n_vals
  ndirs = glob.glob('n_*')
  ndirs.sort()
  n_vals = []
  cpu_time_arr = []
  end_time_arr = []
  zone_cycles_per_second_arr = []
  for ndir in ndirs:
    n = int(ndir[2:])
    n_vals.append(n)
  n_vals.sort()
  n_vals = np.array(n_vals)
  for n in n_vals:
    os.chdir("n_%d" %n)
    if (glob.glob("slur*")==[]): 
      os.chdir("..")
      continue
    get_run_time()
    cpu_time_arr.append(cpu_time)
    end_time_arr.append(end_time)
    zone_cycles_per_second_arr.append(zone_cycles_per_second)
    os.chdir("..")
  cpu_time_arr = np.array(cpu_time_arr)
  zone_cycles_per_second_arr = np.array(zone_cycles_per_second_arr)
  end_time_arr = np.array(end_time_arr)
  projected_time = t_end/(end_time_arr) * cpu_time_arr *n_vals
def print_scaling_results():
  n_cores = [1,2,4,16,28,56,112]
  n_cores = [1,2,4,8,16,23,46,92]

  cpu_time_arr = []
  for n in n_cores:
    os.chdir("n_%d" %n)
    print (n)
    slurm_file = glob.glob("slurm*")[0]
    with open(slurm_file) as f:
      for line in f:
        if "cpu time use" in line: #zone-cycles/cpu_second" in line:
          c1,c2 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          print (line)
      cpu_time = c1*10.**c2
      cpu_time_arr.append(cpu_time)

    os.chdir("..")


def get_gprof_output():
  gmon_list = glob.glob('gmon*')
  for file in gmon_list:
    os.system('gprof ./athena -p %s >> %s' % (file,'output'+file[8:]))
def get_gprof_results():
  global self_seconds_arr,percent_time_arr
  global self_seconds_riemann_arr
  self_seconds_arr = []
  percent_time_arr = []
  self_seconds_riemann_arr = []
  outputs = glob.glob("output*")
  for file in outputs:
    with open(file) as f:
      for line in f:
        if "cons_force" in line: 
          c1,c2,c3,c4,c5,c6 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          print (line)
    self_seconds = c3
    percent_time = c1
    self_seconds_arr.append(self_seconds)
    percent_time_arr.append(percent_time)
    with open(file) as f:
      for line in f:
        if "Hydro::RiemannSolver" in line: 
          c1,c2,c3,c4,c5,c6 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          print (line)
    self_seconds = c3
    self_seconds_riemann_arr.append(self_seconds)



def get_run_time():
  global cpu_time, zone_cycles_per_second
  global end_time
  slurm_dirs = glob.glob('slurm*')
  slurm_dirs.sort
  slurm_file = slurm_dirs[-1]
  with open(slurm_file) as f:
      for line in f:
        if "cpu time use" in line: #zone-cycles/cpu_second" in line:
          c1,c2 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          cpu_time = c1*10.**c2
        if "cpu_second" in line:
          c3,c4 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          zone_cycles_per_second = c3*10.**c4
        if ("time=" in line) and ("dt=" not in line):
          c5,c6,c7 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          end_time = c5*10.**c6




def rdhdf5_new(file):
  import h5py
  f = h5py.File(file,'r')
  n_blocks = f.attrs['NumMeshBlocks']
  global x,y,z,rho,press,vel1,vel2,vel3
  x = []
  y = []
  z = []
  rho = []
  press = []
  vel1 = []
  vel2 = []
  vel3 = []
  for n in range(n_blocks):
    [X,Y,Z] = np.meshgrid(f['x1v'][n],f['x2v'][n],f['x3v'][n])
    x.append(X)
    y.append(Y)
    z.append(Z)
    rho.append(np.transpose(f['prim'][0][n]))
    press.append(np.transpose(f['prim'][1][n]))
    vel1.append(np.transpose(f['prim'][2][n]))
    vel2.append(np.transpose(f['prim'][3][n]))
    vel3.append(np.transpose(f['prim'][4][n]))

  x,y,z = np.array(x),np.array(y),np.array(z)
  rho = np.array(rho)
  press = np.array(rho)
  vel1 = np.array(vel1)
  vel2 = np.array(vel2)
  vel3 = np.array(vel3)


import yt
def yt_load(ifile,gr=False,a=0.0,spherical=False):
  global ds,ds2
  #T_0_kev = (press/rho * mu_highT*mp_over_kev)[0][0]
  #T_unit = mu_highT * mp_over_keV

  units_override = { "length_unit":(1.0,"pc") ,"time_unit":(1.0,"kyr"),"mass_unit":(1.0,"Msun") }
  if (gr==False and spherical==False): file_prefix = glob.glob("*.athdf")[0][:-11]
  elif (spherical==True or gr==True): file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  else: file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  ds = yt.load(file_prefix + "%05d.athdf" %ifile,units_override=units_override)
  if (gr==True and glob.glob("*out3*athdf") != []):
    file_prefix = glob.glob("*out3*.athdf")[0][:-11]
    fname = file_prefix +"%05d.athdf" %ifile
    if os.path.isfile(fname): ds2 = yt.load(fname,units_override=units_override)
  global gam 
  gam = ds.gamma

  a = ds.arr(a*1.0,"code_length")

  def _r(field,data):
    x = data['index','x']
    y = data['index','y']
    z = data['index','z']
    R = np.sqrt(x**2+y**2+z**2)
    if (gr==False): return R
    else: return np.sqrt(R**2-a**2 + np.sqrt((R**2-a**2)**2 + 4.0*a**2*z**2))/np.sqrt(2.0)
  def _theta(field,data):
    return np.arccos(data['index','z']/_r(field,data))
  def _phi(field,data):
    y = data['index','y']
    x = data['index','x']
    r = _r(field,data)
    if (gr==False): return np.arctan2(data['index','y'],data['index','x'])
    else: return np.arctan2(a*x-r*y,a*y+r*x)
  
  if (gr==False and spherical==False):
    def _vr(field,data):
      vx = data['gas',"velocity_x"]
      vy = data['gas',"velocity_y"]
      vz = data['gas',"velocity_z"]
      x = data['index','x']
      y = data['index','y']
      z = data['index','z']
      return (x*vx + y*vy + z*vz)/np.sqrt(x**2. + y**2. + z**2.)
    def _mdot(field,data):
      v_r = _vr(field,data)
      r = _r(field,data)
      rho = data['gas','density']

      return 4.*np.pi * rho * v_r * r**2.
    def _mdot_in(field,data):
      mdot = _mdot(field,data)
      return mdot * (mdot<0)
  def _temperature(field,data):
    mp_over_kev = ds.arr(9.994827,"code_time**2./code_length**2.")
    X = 0.7
    Z = 0.02
    muH = 1./X
    mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
    T_kev = data['gas','pressure']/data['gas','density'] * mu_highT*mp_over_kev
    from yt.units import keV
    return T_kev*keV
  def _kappa(field,data):
    return data['gas','pressure']/data['gas','density']**gam
  if (gr==False and spherical==False):
    def _kappa_dot(field,data):
      return _kappa(field,data) * _mdot(field,data)
    def _ldot_x(field,data):
      lx = data['gas','velocity_z'] * data['index','y'] - data['gas','velocity_y']*data['index','z']
      return lx * _mdot(field,data)
    def _ldot_y(field,data):
      ly = data['gas','velocity_x'] * data['index','z'] - data['gas','velocity_z']*data['index','x']
      return ly * _mdot(field,data)
    def _ldot_z(field,data):
      lz = data['gas','velocity_y'] * data['index','x'] - data['gas','velocity_x']*data['index','y']
      return lz * _mdot(field,data)
    def _ldot_x_in(field,data):
      ldot = _ldot_x(field,data)
      mdot = _mdot(field,data)
      return ldot * (mdot<0)
    def _ldot_y_in(field,data):
      ldot = _ldot_y(field,data)
      mdot = _mdot(field,data)
      return ldot * (mdot<0)
    def _ldot_z_in(field,data):
      ldot = _ldot_z(field,data)
      mdot = _mdot(field,data)
      return ldot * (mdot<0)
    def _bernoulli(field,data):
      vsq = data['gas','velocity_x']**2. + data['gas','velocity_y']**2. + data['gas','velocity_z']**2.
      csq = gam * data['gas','pressure']/data['gas','density']
      gm_ = ds.arr(0.019264,"code_length**3/code_time**2")
      return vsq/2. + csq/(gam-1.) - gm_/_r(field,data)
    def _Edot(field,data):
      return _bernoulli(field,data)*_mdot(field,data)


  # if (gr==False and spherical==False): ds.add_field(('gas','mdot'),function = _mdot, units = "Msun/kyr",particle_type = False,sampling_type="cell")
  # if (gr==False and spherical==False): ds.add_field(('gas','mdot_in'),function = _mdot_in, units = "Msun/kyr",particle_type = False,sampling_type="cell")
  # #ds.add_field(('gas','temperature'),function = _temperature, units = "keV", force_override= True,particle_type = False,sampling_type="cell")
  # if (gr==False and spherical==False): ds.add_field(('gas',"Ldot_x"),function = _ldot_x,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  # if (gr==False and spherical==False): ds.add_field(('gas',"Ldot_y"),function = _ldot_y,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  # if (gr==False and spherical==False): ds.add_field(('gas',"Ldot_z"),function = _ldot_z,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  # if (gr==False and spherical==False): ds.add_field(('gas',"Ldot_x_in"),function = _ldot_x_in,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  # if (gr==False and spherical==False): ds.add_field(('gas',"Ldot_y_in"),function = _ldot_y_in,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  # if (gr==False and spherical==False): ds.add_field(('gas',"Ldot_z_in"),function = _ldot_z_in,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  # ##if (gr==False): ds.add_field(('gas','Edot'),function = _Edot, units = "pc**2/kyr**2*Msun/kyr",particle_type = False,sampling_type="cell")
  # ##if (gr==False): ds.add_field(('gas',"kappa_dot"),function = _kappa_dot,units="code_length**4*code_pressure*code_velocity/code_mass**(2/3)",particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"
  # if (spherical==False): ds.add_field(('gas',"r"),function = _r,units="code_length",particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"
  # if (spherical==False): ds.add_field(('gas',"theta"),function = _theta,particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"
  # if (spherical==False): ds.add_field(('gas',"phi"),function = _phi,particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"


def get_radial_profiles_yt():
  my_sphere = ds.sphere([0,0,0],(ds.domain_right_edge[0],"pc"))

  prof = my_sphere.profile("radius", ["density","pressure","mdot","Ldot_x","Ldot_y","Ldot_z","kappa_dot","Edot","Ldot_x_in","Ldot_y_in","Ldot_z_in","mdot_in"],weight_field = "cell_volume")

  global r 
  r = prof.x
  global rho_avg,p_avg,mdot_avg,Ldotx_avg,Ldoty_avg,Ldotz_avg,kappa_dot_avg
  global Edot_avg,Ldotx_in,Ldoty_in,Ldotz_in, mdot_in  
  rho_avg = prof['density']
  p_avg = prof['pressure']
  mdot_avg = prof['mdot']
  Ldotx_avg = prof['Ldot_x']
  Ldoty_avg = prof['Ldot_y']
  Ldotz_avg = prof['Ldot_z']
  Ldotx_in = prof['Ldot_x_in']
  Ldoty_in = prof['Ldot_y_in']
  Ldotz_in = prof['Ldot_z_in']
  kappa_dot_avg = prof['kappa_dot']
  Edot_avg = prof['Edot']
  mdot_in = prof['mdot_in']


def get_RM_map(file_prefix):

  e_charge = 4.803e-10
  me = 9.109e-28
  cl = 2.997924e10
  mp = 1.6726e-24
  pc = 3.086e18
  kyr = 3.154e10
  msun = 1.989e33

  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar
  from yt.units.yt_array import YTArray

  muH_solar = 1./X_solar
  Z = 3. * Z_solar
  X = 0.
  mue = 2. /(1.+X)

  def _RM_integrand(field,data):
    ne= data["rho"].in_cgs()/mp/mue
    B_par = data["Bcc3"].in_cgs()
    return YTArray(np.array(ne * B_par * e_charge**3/(2.0*np.pi * me**2 * cl**4)),'cm**-3')

  ds.add_field(("gas","RM_integrand"),function = _RM_integrand,units="cm**-3",particle_type = False,sampling_type="cell",force_override=True)

  # prj = ds.proj('RM_integrand','z')
  # frb = prj.to_frb((0.5,'pc'),[1600,1600])
  global x_RM,y_RM,RM_map 

  # x_RM = np.array(frb['x'])
  # y_RM = np.array(frb['y'])
  # RM = np.array(frb['RM_integrand'].in_cgs())

  box_radius = 0.2
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,  #512
            (-box_radius,'pc'):(box_radius,'pc'):256j,   #512
            (-1,'pc'):(1,'pc'):1028j ]

  #box_radius = 1.0/2**5
  # region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,  #512
  #           (-box_radius,'pc'):(box_radius,'pc'):256j,   #512
  #           (-box_radius,'pc'):(box_radius,'pc'):256j]
  RM_map = np.array(region['RM_integrand'].mean(-1).in_cgs()) * 2 * pc



  c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = 'cubehelix',vmin=-1.5,vmax=1.5)
  # c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10((RM_map)),cmap = 'Reds',vmin=-1.5,vmax=1.5)
  # c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(-(RM_map)),cmap = 'Blues',vmin=-1.5,vmax=1.5)

  #c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = 'cubehelix',vmin=-0,vmax=3)

  circ = matplotlib.patches.Circle((dalpha[0],ddelta[0]),radius = .01,fill=False,ls='--',lw=3,color='white')
  matplotlib.pyplot.gca().add_artist(circ)


  plt.xlabel(r'$x$ (pc)',fontsize = 20)
  plt.ylabel(r'$y$ (pc)',fontsize = 20)

  cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
  cb.set_label(r"$ \log_{10}|RM| $ ",fontsize=17)


  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
      label.set_fontsize(10)
  plt.tight_layout()



def get_Xray_Lum(file_prefix,r_out,type="cylinder",make_image = False):
    mp_over_kev = 9.994827
    Z_o_X_solar = 0.0177
    Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
    X_solar = 0.7491
    Z_solar = 1.0-X_solar - Y_solar

    muH_solar = 1./X_solar
    Z = 3.0 * Z_solar
    X = 0.
    mue = 2. /(1.+X)
    mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
    mp = 8.41175e-58
    def Lam_func(TK):
        f1 = file_prefix + "_H_only.dat"
        f2 = file_prefix + "_He_only.dat"
        f3 = file_prefix + "_Metals_only.dat"
        data = np.loadtxt(f1)
        T_tab = data[:,0] * 1.16045e7
        Lam_H = data[:,1]
        data = np.loadtxt(f2)
        Lam_He = data[:,1]
        data = np.loadtxt(f3)
        Lam_metals = data[:,1]
        # T_tab = 10.**data[:,0]
        T_min = np.amin(T_tab)
        T_max = np.amax(T_tab)
        # if isinstance(TK,)
        # TK[TK<T_min] = T_min
        # TK[TK>T_max] = T_max
        # Lam_tab = 10.**data[:,1]

        Lam_tab = (X/X_solar) * Lam_H + ( (1.-X-Z)/Y_solar) * Lam_He + (Z/Z_solar) * Lam_metals
        from scipy.interpolate import InterpolatedUnivariateSpline
        Lam = InterpolatedUnivariateSpline(T_tab,Lam_tab,k = 1,ext =3)  #ext = 3 returns boundary value if outside range of T
        return Lam(TK)
    def _Lam_chandra(field,data):
        T_kev = data['gas','pressure']/data['gas','density'] * mu_highT*mp_over_kev
        T_K = T_kev*1.16e7
        nH = data['gas','density']/mp/muH_solar
        ne = data['gas','density']/mp/mue
        return Lam_func(T_K) * ne*nH 

    ds.add_field(('gas','Lam_chandra'),function = _Lam_chandra,units="code_mass**2/code_length**6",particle_type = False,sampling_type="cell",force_override=True)
    Lz =(ds.domain_right_edge-ds.domain_left_edge)[2]
    
    if (type=="cylinder"):
        ad = ds.disk("c",[0,0,1],(r_out,"pc"),Lz/2.)  #all_data()
    else:
        ad = ds.sphere("c",(r_out,"pc"))
    average_value = ad.quantities.weighted_average_quantity("Lam_chandra","cell_volume")
    average_value = yt.YTQuantity(average_value,"erg/s*cm**3 /pc**6")
    from yt.units import pc 
    box_volume = (ds.domain_right_edge-ds.domain_left_edge)[0]*(ds.domain_right_edge-ds.domain_left_edge
    )[1]*(ds.domain_right_edge-ds.domain_left_edge)[2]

    if (type=="cylinder"):
        volume = np.pi * ad.radius**2. * Lz
    else:
        volume = 4./3.*np.pi * ad.radius**3.

    average_value = (average_value * volume).in_cgs()

    if (make_image==True): #      proj = ds.proj("Lam_chandra", "z", weight_field="cell_volume")
      global x_im,y_im,image
      box_radius = 1.0
      region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,
                    (-box_radius,'pc'):(box_radius,'pc'):256j,
                    (-Lz/2.0,'pc'):(Lz/2.0,'pc'):256j ]
      image = region['Lam_chandra'].mean(-1)*Lz
      max = np.amax(image)
      image = image  #/np.amax(image)
      x_im = region['x'].mean(-1)
      y_im = region['y'].mean(-1)


    #res = 300
    #length_im = r_out
#frb = proj.to_frb((length_im,"pc"),[res,res])
      #image = frb['Lam_chandra']*Lz
      #x_im = np.linspace(-length_im/2.,length_im/2.,res)
      #y_im = np.linspace(-length_im/2.,length_im/2.,res)


    return average_value



def angle_avg(arr):
  return (arr*sin(theta)).mean(-1).mean(-1)/(sin(theta).mean(-1).mean(-1))

def compute_alpha():
  global sh
  global alpha

  Sh = (angle_avg(Lxdot_avg/s/(4.*pi*r**2.)) - angle_avg(vr_avg)*angle_avg(Lz_avg/s) )   # units of rho v^2
  alpha = Sh/angle_avg(press_tavg)
def make_l_histogram(r_in,r_out):
  global hist,l_hist,mdot_hist
  get_mdot()
  dx = x[1,0,0]-x[0,0,0]
  dy = y[0,1,0] - y[0,0,0]
  dz = z[0,0,1] - z[0,0,0]
  dV = dx*dy*dz
  r = np.sqrt(x**2.+ y**2.+z**2.)
  th = np.arccos(z/r)
  phi = np.arctan2(y,x)
  l = np.sqrt(l_x**2. + l_y**2. + l_z**2.)

  index = (r>=r_in)*(r<=r_out) #*(mdot<=0)
  vol = np.sum(dV*(r/r)[index])

  l_phi = np.arctan2(l_y,l_z)
  l_th = np.arccos(l_z/l)

  hist = plt.hist(th[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')
  hist = plt.hist(phi[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')

  #plt.hist(log10(l)[index].flatten(),weights=((-mdot*dV/vol/l)[index]).flatten(),bins=100,log=True,histtype='step')

  hist = plt.hist(l_th[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')
  hist = plt.hist(l_phi[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')


  hist = plt.hist(log10(abs(l))[index].flatten(),weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')
  sign = (l_x>0)*1 - (l_x<0)*1
  hist_x = plt.hist(log10(abs(l_x))[index].flatten(),weights=((-mdot*dV/vol*sign)[index]).flatten(),bins=100,histtype='step')
  sign = (l_y>0)*1 - (l_y<0)*1
  hist_y = plt.hist(log10(abs(l_y))[index].flatten(),weights=((-mdot*dV/vol*sign)[index]).flatten(),bins=hist_x[1],histtype='step')
  sign = (l_z>0)*1 - (l_z<0)*1
  hist_z = plt.hist(log10(abs(l_z))[index].flatten(),weights=((-mdot*dV/vol*sign)[index]).flatten(),bins=hist_x[1],histtype='step')

  hist_z = plt.hist(log10(abs(l))[index].flatten(),weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')

  l_hist = 10.**hist[1][1:]
  mdot_hist = hist[0]



 # hist_z[1][1:] are bins
 # hist_z[0] are y axis
  # l_var = hist_x[1]
  # dmdot = hist_x[0]**2. + hist_x[1]





def t_to_it(t_input):
  if (t_input <=t[0]):
    return 0
  if (t_input >=t[-1]):
    return t.shape[0]-1
  for it in range(t.shape[0]-1):
    if ( (t[it] <t_input) and (t_input<=t[it+1]) ):
      if (t_input/t[it]-1. < 1. - t_input/t[it+1]):
        return it
      else:
        return it+1
def cuadra_plot(t_in=None,r_in = None,type = "angular_momentum"):
  simulation_start_time = -1100
  if ( (type =="angular_momentum") and (t_in is None) ):
    print ("time must be specified to plot angular momentum")
    return
  if ( (type=="accretion_rate") and (r_in is None) ):
    r_in = 0.05 * arc_secs
  if ((type=="ancular_momentum_evolution") and (r_in is None)):
    r_in = 0.3 * arc_secs
  if (type == "angular_momentum"):
    it = t_to_it(t_in)
    plt.semilogx(r[0,:]/arc_secs,(L_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'l')
    plt.semilogx(r[0,:]/arc_secs,(Lx_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'lx',ls="-")
    plt.semilogx(r[0,:]/arc_secs,(Ly_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'ly',ls=":")
    plt.semilogx(r[0,:]/arc_secs,(Lz_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'lz',ls = "--")
    plt.xlim(.1,2)
    plt.ylim(-1,1)
    plt.xlabel(r"R [arcsecs]",fontsize = 20)
    plt.ylabel(r"Normalised Ang Mom",fontsize = 20)
    plt.legend(loc='best',fontsize = 15)
  elif (type == "accretion_rate"):
    ir = r_to_ir(r_in)
    plt.plot(t*1e3+simulation_start_time,-mdot_avg[:,ir]*1e3,lw = 2)
    plt.ylim(0,8)
    plt.xlim(-1000,100)
    plt.xlabel(r"Time [yr]",fontsize = 20)
    plt.ylabel(r"Acc. Rate [$10^{-6} M_{\rm sun}$ yr$^{-1}$]",fontsize = 20)
  elif (type =="angular_momentum_evolution"):
    ir_max = r_to_ir(r_in)
    ir_min = r_to_ir(0.05*arc_secs)
    plt.plot(t*1e3+simulation_start_time,(L_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'l',ls="-")
    plt.plot(t*1e3+simulation_start_time,(Lx_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'lx',ls="-")
    plt.plot(t*1e3+simulation_start_time,(Ly_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'ly',ls=":")
    plt.plot(t*1e3+simulation_start_time,(Lz_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'lz',ls="--")
    plt.xlim(-1000,100)
    plt.ylim(-0.5,0.5)
    plt.xlabel(r"Time [yr]",fontsize = 20)
    plt.ylabel(r"Normalised Ang Mom",fontsize = 20)
    plt.legend(loc='best',fontsize = 12)

def plot_cooling_test():
  rd_hst('star_wind.hst')
  plt.clf()
  gam = 5./3.
  p = E/8.*(gam-1.) # divide by box volume
  d = M/8.
  M_sun = 1.989e33
  pc = 3.086e+18
  mp_over_kev = 9.994827
  mp = 1.6726219e-24
  k_b = 1.380658e-16
  X = 0.7
  Z = 0.02
  muH = 1./X
  mue = 2./(1. + X)
  CUADRA_COOL=0
  if (CUADRA_COOL ==0):
    kT_kev_tab = np.array([8.61733130e-06,   8.00000000e-04,   1.50000000e-03,2.50000000e-03,   7.50000000e-03,   
      2.00000000e-02,3.10000000e-02,   1.25000000e-01,   3.00000000e-01,2.26000000e+00,   1.00000000e+02])
    Lam_tab = np.array([1.24666909e-27,   3.99910139e-26,   1.47470970e-22, 1.09120314e-22,   4.92195285e-22,   
      5.38853593e-22, 2.32144473e-22,   1.38278507e-22,   3.66863203e-23,2.15641313e-23,   9.73848346e-23])
    exp_cc = np.array([0.76546122,  13.06493514,  -0.58959508,   1.37120661, 0.09233853,  -1.92144798,  -0.37157016,
      -1.51560627,-0.26314206,   0.39781441,   0.39781441])
  else:
    kT_kev_tab = np.array([8.61733130e-06,   8.00000000e-04,   1.50000000e-03,2.50000000e-03,   7.50000000e-03,   
      2.00000000e-02,3.10000000e-02,   1.25000000e-01,   3.00000000e-01,2.26000000e+00,   1.00000000e+02])
    Lam_tab = np.array([    1.89736611e-19,   7.95699530e-21,   5.12446122e-21,
      3.58388517e-21,   1.66099838e-21,   8.35970776e-22,
      6.15118667e-22,   2.31779533e-22,   1.25581948e-22,
      3.05517566e-23,   2.15234602e-24])
    exp_cc = np.array([    -0.7,  -0.7,  -0.7,   -0.7,
      -0.7,  -0.7,  -0.7,  -0.7,
      -0.7,   -0.7,  -0.7])
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  T_kev = p/d * mu_highT*mp_over_kev
  T_K = T_kev * 1.16e7
  from scipy.interpolate import interp1d
  from scipy.interpolate import InterpolatedUnivariateSpline
  Lam = InterpolatedUnivariateSpline(kT_kev_tab,Lam_tab,k = 1)
  def Lam(T_kev):
    global k
    k = []

    for i in range(T_kev.size):
      for j in range(kT_kev_tab.size-1,-1,-1):
        if (T_kev[i] >= kT_kev_tab[j]):
          break
      k.append(j)
    
    return Lam_tab[k] * (T_kev/kT_kev_tab[k])** exp_cc[k]
  #Lam = s(x)
  #Lam = interp1d(kT_kev_tab,Lam_tab,fill_value = "extrapolate")
  rdhdf5(0)
  d_0 = rho[0][0] * M_sun/(pc**3.)
  T_0_kev = (press/rho * mu_highT*mp_over_kev)[0][0]
  T_0_K = T_0_kev * 1.16e7
  tcool = (muH * muH/mu_highT * mp *k_b * T_0_K / (gam-1.)/Lam(np.array([T_0_kev]))/d_0)[0][0]

  #(T) * (muH * muH) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
  kyr = 31557600000. 
  plt.loglog(t*kyr/tcool,T_K,lw=0,marker = 'o',label = 'small $\Delta t$')

  t_an = np.logspace(-4,2,1000)
  def dT_dt(T_dimensionless,t):
    T_kev = T_dimensionless * T_0_kev 
    return -Lam(T_kev)/Lam(np.array([T_0_kev]))[0][0]  
  from scipy.integrate import odeint

  T_an = odeint(dT_dt,1., t_an)*T_0_K
  plt.loglog(t_an,T_an,lw = 2,label = r'python odeint')
  plt.legend(loc='best')
  plt.xlabel(r'$t/t_{cool}$',fontsize = 25)
  plt.ylabel(r'$T$ (K)',fontsize = 25)
  plt.tight_layout()
  # alpha = -0.7
  # T_ref = 1e7/1.16e7
  # Lam_ref = 6e-23
  # Y_0 = 1./(1.-alpha) * (1. - (T_ref/T_0_kev)**(alpha-.1) )
  # def Y_inv(Y):
  #   return T_ref * (1-(1.-alpha)*Y)**(1./(1.-alpha))

  # T_an = Y_inv(Y_0 + T_0_kev/T_ref * Lam(np.array([T_ref]))[0] /Lam(np.array([T_0_kev]))[0]  * t_an)
  # T_an = (1.**(1.-alpha) - t_an * (1-alpha) )**(1./(1.-alpha))*T_0_K
class Star:
    eccentricity = 0.
    mean_angular_motion = 0.
    alpha = 0.
    beta = 0.
    tau = 0.
    gamma = 0.
    x10 = 0.
    x20 = 0.
    x30 = 0.
    v10 = 0.
    v20 = 0.
    v30 = 0.
    x1 = 0.
    x2 = 0.
    x3 = 0.
    v1 = 0.
    v2 = 0.
    v3 = 0.
    Mdot = 0.
    vwind = 0.
    radius = 0.
def rd_star_file(file):
    global gm_, simulation_start_time,Nstars
    global star_array
    global gd, mdot_star,vwind,ecc,x10,x20,x30,v10,v20,v30,alpha,beta,gamma,tau,mean_angular_motion
    global period, a,b,r_mean,nx1,nx2,nx3
    global mdot_tot
    star_array = []
    fin = open( file, "rb" )
    header = fin.readline().split()
    n = 0
    Nstars =int(header[n]); n+=1
    simulation_start_time =np.float64(header[n]); n+=1
    gm_ = np.float64(header[n]); n+=1

    if (n!=len(header)):
        print ('Extra or missing header variables')
        print (n, 'variables in script', len(header),  'variables in header')
    n=0
    gd = np.loadtxt( file,
                    dtype=np.float64,
                    skiprows=1,
                    unpack = True )

    tmp = gd[n]; n+=1
    mdot_star = gd[n]; n+=1
    mdot_tot = np.sum(mdot_star)
    vwind =gd[n]; n+=1
    x10 = gd[n]; n+=1
    x20 = gd[n]; n+=1
    x30 = gd[n]; n+=1
    v10 = gd[n]; n+=1
    v20 = gd[n]; n+=1
    v30 = gd[n]; n+=1
    alpha = gd[n]; n+=1
    beta = gd[n]; n+=1
    gamma = gd[n]; n+=1
    tau = gd[n]; n+=1
    mean_angular_motion = gd[n]; n+=1
    ecc = gd[n]; n+=1
    nx1 = gd[n]; n+=1
    nx2 = gd[n]; n+=1
    nx3 = gd[n]; n+=1
    period = 2.*np.pi/mean_angular_motion
    a = (gm_/mean_angular_motion**2.)**(1./3.)
    b = a * np.sqrt(1.- ecc**2.)
    r_mean = a * (1. + ecc**2./2.)  #mean radius of orbit in time
    n = 0

    gd=gd.transpose()
    for i in range(Nstars):
        star = Star()
        [tmp, star.Mdot, star.vwind, star.x10 ,star.x20 ,star.x30 ,star.v10 ,star.v20, star.v30 ,
         star.alpha, star.beta, star.gamma,star.tau ,star.mean_angular_motion, star.eccentricity,star.nx1,star.nx2,star.nx3] = gd[n]; n+=1
        star_array.append(star)

def resave_star_file(fname,VARY_MDOT=0,VARY_VWIND=0):
  rd_star_file(fname)
  if (VARY_MDOT==1):
    fname = fname + ".vary_mdot"
  if (VARY_VWIND==1):
    fname = fname + ".vary_vwind"
  else:
    print ("Nothing to do, quitting")
    return
  f = open(fname,"w")
  header = [str(Nstars), str(simulation_start_time), str(gm_)]
  f.write(" ".join(header) + "\n")
  for i in range(Nstars):
      star = star_array[i]
      if (VARY_MDOT==1):
          star.Mdot = star.Mdot/random.uniform(1.,6.)
      if (VARY_VWIND ==1):
          star.vwind = star.vwind/random.uniform(1,6.)
      star_list = [str(0), str(star.Mdot), str(star.vwind), str(star.x1) ,str(star.x2) ,str(star.x3) ,str(star.v1) ,str(star.v2), str(star.v3) ,
          str(star.alpha), str(star.beta), str(star.gamma) ,str(star.tau) ,str(star.mean_angular_motion), str(star.eccentricity)]
      f.write(" ".join(star_list) + "\n")

  f.close()
def get_R_orbit(t_vals):
  global R_orbit

  R_orbit = []

  for i_star in range(Nstars):
    def eqn(e_anomaly,m_anamoly):
      return m_anamoly - e_anomaly + ecc[i_star] * np.sin(e_anomaly)
    mean_anamoly = mean_angular_motion[i_star] * (t_vals -tau[i_star])
    eccentric_anomaly =  fsolve(eqn,mean_anamoly,args = (mean_anamoly,))

    R_orbit.append( a[i_star] * (1.-ecc[i_star]**2.)/ (1. + ecc[i_star]*np.cos(eccentric_anomaly)))
  R_orbit = np.array(R_orbit)

def plot_R_orbit(t_vals):
  plt.figure()
  for i in range(Nstars):
    plt.plot(t_vals*1e3 + simulation_start_time*1e3,R_orbit[i])


def rd_Lx():
  global Lx,t_arr
  file_list = glob.glob("Lx_*.npz")
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  Lx = np.zeros((len(file_list),2))
  for ifile in range(len(file_list)):
    rdnpz(file_list[ifile])
    t_arr[ifile] = t
    Lx[ifile,0] = Lx_1_5
    Lx[ifile,1] = Lx_10

def rd_RM(moving=False):
  global t_arr,RM_sgra,RM_pulsar,z,RM_pulsar_rand,DM_pulsar
  if (moving == True): file_list = glob.glob("RM_dump_moving*.npz")
  else: file_list = glob.glob("RM_dump_*.npz")
  keys = np.load(file_list[0]).keys()
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  RM_sgra = np.zeros(len(file_list))
  rdnpz(file_list[0])
  RM_pulsar = np.zeros((len(file_list),len(z_los)-1))
  RM_pulsar_rand = np.zeros((len(file_list),10))
  DM_pulsar = np.zeros((len(file_list),len(z_los)-1))

  for ifile in range(len(file_list)):
    rdnpz(file_list[ifile])
    t_arr[ifile] = t
    RM_sgra[ifile] = sgra_RM
    RM_pulsar[ifile] = pulsar_RM
    if "pulsar_RM_rand" in keys:
      RM_pulsar_rand[ifile] = pulsar_RM_rand
    if "pulsar_DM" in keys:
      DM_pulsar[ifile] = pulsar_DM
  z = z_los

def rd_RM_rel():
  global t_arr,r, rm1, rm2, rm3, rm1_, rm2_, rm3_
  file_list = glob.glob("RM_dump_*.npz")
  keys = np.load(file_list[0]).keys()
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  rm1 = np.zeros(len(file_list))
  rm2 = np.zeros(len(file_list))
  rm3 = np.zeros(len(file_list))

  rdnpz(file_list[0])
  rm1_ = np.zeros((len(file_list),len(r)))
  rm2_ = np.zeros((len(file_list),len(r)))
  rm3_ = np.zeros((len(file_list),len(r)))

  for ifile in range(len(file_list)):
    rdnpz(file_list[ifile])
    t_arr[ifile] = t
    rm1[ifile] = RM1
    rm2[ifile] = RM2
    rm3[ifile] = RM3

    rm1_[ifile] = RM1_
    rm2_[ifile] = RM2_
    rm3_[ifile] = RM3_
  
def rd_column_density(x_pos,y_pos):
  global cd_arr, t_arr
  file_list = glob.glob("column_density_z_*.npz")
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  cd_arr = np.zeros(len(file_list))
  n = 0
  rdnpz(file_list[0])
  def get_i(arg,arr):
    dx = np.diff(arr)[0]
    x_min = arr[0]
    return np.int(np.round((arg-x_min)/dx))

  ix = get_i(x_pos,x[:,0])
  iy = get_i(y_pos,y[0,:])



  for fname in file_list:
    rdnpz(fname)
    cd_arr[n] = column_density[ix,iy]
    t_arr[n] = t
    n = n + 1


def get_RM_gr():
  global ne,Bcc1_Gauss,Bcc2_Gauss, Bcc3_Gauss
  set_constants()
  rg = gm_/(cl/pc*kyr)**2

  def set_beta_1e2_units():
    global rho_unit,v_unit,P_unit,B_unit,mdot_unit
    rho_unit = 53939.4
    v_unit = 306.166
    P_unit = 5.05615e+09
    B_unit = 71106.6

  def set_beta_1e6_units():
    global rho_unit,v_unit,P_unit,B_unit,mdot_unit
    rho_unit = 29737.4
    P_unit = 2.78751e+09
    B_unit = 52796.9

  set_beta_1e2_units()
  ne = rho * rho_unit * rho_to_n_cgs/mue 
  Bcc1_Gauss = Bcc1 * B_unit * Bunit 
  Bcc2_Gauss = Bcc2 * B_unit * Bunit
  Bcc3_Gauss = Bcc3 * B_unit * Bunit
  r_cm = r * rg*pc
  dr_cm = np.gradient(r_cm[:,0,0])

  Rm = 1e4 * e_charge**3/(2*pi * me**2 * cl**4)



# figure(1)
# clf()
# loglog(r,rho_avg,lw=2,ls = '-',color= 'red',label = r'$\rho$')
# loglog(x[:,0],rho[:,0,0],lw = 2,ls = '--',color = 'red')
# loglog(r,T_avg*2.,lw=2,ls ='-',color = 'blue',label = r'$T$')
# loglog(x[:,0],(press/rho*2.)[:,0,0],lw = 2,ls = '--',color = 'blue')
# ylim(1e-2,1e1)
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)
# figure(2)
# clf()

# semilogx(r,vr_avg,lw=2,ls = '-',color= 'red',label = r'$v_r$')
# semilogx(x[:,0],vel1[:,0,0],lw = 2,ls = '--',color = 'red')
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)

# ylim(-2,1)

# figure(1)
# clf()
# loglog(r[0,:],rho_avg[2000:4000,:].mean(0),lw=2,ls = '-',color= 'red',label = r'$\rho$')
# loglog(x[:,0],rho[:,0,0],lw = 2,ls = '--',color = 'red')
# loglog(r[0,:],(p_avg/rho_avg)[2000:4000,:].mean(0)*2.,lw=2,ls ='-',color = 'blue',label = r'$T$')
# loglog(x[:,0],(press/rho*2.)[:,0,0],lw = 2,ls = '--',color = 'blue')
# ylim(1e-2,1e2)
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)
# figure(2)
# clf()

# semilogx(r[0,:],vr_avg[2000:4000,:].mean(0),lw=2,ls = '-',color= 'red',label = r'$v_r$')
# semilogx(x[:,0],vel1[:,0,0],lw = 2,ls = '--',color = 'red')
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)

# ylim(-1.5,1)

#nspace = 4
#figure(1)
#clf()
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin_smaller_radii/star_wind.hst')
#loglog(r[0,:],rho_avg[1579,:],lw=2,ls = '-',color= 'red',label = r'$\rho$')
#loglog(x[:,0],rho[:,0,0],lw = 2,ls = '--',color = 'red')
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin/star_wind.hst')
#loglog(r[0,:],rho_avg[1579,:],lw=2,ls = '-.',color= 'red')
##rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_all_angles_uniform/star_wind.hst')
#rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_uniform_2000_stars/star_wind.hst')
#
#loglog(r[0,::nspace],rho_avg[2648,::nspace],marker = 'o',lw=0,color= 'red')
#
#
#
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin_smaller_radii/star_wind.hst')
#loglog(r[0,:],(p_avg/rho_avg)[1579,:]*2.,lw=2,ls ='-',color = 'blue',label = r'$T$')
#loglog(x[:,0],(press/rho*2.)[:,0,0],lw = 2,ls = '--',color = 'blue')
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin/star_wind.hst')
#loglog(r[0,:],(p_avg/rho_avg)[1579,:]*2.,lw=2,ls ='-.',color = 'blue')
##rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_all_angles_uniform/star_wind.hst')
#rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_uniform_2000_stars/star_wind.hst')
#
#loglog(r[0,::nspace],(p_avg/rho_avg)[2648,::nspace]*2.,lw=0,marker ='o',color = 'blue')
#
#
#ylim(1e-2,1e2)
#xlabel(r'$r$',fontsize = 25)
#plt.legend(loc='best',frameon=0,fontsize = 20)
##axvline(rbound,color = 'black',lw = 4)
#figure(2)
#clf()
#
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin_smaller_radii/star_wind.hst')
#semilogx(r[0,:],vr_avg[1579,:],lw=2,ls = '-',color= 'red',label = r'$v_r$')
#semilogx(x[:,0],vel1[:,0,0],lw = 2,ls = '--',color = 'red')
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin/star_wind.hst')
#semilogx(r[0,:],vr_avg[1579,:],lw=2,ls = '-.',color= 'red')
##rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_all_angles_uniform/star_wind.hst')
#rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_uniform_2000_stars/star_wind.hst')
#
#semilogx(r[0,::nspace/2],vr_avg[2648,::nspace/2],lw=0,marker = 'o',color= 'red')
#
#
#
#
#xlabel(r'$r$',fontsize = 25)
#plt.legend(loc='best',frameon=0,fontsize = 20)
##axvline(rbound,color = 'black',lw = 4)
#
#ylim(-1.5,1)



# set_constants()
# rbound = 10./128./2.**6.*2./arc_secs
# figure(1)
# clf()
# rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_smaller_radii/star_wind.hst')
# loglog(r[0,:]/arc_secs,rho_avg[650:750,:].mean(0)*rho_to_n_cgs,lw=2,ls = '-',color= 'red',label = r'3D Sim')
# loglog(x[:,0]/arc_secs,rho[:,0,0]*rho_to_n_cgs,lw = 2,ls = '--',color = 'red',label = r'1D Calc')


# rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_smaller_radii/star_wind.hst')
# loglog(r[0,:]/arc_secs,(p_avg[650:750,:].mean(0)/rho_avg[650:750,:].mean(0)*mp_over_kev)*2.,lw=2,ls ='-',color = 'blue')
# loglog(x[:,0]/arc_secs,(press/rho*2.*mp_over_kev)[:,0,0],lw = 2,ls = '--',color = 'blue')

# ylabel(r'$T$ (keV)     $n_e$ (cm$^{-3}$)',fontsize = 25)

# text(1,2,r'$T$',rotation=0,fontsize=20,color='blue')
# text(1,.6e3,r'$n_e$',rotation=0,fontsize=20,color='red')



# xlim(rbound*.9,2e2)

# ylim(5e-2,1e4)
# xlabel(r'$r$ (arcsecs)',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# leg = plt.gca().get_legend()
# leg.legendHandles[0].set_color('black')
# leg.legendHandles[1].set_color('black')
# plt.draw()
# axvline(rbound,color = 'black',lw = 4)
# plt.setp(plt.gca().get_xticklabels(), fontsize=20)
# plt.setp(plt.gca().get_yticklabels(), fontsize=20)
# plt.tight_layout()
# plt.savefig("rho_T_rand_stars.pdf")
# figure(2)
# clf()

# rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_smaller_radii/star_wind.hst')
# semilogx(r[0,:]/arc_secs,vr_avg[650:750,:].mean(0)/(1000.*km_per_s),lw=2,ls = '-',color= 'red',label = r'3D Sim')
# semilogx(x[:,0]/arc_secs,vel1[:,0,0]/(1000.*km_per_s),lw = 2,ls = '--',color = 'red',label = r'1D Calc')



# ylabel(r'$v_r/v_{wind}$',fontsize = 25)


# xlabel(r'$r$ (arcsecs)',fontsize = 25)
# axvline(rbound,color = 'black',lw = 4)
# plt.legend(loc='best',frameon=0,fontsize = 20)

# leg = plt.gca().get_legend()
# leg.legendHandles[0].set_color('black')
# leg.legendHandles[1].set_color('black')
# plt.setp(plt.gca().get_xticklabels(), fontsize=20)
# plt.setp(plt.gca().get_yticklabels(), fontsize=20)
# plt.draw()

# xlim(rbound*.9,2e2)

# ylim(-1.5,1)
# plt.tight_layout()
# plt.savefig("vr_rand_stars.pdf")



# figure(1)
# clf()
# loglog(r[0,:],rho_avg[110,:],lw=2,ls="-",label = r'$\rho$',color = 'black')
# loglog(r_high[0,:],rho_high[110,:],lw=2,ls="--",color = 'black')
# loglog(r[0,:],p_avg[110,:],lw=2,ls="-",label = r'$P$',color = 'blue')
# loglog(r_high[0,:],p_high[110,:],lw=2,ls="--",color = 'blue')
# loglog(r[0,:],abs(vr_avg[110,:]),lw=2,ls="-",label = r'$|v_r|$',color = 'red')
# loglog(r_high[0,:],abs(vr_high[110,:]),lw=2,ls="--",color = 'red')
# loglog(r[0,:],abs((mdot_avg)[110,:]),lw=2,ls="-",label = r'$\dot M$',color = 'green')
# loglog(r_high[0,:],abs(mdot_high[110,:]),lw=2,ls="--",color = 'green')
# xlabel(r'$r$ (pc)',fontsize = 25)
# plt.legend(loc='best',frameon=0)
# plt.tight_layout()

def compare_two_hsts(f1,f2,t1 = 1.1,t2 = 2.2,t0_1 = -1.1,t0_2 = -2.2,lab1=r"absorbing",lab2=r"none"):
  # f1 = "/global/scratch/smressle/star_cluster/cuadra_comp/min_ecc_run/star_wind.hst"
  # f2 = "/global/scratch/smressle/star_cluster/cuadra_comp/min_ecc_run_t0_2.2/star_wind.hst"
  figure(1)
  plt.clf()
  rd_hst(f1)
  it1 = t_to_it(t1)
  loglog(r[0,:],rho_avg[it1-3:it1+3,:].mean(0),lw=2,ls="-",label = r'$\rho$',color = 'black')
  loglog(r[0,:],np.sqrt(5./3.*p_avg[it1-3:it1+3,:].mean(0)/rho_avg[it1-3:it1+3,:].mean(0)),lw=2,ls="-",label = r'$c_s$',color = 'blue')
  loglog(r[0,:],abs((mdot_avg[it1-3:it1+3,:].mean(0)/rho_avg[it1-3:it1+3,:].mean(0)/(r**2.)[it1-3:it1+3,:].mean(0)/np.pi/4.)),lw=2,ls="-",label = r'$|v_r|$',color = 'red')
  loglog(r[0,:],abs((mdot_avg[it1-3:it1+3,:].mean(0))),lw=2,ls="-",label = r'$\dot M$',color = 'green')
  rd_hst(f2)
  it2 = t_to_it(t2)
  loglog(r[0,:],rho_avg[it2-3:it2+3,:].mean(0),lw=2,ls="--",color = 'black')
  loglog(r[0,:],np.sqrt(5./3.*p_avg[it2-3:it2+3,:].mean(0)/rho_avg[it2-3:it2+3,:].mean(0)),lw=2,ls="--",color = 'blue')
  loglog(r[0,:],abs((mdot_avg[it2-3:it2+3,:].mean(0)/rho_avg[it2-3:it2+3,:].mean(0)/(r**2.)[it2-3:it2+3,:].mean(0)/np.pi/4.)),lw=2,ls="--",color = 'red')
  loglog(r[0,:],abs((mdot_avg[it2-3:it2+3,:].mean(0))),lw=2,ls="--",color = 'green')

  xlabel(r'$r$ (pc)',fontsize = 25)
  plt.legend(loc='best',frameon=0)
  plt.tight_layout()


  figure(2)
  plt.clf()
  rd_hst(f1)
  l_kep = np.sqrt(gm_*r[0,:])
  plt.semilogx(r[0,:],abs(L_avg[it1-3:it1+3,:].mean(0)/rho_avg[it1-3:it1+3,:].mean(0))/l_kep,lw = 2, label = r'$l/l_{kep}$',color = 'red')
  # plt.semilogx(r[0,:],abs(Lx_avg/rho_avg)[it1-3:it1+3,:].mean(0),lw = 2, label = r'$l_x$',ls="-",color = 'blue')
  # plt.semilogx(r[0,:],abs(Ly_avg/rho_avg)[it1-3:it1+3,:].mean(0),lw=2, label = r'$l_y$',ls="-",color = 'green')
  # plt.semilogx(r[0,:],abs(Lz_avg/rho_avg)[it1-3:it1+3,:].mean(0),lw = 2, label = r'$l_z$',ls = "-",color = 'red')

  rd_hst(f2)
  l_kep = np.sqrt(gm_*r[0,:])
  plt.semilogx(r[0,:],abs(L_avg[it2-3:it2+3,:].mean(0)/rho_avg[it2-3:it2+3,:].mean(0))/l_kep,lw = 2,ls = "--",color = 'black')
  # plt.semilogx(r[0,:],abs(Lx_avg/rho_avg)[it2-3:it2+3,:].mean(0),lw = 2,ls="--",color = 'blue')
  # plt.semilogx(r[0,:],abs(Ly_avg/rho_avg)[it2-3:it2+3,:].mean(0),lw=2,ls="--",color = 'green')
  # plt.semilogx(r[0,:],abs(Lz_avg/rho_avg)[it2-3:it2+3,:].mean(0),lw = 2,ls = "--",color = 'red')
  plt.yscale('log')
  plt.xlabel(r'$r$ (pc)',fontsize = 25)
  plt.ylabel(r'$l/l_{kep}$',fontsize = 25)
  #plt.legend(loc='best',frameon=0)
  plt.tight_layout()


  figure(3)
  plt.clf()
  rd_hst(f1)
  ir1 = r_to_ir(0.05*arc_secs)
  plt.plot(t*1e3 + t0_1*1e3,-mdot_avg[:,ir1]*1e3,lw=2,ls = "-",color = 'red',label = lab1)
  rd_hst(f2)
  ir2 = r_to_ir(0.05*arc_secs)
  plt.plot(t*1e3 + t0_2*1e3,-mdot_avg[:,ir2]*1e3,lw =2,ls = "--",color = 'black',label = lab2)
  plt.ylim(0,8)
  plt.xlim(min(t0_2,t0_1)*1e3,100)
  plt.xlabel(r"Time [yr]",fontsize = 20)
  plt.ylabel(r"Acc. Rate [$10^{-6} M_{\rm sun}$ yr$^{-1}$]",fontsize = 20)
  plt.legend(loc = 'best',fontsize = 15,frameon=0)
  plt.tight_layout()

def plot_keplerian_torus(r_out,frac_kep):
    fig = plt.figure(1)
    fig.clf()

    q = 3./2.
    
    def v_phi(s):
        return frac_kep*np.sqrt(gm_/s)
    # def Omega(s):
    #     return np.sqrt(gm_/R_pmax**3.) * (s/R_pmax)**(-q)
    # def v_phi(s):
    #     return np.sqrt(gm_/R_pmax) * (s/R_pmax)**(-q+1.)

    global rho,C,RHS
    global rvals,xvals,zvals;
    rvals = np.logspace(np.log10(0.01*r_out),np.log10(2*r_out))
    thvals = np.linspace(0.0001,np.pi-0.0001)
    rvals,thvals = meshgrid(rvals,thvals)
    zvals = np.cos(thvals)*rvals
    xvals = np.sin(thvals)*rvals
    svals = rvals * np.sin(thvals)
    kappa = 1.

    C = -gm_/r_out + v_phi(r_out)**2./(2.*q-2.)
    #C = -gm_/rvals + l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
        
        
    gam =5./3.
    gm1 = gam-1.

    #RHS = C + gm_/rvals - l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
    RHS =C + gm_/rvals - v_phi(svals)**2./(2.*q-2.)
    rho = (RHS*gm1/gam/kappa)**(1./gm1)
    rho[isnan(rho)] = 0

    plt.contourf(xvals,zvals,log10(rho),isfilled=1,levels = np.linspace(log10(np.amax(rho))-5.,log10(np.amax(rho))+.1,200))
    plt.colorbar()


def plot_torus(r_out, R_pmax,q=2.):
    fig = plt.figure(1)
    fig.clf()
    l_kep = np.sqrt(gm_*R_pmax)
    
    # def v_phi(s):
    #     return frac_kep*np.sqrt(gm_/s)
    def Omega(s):
        return np.sqrt(gm_/R_pmax**3.) * (s/R_pmax)**(-q)
    def v_phi(s):
        return np.sqrt(gm_/R_pmax) * (s/R_pmax)**(-q+1.)

    global rho,C,RHS
    global rvals,xvals,zvals,thvals,svals;
    rvals = np.logspace(np.log10(0.01*r_out),np.log10(10*r_out),200)
    thvals = np.linspace(0.0001,np.pi-0.0001,200)
    rvals,thvals = meshgrid(rvals,thvals)
    zvals = np.cos(thvals)*rvals
    xvals = np.sin(thvals)*rvals
    svals = rvals * np.sin(thvals)
    kappa = 1.

    C = -gm_/r_out + v_phi(r_out)**2./(2.*q-2.)
    #C = -gm_/rvals + l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
        
        
    gam =5./3.
    gm1 = gam-1.

    #RHS = C + gm_/rvals - l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
    RHS =C + gm_/rvals - v_phi(svals)**2./(2.*q-2.)
    rho = (RHS*gm1/gam/kappa)**(1./gm1)
    rho[isnan(rho)] = 0

    RHS_max = C + gm_/R_pmax - v_phi(R_pmax)**2./(2.*q-2.)
    rho_max = (RHS_max*gm1/gam/kappa)**(1./gm1)

    plt.contourf(xvals,zvals,log10(rho),isfilled=1,levels = np.linspace(log10(np.amax(rho))-6.,log10(np.amax(rho))+.1,200))
    plt.colorbar()


def plot_rotating_star(r_star,rho_star=1.0):
    Omega = 0.5*np.sqrt(gm_/r_star**3)
    global rho,C,RHS
    global rvals,xvals,zvals,thvals,svals;
    rvals = np.logspace(np.log10(0.01*r_star),np.log10(10*r_star),200)
    thvals = np.linspace(0.0001,np.pi-0.0001,200)
    rvals,thvals = meshgrid(rvals,thvals)
    zvals = np.cos(thvals)*rvals
    xvals = np.sin(thvals)*rvals
    svals = rvals * np.sin(thvals)
    kappa = 1.

    gam = 5.0/3.0
    gm1 = gam-1.0
    ##C = gam/(gm1) * kappa * rho_star**(gam-1.0)
    C = -gm_/r_star

    RHS =C + gm_/rvals + Omega**2 * svals**2/2.0
    rho = (RHS*gm1/gam/kappa)**(1./gm1)
    plt.contourf(xvals,zvals,log10(rho),isfilled=1,levels = np.linspace(log10(np.nanmax(rho))-6.,log10(np.nanmax(rho))+.1,200))


def reshape_arrays():
  global r,th,ph,rho,press,vel1,vel2,vel3 
  r = r.repeat(4).reshape(nx,ny,4) *1.0
  th = th.repeat(4).reshape(nx,ny,4) *1.0
  ph =  ph.repeat(4).reshape(nx,ny,4) *1.0
  rho = rho.repeat(4).reshape(nx,ny,4) *1.0
  press = press.repeat(4).reshape(nx,ny,4) *1.0
  vel1 = vel1.repeat(4).reshape(nx,ny,4) * 1.0
  vel2 = vel2.repeat(4).reshape(nx,ny,4) * 1.0
  vel3 = vel3.repeat(4).reshape(nx,ny,4) * 1.0
def rescale_prim_for_inits(axisym=False):
    def fold_theta(arr,anti=False):
        fac = 1
        if (anti ==True): fac = -1
        return arr[:,::-1,:]/2.0 * fac + arr[:,:,:]/2.0
#        return arr[:,::-1,:]*(th_tavg<=pi/2.)*fac + arr[:,:,:]*(th_tavg>pi/2.)
    def axisymmetrize(arr):
        return arr.mean(-1).repeat(arr.shape[-1]).reshape(arr.shape)
    global r,th,ph,rho,press,vel1,vel2,vel3
    rg = 2.058e-7 #pc
    rin_9 = 2.*2./128./2**9.0
    #rg = rin_9
    r = r_tavg * (rg/rin_9)
    th = th_tavg
    ph = ph_tavg
    press = fold_theta(press_tavg * (rin_9/rg)**2.0)
    rho = fold_theta(rho_tavg * (rin_9/rg))
    vel1 = fold_theta(vr_avg * (rin_9/rg)**0.5)
    vel2 = fold_theta(vth_avg * (rin_9/rg)**0.5,anti=True)
    vel3 = fold_theta(vphi_avg * (rin_9/rg)**0.5)

    if (axisym==True):
      press = axisymmetrize(press)
      rho = axisymmetrize(rho)
      vel1 = axisymmetrize(vel1)
      vel2 = axisymmetrize(vel2)
      vel3 = axisymmetrize(vel3)
      #rand = np.random.uniform(0,1,size=rho.shape)
      #press = press * (1. + 4.0e-2*(rand-0.5))


def read_vector_potential(idump,low_res=False,spherical=True,th = 0, ph = 0,nr=356,nth=128,nphi=128):
  rd_yt_convert_to_spherical(idump,MHD=False,th=th,ph=ph,omega_phi = None,
    dump_name=None,low_res=low_res,nr=nr,nth=nth,nphi=nphi,double_precision=True)
  global A1,A2,A3, Ar, Ath,Aphi
  A1 = vel1
  A2 = vel2
  A3 = vel3 

  if (spherical==True):
    get_mdot(False)
    Ar = vr 
    Ath = vth
    Aphi = vphi


def make_athena_inits(fname,spherical=False,gr=False,mhd=False, vector_potential=False,electrons=False):
  if (spherical==False): data = [x.transpose(),y.transpose(),z.transpose(),rho.transpose(),vel1.transpose(),vel2.transpose(),vel3.transpose(),press.transpose()]
  elif (gr==False): data = [r.transpose(),th.transpose(),ph.transpose(),rho.transpose(),vel1.transpose(),vel2.transpose(),vel3.transpose(),press.transpose()]
  else: data = [r.transpose(),th.transpose(),ph.transpose(),rho.transpose(),vr.transpose(),vth.transpose(),vphi.transpose(),press.transpose()]

  if (mhd==True): 
    if (gr==False): data = data + ([Bcc1.transpose(),Bcc2.transpose(),Bcc3.transpose()])
    else: data = data + ([Br.transpose(),Bth.transpose(),Bphi.transpose()])

  if (vector_potential==True): 
    if (gr==False): data = data + ([np.array(A1.transpose()), np.array(A2.transpose()), np.array(A3.transpose())])
    else: data = data + ([np.array(Ar.transpose()), np.array(Ath.transpose()), np.array(Aphi.transpose())])

  n_electrons = 0
  if (electrons==True):
    n_electrons = 1
    data = data + ([np.array(ke_ent.transpose())])
    try:
      ke_ent2
    except NameError:
      print("ke_ent2 not defined")
    else:
      data = data + ([np.array(ke_ent2.transpose())])
      n_electrons += 1
    try:
      ke_ent3
    except NameError:
      print("ke_ent3 not defined")
    else: 
      data = data + ([np.array(ke_ent3.transpose())])
      n_electrons += 1
  data = np.array(data)
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]
  header = [np.str(nx),np.str(ny),np.str(nz)]
  if (electrons==True): header = header + ([np.str(n_electrons)])

  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()


def make_mhd_restart_file(path_to_sim_data,path_to_vector_potential,idump_sim,idump_vector_potential,fname = "vector_potential_dump",electrons=False):
  global r,th,phi,A1,A2,A3,vel1,vel2,vel3,rho,press
  nr = 320
  nth=200
  nphi = 400
  os.chdir(path_to_vector_potential)
  read_vector_potential(idump_vector_potential,spherical=False,nr=nr,nth=nth,nphi=nphi,low_res=True)

  os.chdir(path_to_sim_data)
  rd_yt_convert_to_spherical(idump_sim,MHD=True,nr=nr,nth=nth,nphi=nphi,low_res=True,double_precision=True)
  make_athena_inits(fname,spherical=True,gr=False,mhd=True, vector_potential=True,electrons=electrons)

def make_grmhd_restart_file(path_to_sim_data,path_to_vector_potential,idump_sim,idump_vector_potential,fname = "gr_vector_potential_dump",
  low_res=False,th_tilt=0,ph_tilt = 0,electrons=False):
  global r,th,phi,A1,A2,A3,vel1,vel2,vel3,rho,press
  nr = 320
  nth=200
  nphi = 400

  # low_res = True
  # nr = 32https://aka.ms/dotnet/8.0/windowsdesktop-runtime-win-x64.exe
  # nth = 32
  # nphi = 32
  os.chdir(path_to_vector_potential)
  read_vector_potential(idump_vector_potential,spherical=False,nr=nr,nth=nth,nphi=nphi,low_res=True,th=th_tilt,ph=ph_tilt)

  os.chdir(path_to_sim_data)
  rd_yt_convert_to_spherical(idump_sim,MHD=True,nr=nr,nth=nth,nphi=nphi,low_res=True,th=th_tilt,ph=ph_tilt,double_precision=True)
  make_athena_inits(fname,spherical=True,gr=False,mhd=True, vector_potential=True,electrons=electrons)

def make_grmhd_restart_file_mks(path_to_sim_data,path_to_vector_potential,idump_sim,idump_vector_potential,fname = "gr_vector_potential_dump",
  low_res=False):
  global r,th,ph, phi,A1,A2,A3,vel1,vel2,vel3,rho,press
  nr = 320
  nth=200
  nphi = 400

  # low_res = True
  # nr = 32
  # nth = 32
  # nphi = 32
  os.chdir(path_to_vector_potential)
  read_vector_potential(idump_vector_potential,spherical=True,nr=nr,nth=nth,nphi=nphi,low_res=True,th =1.27542452,ph= -1.74037675)

  os.chdir(path_to_sim_data)
  rd_yt_convert_to_spherical(idump_sim,MHD=True,nr=nr,nth=nth,nphi=nphi,low_res=True,th =1.27542452,ph= -1.74037675)
  th = theta
  ph = phi

  get_mdot(mhd=True)
  make_athena_inits(fname,spherical=True,gr=True,mhd=True, vector_potential=True)


def rd_athena_inits(fname):
  global r, th, phi, rho, vr, vth, vphi, press, Ar, Ath,Aphi
  global Br,Bth,Bphi
  global header
  f = open(fname,"rb")
  header = f.readline().split()
  nx = int(header[0])
  ny = int(header[1])
  nz = int(header[2])
  body = np.fromfile(f,dtype=np.float64,count=-1)
  gd = body.view().reshape((-1,nx,ny,nz),order="F")

  gd = (gd.transpose(0,3,2,1))
  f.close()


  r = gd[0].transpose()
  th = gd[1].transpose()
  phi = gd[2].transpose()
  rho = gd[3].transpose()
  vr = gd[4].transpose()
  vth = gd[5].transpose()
  vphi = gd[6].transpose()
  press = gd[7].transpose()
  Br = gd[8].transpose()
  Bth = gd[9].transpose()
  Bphi = gd[10].transpose()
  Ar = gd[11].transpose()
  Ath = gd[12].transpose()
  Aphi = gd[13].transpose()

def rd_binary_orbits(fname):
  global t_array,x1_array,y1_array,z1_array,x2_array,y2_array,z2_array
  global a1x_array,a1y_array,a1z_array,a2x_array,a2y_array,a2z_array
  global v1x_array,v1y_array,v1z_array,v2x_array,v2y_array,v2z_array
  global header
  global nt, q
  f = open(fname,"rb")
  header = f.readline().split()
  nt = int(header[0])
  q = double(header[1])
  body = np.fromfile(f,dtype=np.float64,count=-1)
  gd = body.view().reshape((-1,nt),order="F")

  gd = (gd.transpose(0,1))
  f.close()


  t_array = gd[0].transpose()
  x1_array = gd[1].transpose()
  y1_array = gd[2].transpose()
  z1_array = gd[3].transpose()
  x2_array = gd[4].transpose()
  y2_array = gd[5].transpose()
  z2_array = gd[6].transpose()
  a1x_array = gd[7].transpose()
  a1y_array = gd[8].transpose()
  a1z_array = gd[9].transpose()
  a2x_array = gd[10].transpose()
  a2y_array = gd[11].transpose()
  a2z_array = gd[12].transpose()
  v1x_array = gd[13].transpose()
  v1y_array = gd[14].transpose()
  v1z_array = gd[15].transpose()
  v2x_array = gd[16].transpose()
  v2y_array = gd[17].transpose()
  v2z_array = gd[18].transpose()
def make_athena_inits_wrapper():
  os.chdir("/global/scratch/smressle/star_cluster/poisson_from_restart/beta_1e2_t_120")
  read_vector_potential(123,spherical=True,th = 1.3,ph=-1.8)
  os.chdir("/global/scratch/smressle/star_cluster/test_new_code/without_S2_9_levels_beta_1e2_v3_orbits_comet") 
  rdnpz("dump_spher_120_th_1.3_phi_-1.8.npz")
  get_mdot(mhd=True)
  make_athena_inits("init_beta_1e2_120_Aphi.init",spherical = True,gr=True,mhd=True,vector_potential=True)

def rd_binary(fname,nx,ny,nz):
  global gd,header
  f = open(fname,"rb")
  header = f.readline()
  body = np.fromfile(f,dtype=np.float64,count=-1)
  gd = body.view().reshape((-1,nz,ny,nx),order="F")
  gd = (gd.transpose(0,3,2,1))
  f.close()
def rdnpz(file):
    dic = np.load(file)
    for key in dic.keys():
        exec("globals()['%s'] = dic['%s']" % (key,key))
def rd_yt_convert_to_spherical(idump,MHD=False,th=0,ph=0,omega_phi = None,dump_name=None,low_res =False,
  method='nearest',fill_value = 0.0,gr=False,ax=0.0,ay=0.0,az=0.0,ISOTHERMAL = False, nr = 356, nth = 128,nphi=128,
  double_precision=False,slice = False,rmin=None,rmax=None,midplane_slice = False,q=0.0,rbh2=60.0,aprime=0.0,uov=False):
  a = np.sqrt(ax**2.0+ay**2.0+az**2.0)
  if (dump_name is None): 
    if (low_res == True): dump_name = "dump_spher_%d_th_%.2g_phi_%.2g_low_res.npz" %(idump,th,ph)
    elif (slice==True): dump_name = "dump_spher_%d_th_%.2g_phi_%.2g_slice.npz" %(idump,th,ph)
    elif (midplane_slice==True): dump_name = "dump_spher_%d_th_%.2g_phi_%.2g_midplane_slice.npz" %(idump,th,ph)
    else: dump_name = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(idump,th,ph)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    yt_load(idump,gr=gr,a=a)
    if (omega_phi is not None):
      ph = ph + omega_phi * ds.current_time/ds.time_unit
      dump_name = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(idump,th,ph)
    global r,phi,theta,xi,yi,zi
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t 
    global k_ent,ke_ent,ke_ent2,ke_ent3
    global divb
    # ax = 0
    # ay = 0 
    # az = a 
    #unit vectors for the new coordinate system in the frame of the old coordinates
    z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])   #r
    x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  #theta
    y_hat = np.array([-sin(ph),cos(ph),0])                        #phi

    #unit vectors for original coordinate system in the frame of the new coordinates
    #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
    x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #in terms of [theta_hat, phi_hat, r_hat] 
    y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    z_hat_prime = [-np.sin(th),0,np.cos(th)]

    index = np.arange(ds.r['density'].shape[0])

    def r_func(x,y,z,ax,ay,az):
        def SQR(var):
          return var**2.0

        a = np.sqrt(ax**2.0+ay**2.0+az**2.0)
        a_dot_x = ax * x + ay * y + az * z;

        R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
        return np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a_dot_x) )  )/np.sqrt(2.0);

    # if (rmin is None): rmin = np.amin(r_func(np.array(ds.r['x']),np.array(ds.r['y']),np.array(ds.r['z']),ax,ay,az))
    # if (rmax is None): rmax = np.amax(r_func(np.array(ds.r['x']),np.array(ds.r['y']),np.array(ds.r['z']),ax,ay,az))

    if (rmin is None and gr==True): rmin = 0.5
    if (rmax is None and gr==True): rmax = 1e3

    if (rmin is None and gr==False): rmin = np.amin(ds.r['x']**2+ds.r['y']**2+ds.r['z']**2)
    if (rmax is None and gr==False): rmax = np.amax(ds.r['x']**2+ds.r['y']**2+ds.r['z']**2)
 
    # r_extrema = ds.all_data().quantities.extrema("r")
    # if (rmin is None): rmin =r_extrema[0]*2. 
    # if (rmax is None): rmax = r_extrema[1]*.9

    #faces
    theta = np.linspace(0.,np.pi,nth+1)
    phi = np.linspace(0,2.*np.pi,nphi+1)

    if (slice==True): phi = np.array([0.0,np.pi])
    if (midplane_slice==True): theta = np.array([np.pi/2.0])

    #centers
    r = np.logspace(log10(rmin),log10(rmax),nr)

    if (midplane_slice==False):
      dth = np.diff(theta)[0]
      theta = (theta + dth/2.0)[:-1]
    if (slice==False):
      dphi = np.diff(phi)[0]
      phi = (phi + dphi/2.0)[:-1]
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')



    ##new x,y,z coords in terms of new r,th,phi coords
    if (gr==True):

      [xi_prime,yi_prime,zi_prime] = convert_spherical_to_cartesian_ks(r,theta,phi,ax,ay,az)
      # xi_prime = r*np.cos(phi)*np.sin(theta) - a*np.sin(phi)*np.sin(theta)
      # yi_prime = r*np.sin(phi)*np.sin(theta) + a*np.cos(phi)*np.sin(theta)
      # zi_prime = r*np.cos(theta)
    else:
      xi_prime = r*np.cos(phi)*np.sin(theta) 
      yi_prime = r*np.sin(phi)*np.sin(theta) 
      zi_prime = r*np.cos(theta)

    #r_vec = x_new x_hat_new + y_new y_hat_new + z_new z_hat_new  
    # x_old = r_vec dot x_hat_old 
    #original x,y,z coords in terms of new coords
    if (gr==False):
      xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
      yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
      zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2]
    else: #assume cooordinate system aligned with spin of black hole
      xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
      yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
      zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2] 
    new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = method,fill_value = fill_value).astype(np.int64)


    from yt.units import pc, msun,kyr
    rho = ds.r['density'][new_index] * pc**3/msun
    if (ISOTHERMAL == False): press = ds.r['press'][new_index] * pc**3/msun * kyr**2/pc**2
    vel1 = ds.r['vel1'][new_index] * kyr/pc
    vel2 = ds.r['vel2'][new_index] * kyr/pc
    vel3 = ds.r['vel3'][new_index] * kyr/pc

    global uu,bu,bsq
    uu = [0,0,0,0]
    bu = [0,0,0,0]

    if (gr==True):
        cks_metric(xi,yi,zi,ax,ay,az)
        tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
        gamma = np.sqrt(1.0 + tmp);

        # Calculate 4-velocity
        #invert_metric(g)
        cks_inverse_metric(xi,yi,zi,ax,ay,az)
        alpha = np.sqrt(-1.0/gi[0,0]);
        uu[0] = gamma/alpha;
        uu[1] = vel1 - alpha * gamma * gi[0,1];
        uu[2] = vel2 - alpha * gamma * gi[0,2];
        uu[3] = vel3 - alpha * gamma * gi[0,3];
      # uu[0] = ds2.r['user_out_var1'][new_index]
      # uu[1] = ds2.r['user_out_var2'][new_index]
      # uu[2] = ds2.r['user_out_var3'][new_index]
      # uu[3] = ds2.r['user_out_var4'][new_index]
        uu = np.array(uu)

        uu_tmp = uu*1.0

    
    vx_tmp = vel1
    vy_tmp = vel2
    vz_tmp = vel3 


    #new coords in terms of new r,th,phi
    if (gr==True):
      [x,y,z] = convert_spherical_to_cartesian_ks(r,theta,phi,ax,ay,az)
      # x = r*np.cos(phi)*np.sin(theta) - a*np.sin(phi)*np.sin(theta)
      # y = r*np.sin(phi)*np.sin(theta) + a*np.cos(phi)*np.sin(theta)
      # z = r*np.cos(theta)
    else:
      x = r*np.cos(phi)*np.sin(theta) 
      y = r*np.sin(phi)*np.sin(theta) 
      z = r*np.cos(theta)

    if (MHD==True):
        B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
        Bcc1 = ds.r['Bcc1'][new_index]/B_unit
        Bcc2 = ds.r['Bcc2'][new_index]/B_unit
        Bcc3 = ds.r['Bcc3'][new_index]/B_unit

        if (gr==False):
          Bx_tmp = Bcc1
          By_tmp = Bcc2
          Bz_tmp = Bcc3
          Bcc1 = Bx_tmp*x_hat_prime[0] + By_tmp*y_hat_prime[0] + Bz_tmp*z_hat_prime[0]
          Bcc2 = Bx_tmp*x_hat_prime[1] + By_tmp*y_hat_prime[1] + Bz_tmp*z_hat_prime[1]   
          Bcc3 = Bx_tmp*x_hat_prime[2] + By_tmp*y_hat_prime[2] + Bz_tmp*z_hat_prime[2]

        if (gr==True):
          # bsq = ds2.r['user_out_var5'][new_index]*2.0
          B_vec = np.zeros(uu.shape)
          B_vec[1] = Bcc1 
          B_vec[2] = Bcc2 
          B_vec[3] = Bcc3
          cks_metric(xi,yi,zi,ax,ay,az)
          for i in range(1,4):
            for mu in range(0,4):
              bu[0] += g[i,mu]*uu[mu]*B_vec[i]
          bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
          bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
          bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
          bu = np.array(bu)
          bu_tmp = bu* 1.0

          bsq = 0
          for i in range(4):
            for j in range(4):
              bsq += g[i,j] * bu[i] * bu[j]
          if (uov==True): divb = ds2.r['user_out_var6'][new_index]*2.0


    if (gr==False):
        vel1 = vx_tmp*x_hat_prime[0] + vy_tmp*y_hat_prime[0] + vz_tmp*z_hat_prime[0]
        vel2 = vx_tmp*x_hat_prime[1] + vy_tmp*y_hat_prime[1] + vz_tmp*z_hat_prime[1]
        vel3 = vx_tmp*x_hat_prime[2] + vy_tmp*y_hat_prime[2] + vz_tmp*z_hat_prime[2]
    else:
        vel1 = vx_tmp*1.0
        vel2 = vy_tmp*1.0
        vel3 = vz_tmp*1.0
        # x_old = x \cos(\theta)\cos(\varphi) + y (-\sin(\varphi)) + z \sin(\theta)\cos(\varphi)\\
        # y_old = x \cos(\theta)\sin(\varphi) + y \cos(\varphi) + z \sin(\theta)\sin(\varphi) \\
        # z_old = x (-\sin(\theta)) + z \cos(\theta)
        # x_new = x_old * xhat_prime[0] + y_old * yhat_prime[0] + z_old * zhat_prime[0]
        # y_new = x_old * xhat_prime[1] + y_old * yhat_prime[1] + z_old * zhat_prime[1]
        # z_new = x_old * xhat_prime[2] + y_old * yhat_prime[2] + z_old * zhat_prime[2]
          # dxnew_dxold = np.cos(th) * np.cos(ph)
          # dxnew_dyold = - np.sin(ph)
          # dxnew_dzold = np.sin(th) * np.cos(ph)

          # dynew_dxold = np.cos(th) * np.sin(ph)
          # dynew_dyold = np.cos(ph)
          # dynew_dzold = np.sin(th)*np.sin(ph)

          # dznew_dxold = -np.sin(th)
          # dznew_dyold = 0.0
          # dznew_dzold = np.cos(th)
          # uu[1] = uu_tmp[1] * dxnew_dxold + uu_tmp[2] * dxnew_dyold + uu_tmp[3] * dxnew_dzold
          # uu[2] = uu_tmp[1] * dynew_dxold + uu_tmp[2] * dynew_dyold + uu_tmp[3] * dynew_dzold
          # uu[3] = uu_tmp[1] * dznew_dxold + uu_tmp[2] * dznew_dyold + uu_tmp[3] * dznew_dzold

        uu[1] = uu_tmp[1] * x_hat_prime[0] + uu_tmp[2] * y_hat_prime[0]  + uu_tmp[3] * z_hat_prime[0]
        uu[2] = uu_tmp[1] * x_hat_prime[1] + uu_tmp[2] * y_hat_prime[1]  + uu_tmp[3] * z_hat_prime[1]
        uu[3] = uu_tmp[1] * x_hat_prime[2] + uu_tmp[2] * y_hat_prime[2]  + uu_tmp[3] * z_hat_prime[2]

        if (MHD==True):
          bu[1] = bu_tmp[1] * x_hat_prime[0] + bu_tmp[2] * y_hat_prime[0]  + bu_tmp[3] * z_hat_prime[0]
          bu[2] = bu_tmp[1] * x_hat_prime[1] + bu_tmp[2] * y_hat_prime[1]  + bu_tmp[3] * z_hat_prime[1]
          bu[3] = bu_tmp[1] * x_hat_prime[2] + bu_tmp[2] * y_hat_prime[2]  + bu_tmp[3] * z_hat_prime[2]

          # bu[1] = bu_tmp[1] * dxnew_dxold + bu_tmp[2] * dxnew_dyold + bu_tmp[3] * dxnew_dzold
          # bu[2] = bu_tmp[1] * dynew_dxold + bu_tmp[2] * dynew_dyold + bu_tmp[3] * dynew_dzold
          # bu[3] = bu_tmp[1] * dznew_dxold + bu_tmp[2] * dznew_dyold + bu_tmp[3] * dznew_dzold

          Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
          Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
          Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])


    global nx,ny,nz 
    nx = x.shape[0]
    if (midplane_slice==False): ny = x.shape[1]
    else: ny = 1
    if (slice==False): nz = x.shape[2]
    else: nz = 2

    if (double_precision==False):
      rho = np.array(np.float32(rho))
      press = np.array(np.float32(press))
      vel1 = np.array(np.float32(vel1))
      vel2 = np.array(np.float32(vel2))
      vel3 = np.array(np.float32(vel3))
      r = np.array(np.float32(r))
      theta = np.array(np.float32(theta))
      x = np.array(np.float32(x))
      y = np.array(np.float32(y))
      z = np.array(np.float32(z))
      phi = np.array(np.float32(phi))

      if (MHD==True):
        Bcc1 = np.array(np.float32(Bcc1))
        Bcc2 = np.array(np.float32(Bcc2))
        Bcc3 = np.array(np.float32(Bcc3))
      if (gr==True):
        uu = np.array(np.float32(uu))
        if (MHD==True):
          bu = np.array(np.float32(bu))
          bsq = np.array(np.float32(bsq))

    if (ISOTHERMAL==False): dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz,"th_tilt":th,"phi_tilt":ph, "t":ds.current_time, "r": r,"th": theta, "ph": phi  }
    else: dic = {"rho": rho, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz,"th_tilt":th,"phi_tilt":ph, "t":ds.current_time, "r": r,"th": theta, "ph": phi }
    if (gr==True):
      dic['uu'] = uu
    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        if (gr==True): dic["bu"] = bu
        if (gr==True): dic['bsq'] = bsq
    if (('athena_pp','r0') in ds.field_list ): 
      k_ent = ds.r['r0'][new_index]
    if (('athena_pp','r1') in ds.field_list ): 
      ke_ent = ds.r['r1'][new_index]
      dic["ke_ent"] = ke_ent
    if (('athena_pp','r2') in ds.field_list ): 
      ke_ent2 = ds.r['r2'][new_index]
      dic["ke_ent2"] = ke_ent2
    if (('athena_pp','r3') in ds.field_list ): 
      ke_ent3 = ds.r['r3'][new_index]
      dic["ke_ent3"] = ke_ent3




    t = ds.current_time

    np.savez(dump_name,**dic)


def get_1d_arrays(idump, gr=True, a = 0.0):
  yt_load(idump,gr=gr,a=a)
  from yt.units import pc, msun,kyr
  dic = {}
  dic["rho"] = np.array(ds.r['rho']* pc**3/msun)
  dic["press"] = np.array(ds.r['press']* pc**3/msun * kyr**2/pc**2)
  dic['x'] = np.array(ds.r['x'])
  dic['y'] = np.array(ds.r['y'])
  dic['z'] = np.array(ds.r['z'])

  B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
  dic['Bx'] = np.array(ds.r['Bcc1']/B_unit)
  dic['By'] = np.array(ds.r['Bcc2']/B_unit)

  dic['Bz'] = np.array(ds.r['Bcc3']/B_unit)

  uu = [0,0,0,0]
  bu = [0,0,0,0]
  vel1 = np.array(ds.r['vel1'] * kyr/pc)
  vel2 = np.array(ds.r['vel2'] * kyr/pc)
  vel3 = np.array(ds.r['vel3'] * kyr/pc)


  if (gr==True):
        cks_metric(dic['x'],dic['y'],dic['z'],0,0,a,ONED=True)
        tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
        gamma = np.sqrt(1.0 + tmp);

        # Calculate 4-velocity
        cks_inverse_metric(dic['x'],dic['y'],dic['z'],0,0,a,ONED=True)
        alpha = np.sqrt(-1.0/gi[0,0]);
        uu[0] = gamma/alpha;
        uu[1] = vel1 - alpha * gamma * gi[0,1];
        uu[2] = vel2 - alpha * gamma * gi[0,2];
        uu[3] = vel3 - alpha * gamma * gi[0,3];

        uu = np.array(uu)

        if (gr==True):
          B_vec = np.zeros(uu.shape)
          B_vec[1] = dic['Bx'] 
          B_vec[2] = dic['By']
          B_vec[3] = dic['Bz']
          cks_metric(dic['x'],dic['y'],dic['z'],0,0,a,ONED=True)
          for i in range(1,4):
            for mu in range(0,4):
              bu[0] += g[i,mu]*uu[mu]*B_vec[i]
          bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
          bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
          bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
          bu = np.array(bu)
          bu_tmp = bu* 1.0

          bsq = 0
          for i in range(4):
            for j in range(4):
              bsq += g[i,j] * bu[i] * bu[j]
          dic['bsq'] = bsq
          dic['uu'] = uu
          dic['bu'] = bu
  np.savez("1d_arrays_%d.npz" %idump,**dic)

def rd_yt_convert_to_gammie(idump,MHD=False,dump_name=None,low_res =False,method='nearest',
  fill_value = 0.0,gr=False,a=0.0, ISOTHERMAL = False, nr = 356, nth = 128,nphi=128,double_precision=False,hslope=0.3):
  if (dump_name is None): 
    if (low_res == False): dump_name = "dump_gammie_%d.npz" %(idump)
    else: dump_name = "dump_gammie_%d_low_res.npz" %(idump)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    yt_load(idump,gr=gr,a=a)
    global r,phi,theta,xi,yi,zi
    global x1,x2,x3
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t 
    global k_ent,ke_ent,ke_ent2,ke_ent3
    #unit vectors for the new coordinate system in the frame of the old coordinates
    z_hat = np.array([0,0,1])   #r
    x_hat = np.array([1,0,0])  #theta
    y_hat = np.array([0,1,0])                        #phi

    #unit vectors for original coordinate system in the frame of the new coordinates
    #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
    x_hat_prime = [1,0,0]   #in terms of [theta_hat, phi_hat, r_hat] 
    y_hat_prime = [0,1,0]
    z_hat_prime = [0,0,1]

    index = np.arange(ds.r['density'].shape[0])
    r_extrema = ds.all_data().quantities.extrema("r")
    rmin = 1.0 
    rmax = r_extrema[1]*.9

    #faces
    x2 = np.linspace(0.,1.0,nth+1)
    x3 = np.linspace(0,2.*np.pi,nphi+1)

    #centers
    x1 = np.linspace(np.log(rmin),np.log(rmax),nr+1)

    dx1 = np.diff(x1)[0]
    x1 = (x1 + dx1/2.0)[:-1]
    dx2 = np.diff(x2)[0]
    x2 = (x2 + dx2/2.0)[:-1]
    dx3 = np.diff(x3)[0] 
    x3 = (x3 + dx3/2.0)[:-1]
    x1,x2,x3 = np.meshgrid(x1,x2,x3,indexing='ij')


    ##new x,y,z coords in terms of new r,th,phi coords
    r = np.exp(x1)
    theta = np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
    phi = x3*1.0
    xi_prime = r*np.cos(phi)*np.sin(theta) - a*np.sin(phi)*np.sin(theta)
    yi_prime = r*np.sin(phi)*np.sin(theta) + a*np.cos(phi)*np.sin(theta)
    zi_prime = r*np.cos(theta)


    #r_vec = x_new x_hat_new + y_new y_hat_new + z_new z_hat_new  
    # x_old = r_vec dot x_hat_old 
    #original x,y,z coords in terms of new coords

    xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
    yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
    zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2] 
    new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = method,fill_value = fill_value)


    from yt.units import pc, msun,kyr
    rho = ds.r['density'][new_index] * pc**3/msun
    if (ISOTHERMAL == False): press = ds.r['press'][new_index] * pc**3/msun * kyr**2/pc**2
    vel1 = ds.r['vel1'][new_index] * kyr/pc
    vel2 = ds.r['vel2'][new_index] * kyr/pc
    vel3 = ds.r['vel3'][new_index] * kyr/pc

    global uu,bu,bsq
    uu = [0,0,0,0]
    bu = [0,0,0,0]

    if (gr==True):
        cks_metric(xi,yi,zi,0,0,a)
        tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
        gamma = np.sqrt(1.0 + tmp);

        # Calculate 4-velocity
        cks_inverse_metric(xi,yi,zi,0,0,a)
        alpha = np.sqrt(-1.0/gi[0,0]);
        uu[0] = gamma/alpha;
        uu[1] = vel1 - alpha * gamma * gi[0,1];
        uu[2] = vel2 - alpha * gamma * gi[0,2];
        uu[3] = vel3 - alpha * gamma * gi[0,3];

        uu = np.array(uu)

        uu_tmp = uu*1.0

    
    vx_tmp = vel1
    vy_tmp = vel2
    vz_tmp = vel3 


    #new coords in terms of new r,th,phi
    x = r*np.cos(phi)*np.sin(theta) - a*np.sin(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta) + a*np.cos(phi)*np.sin(theta)
    z = r*np.cos(theta)


    if (MHD==True):
        B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
        Bcc1 = ds.r['Bcc1'][new_index]/B_unit
        Bcc2 = ds.r['Bcc2'][new_index]/B_unit
        Bcc3 = ds.r['Bcc3'][new_index]/B_unit

        B_vec = np.zeros(uu.shape)
        B_vec[1] = Bcc1 
        B_vec[2] = Bcc2 
        B_vec[3] = Bcc3
        cks_metric(xi,yi,zi,0,0,a)
        for i in range(1,4):
          for mu in range(0,4):
            bu[0] += g[i,mu]*uu[mu]*B_vec[i]
        bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
        bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
        bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
        bu = np.array(bu)
        bu_tmp = bu* 1.0

        bsq = 0
        for i in range(4):
          for j in range(4):
            bsq += g[i,j] * bu[i] * bu[j]


    #convert to gammie 4 vectors

    #first convert to ks:

    uu_ks = cks_vec_to_ks(uu,xi,yi,zi,0,0,a)
    bu_ks = cks_vec_to_ks(bu,xi,yi,zi,0,0,a)

    #then to gammie

    uu = ks_vec_to_gammie(uu_ks,x1,x2,x3,a=a,hslope=hslope)
    bu = ks_vec_to_gammie(bu_ks,x1,x2,x3,a=a,hslope=hslope)

    Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
    Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
    Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])



    global nx,ny,nz 
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]

    if (double_precision==False):
      rho = np.float32(rho)
      press = np.float32(press)
      vel1 = np.float32(vel1)
      vel2 = np.float32(vel2)
      vel3 = np.float32(vel3)
      r = np.float32(r)
      theta = np.float32(theta)
      x = np.float32(x)
      y = np.float32(y)
      z = np.float32(z)
      phi = np.float32(phi)

      if (MHD==True):
        Bcc1 = np.float32(Bcc1)
        Bcc2 = np.float32(Bcc2)
        Bcc3 = np.float32(Bcc3)
      if (gr==True):
        uu = np.float32(uu)
        bu = np.float32(bu)
        bsq = np.float32(bsq)

    if (ISOTHERMAL==False): dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz, "t":ds.current_time, "r": r,"th": theta, "ph": phi,"x1": x1,"x2":x2,"x3":x3 }
    else: dic = {"rho": rho, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz, "t":ds.current_time, "r": r,"th": theta, "ph": phi,"x1": x1,"x2":x2,"x3":x3 }
    if (gr==True):
      dic['uu'] = uu
    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        if (gr==True): dic["bu"] = bu
        if (gr==True): dic['bsq'] = bsq
    if (('athena_pp','r0') in ds.field_list ): 
      k_ent = ds.r['r0'][new_index]
    if (('athena_pp','r1') in ds.field_list ): 
      ke_ent = ds.r['r1'][new_index]
      dic["ke_ent"] = ke_ent
    if (('athena_pp','r2') in ds.field_list ): 
      ke_ent2 = ds.r['r2'][new_index]
      dic["ke_ent2"] = ke_ent2
    if (('athena_pp','r3') in ds.field_list ): 
      ke_ent3 = ds.r['r3'][new_index]
      dic["ke_ent3"] = ke_ent3




    t = ds.current_time

    np.savez(dump_name,**dic)


def rd_spherical_smr(idump,MHD=False,dump_name=None,
  method='nearest',fill_value = 0.0, nr = 356, nth = 128,nphi=128, double_precision=False,rmin=None,rmax=None,):
  if (dump_name is None): 
    dump_name = "dump_spher_%d.npz" %(idump)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    file_prefix = glob.glob("*out2*.athdf")[0][:-11]
    get_hdf5_data(file_prefix + "%05d.athdf" %idump,mhd=MHD)
    file_prefix = glob.glob("*user*.athdf")[0][:-11]
    get_hdf5_data(file_prefix + "%05d.athdf" %idump,mhd=MHD,user_file=True)
    global r,phi,theta,xi,yi,zi
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t, len_th,len_ph
    global fluxr, fluxth, dM_floor,dM_floor_tot
    global dr,dth,dph,bsq


    index = np.arange(rho_.shape[0])
    if (rmin is None): rmin = np.amin(r_) 
    if (rmax is None): rmax = np.amax(r_) 

    #faces
    theta = np.linspace(0.,np.pi,nth+1)
    phi = np.linspace(0,2.*np.pi,nphi+1)

    #centers
    r = np.logspace(log10(rmin),log10(rmax),nr)

    dth = np.diff(theta)[0]
    theta = (theta + dth/2.0)[:-1]
    dphi = np.diff(phi)[0]
    phi = (phi + dphi/2.0)[:-1]
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')


    new_index = scipy.interpolate.griddata((r_,th_,ph_),index,(r,theta,phi),method = method,fill_value = fill_value)

    rho = rho_[new_index]
    press = press_[new_index] 
    vel1 = vel1_[new_index] 
    vel2 = vel2_[new_index] 
    vel3 = vel3_[new_index] 


    fluxr = flux_r_[new_index]
    fluxth = flux_th_[new_index]
    dM_floor = dM_floor_[new_index]

    len_th = dth_*r_ 
    len_ph = dph_*r_*np.sin(th_)
    dr = dr_[new_index] 
    dth = dth_[new_index] 
    dph = dph_[new_index]
    len_th = len_th[new_index]*1.0 
    len_ph = len_ph[new_index]*1.0

    if (MHD==True):
      Bcc1 = Bcc1_[new_index] 
      Bcc2 = Bcc2_[new_index] 
      Bcc3 = Bcc3_[new_index] 
      bsq = Bcc1**2 + Bcc2**2 + Bcc3**2 


    x = r*np.cos(phi)*np.sin(theta) 
    y = r*np.sin(phi)*np.sin(theta) 
    z = r*np.cos(theta)


    global nx,ny,nz 
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]

    if (double_precision==False):
      rho = np.array(np.float32(rho))
      press = np.array(np.float32(press))
      vel1 = np.array(np.float32(vel1))
      vel2 = np.array(np.float32(vel2))
      vel3 = np.array(np.float32(vel3))
      r = np.array(np.float32(r))
      theta = np.array(np.float32(theta))
      x = np.array(np.float32(x))
      y = np.array(np.float32(y))
      z = np.array(np.float32(z))
      phi = np.array(np.float32(phi))

      dr = np.array(np.float32(dr))
      dth = np.array(np.float32(dth))
      dph = np.array(np.float32(dph))
      len_th = np.array(np.float32(len_th))
      len_ph = np.array(np.float32(len_ph))

      if (MHD==True):
        Bcc1 = np.array(np.float32(Bcc1))
        Bcc2 = np.array(np.float32(Bcc2))
        Bcc3 = np.array(np.float32(Bcc3))
        bsq = np.array(np.float32(bsq))

    dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz, "t":t_, "r": r,"th": theta, "ph": phi  }


    dic['fluxr'] = fluxr
    dic['fluxth'] = fluxth
    dic['dM_floor'] = dM_floor
    dic['dr'] = dr 
    dic['dth'] = dth 
    dic['dph'] = dph
    dic['len_th'] = len_th
    dic['len_ph'] = len_ph

    dic['dM_r_inner'] = dM_r_inner
    dic['dM_r_inner_pos'] = dM_r_inner_pos

    dic['dM_r_outer'] = dM_r_outer
    dic['dM_th_inner'] = dM_th_inner
    dic['dM_th_outer'] = dM_th_outer
    dM_floor_tot = dM_floor_.sum()
    dic['dM_floor_tot'] = dM_floor_tot

    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        dic["bsq"] = bsq

    t =  t_

    np.savez(dump_name,**dic)

def make_movies_from_frames():
  for var in ["rho","T"]:
    for scale in ['outer','inner']:
      for direction in ['x','y','z']:
        os.system('ffmpeg -i frame_%s_%s_%s_%s.png -vcodec mpeg4 -qmax 5 movie_%s_%s_%s.mp4' %(var,scale,direction,"%d",var,scale,direction))
def make_lunch_talk_plots():
    r_in  = 2.*2./128./2.**8.
    plt.figure(1)
    plt.clf()
    f1 = "/global/scratch/smressle/star_cluster/restart/absorbing_8_levels_more_output/star_wind.hst"
    f2 = "/global/scratch/smressle/star_cluster/cuadra_comp/without_S2_new_outputs/star_wind.hst"

    f1 = "/global/scratch/smressle/star_cluster/cuadra_comp/test_new_output_8_levels/star_wind.hst"
    f2 = f1
    i_s = 0
    ie = 14
    rd_hst(f1)
    l_kep_in = sqrt(gm_*r_in)
    plt.loglog(r[0,:]/2.167e-7,-mdot_avg[i_s:ie,:].mean(0),lw = 2 ,ls='-',color='red')
    rd_hst(f2)
    plt.loglog(r[0,:]/2.167e-7,mdot_avg[i_s:ie,:].mean(0),lw = 2 ,ls='--',color = 'red')
    plt.xlabel(r"$r/r_g$",fontsize = 30)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+plt.gca().get_yticklabels():
        label.set_fontsize(20)
    plt.gca().axvspan(0.35*0.04/2.167e-7,2*0.04/2.167e-7, alpha=0.5, color='grey')
    plt.tight_layout()
    plt.ylim(1e-4,1.5e0)
    plt.savefig("mdot.png")
    plt.figure(2)
    plt.clf()
    rd_hst(f1)
    def Ldot_avg_func(istart,iend):
        return np.sqrt(Lxdot_avg[istart:iend,:].mean(0)**2. + Lydot_avg[istart:iend,:].mean(0)**2. + Lzdot_avg[istart:iend,:].mean(0)**2.)
    def L_avg_func(istart,iend):
        return np.sqrt(Lx_avg[istart:iend,:].mean(0)**2. + Ly_avg[istart:iend,:].mean(0)**2. + Lz_avg[istart:iend,:].mean(0)**2.)
    plt.loglog(r[0,:]/2.167e-7,abs(Ldot_avg_func(i_s,ie)/mdot_avg[i_s:ie,:].mean(0))/l_kep_in,lw = 2,label = r'$\langle l \rangle_\rho$')
    plt.loglog(r[0,:]/2.167e-7,abs(L_avg_func(i_s,ie)/rho_avg[i_s:ie,:].mean(0))/l_kep_in,lw=2,label = r'$\langle l \rangle_{\dot M}$')
#plt.ylim(5e-5,1e-2)
#plt.xlim(3e1,1e4)
    loglog(r[0,:]/2.167e-7,0.5*l_kep[0,:]/l_kep_in,lw = 2,ls = '--',label = r'$0.5 l_{\rm kep}$')
    plt.xlabel(r"$r/r_g$",fontsize = 30)
#plt.ylabel(r"$l/l_{\rm kep, in}$",fontsize = 30)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+plt.gca().get_yticklabels():
        label.set_fontsize(20)
    plt.tight_layout()
    plt.legend(loc = 'best',frameon = 0,fontsize = 20)

    plt.savefig("l_kep.png")

    plt.figure(3)
    plt.clf()

    rdnpz("dump_spher_avg_102_114.npz")

    plt.contourf(-x_tavg[:,:,0],z_tavg[:,:,0],log10(((-mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Blues')
    plt.contourf(-x_tavg[:,:,0],z_tavg[:,:,0],log10(((mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Reds')
    plt.contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((-mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Blues')
    plt.contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Reds')
    plt.xlim(-0.003*3,.003*3)
    plt.ylim(-0.003*3,.003*3)

    plot_streamlines(box_radius = 0.003*3.)

    plt.axis('off')
    plt.savefig("streamlines.png")

def Eliot_plots():
  plt.figure(1)
  plt.clf()
  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  theta = np.arccos(z_tavg/r)
  gam = 5./3.
  gm_ = 0.0191744
  csq = gam * (press_tavg*np.sin(theta)).mean(-1).mean(-1)/(rho_tavg*np.sin(theta)).mean(-1).mean(-1)
  l = (Lz_avg*np.sin(theta)).mean(-1).mean(-1)/(rho_tavg*np.sin(theta)).mean(-1).mean(-1)
  a = gm_/r[:,0,0]
  plt.loglog(r[:,0,0],csq/(gam-1.)/a,label = r"$\frac{c_s^2}{\gamma-1} \frac{r}{GM}$",lw=2,ls = '-')
  plt.loglog(r[:,0,0],l**2./(a*r[:,0,0]**2.),label = r"$\frac{l^2}{l_{\rmkep}^2}$",lw=2,ls = "--")

  plt.xlabel(r'$r$(pc)',fontsize = 25)

  plt.ylim(.5e-2,2e0)
  plt.xlim(1e-4,1e-1)

  plt.legend(loc='best',fontsize = 15,frameon=0)
  plt.tight_layout()

  plt.figure(2)
  plt.clf()

  plt.contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(abs((Lz_avg).mean(-1))),levels = np.linspace(-1.5,0,200),extend = 'both')
  plt.xlim(0,.001*3)
  plt.ylim(-0.5e-3*3,0.5e-3*3)
  cb = plt.colorbar()

  cb.set_label(r'$\log_{10}\left(\langle L_z\rangle\right)$',fontsize=25)
  plt.xlabel(r'$R (pc)$',fontsize =25)
  plt.ylabel(r'$z (pc)$',fontsize = 25)
  plt.tight_layout()

  plt.figure(3)
  plt.clf()

  plt. contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((-mdot_avg).mean(-1))),levels = np.linspace(-4.5,-3,200),extend='both')
  plt.xlim(0,.001*3)
  plt.ylim(-0.5e-3*3,0.5e-3*3)
  cb = plt.colorbar()

  cb.set_label(r'$\log_{10}\left(-\langle\dot M\rangle\right)$',fontsize=25)
  plt.xlabel(r'$R (pc)$',fontsize =25)
  plt.ylabel(r'$z (pc)$',fontsize = 25)
  plt.tight_layout()

  plt.figure(4)
  plt.clf()

  plt. contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((rho_tavg).mean(-1))),levels = np.linspace(1.5,2.5,200),extend='both')
  plt.xlim(0,.001*3)
  plt.ylim(-0.5e-3*3,0.5e-3*3)
  cb = plt.colorbar()

  cb.set_label(r'$\log_{10}\left(\langle \rho\rangle\right)$',fontsize=25)
  plt.xlabel(r'$R (pc)$',fontsize =25)
  plt.ylabel(r'$z (pc)$',fontsize = 25)
  plt.tight_layout()


def plot_vectors(A_r,A_th,A_phi=None,box_radius = 0.003,spherical_coords=False):
  global x,y,vx,vz
  global vr_avg
  x,z = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  s = np.sqrt(x_tavg**2. + y_tavg**2.)
  theta = np.arccos(z_tavg/r)

  #if (spherical_coords ==False ): vth_avg = v1 * (x_tavg*z_tavg)/(r*s+1e-15)+ v2 * (y_tavg*z_tavg)/(r*s+1e-15) + v3 * (-s/(r+1e-15))

  vxi = (A_r).mean(-1) * np.sin(theta[:,:,0]) + (A_th).mean(-1)  * np.cos(theta[:,:,0])
  vzi = (A_r).mean(-1) * np.cos(theta[:,:,0]) + (A_th).mean(-1)  * -np.sin(theta[:,:,0])

  vx = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vxi.flatten(),(x,z),method = 'nearest')
  vz = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vzi.flatten(),(x,z),method = 'nearest')

  plt.streamplot(x.transpose(),z.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  plt.streamplot(-x.transpose(),z.transpose(),-vx.transpose(),vz.transpose(),color = 'black')



def plot_streamlines(box_radius = 0.003,spherical_coords=False,gr=False):
  global x,y,vx,vz
  global vr_avg
  x,z = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  s = np.sqrt(x_tavg**2. + y_tavg**2.)
  theta = np.arccos(z_tavg/r)

  if (gr==True):
      vth_avg = (uu[2]/uu[0])*r
      vr_avg = (uu[1]/uu[0])
  elif (spherical_coords ==False ): vth_avg = vel1_tavg * (x_tavg*z_tavg)/(r*s+1e-15)+ vel2_tavg * (y_tavg*z_tavg)/(r*s+1e-15) + vel3_tavg * (-s/(r+1e-15))

  vxi = (rho_tavg*vr_avg).mean(-1)/rho_tavg.mean(-1) * np.sin(theta[:,:,0]) + (rho_tavg*vth_avg).mean(-1)/rho_tavg.mean(-1)  * np.cos(theta[:,:,0])
  vzi = (rho_tavg*vr_avg).mean(-1)/rho_tavg.mean(-1) * np.cos(theta[:,:,0]) + (rho_tavg*vth_avg).mean(-1)/rho_tavg.mean(-1) * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vxi.flatten(),(x,z),method = 'nearest')
  vz = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vzi.flatten(),(x,z),method = 'nearest')

  plt.streamplot(x.transpose(),z.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  plt.streamplot(-x.transpose(),z.transpose(),-vx.transpose(),vz.transpose(),color = 'black')

def plot_fieldlines(box_radius = 0.003,spherical_coords=False):
  global x,y,vx,vz
  x,z = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  s = np.sqrt(x_tavg**2. + y_tavg**2.)
  theta = np.arccos(z_tavg/r)

  if (spherical_coords ==False ): Bth_avg = Bcc1_tavg * (x_tavg*z_tavg)/(r*s+1e-15)+ Bcc2_tavg * (y_tavg*z_tavg)/(r*s+1e-15) + Bcc3_tavg * (-s/(r+1e-15))

  vxi = Br_avg.mean(-1) * np.sin(theta[:,:,0]) + Bth_avg.mean(-1) * np.cos(theta[:,:,0])
  vzi = Br_avg.mean(-1) * np.cos(theta[:,:,0]) + Bth_avg.mean(-1) * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vxi.flatten(),(x,z),method = 'nearest')
  vz = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vzi.flatten(),(x,z),method = 'nearest')

  plt.streamplot(x.transpose(),z.transpose(),vx.transpose(),vz.transpose(),color = 'white',density=0.5)
  plt.streamplot(-x.transpose(),z.transpose(),-vx.transpose(),vz.transpose(),color = 'white',density = 0.5)

def plot_fieldlines_slice(box_radius = 0.003,xbox_radius=None,ybox_radius=None,spherical_coords=False,iphi=0,arrowstyle='->',lw=1,density=1,color='black'):
  global x,y,vx,vz
  global x_grid_r,x_grid_l,x_grid,z_grid
  if (xbox_radius is None and ybox_radius is None): 
    xbox_radius = box_radius
    ybox_radius = box_radius
  elif (ybox_radius is None): ybox_radius = xbox_radius
  elif (xbox_radius is None): xbox_radius = ybox_radius
  ym = -ybox_radius
  yp = ybox_radius
  xp = xbox_radius
  xm = -xbox_radius

  print(xm,xp,ym,yp)
  if (xm<0): x_grid,z_grid = np.meshgrid(np.linspace(0,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')
  else: x_grid,z_grid = np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)



  if (spherical_coords==False): Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
  else:
    Br = Bcc1
    Bth = Bcc2


  vxi = (Br[:,:,iphi] * np.sin(theta[:,:,0]) + Bth[:,:,iphi] * np.cos(theta[:,:,0]) ) 
  vzi = (Br[:,:,iphi] * np.cos(theta[:,:,0]) + Bth[:,:,iphi] * -np.sin(theta[:,:,0]) ) 

  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  vxr = vx
  vzr = vz
  x_grid_r = x_grid
  z_grid_r = z_grid
  #plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')


  if (xp>0): x_grid,z_grid = np.meshgrid(np.linspace(xm,0,128),np.linspace(ym,yp ,128),indexing = 'ij')
  else: np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')
  vxi = (Br[:,:,iphi+nz//2] * np.sin(theta[:,:,0]) + Bth[:,:,iphi+nz//2] * np.cos(theta[:,:,0])) * np.cos(ph[:,:,iphi+nz//2])
  vzi = Br[:,:,iphi+nz//2] * np.cos(theta[:,:,0]) + Bth[:,:,iphi+nz//2] * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  vxl = vx
  vzl = vz

  x_grid_l = x_grid
  z_grid_l = z_grid

  vx = np.concatenate((vxl,vxr),axis=0)
  vz = np.concatenate((vzl,vzr),axis=0)
  x_grid = np.concatenate((x_grid_l,x_grid_r),axis=0)
  z_grid = np.concatenate((z_grid_l,z_grid_r),axis=0)
  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)

def plot_fieldlines_midplane_slice(box_radius = 0.003,xbox_radius=None,ybox_radius=None,spherical_coords=False,iphi=0,arrowstyle='->',lw=1,density=1,color='black'):
  global x,y,vx,vz
  global x_grid_r,x_grid_l,x_grid,y_grid
  if (xbox_radius is None and ybox_radius is None): 
    xbox_radius = box_radius
    ybox_radius = box_radius
  elif (ybox_radius is None): ybox_radius = xbox_radius
  elif (xbox_radius is None): xbox_radius = ybox_radius
  ym = -ybox_radius
  yp = ybox_radius
  xp = xbox_radius
  xm = -xbox_radius


  x_grid,y_grid = np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)
  phi = np.arctan2(y,x)



  if (spherical_coords==False): 
    Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
    vxi = Bcc1[:,ny//2,:] 
    vyi = Bcc2[:,ny//2,:] 
  else:
    Br = Bcc1
    Bth = Bcc2
    Bphi = Bcc3


    vxi = (Br[:,ny//2,:] * np.cos(phi[:,ny//2,:]) + Bth[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))  ) 
    vyi = (Br[:,ny//2,:] * np.sin(phi[:,ny//2,:]) + Bth[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))  ) 

  vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),vxi.flatten(),(x_grid,y_grid),method = 'nearest')
  vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),vyi.flatten(),(x_grid,y_grid),method = 'nearest')


  plt.streamplot(x_grid.transpose(),y_grid.transpose(),vx.transpose(),vy.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)



def plot_fieldlines_right(xm,xp,ym,yp,spherical_coords=False,iphi=0,arrowstyle='->',lw=1,density=1,color='black'):
  global x,y,vx,vz
  global x_grid_r,x_grid_l,x_grid,z_grid

  x_grid,z_grid = np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')

  th_grid = np.arccos(z_grid/sqrt(x_grid**2+z_grid**2))

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)



  if (spherical_coords==False): Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
  else:
    Br = Bcc1
    Bth = Bcc2


  vxi = (Br[:,:,iphi] * np.sin(theta[:,:,0]) + Bth[:,:,iphi] * np.cos(theta[:,:,0]) ) 
  vzi = (Br[:,:,iphi] * np.cos(theta[:,:,0]) + Bth[:,:,iphi] * -np.sin(theta[:,:,0]) ) 

  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  vx[th_grid<np.amin(theta)] = 0.0
  vx[th_grid>np.amax(theta)] = 0.0
  vz[th_grid<np.amin(theta)] = 0.0
  vz[th_grid>np.amax(theta)] = 0.0
  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)

def plot_streamlines_phi_slice(box_radius = 0.003,iphi = 0):
  global x,y,vx,vz
  x_grid,z_grid = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)



  vth = vel1 * (x*z)/(r*s+1e-15)+ vel2 * (y*z)/(r*s+1e-15) + vel3 * (-s/(r+1e-15))

  vxi = (vr[:,:,iphi] * np.sin(theta[:,:,0]) + vth[:,:,iphi] * np.cos(theta[:,:,0]) ) 
  vzi = (vr[:,:,iphi] * np.cos(theta[:,:,0]) + vth[:,:,iphi] * -np.sin(theta[:,:,0]) ) 

  vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')


  x_grid,z_grid = np.meshgrid(np.linspace(-box_radius,0,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')
  vxi = (vr[:,:,iphi+nz//2] * np.sin(theta[:,:,0]) + vth[:,:,iphi+nz//2] * np.cos(theta[:,:,0])) * np.cos(ph[:,:,iphi+nz//2])
  vzi = vr[:,:,iphi+nz//2] * np.cos(theta[:,:,0]) + vth[:,:,iphi+nz//2] * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')



def plot_streamlines_midplane(box_radius = 0.003,density=1,is_tavg=False,color='black',lw=1):
  global x,y,vx,vz
  x_grid,y_grid = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  if (is_tavg==False):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
  else:
    r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
    s = np.sqrt(x_tavg**2. + y_tavg**2.)
    theta = np.arccos(z_tavg/r)
    phi = np.arctan2(y_tavg,x_tavg)

  if (is_tavg==False):
    vxi = vr[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:]) + vphi[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    vyi = vr[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:]) + vphi[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),vxi.flatten(),(x_grid,y_grid),method = 'nearest')
    vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),vyi.flatten(),(x_grid,y_grid),method = 'nearest')
  else:
    vxi = vr_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:]) + vphi_avg[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    vyi = vr_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:]) + vphi_avg[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    vx = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),vxi.flatten(),(x_grid,y_grid),method = 'nearest')
    vy = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),vyi.flatten(),(x_grid,y_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),y_grid.transpose(),vx.transpose(),vy.transpose(),color = color,density = density,linewidth=lw)

def plot_fieldlines_midplane(box_radius = 0.003, is_tavg =False):
  global x,y,vx,vz
  x_grid,y_grid = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  if (is_tavg==False):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
  else:
    r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
    s = np.sqrt(x_tavg**2. + y_tavg**2.)
    theta = np.arccos(z_tavg/r)
    phi = np.arctan2(y_tavg,x_tavg)


  if (is_tavg==False):
    Bxi = Br[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:])  + Bphi[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    Byi = Br[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:])  + Bphi[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    Bx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Bxi.flatten(),(x_grid,y_grid),method = 'nearest')
    By = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Byi.flatten(),(x_grid,y_grid),method = 'nearest')
  else:
    Bxi = Br_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:])  + Bphi_avg[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    Byi = Br_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:])  + Bphi_avg[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    Bx = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),Bxi.flatten(),(x_grid,y_grid),method = 'nearest')
    By = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),Byi.flatten(),(x_grid,y_grid),method = 'nearest')
  plt.streamplot(x_grid.transpose(),y_grid.transpose(),Bx.transpose(),By.transpose(),color = 'black')


  # vxi = vr[:,:,iphi+nz/2] * np.sin(theta[:,:,0]) + vth[:,:,iphi] * np.cos(theta[:,:,0])
  # vyi = vr[:,:,iphi+nz/2] * np.cos(theta[:,:,0]) + vth[:,:,iphi] * -np.sin(theta[:,:,0])

  # vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  # vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  #plt.streamplot(-x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')
def plot_streamlines_spherical(box_radius = 0.003):
  global vx,vz 
  r = x 
  theta = y 
  x_grid,z_grid = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  vxi = vel1.mean(-1) * np.sin(theta) + vel2.mean(-1) * np.cos(theta)
  vzi = vel1.mean(-1) * np.cos(theta) + vel2.mean(-1) * -np.sin(theta)

  x_tmp = r*sin(theta) 
  z_tmp = r*cos(theta)
  vx= scipy.interpolate.griddata((x_tmp.flatten(),z_tmp.flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x_tmp.flatten(),z_tmp.flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  plt.streamplot(-x_grid.transpose(),z_grid.transpose(),-vx.transpose(),vz.transpose(),color = 'black')

def rd_cooling_files(Z_solar = 1.0):
  global T,Lam_tot,Lam_metal,Lam_non_metal
  data = np.loadtxt('lambda.dat')
  T = data[:,0]
  Lam_non_metal = data[:,13] 
  Lam_metal = data[:,15]
  Lam_tot = Lam_non_metal + Lam_metal*Z_solar

def mk_lambda_file(fname):
  data = np.loadtxt('lambda.dat')
  T = data[:,0][::8]
  Lam_non_metal = data[:,13][::8]
  Lam_metal = data[:,15][::8]

  f = open(fname,"w")
  for i in range(T.shape[0]):
      array = [str(T[i]),str(Lam_non_metal[i]),str(Lam_metal[i])]
      f.write(" ".join(array) + "\n")
  f.close()


def spex_lam():
  global T_kev, L_arr
  
  isfine = 0
  limited_band = 0
  X_ray_image_band = 1
  T_kev = np.array([8.00000000e-04,   1.50000000e-03, 2.50000000e-03,   7.50000000e-03,
    2.00000000e-02, 3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
     8.22000000e-01, 2.26000000e+00, 3.010000000e+00, 3.4700000000e+01, 1.00000000e+02 ])
  if (isfine==1):
      T_K = np.logspace(4,9,100)
      T_kev = T_K/1.16045e7
  if (limited_band ==1 or X_ray_image_band ==1):
    T_K = np.logspace(4,9,100)
    T_kev = T_K/1.16045e7
  L_arr = []
  n = 0
  W_to_erg_s = 1e7
  with_H = 1
  n_metals = 1.0 #Metallicity in Solar Units
  norm = 1e64 * 1e-6  #in cm^-3

  for i in range(3):
      if (i ==0):
        H_only = 1
        He_only = 0
        Metals_only = 0
      elif (i==1):
        H_only = 0
        He_only = 1
        Metals_only = 1
      else:
        H_only = 0
        He_only = 0
        Metals_only = 1
      L_arr = []
      n = 0

      #Lodders 2003
      Z_o_X_solar = 0.0177
      Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
      X_solar = 0.7491
      Z_solar = 1.0-X_solar - Y_solar

      if (with_H==0):
        Y = Y_solar/(1.-X)
        Z = Z_solar/(1.-X) * n_metals
        X = 0.
      for T in T_kev:
        f = open("spex.com",'w')
        f.write("spex << STOP\n")
        f.write("com cie\n")
        f.write("abundance solar\n")
        if (limited_band ==1):
          f.write("elim 2:10\n")
        elif (X_ray_image_band ==1):
          f.write("elim 2:8\n")
        else:  
          f.write("elim 0.0000001:1000000\n")
        f.write("par t val %g \n" %T)
        f.write("par it val %g \n" %T)
#f.write("par ed val 1.\n")
        if (H_only==1):
          f.write("par 02 val 0.0\n")
          for i_el in range(3,31):
                f.write("par %02d val %g\n" %(i_el,0.0))
        elif (He_only==1):
          f.write("par ref val 02\n")
          f.write("par 01 val 0.00001\n")
          for i_el in range(3,31):
                f.write("par %02d val %g\n" %(i_el,0.0))
        elif (Metals_only==1):
          f.write("par ref val 08\n")
          f.write("par 01 val 0.00001\n")
          f.write("par 02 val 0.0\n")
          for i_el in range(3,31):
                f.write("par %02d val %g\n" %(i_el,n_metals))
        f.write("calc\n")
        f.write("log out spex_%d o\n" %n)
        f.write("par show\n")
        f.write("quit\n")
        f.write("STOP\n")
        f.close()
        os.system("chmod +x spex.com")
        os.system("./spex.com")
        save_line = 0
        with open("spex_%d.out"%n) as f:
            for num,line in enumerate(f,0):
                if "(W)" in line:
                    i_line = num+1
                    break
        f = open("spex_%d.out"%n)
        lines = f.readlines()
        f.close()
        tmp = list(map(float,re.findall(r'[+-]?[0-9.]+', lines[i_line])) )
        Lum = tmp[-2] * 10.**(tmp[-1])
        L_arr.append(Lum)
        os.system("rm spex_%d.out"%n)

        n = n+1
      L_arr = np.array(L_arr)*W_to_erg_s / norm

      if (isfine==0 and limited_band==0):
        L_arr[-1] = L_arr[-2] * (T_kev[-1]/T_kev[-2])**(np.log(L_arr[-2]/L_arr[-3])/np.log(T_kev[-2]/T_kev[-3]))



      fname = "Lam_spex_Z_solar"

      if (limited_band==1):
        fname = fname + "_2_10_kev"

      if (X_ray_image_band==1):
        fname = fname + "_2_8_kev"

      if (isfine==1):
        file_suffix = "_fine.dat"
      else:
        file_suffix = ".dat"
      if (H_only==1):
        fname = fname + "_H_only" + file_suffix
      elif (He_only ==1):
        fname = fname + "_He_only" + file_suffix
      elif (Metals_only ==1):
        fname =fname + "_Metals_only" + file_suffix
      f = open(fname,"w")

      for i in range(T_kev.shape[0]):
          array = [str(T_kev[i]),str(L_arr[i])]
          f.write(" ".join(array) + "\n")
      f.close()

def rd_spex_lam(fname):
  global T_kev,T_K,Lam
  data = np.loadtxt(fname)
  T_kev = data[:,0]
  T_K = T_kev * 1.16045e7
  Lam = data[:,1]

def compute_total_lam(file_prefix,X=0,Z=3,isfine = 0):
  global T_K, T_kev,Lam,Lam_He,Lam_metals,Lam_H,muH_solar,mue,mu

  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar


  Z  = Z * Z_solar
  muH_solar = 1./X_solar
  mu = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  mue = 2./(1+X)

  if (isfine==0):
    f1 = file_prefix + "_H_only.dat"
    f2 = file_prefix + "_He_only.dat"
    f3 = file_prefix + "_Metals_only.dat"
  else:
    f1 = file_prefix + "_H_only_fine.dat"
    f2 = file_prefix + "_He_only_fine.dat"
    f3 = file_prefix + "_Metals_only_fine.dat" 
  data = np.loadtxt(f1)
  T_kev = data[:,0]
  T_K = data[:,0] * 1.16045e7
  Lam_H = data[:,1]
  data = np.loadtxt(f2)
  Lam_He = data[:,1]
  data = np.loadtxt(f3)
  Lam_metals = data[:,1]

  Lam = Lam_H * X/X_solar + Lam_He * (1-X-Z)/Y_solar + Lam_metals * Z/Z_solar



def spex_ne_nH():
    global T_kev, nerat_arr

    T_kev = np.array([8.00000000e-04,   1.50000000e-03, 2.50000000e-03,   7.50000000e-03,
                  2.00000000e-02, 3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
                  8.22000000e-01, 2.26000000e+00, 3.010000000e+00, 3.4700000000e+01,1.00000000e+02 ])
    T_K = np.logspace(4,9,100)
    T_kev = T_K/1.16045e7
    nerat_arr = []
    n = 0
    W_to_erg_s = 1e7
    with_H = 1
    n_metals = 1.0 #Metallicity in Solar Units
    norm = 1e64 * 1e-6  #in cm^-3


    #Lodders 2003
    Z_o_X_solar = 0.0177
    Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
    X_solar = 0.7491
    Z_solar = 1.0-X_solar - Y_solar
    
    X = 0
    Z = 3.*Z_solar
    Y = 1 - X - Z
    
    metal_frac = Z/Y / (Z_solar/Y_solar)

    for T in T_kev:
        f = open("spex.com",'w')
        f.write("spex << STOP\n")
        f.write("com cie\n")
        f.write("abundance solar\n")
        f.write("elim 0.00000001:100000\n")
        f.write("par t val %g \n" %T)
        f.write("par it val %g \n" %T)
        f.write("par ed val 1.\n")
        for i_el in range(3,31):
            f.write("par %02d val %g\n" %(i_el,metal_frac))
        f.write("par ref val 02\n")
        f.write("par 01 val 0.00001\n")
        f.write("log out spex_%d o\n" %n)
        f.write("ascdump terminal 1 1 plas\n")
        f.write("quit\n")
        f.write("STOP\n")
        f.close()
        os.system("chmod +x spex.com")
        os.system("./spex.com")
        save_line = 0
        with open("spex_%d.out"%n) as f:
            for num,line in enumerate(f,0):
                if "Electron/Hydrogen density" in line:
                    i_line = num
                    break
        f = open("spex_%d.out"%n)
        lines = f.readlines()
        f.close()
        tmp = map(float,re.findall(r'[+-]?[0-9.]+', lines[i_line]))
        if (np.array(tmp).shape[0]>1):
            ne_o_nh  = tmp[-2] * 10.**(tmp[-1])
        else:
            ne_o_nh =tmp[-1]
        nerat_arr.append(ne_o_nh)
        os.system("rm spex_%d.out"%n)

        n = n+1
    nerat_arr = np.array(nerat_arr)/ (1.-X_solar)

    fname = "ne_rat_fine.dat"
    f = open(fname,"w")

    for i in range(T_kev.shape[0]):
        array = [str(T_kev[i]),str(nerat_arr[i])]
        f.write(" ".join(array) + "\n")
    f.close()

def rd_ne_nh(fname):
  global T_kev,T_K,ne_o_nh
  data = np.loadtxt(fname)
  T_kev = data[:,0]
  T_K = T_kev * 1.16045e7
  ne_o_nh = data[:,1]


def cuadra_cool(T_K):
    Tbr1 = 3.0e4
    Tbr2 = 4.0e7
    case1 = -6.4e-23 * (Tbr2/1e7)**-0.7 * (T_K/Tbr2)**0.5
    case2 = -6.4e-23 * (T_K/1e7)**-0.7
    case3 = -6.4e-23 * (T_K/1e7)**-0.7 * (T_K/Tbr1)**2.0
    return (T_K>Tbr2)*case1 + (T_K>Tbr1) * (T_K<Tbr2) *case2 + (T_K<=Tbr1) * case3

def tavg(var,t_i,t_f):
  iti = t_to_it(t_i)
  itf = t_to_it(t_f)
  if (iti==itf): return var[itf,:]
  else: return var[iti:itf,:].mean(0)

def get_l_angles(t_i =1.0, t_f = 1.2,levels = 8):
  global th_l,phi_l,x_rat,y_rat,z_rat
  L_tot = np.sqrt( tavg(Lx_avg,t_i,t_f)**2. + tavg(Ly_avg,t_i,t_f)**2. + tavg(Lz_avg,t_i,t_f)**2.)
  r_in = 2.*2./2.**levels/128.

  x_rat = (tavg(Lx_avg,t_i,t_f)/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)
  y_rat = (tavg(Ly_avg,t_i,t_f)/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)
  z_rat = (tavg(Lz_avg,t_i,t_f)/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)
  norm = np.sqrt(x_rat**2+y_rat**2+z_rat**2)
  x_rat = x_rat/norm
  y_rat = y_rat/norm
  z_rat = z_rat/norm
  th_l = np.arccos(z_rat)
  phi_l = np.arctan2(y_rat,x_rat)

def get_l_angles_grmhd():
  global th_l,phi_l
  L_tot = np.sqrt(Lx**2 + Ly**2 + Lz**2)
  i1 = 30
  i2 = 56
  it1 = 800
  it2 = -1

  nx_avg = (Lx/L_tot)[it1:it2,i1:i2].mean(-1).mean(-1)
  ny_avg = (Ly/L_tot)[it1:it2,i1:i2].mean(-1).mean(-1)
  nz_avg = (Lz/L_tot)[it1:it2,i1:i2].mean(-1).mean(-1)

  th_l = np.arccos(nz_avg)
  ph_l = np.arctan2(ny_avg,nx_avg)



def mdot_bondi(p,rho):
    qs = 1./4.
    cs = np.sqrt(5./3. * p/rho)
    return 4.*pi * qs * gm_**2. * rho / (cs**3.)

def get_bh_spin_vector(tilt_angle=0,th=0,ph=0):
  l_vector = [np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]



def render_3d():
  import yt
  from yt.visualization.volume_rendering.api import Scene, VolumeSource 
  import numpy as np
  yt.enable_parallelism()
  sc  = Scene()
  vol = VolumeSource(ds, field="density")
  bounds = (1e-2, 10.**1.5)
  tf = yt.ColorTransferFunction(np.log10(bounds))
  tf.add_layers(8, colormap='bone')
  tf.grey_opacity = False
  vol.transfer_function = tf
  vol.tfh.tf = tf
  vol.tfh.bounds = bounds
  # vol.tfh.plot('transfer_function.png', profile_field=('gas', 'mass'))
  cam = sc.add_camera(ds, lens_type='plane-parallel')
  cam.resolution = [512,512]
  # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
  # cam.switch_orientation(normal_vector=normal_vector,
  #                        north_vector=north_vector)
  cam.set_width(ds.domain_width*0.25)

  cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')
  normal_vector = [0,0,-1]  #camera to focus
  north_vector = [0,1,0]  #up direction
  cam.switch_orientation(normal_vector=normal_vector,
                           north_vector=north_vector)
  sc.add_source(vol)
  sc.render()
  sc.save('tmp2.png',sigma_clip = 6.0)
  #sc.save('./RENDERING/rendering_temperature_2_'+str(i).zfill(3)+'.png', sigma_clip=6.0)

def yt_region(box_radius):
  global region,rho,press,vel1,vel2,vel3,x,y,z
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,(-box_radius,'pc'):(box_radius,'pc'):256j,
      (-box_radius,'pc'):(box_radius,'pc'):256j]
  rho = region['rho']
  press = region['press']
  vel1 = region['vel1']
  vel2 = region['vel2']
  vel3 = region['vel3']
  x = region['x']
  y = region['y']
  z = region['z']


def Lambda_cool(TK,file_prefix,X=0,Z=3):
  global mue,mu_highT,muH_solar
  mp_over_kev = 9.994827
  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = Z * Z_solar
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  mp = 8.41175e-58
  f1 = file_prefix + "_H_only.dat"
  f2 = file_prefix + "_He_only.dat"
  f3 = file_prefix + "_Metals_only.dat"
  data = np.loadtxt(f1)
  T_tab = data[:,0] * 1.16045e7
  Lam_H = data[:,1]
  data = np.loadtxt(f2)
  Lam_He = data[:,1]
  data = np.loadtxt(f3)
  Lam_metals = data[:,1]
  # T_tab = 10.**data[:,0]
  T_min = np.amin(T_tab)
  T_max = np.amax(T_tab)
  # if isinstance(TK,)
  # TK[TK<T_min] = T_min
  # TK[TK>T_max] = T_max
  # Lam_tab = 10.**data[:,1]

  Lam_tab = (X/X_solar) * Lam_H + ( (1.-X-Z)/Y_solar) * Lam_He + (Z/Z_solar) * Lam_metals
  from scipy.interpolate import InterpolatedUnivariateSpline
  Lam = InterpolatedUnivariateSpline(T_tab,Lam_tab,k = 1,ext =3)  #ext = 3 returns boundary value if outside range of T
  return Lam(TK)

def t_cool_func(press,rho,file_prefix,X=0,Z=3):
  mp_over_kev = 9.994827
  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = Z * Z_solar
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  mp = 8.41175e-58

  T_kev = press/rho * mu_highT*mp_over_kev
  TK= T_kev*1.16e7
  Lam_cgs = Lambda_cool(TK,file_prefix,X,Z)
  gm1 = 5./3. -1.
  UnitLambda_times_mp_times_kev = 1.255436328493696e-21

  return (T_kev) * (mu_highT) / ( gm1 * rho *             Lam_cgs/UnitLambda_times_mp_times_kev )

def matrix_vec_mult(A,b):
    result = [0,0,0]
    for i in range(3):
        for j in range(3):
            result[i] += A[i,j]*b[j]

    return result

def transpose(A):
    result = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            result[i,j] = A[j,i]
    return result
def get_rotation_matrix(alpha,beta,gamma=0):
    X_rot = np.zeros((3,3))
    Z_rot = np.zeros((3,3))
    Z_rot2 = np.zeros((3,3))
    rot = np.zeros((3,3))
    rot_tmp = np.zeros((3,3))


    Z_rot2[0,0] = np.cos(gamma)
    Z_rot2[0,1] = -np.sin(gamma)
    Z_rot2[0,2] = 0.
    Z_rot2[1,0] = np.sin(gamma)
    Z_rot2[1,1] = np.cos(gamma)
    Z_rot2[1,2] = 0.
    Z_rot2[2,0] = 0.
    Z_rot2[2,1] = 0.
    Z_rot2[2,2] = 1.

    X_rot[0,0] = 1.
    X_rot[0,1] = 0.
    X_rot[0,2] = 0.
    X_rot[1,0] = 0.
    X_rot[1,1] = np.cos(beta)
    X_rot[1,2] = -np.sin(beta)
    X_rot[2,0] = 0.
    X_rot[2,1] = np.sin(beta)
    X_rot[2,2] = np.cos(beta)

    Z_rot[0,0] = np.cos(alpha)
    Z_rot[0,1] = -np.sin(alpha)
    Z_rot[0,2] = 0.
    Z_rot[1,0] = np.sin(alpha)
    Z_rot[1,1] = np.cos(alpha)
    Z_rot[1,2] = 0.
    Z_rot[2,0] = 0.
    Z_rot[2,1] = 0.
    Z_rot[2,2] = 1.

    for i in range(3):
        for j in range(3):
            for k in range(3):
                rot_tmp[i,j] += X_rot[i,k] * Z_rot[k,j]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                rot[i,j] += Z_rot2[i,k] * rot_tmp[k,j]


    return rot

def get_orbit(star,t_vals):
    period = 2.*np.pi/star.mean_angular_motion
    a = (gm_/star.mean_angular_motion**2.)**(1./3.)
    
    if star.eccentricity <1 :
        b = a * np.sqrt(1. - star.eccentricity**2. )
    else:
        b = a * np.sqrt(star.eccentricity**2.-1.)

    def eqn(e_anomaly,m_anamoly):
        if (star.eccentricity<1):
            return m_anamoly - e_anomaly + star.eccentricity * np.sin(e_anomaly)
        else:
            return m_anamoly + e_anomaly - star.eccentricity * np.sinh(e_anomaly)

    mean_anomaly = star.mean_angular_motion * (t_vals - star.tau)

    # = mean_angular_motion * (t + simulation_start_time + mean_anomaly_0/mean_angular_motion)

    eccentric_anomaly =  fsolve(eqn,mean_anomaly,args = (mean_anomaly,))


    if (star.eccentricity<1):
        x1_t= a * (np.cos(eccentric_anomaly) - star.eccentricity)
        x2_t= b * np.sin(eccentric_anomaly)
        Edot = star.mean_angular_motion/ (1.-star.eccentricity * np.cos(eccentric_anomaly))
        v1_t = - a * np.sin(eccentric_anomaly) * Edot
        v2_t = b * np.cos(eccentric_anomaly) * Edot
    else:
        x1_t = a * ( star.eccentricity - np.cosh(eccentric_anomaly) )
        x2_t = b * np.sinh(eccentric_anomaly)
        Edot = -star.mean_angular_motion/ (1. - star.eccentricity * np.cosh(eccentric_anomaly))
        v1_t = a * (- np.sinh(eccentric_anomaly) * Edot)
        v2_t = b * np.cosh(eccentric_anomaly) * Edot

    return [x1_t,x2_t,0.], [v1_t,v2_t,0.]



def get_star_loc(t):
    for star in star_array:
      rotation_matrix = get_rotation_matrix(star.alpha,star.beta,star.gamma)
      inverse_rotation_matrix = transpose(rotation_matrix)
      X_orbit,V_orbit = get_orbit(star,t)
      x1_orbit,x2_orbit,x3_orbit = matrix_vec_mult(inverse_rotation_matrix,X_orbit)
      v1_orbit,v2_orbit,v3_orbit = matrix_vec_mult(inverse_rotation_matrix,V_orbit)
      star.x1 = np.float(x1_orbit)
      star.x2 = np.float(x2_orbit)
      star.x3 = np.float(x3_orbit)
      star.v1 = np.float(v1_orbit)
      star.v2 = np.float(v2_orbit)
      star.v3 = np.float(v3_orbit)
def set_star_size():
  for star in star_array:
    level_0 = 1.0
    level_1 = level_0/2.0
    level_2 = level_1/2.0
    level_3 = level_2/2.0
    level_4 = level_3/2.0
    dx = 2./128.
    star.radius = 2.* np.sqrt(3.0)*dx 

    if ( (fabs(star.x1)< level_1) and (fabs(star.x2) <level_1) and (fabs(star.x3)<level_1) ): star.radius= star.radius/2.0
    if ( (fabs(star.x1)< level_2) and (fabs(star.x2) <level_2) and (fabs(star.x3)<level_2) ): star.radius= star.radius/2.0
    if ( (fabs(star.x1)< level_3) and (fabs(star.x2) <level_3) and (fabs(star.x3)<level_3) ): star.radius= star.radius/2.0
    if ( (fabs(star.x1)< level_4) and (fabs(star.x2) <level_4) and (fabs(star.x3)<level_4) ): star.radius= star.radius/2.0
def mask_grid(x_arr,y_arr,z_arr):
  global mask_array
  mask_array = np.zeros(x_arr.shape) +1.0
  for star in star_array:
    print (star.Mdot)
    dr = np.sqrt( (np.array(x_arr)-star.x1)**2.0 + (np.array(y_arr)-star.x2)**2.0 + (np.array(z_arr)-star.x3)**2.0 )   
    mask_array = mask_array * (dr>star.radius)

def make_rendering():
  nx_disk = -0.12
  ny_disk = -0.79
  nz_disk = 0.6
  box_radius = 1.0 
  box_radius = 0.01
  box_radius = 1.0
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):512j,
  (-box_radius,'pc'):(box_radius,'pc'):512j,
  (-box_radius,'pc'):(box_radius,'pc'):512j ]
  x,y,z = region['x'],region['y'],region['z']
  import numpy as np

  bbox = np.array([[-box_radius,box_radius],[-box_radius,box_radius],[-box_radius,box_radius]])
  rho = region['density']
  #rho = rho * mask_array
  r = np.sqrt(x**2. + y**2. + z**2.)
  rho_dot_r = rho*r
  press = region['press']
  set_constants()
  keV_to_Kelvin = 1.16045e7

  T  = press/rho *  mu_highT*mp_over_kev*keV_to_Kelvin
  data =  dict(density = (np.array(rho),"Msun/pc**3"),temperature = (np.array(T),"K"),rho_dot_r = (np.array(rho_dot_r),"Msun/pc**2"),x = (np.array(x),"pc"), y = (np.array(y),"pc"),z = (np.array(z),"pc"))
  ds = yt.load_uniform_grid(data,rho.shape,length_unit="pc",bbox=bbox)

  #phi = np.linspace(0,2*pi,100)
  #   for iphi in range(100):
  from yt.visualization.volume_rendering.api import Scene, VolumeSource
  sc  = Scene()
  vol = VolumeSource(ds, field="density")
  #vol.set_log(True)
  vol.set_log(True)
#  bound_min = ds.arr(1e5,"K").in_cgs()
#  bound_max = ds.arr(1e9,"K").in_cgs()

  bound_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
  bound_max = ds.arr(10.**2.0,"Msun/pc**3.").in_cgs()
  tf_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
  tf_max = ds.arr(10.**0.5,"Msun/pc**3.").in_cgs()

  # bound_min = ds.arr(0.04/1e-2,"Msun/pc**3.").in_cgs()
  # bound_max = ds.arr(0.24/1e-3,"Msun/pc**3.").in_cgs()
  # bound_min = ds.arr(0.04/2,"Msun/pc**2.") #.in_cgs()
  # bound_max = ds.arr(0.24*10,"Msun/pc**2.")#.in_cgs()
  # tf_min = bound_min #ds.arr(1e-2,"Msun/pc**3.").in_cgs()
  # tf_max = bound_max #ds.arr(10.**0.5,"Msun/pc**3.").in_cgs()

  bounds = (bound_min, bound_max)

  tf = yt.ColorTransferFunction(np.log10(bounds))
  def linramp(vals, minval, maxval):
    return (vals - vals.min())/(vals.max() - vals.min())
  #tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
  tf.add_layers(8, colormap='ocean',mi = np.log10(tf_min),ma = np.log10(tf_max),col_bounds=([np.log10(tf_min),np.log10(tf_max)])) #,w = 0.01,alpha = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])  #ds_highcontrast
  tf.add_step(np.log10(tf_max*2),np.log10(bound_max), [0.5,0.5,0.5,1.0])
  #tf.sample_colormap(5.,w=0.001,colormap='inferno')
#  tf.add_gaussian(np.log10(tf_max*2),width=0.01,height = [1.0,1.0,1.0,1.0])
#  tf.add_gaussian(np.log10(tf_max*4),width=0.01,height = [1.0,1.0,1.0,1.0])
#  tf.add_gaussian(np.log10(tf_max*8),width=0.01,height = [1.0,1.0,1.0,1.0])

  #tf.add_gaussian(np.log10(5e7),width = 0.01,height=[1.0, 0.0, 0, 0.9])
#  tf.add_step(np.log10(3e7),np.log10(1e9),[1.0,0.0,0.0,0.5])
#  tf.add_step(np.log10(1e6),np.log10(3e7),[0.5,0.0,0.5,0.2])
#  tf.add_step(np.log10(1e5),np.log10(1e6),[0.0,0.0,1.0,1.0])
  tf.grey_opacity = False
  vol.transfer_function = tf
  vol.tfh.tf = tf
  vol.tfh.bounds = bounds
  vol.tfh.plot('transfer_function.png', profile_field="density")
  cam = sc.add_camera(ds, lens_type='perspective')
  cam.resolution = [512,512]
  # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
  # cam.switch_orientation(normal_vector=normal_vector,
  #                        north_vector=north_vector)
  cam.set_width(ds.domain_width*0.25)

  #cam.position = ds.arr(np.array([0.5*np.sin(phi),,0.5*np.cos(phi)]),'code_length')
  cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')   #CHANGE WITH PHI
  #cam.position = ds.arr(np.array([0.0,0,-0.01]), 'code_length')
  normal_vector = [0,0,-1]  #camera to focus  #CHANGE WITH PHI
  north_vector = [0,1,0]  #up direction 
  #north_vector = [nx_disk,ny_disk,nz_disk]  
  cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
  sc.add_source(vol)
  sc.render()
  # sc.save('tmp2.png',sigma_clip = 6.0)
  # sc = yt.create_scene(asc.ds,lens_type = 'perspective')
  # sc.camera.zoom(2.0)
  # sc[0].tfh.set_bounds([1e-4,1e2])
  fname = "test.png"
  sc.save(fname,sigma_clip = 6.0)
  for i in cam.iter_rotate(np.pi*2.0,50,rot_center=[0,0,0]):
    sc.render()
    sc.save("test_rotate_%04d.png" %i)

def fold_theta(arr):
  return arr[:,::-1,:]/2.0 + arr[:,:,:]/2.0

def fit_theta():
  rd_hst("star_wind.hst")
  rdnpz("dump_spher_avg_100_110.npz")
  plotting_radius = 0.1 * arc_secs
  theta = np.arccos(z_tavg/r_tavg)
  def get_ir(r_in):
    dlog10r = np.diff(np.log10(r_tavg[:,0,0]))[0]
    r_min = r_tavg[0,0,0]
    r_out = r_tavg[-1,0,0]
    #r = r_min * 10**(ir*dlog10r)
    return np.int(np.round(np.log10(r_in/r_min)/dlog10r))

  ir = get_ir(plotting_radius)

  os.system("export LD_PRELOAD=/global/software/sl-7.x86_64/modules/langs/intel/2016.4.072/mkl/lib/intel64_lin/libmkl_core.so:/global/software/sl-7.x86_64/modules/langs/intel/2016.4.072/mkl/lib/intel64_lin/libmkl_sequential.so")
  from scipy.optimize import curve_fit
  def rho_func(h,rho_0,rho_peak):
    return rho_0 + (rho_peak-rho_0) * np.sin(h)**3.0
  def press_func(h,rho_0,rho_peak):
    return rho_0 + (rho_peak-rho_0) * np.sin(h)**2.5
  def vr_func(h,vr_0,vr_peak):
    return vr_0 + (vr_peak-vr_0) * np.sin(h)**2.0
  def T_func(h,T0):
    return T0
  def vphi_func(h,vphi_peak):
    return vphi_peak * sin(h)

  popt,pcov = curve_fit(rho_func,theta[0,:,0],fold_theta(rho_tavg)[ir,:,:].mean(-1))
  rho_0,rho_peak = popt[0],popt[1]
  popt,pcov = curve_fit(vr_func,theta[0,:,0],fold_theta(vr_avg)[ir,:,:].mean(-1))
  vr_0, vr_peak = popt[0],popt[1]
  popt,pcov = curve_fit(vphi_func,theta[0,:,0],fold_theta(vphi_avg)[ir,:,:].mean(-1))
  vphi_peak = popt[0]
  popt,pcov = curve_fit(press_func,theta[0,:,0],fold_theta(press_tavg)[ir,:,:].mean(-1))
  p_0,p_peak = popt[0],popt[1]

  vr_pole = vr_0 * (r_tavg/plotting_radius)**-1.0
  vr_an = vr_pole + (vr_peak-vr_pole) * sin(theta)**2.0
  rho_an = rho_func(theta,rho_0,rho_peak) * (r_tavg/plotting_radius)**(-1.0)
  press_an = press_func(theta,p_0,p_peak) * (r_tavg/plotting_radius)**(-2.0)
  vphi_an = vphi_func(theta,vphi_peak) * (r_tavg/plotting_radius)**(-0.5)


def mk_spherical_frame(var,min = 4.5,max = 6,cmap="ocean",length_scale=40.0,magnetic = False,gr=False):
  plt.clf()
  rg = 2.058e-7
  if (gr==True): rg = 1;
  tm = 6.7161e-10
#  plt.figure(1)
#  plt.clf()
  plt.xlim(-length_scale,length_scale)
  plt.ylim(-length_scale,length_scale)
  plt.pcolormesh(-(r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,log10((var[:,:,nz//2]) ),vmin=min,vmax=max,cmap = cmap)
  plt.pcolormesh((r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,log10((var[:,:,0]) ),vmin=min,vmax=max,cmap = cmap)
  plt.colorbar()
  if magnetic:
    plt.contour((r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,(psicalc(B1=Bcc1)),0,colors='black')
    plt.contour((-r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,(psicalc(B1=Bcc1)),30,colors='black')

  


def psicalc(B1 = None,gr=False,xy=False):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Bcc1
    _dx2 = np.diff(x2f)
    daphi = -(r*np.sin(th)*B1).mean(-1)*_dx2[None,:]
    if (gr==True): daphi = -(gdet*B1).mean(-1)*_dx2[None,:]
    if (gr== False): aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/np.sin(th)[:,:,0]
    else: aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


    return(aphi)

def psicalc_slice(B1 = None,gr=False,xy=False,iphi = 0):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Bcc1
    if (xy==False):
      _dx2 = np.diff(x2f)
      daphi = -(r*np.sin(th)*B1)[:,:,iphi]*_dx2[None,:]
      if (gr==True): daphi = -(gdet*B1)[:,:,iphi]*_dx2[None,:]
      if (gr== False): aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/np.sin(th)[:,:,0]
      else: aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    else: #Calculate Ay assuming By = 0 (i.e. projecting the magnetic field onto the plane)
      daphi = -B1[:,ny//2,:]
      aphi = daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi


    return(aphi)


#def compute_3D_vector_potential():
  

def psicalc_npz(B1 = None):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Br
    _dx2 = np.diff(th[0,:,0])[0]
    daphi = -(r*np.sin(th)*B1).mean(-1)*_dx2
    aphi  = daphi[:,::-1].cumsum(axis=1)[:,::-1]/(np.sin(th)[:,:,0]+1e-15)
    aphi2 = -daphi[:,:].cumsum(axis=1)[:,:]/(np.sin(th)[:,:,0]+1e-15)

    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    aphi2-=0.5*daphi

    aphi_avg = aphi * np.sin(th[:,:,0]/2.0)**2.0 + aphi2 * np.cos(th[:,:,0]/2.0)**2.0

    #aphi_avg = 0 
    return(aphi_avg)

def psicalc_npz_avg(B1 = None):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Br_avg
    _dx2 = np.diff(th_tavg[0,:,0])[0]
    daphi = -(r_tavg*np.sin(th_tavg)*B1).mean(-1)*_dx2
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/(np.sin(th_tavg)[:,:,0]+1e-15)
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


    return(aphi)

def psicalc_npz_gr(B1 = None,a = 0, gdet = None):
    """
    Computes the field vector potential

    """

  ##Bcc1 = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])

    if (B1 is None): B1 = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])
    _dx2 = np.diff(th[0,:,0])[0]
    daphi = -(r*np.sin(th)*B1).mean(-1)*_dx2
    if (gdet is None and "th" in globals() ): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(th)
    elif (gdet is None): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(theta)
    daphi = -(gdet*B1).mean(-1)*_dx2
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


    return(aphi)

# def psicalc_npz_avg(B2 = None):
#     """
#     Computes the field vector potential
#     """
#     if (B2 is None): B2 = Bth_avg
#     _dx1 = np.diff(th_tavg[0,:,0])[0]
#     daphi = -(r_tavg*np.sin(th_tavg)*B1).mean(-1)*_dx2
#     aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/(np.sin(th_tavg)[:,:,0]+1e-15)
#     aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


#     return(aphi)

def psicalc_xy(B1=None,B2 = None,x1 = None,x2 = None,slice=False):
    """
    Computes the field vector potential
    """
    if (B2 is None): B2 = Bcc2
    if (B1 is None): B1 = Bcc1
    if (x1 is None): x1 = x1v
    if (x2 is None): x2 = x2v

    _dx1 = np.gradient(x1)
    _dx2 = np.gradient(x2)
    if (slice==False):
      daz1 = -(B2).mean(-1)*(_dx1[:,None])
      daz2 = (B1).mean(-1)*_dx2[None,:]
    else:
      daz1 = -(B2)[:,:,B2.shape[-1]//2]*_dx1[:,None]
      daz2 = (B1)[:,:,B1.shape[-1]//2]*_dx2[None,:]   
    az1=daz1[:,:].cumsum(axis=0)[:,:]
    az2=daz2[:,:].cumsum(axis=1)[:,:]
    az1-=0.5*daz1 #correction for half-cell shift between face and center in theta
    az2-=0.5*daz2

    az = az1 - np.gradient(az1,axis=1).cumsum(axis=1) + az2


    return(az)
def bl_metric(r,th,a=0):
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  gbl = np.zeros((4,4))
  Sigma = r**2 + a**2.*np.cos(th)**2;
  Delta = r**2 -2.0*r + a**2;
  A = (r**2 + a**2)**2 - a**2*Delta*np.sin(th)**2.;


  nx = r.shape[0]
  ny = r.shape[1]
  nz = r.shape[2]
  gbl  = np.zeros((4,4,nx,ny,nz))
  gbl[0][0] = - (1 - 2.0*r/Sigma);
  gbl[1][0] = 0;
  gbl[0][1] = 0;
  gbl[1][1] = Sigma/Delta;
  gbl[2][0] = 0;
  gbl[0][2] = 0;
  gbl[2][1] = 0;
  gbl[1][2] = 0;
  gbl[2][2] = Sigma;
  gbl[3][0] = -2*a*r*sin2/Sigma;
  gbl[0][3] = gbl[3][0];
  gbl[1][3] = 0;
  gbl[3][1] = 0;
  gbl[3][2] = 0;
  gbl[3][3] = A*sin2/Sigma;

  return gbl

def bl_vec_to_ks(A,a=0):
  Delta = r**2 -2.0*r + a**2;
  tmp = A*0
  tmp[0] = A[0] + A[1] * 2.*r/Delta
  tmp[1] = A[1] 
  tmp[2] = A[2]
  tmp[3] = A[3] + A[1] * a/Delta
  return tmp 
def ks_vec_to_bl(A,a=0):
  Delta = r**2 -2.0*r + a**2;
  tmp = A*0
  tmp[0] = A[0] - A[1] * 2.*r/Delta
  tmp[1] = A[1] 
  tmp[2] = A[2]
  tmp[3] = A[3] - A[1] * a/Delta
  return tmp


#uu_ks^\mu = uu_cks^\nu dx_ks^\mu_/dx_cks^\nu
def cks_vec_to_ks(A,x,y,z,ax,ay,az):
    a = np.sqrt(ax**2.0+ay**2.0+az**2.0)
    # R = np.sqrt(x**2+y**2+z**2)
    # r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*a**2*z**2 ) )/np.sqrt(2.0)
    # A_ks = A*0
    # #uu_ks^\mu = uu_cks^\nu dx_ks^\mu_/dx_cks^\nu
    # SMALL = 1e-15
    # sqrt_term = 2.0*r**2-R**2+a**2

    # ax = 0.0
    # ay = 0.0
    # az = a*1.0
    [pr,pth,pphi] = GetBoyerLindquistCoordinates(x,y,z,ax,ay,az)

    jac = np.zeros((3,3,x.shape[0],x.shape[1],x.shape[2]))

    ##jac = dx^cks/dx^ks row column
    jac[0,0] = np.sin(pth) * np.cos(pphi)
    jac[1,0] = np.sin(pth) * np.sin(pphi)
    jac[2,0] = np.cos(pth)

    jac[0,1] = ( pr*np.cos(pphi) - az*np.sin(pphi) )*np.cos(pth) - ay*np.sin(pth)
    jac[1,1] = ( az*np.cos(pphi) + pr*np.sin(pphi) )*np.cos(pth) + ax*np.sin(pth)
    jac[2,1] =-( ay*np.cos(pphi) - ax*np.sin(pphi) )*np.cos(pth) - pr*np.sin(pth)

    jac[0,2] =-( az*np.cos(pphi) + pr*np.sin(pphi) )*np.sin(pth)
    jac[1,2] = ( pr*np.cos(pphi) - az*np.sin(pphi) )*np.sin(pth)
    jac[2,2] = ( ax*np.cos(pphi) + ay*np.sin(pphi) )*np.sin(pth)

    jac_inv = np.linalg.inv(jac.transpose(2,3,4,0,1)).transpose(3,4,0,1,2)

    # A_ks[0] = A[0] 
    # A_ks[1] = A[1] * (x*r)/sqrt_term + \
    #            A[2] * (y*r)/sqrt_term + \
    #            A[3] * z/r * (r**2+a**2)/sqrt_term
    # A_ks[2] = A[1] * (x*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) + \
    #            A[2] * (y*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) + \
    #            A[3] * ( (z*z)*(r**2+a**2)/(r**3 * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) - 1.0/(r*np.sqrt(1.0-z**2/r**2) + SMALL) )
    # A_ks[3] = A[1] * (-y/(x**2+y**2+SMALL) + a*r*x/((r**2+a**2)*sqrt_term)) + \
    #            A[2] * (x/(x**2+y**2+SMALL) + a*r*y/((r**2+a**2)*sqrt_term)) + \
    #            A[3] * (a* z/r/sqrt_term) 

    A_ks = A*0
    A_ks[0] = A[0] 
    A_ks[1] = A[1] * jac_inv[0,0] + A[2] * jac_inv[0,1] + A[3] * jac_inv[0,2]
    A_ks[2] = A[1] * jac_inv[1,0] + A[2] * jac_inv[1,1] + A[3] * jac_inv[1,2]
    A_ks[3] = A[1] * jac_inv[2,0] + A[2] * jac_inv[2,1] + A[3] * jac_inv[2,2]

    return A_ks

def ks_vec_to_cks(A,x,y,z,a=0):
    def SQR(var):
      return var**2.0
    R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
    r = np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/np.sqrt(2.0);
    a0_ks = A[0]
    a1_ks = A[1]
    a2_ks = A[2]
    a3_ks = A[3]
    pa0 = a0_ks ;
    pa1 = a1_ks * ( (r*x+a*y)/(SQR(r) + SQR(a))) + \
           a2_ks * x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - \
           a3_ks * y; 
    pa2 = a1_ks * ( (r*y-a*x)/(SQR(r) + SQR(a))) + \
           a2_ks * y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + \
           a3_ks * x;
    pa3 = a1_ks * z/r - \
           a2_ks * r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)))
    return np.array([pa0,pa1,pa2,pa3])

## A_ks^mu_\nu = A_cks^_ dx_ks^\mu/dx_cks^\lam dx_cks^\beta/dx_ks^\nu
# def cks_udtensor_to_ks(A,x,y,z,a=0):

def cks_coord_to_ks(x,y,z,a=0):
    global r, th,ph
    def SQR(b):
      return b**2.0
    R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
    r = np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/np.sqrt(2.0);

    th = np.arccos(z/r)

    ph = np.arctan2((r*y-a*x), (a*y+r*x) )

def bl_vec_to_cks(x,y,z,A,a=0):
    A_cks = np.array(A)*0
    def SQR(q):
      return q**2.0
    m = 1

    R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
    r = np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/np.sqrt(2.0);
    delta = SQR(r) - 2.0*m*r + SQR(a);
    A_cks[0] = A[0] + 2.0*r/delta * A[1];
    A_cks[1] = A[1] * ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta) + A[2] * x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - A[3] * y
    A_cks[2] = A[1] * ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta) + A[2] * y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + A[3] * x
    A_cks[3] = A[1] * z/r - A[2] * r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)))

    return A_cks


def rotate_cks_coord(x,y,z,idir):
  if (idir==0): ##about x
    x1 = y*1.0
    x2 = z*1.0
    x3 = x*1.0

  elif (idir==1): ##about y
    x1 = z*1.0
    x2 = x*1.0
    x3 = y*1.0

  else: ##about z
    x1 = x*1.0
    x2 = y*1.0
    x3 = z*1.0

  return (x1,x2,x3)

def rotate_cks_4_vec(A,idir):
  uu_rot = A*1.0
  if (idir==0): ##about x

    uu_rot[1] = A[2]*1.0
    uu_rot[2] = A[3]*1.0
    uu_rot[3] = A[1]*1.0

  elif (idir==1): ##about y

    uu_rot[1] = A[3]*1.0
    uu_rot[2] = A[1]*1.0
    uu_rot[3] = A[2]*1.0

  else: ##about z

    uu_rot[1] = A[1]*1.0
    uu_rot[2] = A[2]*1.0
    uu_rot[3] = A[3]*1.0

  return uu_rot
# def get_uu_ks_rotated(uu,x,y,z,ax,ay,az,idir):

#   uu_rot = uu*1.0
#   if (idir==0): ##about x
#     x1 = y*1.0
#     x2 = z*1.0
#     x3 = x*1.0

#     uu_rot[1] = uu[2]*1.0
#     uu_rot[2] = uu[3]*1.0
#     uu_rot[3] = uu[1]*1.0

#     a1 = ay
#     a2 = az
#     a3 = ax
#   elif (idir==1): ##about y
#     x1 = z*1.0
#     x2 = x*1.0
#     x3 = y*1.0

#     uu_rot[1] = uu[3]*1.0
#     uu_rot[2] = uu[1]*1.0
#     uu_rot[3] = uu[2]*1.0

#     a1 = az
#     a2 = ax
#     a3 = ay
#   else: ##about z
#     x1 = x*1.0
#     x2 = y*1.0
#     x3 = z*1.0

#     uu_rot[1] = uu[1]*1.0
#     uu_rot[2] = uu[2]*1.0
#     uu_rot[3] = uu[3]*1.0

#     a1 = ax
#     a2 = ay
#     a3 = az



#   uu_ks_rot  = ks_vec_to_ks(uu_rot,x1,x2,x3,a1,a2,a3)

#   return x1,x2,x3,a1,a2,a3,uu_ks_rot


def get_Tud_gdet_ks_rotated(x,y,z,ax,ay,az,idir,is_magnetic=True,gam=5.0/3.0):

  x1,x2,x3 = rotate_cks_coord(x,y,z,idir)
  a1,a2,a3 = rotate_cks_coord(ax,ay,az,idir)
  uu_rot = rotate_cks_4_vec(uu,idir)
  bu_rot = rotate_cks_4_vec(bu,idir)

  uu_ks_rot = cks_vec_to_ks(uu_rot,x1,x2,x3,a1,a2,a3)
  bu_ks_rot = cks_vec_to_ks(bu_rot,x1,x2,x3,a1,a2,a3)

  g = ks_metric_general(x1,x2,x3,a1,a2,a3)

  ud_ks_rot = nan_to_num(Lower(uu_ks_rot,g))
  bd_ks_rot = nan_to_num(Lower(bu_ks_rot,g))

  gdet = Determinant_4b4(g)
  Tud_calc(uu_ks_rot,ud_ks_rot,bu_ks_rot,bd_ks_rot,is_magnetic = is_magnetic,gam=gam)

  return Tud,gdet

  #telemarketing


def ks_metric(r,th,a):
  global g
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  g  = np.zeros((4,4,nx,ny,nz))
  g[0][0] = -(1.0 - 2.0*m*r/sigma);
  g[0][1] = 2.0*m*r/sigma;
  g[1][0] = g[0][1]
  g[0][3] = -2.0*m*a*r/sigma * sin2
  g[3][0] = g[0][3]
  g[1][1] = 1.0 + 2.0*m*r/sigma
  g[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2
  g[3][1] = g[1][3]
  g[2][2] = sigma
  g[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2

def ks_metric_general(x,y,z,ax,ay,az):

    cks_metric(x,y,z,ax,ay,az)

    [pr,pth,pphi] = GetBoyerLindquistCoordinates(x,y,z,ax,ay,az)

    jac = np.zeros((4,4,x.shape[0],x.shape[1],x.shape[2]))

    ##jac = dx^cks/dx^ks row column
    jac[0,0] = 1.0
    jac[1,0] = 0.0
    jac[2,0] = 0.0
    jac[3,0] = 0.0
    jac[0,1] = 0.0
    jac[0,2] = 0.0
    jac[0,3] = 0.0

    jac[1,1] = np.sin(pth) * np.cos(pphi)
    jac[2,1] = np.sin(pth) * np.sin(pphi)
    jac[3,1] = np.cos(pth)

    jac[1,2] = ( pr*np.cos(pphi) - az*np.sin(pphi) )*np.cos(pth) - ay*np.sin(pth)
    jac[2,2] = ( az*np.cos(pphi) + pr*np.sin(pphi) )*np.cos(pth) + ax*np.sin(pth)
    jac[3,2] =-( ay*np.cos(pphi) - ax*np.sin(pphi) )*np.cos(pth) - pr*np.sin(pth)

    jac[1,3] =-( az*np.cos(pphi) + pr*np.sin(pphi) )*np.sin(pth)
    jac[2,3] = ( pr*np.cos(pphi) - az*np.sin(pphi) )*np.sin(pth)
    jac[3,3] = ( ax*np.cos(pphi) + ay*np.sin(pphi) )*np.sin(pth)

    jac_inv = np.linalg.inv(jac.transpose(2,3,4,0,1)).transpose(3,4,0,1,2)

    # g_ks_mu_nu = g_cks_alpha_beta dxcks^alpha/dx_ks^beta
    g_ks = g*0.0
    for i in range(4):
      for j in range(4):
        for k in range(4):
          for m in range(4):
            g_ks[i][j] += g[k][m] * jac[k][i] * jac[m][j]

    return g_ks

def ks_Gamma_ud(radius,theta,a,m=1):
  global gammaud
  gammaud = np.zeros((4,4,4,nx,ny))
  r= radius[:,:,0]
  th = theta[:,:,0]
  gammaud[0][0][0] = -2*(a**2*m**2*r*cos(th)**2 - m**2*r**3)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][0][1] = -(a**4*m*cos(th)**4 + 2*a**2*m**2*r*cos(th)**2 - 2*m**2*r**3 - m*r**4)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][0][2] = -2*a**2*m*r*cos(th)*sin(th)/(a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4) 
  gammaud[0][0][3] = 2*(a**3*m**2*r*cos(th)**2 - a*m**2*r**3)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][1][1] = -2*(a**4*m*cos(th)**4 + a**2*m**2*r*cos(th)**2 - m**2*r**3 - m*r**4)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][1][2] = -2*a**2*m*r*cos(th)*sin(th)/(a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4) 
  gammaud[0][1][3] = (a**5*m*cos(th)**4 + 2*a**3*m**2*r*cos(th)**2 - 2*a*m**2*r**3 - a*m*r**4)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][2][2] = -2*m*r**2/(a**2*cos(th)**2 + r**2) 
  gammaud[0][2][3] = 2*a**3*m*r*cos(th)*sin(th)**3/(a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4) 
  gammaud[0][3][3] = -2*((a**4*m**2*r*cos(th)**2 - a**2*m**2*r**3)*sin(th)**4 + (a**4*m*r**2*cos(th)**4 + 2*a**2*m*r**4*cos(th)**2 + m*r**6)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][0][0] = (a**2*m*r**2 - 2*m**2*r**3 + m*r**4 - (a**4*m - 2*a**2*m**2*r + a**2*m*r**2)*cos(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][0][1] = (2*a**2*m**2*r*cos(th)**2 - 2*m**2*r**3 - (a**4*m*cos(th)**2 - a**2*m*r**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][0][3] = -(a**3*m*r**2 - 2*a*m**2*r**3 + a*m*r**4 - (a**5*m - 2*a**3*m**2*r + a**3*m*r**2)*cos(th)**2)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][1][1] = (2*a**4*m*cos(th)**4 + a**2*m*r**2 - 2*m**2*r**3 - m*r**4 - (a**4*m - 2*a**2*m**2*r + a**2*m*r**2)*cos(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][1][2] = -a**2*cos(th)*sin(th)/(a**2*cos(th)**2 + r**2) 
  gammaud[1][1][3] = ((a**5*m*cos(th)**2 - a**3*m*r**2)*sin(th)**4 + (a**5*r*cos(th)**4 + 2*a*m**2*r**3 + a*r**5 - 2*(a**3*m**2*r - a**3*r**3)*cos(th)**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][2][2] = -(a**2*r - 2*m*r**2 + r**3)/(a**2*cos(th)**2 + r**2) 
  gammaud[1][3][3] = ((a**4*m*r**2 - 2*a**2*m**2*r**3 + a**2*m*r**4 - (a**6*m - 2*a**4*m**2*r + a**4*m*r**2)*cos(th)**2)*sin(th)**4 - (a**2*r**5 - 2*m*r**6 + r**7 + (a**6*r - 2*a**4*m*r**2 + a**4*r**3)*cos(th)**4 + 2*(a**4*r**3 - 2*a**2*m*r**4 + a**2*r**5)*cos(th)**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][0][0] = -2*a**2*m*r*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][0][1] = -2*a**2*m*r*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][0][3] = 2*(a**3*m*r + a*m*r**3)*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][1][1] = -2*a**2*m*r*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][1][2] = r/(a**2*cos(th)**2 + r**2) 
  gammaud[2][1][3] = (a**5*cos(th)**5 + 2*a**3*r**2*cos(th)**3 + (2*a**3*m*r + 2*a*m*r**3 + a*r**4)*cos(th))*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][2][2] = -a**2*cos(th)*sin(th)/(a**2*cos(th)**2 + r**2) 
  gammaud[2][3][3] = -((a**6 - 2*a**4*m*r + a**4*r**2)*cos(th)**5 + 2*(a**4*r**2 - 2*a**2*m*r**3 + a**2*r**4)*cos(th)**3 + (2*a**4*m*r + 4*a**2*m*r**3 + a**2*r**4 + r**6)*cos(th))*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][0][0] = -(a**3*m*cos(th)**2 - a*m*r**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][0][1] = -(a**3*m*cos(th)**2 - a*m*r**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][0][2] = -2*a*m*r*cos(th)/((a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4)*sin(th)) 
  gammaud[3][0][3] = (a**4*m*cos(th)**2 - a**2*m*r**2)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][1][1] = -(a**3*m*cos(th)**2 - a*m*r**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][1][2] = -(a**3*cos(th)**3 + (2*a*m*r + a*r**2)*cos(th))/((a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4)*sin(th)) 
  gammaud[3][1][3] = (a**4*r*cos(th)**4 + 2*a**2*r**3*cos(th)**2 + r**5 + (a**4*m*cos(th)**2 - a**2*m*r**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][2][2] = -a*r/(a**2*cos(th)**2 + r**2) 
  gammaud[3][2][3] = (a**4*cos(th)**5 - 2*(a**2*m*r - a**2*r**2)*cos(th)**3 + (2*a**2*m*r + r**4)*cos(th))/((a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4)*sin(th)) 
  gammaud[3][3][3] = -((a**5*m*cos(th)**2 - a**3*m*r**2)*sin(th)**4 + (a**5*r*cos(th)**4 + 2*a**3*r**3*cos(th)**2 + a*r**5)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6)

  for i in arange(4):
    for j in arange(1,4):
      for k in arange(0,j):
        gammaud[i][j][k] = gammaud[i][k][j]

def ks_inverse_metric(r,th,a):
  global gi
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  delta = r2 - 2.0*m*r + a2
  gi  = np.zeros((4,4,nx,ny,nz))
  gi[0][0] = -(1.0 + 2.0*m*r/sigma);
  gi[0][1] = 2.0*m*r/sigma;
  gi[1][0] = g[0][1]
  gi[0][3] = 0
  gi[3][0] = g[0][3]
  gi[1][1] = delta/sigma
  gi[1][3] = a/sigma
  gi[3][1] = g[1][3]
  gi[2][2] = 1.0/sigma
  gi[3][3] = 1.0 / (sigma * sin2)

def cks_metric(x,y,z,ax,ay,az,ONED=False):
  global g

  def SQR(arg_):
    return arg_*arg_
  m = 1.0

  a = np.sqrt(ax**2+ay**2+az**2)

  a_dot_x = ax * x + ay * y + az * z;

  small = 1e-5
  diff =  (abs(a_dot_x)<small) * ( small * (a_dot_x>=0) - small * (a_dot_x<0)  -  a_dot_x/(a+small)  )

  x = x + diff*ax/(a+small);
  y = y + diff*ay/(a+small);
  z = z + diff*az/(a+small);

  a_dot_x = ax * x + ay * y + az * z;

  x[abs(x)<0.1] = 0.1
  y[abs(y)<0.1] = 0.1
  z[abs(z)<0.1] = 0.1

  [r,th,phi] = GetBoyerLindquistCoordinates(x,y,z,ax,ay,az);

  rh =  ( m + np.sqrt(SQR(m)-SQR(a)) );
  r[r<rh*0.5] = rh * 0.5
  [x,y,z] = convert_spherical_to_cartesian_ks(r,th,phi, ax,ay,az)
  
  a_dot_x = ax * x + ay * y + az * z;


  a_cross_x = [ay * z - az * y,
               az * x - ax * z,
               ax * y - ay * x]


  rsq_p_asq = SQR(r) + SQR(a);



  f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a_dot_x));

  l0 = 1.0;
  l1 = (r * x - a_cross_x[0] + a_dot_x * ax/r)/(rsq_p_asq);
  l2 = (r * y - a_cross_x[1] + a_dot_x * ay/r)/(rsq_p_asq);
  l3 = (r * z - a_cross_x[2] + a_dot_x * az/r)/(rsq_p_asq);


  # nx_ = x.shape[0]
  # ny_ = x.shape[1]
  # nz_ = x.shape[2]

  if (ONED==False):
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]
    g  = np.zeros((4,4,nx,ny,nz))
  else: g = np.zeros((4,4,x.shape[0]))
  # g  = np.zeros((4,4,nx_,ny_,nz_))
  g[0][0] = -1.0 + f * l0*l0
  g[0][1] = f * l0*l1
  g[0][2] = f * l0*l2
  g[0][3] = f * l0*l3
  g[1][1] = 1.0 + f * l1*l1
  g[1][3] = f * l1*l3
  g[2][2] = 1.0 + f * l2*l2
  g[2][3] = f * l2*l3 
  g[1][2] = f * l1*l2
  g[3][3] = 1.0 + f * l3*l3

  g[1][0] = g[0][1]
  g[2][0] = g[0][2]
  g[3][0] = g[0][3]
  g[3][1] = g[1][3]
  g[3][2] = g[2][3]
  g[2][1] = g[1][2]

  # m = 1
  # R = np.sqrt(x**2+y**2+z**2)
  # r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)

  # f = 2.0*r**3/(r**4 + a**2*z**2)
  # l0_ = 1.0
  # l1_ = (r*x+a*y)/(r**2+a**2)
  # l2_ = (r*y-a*x)/(r**2+a**2)
  # l3_ = z/r

  # if (ONED==False):
  #   nx = x.shape[0]
  #   ny = x.shape[1]
  #   nz = x.shape[2]
  #   g_  = np.zeros((4,4,nx,ny,nz))
  # else: g_ = np.zeros((4,4,x.shape[0]))
  # g_[0][0] = -1.0 + f * l0_*l0_;
  # g_[0][1] = f*l0_*l1_;
  # g_[1][0] = g_[0][1]
  # g_[0][2] = f*l0_*l2_
  # g_[2][0] = g_[0][2]
  # g_[0][3] = f*l0_*l3_
  # g_[3][0] = g_[0][3]
  # g_[1][1] = 1.0 + f*l1_*l1_
  # g_[1][3] =  f*l1_*l3_
  # g_[3][1] = g_[1][3]
  # g_[2][2] = 1.0 + f*l2_*l2_
  # g_[2][3] = f*l2_*l3_ 
  # g_[3][2] = g_[2][3]
  # g_[1][2] = f*l1_*l2_
  # g_[2][1] = g_[1][2]
  # g_[3][3] = 1.0 + f*l3_*l3_

  # def SQR(c):
  #   return c**2
  # def pow(c,d):
  #   return c**d

  # sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a)
  # rsq_p_asq = SQR(r) + SQR(a)

  # df_dx1 = SQR(f)*x/(2.0*pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  # df_dx2 = SQR(f)*y/(2.0*pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  # df_dx3 = SQR(f)*z/(2.0*pow(r,5)) * ( ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(a*r)) ;
  # dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  # dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ a/( rsq_p_asq );
  # dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (rsq_p_asq) * ( sqrt_term ) ) ;
  # dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - a/( rsq_p_asq );
  # dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  # dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (rsq_p_asq) * ( sqrt_term ) );
  # dl3_dx1 = - x*z/(r) /( sqrt_term );
  # dl3_dx2 = - y*z/(r) /( sqrt_term );
  # dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( rsq_p_asq )/( sqrt_term ) + 1.0/r;

def cks_inverse_metric(x,y,z,ax,ay,az,ONED = False):
  global gi,gi_

  def SQR(arg_):
    return arg_*arg_
  m = 1.0

  a = np.sqrt(ax**2+ay**2+az**2)

  a_dot_x = ax * x + ay * y + az * z;

  small = 1e-5
  diff =  (abs(a_dot_x)<small) * ( small * (a_dot_x>=0) - small * (a_dot_x<0)  -  a_dot_x/(a+small)  )

  x = x + diff*ax/(a+small);
  y = y + diff*ay/(a+small);
  z = z + diff*az/(a+small);

  a_dot_x = ax * x + ay * y + az * z;

  x[abs(x)<0.1] = 0.1
  y[abs(y)<0.1] = 0.1
  z[abs(z)<0.1] = 0.1

  [r,th,phi] = GetBoyerLindquistCoordinates(x,y,z,ax,ay,az);

  rh =  ( m + np.sqrt(SQR(m)-SQR(a)) );
  r[r<rh*0.5] = rh * 0.5
  [x,y,z] = convert_spherical_to_cartesian_ks(r,th,phi, ax,ay,az)
  
  a_dot_x = ax * x + ay * y + az * z;


  a_cross_x = [ay * z - az * y,
               az * x - ax * z,
               ax * y - ay * x]


  rsq_p_asq = SQR(r) + SQR(a);



  f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a_dot_x));

  l0 = -1.0;
  l1 = (r * x - a_cross_x[0] + a_dot_x * ax/r)/(rsq_p_asq);
  l2 = (r * y - a_cross_x[1] + a_dot_x * ay/r)/(rsq_p_asq);
  l3 = (r * z - a_cross_x[2] + a_dot_x * az/r)/(rsq_p_asq);


  # nx_ = x.shape[0]
  # ny_ = x.shape[1]
  # nz_ = x.shape[2]
  # gi  = np.zeros((4,4,nx_,ny_,nz_))

  if (ONED ==False):
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]
    gi  = np.zeros((4,4,nx,ny,nz))
  else: gi  = np.zeros((4,4,x.shape[0]))
  gi[0][0] = -1.0 - f * l0*l0
  gi[0][1] = -f * l0*l1
  gi[0][2] = -f * l0*l2
  gi[0][3] = -f * l0*l3
  gi[1][1] = 1.0 - f * l1*l1
  gi[1][3] = -f * l1*l3
  gi[2][2] = 1.0 - f * l2*l2
  gi[2][3] = -f * l2*l3 
  gi[1][2] = -f * l1*l2
  gi[3][3] = 1.0 - f * l3*l3

  gi[1][0] = gi[0][1]
  gi[2][0] = gi[0][2]
  gi[3][0] = gi[0][3]
  gi[3][1] = gi[1][3]
  gi[3][2] = gi[2][3]
  gi[2][1] = gi[1][2]



  # m = 1

  # a = np.sqrt(ax**2+ay**2+az**2)
  # R = np.sqrt(x**2+y**2+z**2)
  # r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)

  # f = 2.0*r**3/(r**4 + a**2*z**2)
  # l0 = - 1.0
  # l1 = (r*x+a*y)/(r**2+a**2)
  # l2 = (r*y-a*x)/(r**2+a**2)
  # l3 = z/r

  # if (ONED ==False):
  #   nx = x.shape[0]
  #   ny = x.shape[1]
  #   nz = x.shape[2]
  #   gi_  = np.zeros((4,4,nx,ny,nz))
  # else: gi_  = np.zeros((4,4,x.shape[0]))
  # gi_[0][0] = -1.0 - f * l0*l0;
  # gi_[0][1] = -f*l0*l1;
  # gi_[1][0] = gi_[0][1]
  # gi_[0][2] = -f*l0*l2
  # gi_[2][0] = gi_[0][2]
  # gi_[0][3] = -f*l0*l3
  # gi_[3][0] = gi_[0][3]
  # gi_[1][1] = 1.0 - f*l1*l1
  # gi_[1][3] = -f*l1*l3
  # gi_[3][1] = gi_[1][3]
  # gi_[2][2] = 1.0 - f*l2*l2
  # gi_[2][3] = -f*l2*l3 
  # gi_[3][2] = gi_[2][3]
  # gi_[1][2] = -f*l1*l2
  # gi_[2][1] = gi_[1][2]
  # gi_[3][3] = 1.0 - f*l3*l3

def Determinant_4b4(A):
  a11 = A[0][0];
  a12 = A[0][1];
  a13 = A[0][2];
  a14 = A[0][3];
  a21 = A[1][0];
  a22 = A[1][1];
  a23 = A[1][2];
  a24 = A[1][3];
  a31 = A[2][0];
  a32 = A[2][1];
  a33 = A[2][2];
  a34 = A[2][3];
  a41 = A[3][0];
  a42 = A[3][1];
  a43 = A[3][2];
  a44 = A[3][3];
  det = (a11 * Determinant_3b3(a22, a23, a24, a32, a33, a34, a42, a43, a44) 
           - a12 * Determinant_3b3(a21, a23, a24, a31, a33, a34, a41, a43, a44) 
           + a13 * Determinant_3b3(a21, a22, a24, a31, a32, a34, a41, a42, a44) 
           - a14 * Determinant_3b3(a21, a22, a23, a31, a32, a33, a41, a42, a43) )
  return det

def Determinant_3b3(a11, a12, a13, a21, a22, a23,a31, a32, a33):
  det = (a11 * Determinant_2b2(a22, a23, a32, a33) - 
              a12 * Determinant_2b2(a21, a23, a31, a33) + 
              a13 * Determinant_2b2(a21, a22, a31, a32) )
  return det

def Determinant_2b2(a11, a12, a21, a22):
  return a11 * a22 - a12 * a21

def get_bl_coords(x,y,z,a=0):
  global r,th

  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)

  th = np.arccos(z/r)

def cks_bl_jac(x,y,z,a=0):
  global jac_cks_bl
  jac_cks_bl = np.zeros((4,4,nx,ny,nz))

  ##dx^\mu_cks/dx^\nu_Bl
  m=1
  def SQR(c):
    return c**2
  def pow(c,d):
    return c**d
  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)
  delta = SQR(r) - 2.0*m*r + SQR(a);


    # *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    # *pa1 = a1_bl * ( (r*x-a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
    #        a2_bl * x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
    #        a3_bl * y; 
    # *pa2 = a1_bl * ( (r*y+a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
    #        a2_bl * y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
    #        a3_bl * x;
    # *pa3 = a1_bl * z/r - 
    #        a2_bl * r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_bl[0][0] = 1.0;
  jac_cks_bl[0][1] = 2.0*r/delta
  jac_cks_bl[0][2] = 0.0
  jac_cks_bl[0][3] = 0.0
  jac_cks_bl[1][0] = 0.0
  jac_cks_bl[2][0] = 0.0
  jac_cks_bl[3][0] = 0.0
  jac_cks_bl[1][1] = ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta)
  jac_cks_bl[1][2] = x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)))
  jac_cks_bl[1][3] = -y
  jac_cks_bl[2][1] = ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta)

  jac_cks_bl[2][2] = y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)));
  jac_cks_bl[2][3] = +x
  jac_cks_bl[3][1] = z/r
  jac_cks_bl[3][2] = -r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_bl[3][3] = 0.0

def cks_ks_jac(x,y,z,a=0):
  global jac_cks_ks
  jac_cks_ks = np.zeros((4,4,nx,ny,nz))

  ##dx^\mu_cks/dx^\nu_Bl
  m=1
  def SQR(c):
    return c**2
  def pow(c,d):
    return c**d
  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)
  delta = SQR(r) - 2.0*m*r + SQR(a);


    # *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    # *pa1 = a1_bl * ( (r*x-a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
    #        a2_bl * x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
    #        a3_bl * y; 
    # *pa2 = a1_bl * ( (r*y+a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
    #        a2_bl * y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
    #        a3_bl * x;
    # *pa3 = a1_bl * z/r - 
    #        a2_bl * r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_ks[0][0] = 1.0;
  jac_cks_ks[0][1] = 0.0
  jac_cks_ks[0][2] = 0.0
  jac_cks_ks[0][3] = 0.0
  jac_cks_ks[1][0] = 0.0
  jac_cks_ks[2][0] = 0.0
  jac_cks_ks[3][0] = 0.0
  jac_cks_ks[1][1] = ( (r*x+a*y)/(SQR(r) + SQR(a)) )
  jac_cks_ks[1][2] = x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)))
  jac_cks_ks[1][3] = -y
  jac_cks_ks[2][1] = ( (r*y-a*x)/(SQR(r) + SQR(a)))

  jac_cks_ks[2][2] = y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)));
  jac_cks_ks[2][3] = +x
  jac_cks_ks[3][1] = z/r
  jac_cks_ks[3][2] = -r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_ks[3][3] = 0.0

def cks_cks_prime_jac(r_bh2=10.0,t0=0.0,inclination=0.0):
  global jac_cks_cks_prime
  jac_cks_cks_prime = np.zeros((4,4,nx,ny,nz))



  vxbh,vybh,vzbh = bh2_vel(t,r_bh2,t0=t0,inclination=inclination)
    # *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    # *pa1 = a1_bl * ( (r*x-a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
    #        a2_bl * x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
    #        a3_bl * y; 
    # *pa2 = a1_bl * ( (r*y+a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
    #        a2_bl * y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
    #        a3_bl * x;
    # *pa3 = a1_bl * z/r - 
    #        a2_bl * r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_cks_prime[0][0] = 1.0;
  jac_cks_cks_prime[0][1] = 0.0
  jac_cks_cks_prime[0][2] = 0.0
  jac_cks_cks_prime[0][3] = 0.0
  jac_cks_cks_prime[1][0] = vxbh
  jac_cks_cks_prime[2][0] = vybh
  jac_cks_cks_prime[3][0] = vzbh
  jac_cks_cks_prime[1][1] = 1.0
  jac_cks_cks_prime[1][2] = 0.0
  jac_cks_cks_prime[1][3] = 0.0
  jac_cks_cks_prime[2][1] = 0.0

  jac_cks_cks_prime[2][2] = 1.0
  jac_cks_cks_prime[2][3] = 0.0
  jac_cks_cks_prime[3][1] = 0.0
  jac_cks_cks_prime[3][2] = 0.0
  jac_cks_cks_prime[3][3] = 1.0

def Tud_calc(uu,ud,bu,bd,is_magnetic = False,gam=5.0/3.0):
  global Tud,TudMA,TudEM
  w = rho+press * (gam)/(gam-1.0)
  Tud = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
  TudMA = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
  TudEM = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
  for kapa in np.arange(4):
    for nu in np.arange(4):
      if(kapa==nu): delta = 1
      else: delta = 0
      TudMA[kapa,nu] = w*uu[kapa]*ud[nu]+press*delta
      if (is_magnetic==True): TudEM[kapa,nu] = bsq*uu[kapa]*ud[nu] + 0.5*bsq*delta - bu[kapa]*bd[nu]
      if (is_magnetic==True): Tud[kapa,nu] = TudEM[kapa,nu] + TudMA[kapa,nu]
      else: Tud[kapa,nu] = TudMA[kapa,nu]

def Tdd_calc(Tud,g):
  global Tdd
  Tdd = 0*Tud
  for i in range(4):
    for j in range(4):
      for k in range(4):
        Tdd[i][j] += g[k,i]*Tud[k,j]

def Tdd_cks_to_ks(Tdd,x,y,z,a=0):
  global Tdd_ks
  cks_ks_jac(x,y,z,a)
  Tdd_ks = 0*Tdd
  for i in range(4):
    for j in range(4):
      for k in range(4):
        for m in range(4):
          Tdd_ks[i][j] += Tdd[k][m] * jac_cks_ks[k][i] * jac_cks_ks[m][j]

def raise_Tdd_ks(Tdd,gi):
  global Tud_ks
  Tud_ks = 0 *Tdd
  for i in range(4):
    for j in range(4):
      for k in range(4):
        Tud_ks[i,j] += gi[i,k]*Tdd[k,j]

def convert_to_gr_units():
  rho_max = np.amax(rho)
  SMALL=1e-20
  cl = 306.4
  press = press/cl**2./rho_max
  rho = rho/rho_max
  vel1 = vel1/cl
  vel2 = vel2/cl/(r)
  vel3 = vel3/cl/(r*sin(th)+SMALL)


def Lower(uu,g):
  ud = 0
  for i in range(4):
    ud += g[i,:]*uu[i]
  return ud
# def gr_dot(uu1,uu2,g):
#   sum = 0
#   for i in range(4):
#     for j in range(4):
#       sum += g[i][j] * uu1[i] * uu2[j]
#   return sum
def angle_average_npz(arr,weight=None,gr=False,a=0,gdet = None):
  dx3 = 1
  if "th" in globals(): dx2 = np.diff(th[0,:,0])[0]
  else: dx2 = np.diff(theta[0,:,0])[0]
  if (gr==False): 
    if ("th" in globals()): dOmega = (np.sin(th)*dx2)*dx3
    else: dOmega = (np.sin(theta)*dx2)*dx3
  else: 
    if (gdet is None and "th" in globals() ): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(th)
    elif (gdet is None): gdet =  (r**2 + a**2*np.cos(theta)**2)*np.sin(theta)
    dOmega = gdet *dx2*dx3
  if weight is None: weight = 1.0
  return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)


def angle_integral_npz(arr,weight=None,gr=False,a=0,gdet = None):
  dx3 = 1
  if "th" in globals(): dx2 = np.diff(th[0,:,0])[0]
  else: dx2 = np.diff(theta[0,:,0])[0]
  if "ph" in globals():dx3 = np.diff(ph[0,0,:])[0]
  else: dx3 = np.diff(phi[0,0,:])[0]
  if (gr==False): 
    if ("th" in globals()): dOmega = (np.sin(th)*dx2)*dx3
    else: dOmega = (np.sin(theta)*dx2)*dx3
  else: 
    if (gdet is None and "th" in globals() ): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(th)
    elif (gdet is None): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(theta)
    dOmega = gdet *dx2*dx3
  if weight is None: weight = 1.0
  return (arr * weight * dOmega).sum(-1).sum(-1)
def angle_average(arr,weight=None,gr=False):
  if ("x3f" in globals()): dx3 = np.diff(x3f)
  else: dx3 = np.gradient(ph[0,0,:])
  if ("x2f" in globals()): dx2 = np.diff(x2f)
  else: dx2 = np.gradient(th[0,:,0])
  dOmega = (np.sin(th)*dx2[None,:,None])*dx3[None,None,:]
  if weight is None: weight = 1.0
  if gr==True: dOmega = (gdet * dx2[None,:,None]) * dx3[None,None,:]
  return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)

def get_stress(mhd=True):
  global alpha_m, alpha_h,F_J,F_maxwell,F_reynolds,F_advection
  drhovr = rho*vel1 - angle_average(rho*vel1)[:,None,None]*(rho/rho)
  dvphi = vel3 - angle_average(vel3)[:,None,None]*(rho/rho)
  if (mhd==True): F_J = angle_average((rho*vel1*vel3-Bcc1*Bcc3)*sin(th)*r**3,gr=False)
  else: F_J = angle_average((rho*vel1*vel3)*sin(th)*r**3,gr=False)
  if (mhd==True): F_maxwell = angle_average(r**3*Bcc1*Bcc3*sin(th))*-1
  F_reynolds = angle_average(r**3*sin(th)*drhovr*dvphi)
  if (mhd==True): F_advection = F_J-F_maxwell-F_reynolds
  else: F_advection = F_J-F_reynolds
  Sh = angle_average(rho*vel1*vel3*sin(th)) - angle_average(rho*vel1)*angle_average(vel3*sin(th))
  if (mhd==True): Sm = angle_average(Bcc1*Bcc3*-1.0*sin(th))
  if (mhd==True): Ptot = angle_average(press + bsq/2.0) 
  else: Ptot = angle_average(press)
  if (mhd==True): alpha_m = Sm/Ptot
  alpha_h = Sh/Ptot

def get_stress_cart(mhd = True):
  global alpha_m, alpha_h,F_J,F_maxwell,F_reynolds,F_advection,S
  if (mhd==True):bsq = B1**2 + B2**2 + B3**2
  def angle_average(arr,weight=None):
    dx3 = np.diff(ph[0,0,:])[0]
    dx2 = np.diff(th[0,:,0])[0]
    dOmega = (np.sin(th)*dx2*dx3)
    if weight is None: weight = 1.0
    return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)
  global alpha_m, alpha_h,F_J,F_maxwell,F_reynolds,F_advection
  drhovr = rho*vr - angle_average(rho*vr)[:,None,None]*(rho/rho)
  dvphi = vphi - angle_average(vphi)[:,None,None]*(rho/rho)
  if (mhd==True): F_J = angle_average((rho*vr*vphi-Br*Bphi)*sin(th)*r**3)
  else: F_J = angle_average((rho*vr*vphi)*sin(th)*r**3)
  if (mhd==True): F_maxwell = angle_average(r**3*Br*Bphi*sin(th))*-1
  else: F_maxwell = 0
  F_reynolds = angle_average(r**3*sin(th)*drhovr*dvphi)
  F_advection = F_J-F_maxwell-F_reynolds
  Sh = angle_average(rho*vr*vphi*sin(th)) - angle_average(rho*vr)*angle_average(vphi*sin(th))
  if (mhd==True):Sm = angle_average(Br*Bphi*-1.0*sin(th))
  else: Sm = 0
  if (mhd==True): Ptot = angle_average(press + bsq/2.0)
  else: Ptot = angle_average(press)
  alpha_m = Sm/Ptot
  alpha_h = Sh/Ptot

def gr_dot(A,B):
  return A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3]

def get_tetrad(uu_avg,ud_avg):
  global omega_t,omega_r,omega_th,omega_ph
  C0 = uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3]
  C1 = uu_avg[1]*ud_avg[0]
  C2 = uu_avg[1]*ud_avg[3]
  l = ud_avg[3]/ud_avg[0]
  s = -C0/abs(C0)
  N1 = np.sqrt(- gbl[1,1] * (ud_avg[0]*uu_avg[0]+ud_avg[3]*uu_avg[3])*(1+ud_avg[2]*uu_avg[2])) ##g[1,1] * np.sqrt(g[0,0]*C1**2 + g[1,1]*C0**2 + g[3,3]*C2**2 + 2*g[0,3]*C1*C2)
  N2 = np.sqrt(gbl[2,2]*(1+uu_avg[2]*ud_avg[2]))
  N3 = np.sqrt(gbl[0,0]*l**2 -2*gbl[0,3]*l + gbl[3,3])
  omega_t = uu_avg
  omega_r = uu_avg*0
  omega_r[0] = s/N1 * ud_avg[1]*uu_avg[0]
  omega_r[1] = s/N1 * - (uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3])
  omega_r[2] = 0
  omega_r[3] = s/N1 * ud_avg[1]*uu_avg[3]
  omega_th = uu_avg*0
  omega_th[0] = 1./N2 * ud_avg[2]*uu_avg[0]
  omega_th[1] = 1./N2 * ud_avg[2]*uu_avg[1]
  omega_th[2] = 1./N2 * (1.0 + ud_avg[2]*uu_avg[2])
  omega_th[3] = 1./N2 * ud_avg[2]*uu_avg[3]
  omega_ph = uu_avg*0
  omega_ph[0] = 1./N3 * (-l)
  omega_ph[3] = 1./N3


# def get_LNRF_ks(a=0):
#   global econ_t,econ_r,econ_th,econ_phi
#   r2 = r**2.
#   a2 = a**2.
#   sin2 = np.sin(th)**2.
#   gbl = np.zeros((4,4))
#   Sigma = r**2 + a**2.*np.cos(th)**2;
#   Delta = r**2 -2.0*r + a**2;
#   A = (r**2 + a**2)**2 - a**2*Delta*np.sin(th)**2.;


#   alpha = 1.0/np.sqrt(1.0 + 2.0*r/Sigma)
#   betar = 2.0*r/Sigma /(1.0 + 2.0*r/Sigma)
#   gammarr = 1.0 + 2.0*r/(Sigma)
#   gammathth = Sigma
#   gammaphiphi = A * sin2/Sigma
#   gammarphi = - a * sin2*(1.0+2.0*r/Sigma)

#   ecov_t = [alpha,0,0,0]
#   ecov_r = [betar/np.sqrt(gammarr),1.0/sqrt(gammarr),0,0]
#   ecov_th  = [0,0,np.sqrt(gammathth),0]
#   ecov_phi = [betar*gammarphi/np.sqrt(gammaphiphi),gammarphi/np.sqrt(gammaphiphi),0,np.sqrt(gammaphiphi)]

#   econ_t = [1.0/alpha,- betar/alpha,0,0]
#   econ_r = [0,1.0/np.sqrt(gammarr),0,-gammarphi/np.sqrt(gammarr)/gammaphiphi]
#   econ_th = [0,0,1.0/np.sqrt(gammathth),0]
#   econ_phi = [0,0,0,1.0/np.sqrt(gammaphiphi)]


def get_LNRF_bl(a=0):
  global econ_t,econ_r,econ_th,econ_phi
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  gbl = np.zeros((4,4))
  Sigma = r**2 + a**2.*np.cos(th)**2;
  Delta = r**2 -2.0*r + a**2;
  A = (r**2 + a**2)**2 - a**2*Delta*np.sin(th)**2.;
  rhosq = r2 + a2*np.cos(th)**2.0
  omega = 2.0*a*r/A 


  alpha = np.sqrt(Delta*rhosq/A)
  betacon_phi = -omega 
  gammacov_rr = rhosq/Delta
  gammacov_thth = rhosq 
  gammacov_phiphi = A*sin2/rhosq
  gammacon_rr = Delta/rhosq 
  gammacon_thth = 1.0/rhosq 
  gammacon_phiphi = rhosq/(A*sin2)


  ecov_t = [alpha,0,0,0]
  ecov_r = [0,np.sqrt(gammacov_rr),0,0]
  ecov_th  = [0,0,np.sqrt(gammacov_thth),0]
  ecov_phi = [betacon_phi*np.sqrt(gammacov_phiphi),0,0,np.sqrt(gammacov_phiphi)]

  zero = 0.0*r 
  econ_t = np.array([1.0/alpha,zero,zero,-betacon_phi/alpha])
  econ_r = np.array([zero,np.sqrt(gammacon_rr),zero,zero])
  econ_th = np.array([zero,zero,np.sqrt(gammacon_thth),zero])
  econ_phi = np.array([zero,zero,zero,np.sqrt(gammacon_phiphi)])



def ks_metric(r,th,a):
  global g
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  g  = np.zeros((4,4,nx,ny,nz))
  g[0][0] = -(1.0 - 2.0*m*r/sigma);
  g[0][1] = 2.0*m*r/sigma;
  g[1][0] = g[0][1]
  g[0][3] = -2.0*m*a*r/sigma * sin2
  g[3][0] = g[0][3]
  g[1][1] = 1.0 + 2.0*m*r/sigma
  g[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2
  g[3][1] = g[1][3]
  g[2][2] = sigma
  g[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2


def ks_binary_metric(t,x,y,z,aprime_=0.0,ONED=False,q=0,r_bh2=10.0,a=0.0,t0=0.0,inclination=0.0):
  cks_binary_metric(t,x,y,z,a1z=a,a2z_=aprime_,q=q,r_bh2=r_bh2,t0=t0,inclination=inclination)
  global g
  ## g_ks _mu _nu = g_cks _alpha _beta d x_cks^alpha/dx_ks^mu d x_cks^beta/dx_ks^nu

  cks_ks_jac(x,y,z,a)
  g_tmp = 0*g
  for i in range(4):
    for j in range(4):
      for k in range(4):
        for m in range(4):
          g_tmp[i][j] += g[k][m] * jac_cks_ks[k][i] * jac_cks_ks[m][j]

  g = g_tmp * 1.0



def cks_binary_metric_2nd_BH(t,xprime,yprime,zprime,aprime_=0.0,ONED=False,q=0,r_bh2=10.0,a=0.0,t0=0.0,inclination=0.0):
  xbh,ybh,zbh = bh2_pos(t,r_bh2,t0=t0,inclination=inclination)

  x = xprime + xbh 
  y = yprime + ybh 
  z = zprime + zbh 

  cks_binary_metric(t,x,y,z,a2z_=aprime_,q=q,r_bh2=r_bh2,a1z=a,t0=t0,inclination=inclination)
  global g
  ## g_cks' _mu _nu = g_cks _alpha _beta d x_cks^alpha/dx_cks'^mu d x_cks^beta/dx_cks'^nu

  # x = xprime + xbh
  # y = yprime + ybh
  # z = zprime + zbh

  #dx/dxprime = 1
  #dy/dyprime = 1
  #dz/dzprime = 1
  #dx/dt = vxbh
  #dy/dt = vybh
  #dz/dt = vybh
  cks_cks_prime_jac(r_bh2=r_bh2,t0=t0,inclination=inclination)
  g_tmp = 0*g
  for i in range(4):
    for j in range(4):
      for k in range(4):
        for m in range(4):
          g_tmp[i][j] += g[k][m] * jac_cks_cks_prime[k][i] * jac_cks_cks_prime[m][j]

  g = g_tmp * 1.0

def ks_binary_metric_2nd_BH(t,xprime,yprime,zprime,aprime_=0.0,ONED=False,q=0,r_bh2=10.0,a=0.0,t0=0.0,inclination=0.0):
  cks_binary_metric_2nd_BH(t,xprime,yprime,zprime,aprime_=aprime_,ONED=ONED,q=q,r_bh2=r_bh2,a=a,t0=t0,inclination=inclination)
  global g
  ## g_ks _mu _nu = g_cks _alpha _beta d x_cks^alpha/dx_ks^mu d x_cks^beta/dx_ks^nu

  cks_ks_jac(xprime,yprime,zprime,aprime_)
  g_tmp = 0*g
  for i in range(4):
    for j in range(4):
      for k in range(4):
        for m in range(4):
          g_tmp[i][j] += g[k][m] * jac_cks_ks[k][i] * jac_cks_ks[m][j]

  g = g_tmp * 1.0

def get_mri_q():
  global bphi,bth,dphi,dth,Qphi,Qth,gbl
  gbl =  bl_metric(r,th)
  uu_bl = ks_vec_to_bl(uu)
  ud_bl = Lower(uu_bl,gbl)
  get_tetrad(uu_bl,ud_bl)
  omega_t = bl_vec_to_ks(omega_t)
  omega_r = bl_vec_to_ks(omega_r)
  omega_th = bl_vec_to_ks(omega_th)
  omega_ph = bl_vec_to_ks(omega_ph)

  bphi = gr_dot(bd,omega_ph)
  bth = gr_dot(bd,omega_th)

  dx3 = np.diff(x3f)
  dx2 = np.diff(x2f)
  dx1 = np.diff(x1f)

  dx3 = (gdet/gdet) * dx3[None,None,:]
  dx2 = (gdet/gdet) * dx2[None,:,None]
  dx1 = (gdet/gdet) * dx1[:,None,None]
  
  dx_mu = uu*0 
  dx_mu[0] = 1.87213890018754e-03
  dx_mu[1] = dx1 
  dx_mu[2] = dx2 
  dx_mu[3] = dx3 

  dphi = gr_dot(Lower(dx_mu,g),omega_ph)
  dth = gr_dot(Lower(dx_mu,g),omega_th)

  Omega = uu[3]/uu[0]
  Qphi = 2.*np.pi / (Omega * dphi) * bphi/np.sqrt(rho)
  Qth = 2.*np.pi / (Omega * dth) * bth/np.sqrt(rho)

def get_mri_q_cartesian(x,y,z,a=0,xmax = 50.0,refinement_levels = 4):
  global bphi,bth,dphi,dth,Qphi,Qth,gbl,Qr
  nx = x.shape[0]
  ny = x.shape[1]
  nz = x.shape[2]
  cks_metric(x,y,z,0,0,)
  bd = Lower(bu,g)

  uu_ks = cks_vec_to_ks(uu,x,y,z,0,0,a)


  gbl = bl_metric(r,th,a) 

  global econ_t,econ_r,econ_th,econ_phi
  get_LNRF_bl(a=a)

  econ_t = bl_vec_to_cks(x,y,z,econ_t,a=a)
  econ_r = bl_vec_to_cks(x,y,z,econ_r,a=a)
  econ_th = bl_vec_to_cks(x,y,z,econ_th,a=a)
  econ_phi = bl_vec_to_cks(x,y,z,econ_phi,a=a)

  bphi = gr_dot(bd,econ_phi)
  bth = gr_dot(bd,econ_th)
  br = gr_dot(bd,econ_r)


  DX = xmax*2.0/(nx*1.0)
  dx_list = []
  r_lim_list = []
  for n in range(refinement_levels):
    dx_list.append(DX/2.0**n)
    r_lim_list.append(xmax/2.0**n)

  dx_arr = rho*0 + DX

  for i in range(len(dx_list)-1):
    r_lim = r_lim_list[i]
    ind = (x<r_lim)*(x>-r_lim) * (y<r_lim)*(y>-r_lim) * (z<r_lim)*(z>-r_lim)
    dx_arr[ind] = dx_list[i]
  
  dx_mu = uu*0 
  dx_mu[0] = 1.46484375000429e-02
  dx_mu[1] = dx_arr 
  dx_mu[2] = dx_arr
  dx_mu[3] = dx_arr

  dphi = gr_dot(Lower(dx_mu,g),econ_phi)
  dth = gr_dot(Lower(dx_mu,g),econ_th)
  dr = gr_dot(Lower(dx_mu,g),econ_r)

  Omega = uu_ks[3]/uu_ks[0]
  Qphi = 2.*np.pi / (Omega * dphi) * bphi/np.sqrt(rho)
  Qth = 2.*np.pi / (Omega * dth) * bth/np.sqrt(rho)
  Qr = 2.*np.pi/(Omega * dr) * br/np.sqrt(rho)
def get_mri_q_newt(ndim=3,npz=False,smr=False,r_max=500.0,max_level=5,ny_block=4):
  global Qphi,Qth,Qr,len_ph,len_th,dr
  
  if (npz==False):
    if(ndim==3): dphi = np.gradient(ph,axis=2)
    dth = np.gradient(th,axis=1)
    dr = np.gradient(r,axis=0)

    if (smr==True):
      get_dr_smr(r_max=r_max,max_level=max_level,ny_block=ny_block)
      dth = 1.0*dth_grid
      dphi = 1.0*dph_grid
      dr = 1.0*dr_grid

    len_th = dth * r 
    len_ph = dphi*r*sin(th)



  Omega = vel3/(r*sin(th))
  if (ndim==3): Qphi = 2.*np.pi / (Omega * len_ph) * Bcc3/np.sqrt(rho)
  Qr = 2.*np.pi / (Omega * dr) * Bcc1/np.sqrt(rho)
  Qth = 2.*np.pi / (Omega * len_th) * Bcc2/np.sqrt(rho)

def get_time_step_limit():
  dphi = np.gradient(ph,axis=2)
  dth = np.gradient(th,axis=1)
  dr = np.gradient(r,axis=0)

  dx1_min = np.amin(abs(dr/vel1),axis=2)
  dx2_min = np.amin(abs(dth*r/vel2),axis=2)
  dx3_min = np.amin(abs(dphi*r*sin(th)/vel3),axis=2)

  va = sqrt(bsq/rho)
  dx1_min_B = np.amin(dr/va,axis=2)
  dx2_min_B = np.amin(dth*r/va,axis=2)
  dx3_min_B = np.amin(dphi*r*sin(th)/va,axis=2)


def get_gr_stress(a=0):
  global omega_t,omega_r,omega_th,omega_ph
  global l,C0,C1,C2,N1,s,N2,N3
  gbl =  bl_metric(r,th)  
  uu_avg = uu*0

  vr_avg = angle_average(uu[1]/uu[0],weight=rho,gr=True)[:,None,None]*(rho/rho)
  vth_avg = angle_average(uu[2]/uu[0],weight=rho,gr=True)[:,None,None]*(rho/rho)
  vph_avg = angle_average(uu[3]/uu[0],weight=rho,gr=True)[:,None,None]*(rho/rho)
  vu = uu*0
  vu[0] = (rho/rho)
  vu[1] = vr_avg
  vu[2] = vth_avg
  vu[3] = vph_avg

  tmp = 0
  for mu in range(4):
    for nu in range(4):
      tmp += g[mu,nu]*vu[mu]*vu[nu]
  norm = np.sqrt(-1./tmp)

  uu_avg = vu * norm

  #for i in range(4): uu_avg[i] = angle_average(uu[i],gr=True)[:,None,None]*(rho/rho)

  #uu_avg = uu 
  #ud_avg = ud

  uu_avg = ks_vec_to_bl(uu_avg)
  ud_avg = Lower(uu_avg,gbl)
  C0 = uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3]
  C1 = uu_avg[1]*ud_avg[0]
  C2 = uu_avg[1]*ud_avg[3]
  l = ud_avg[3]/ud_avg[0]
  s = -C0/abs(C0)
  N1 = np.sqrt(- gbl[1,1] * (ud_avg[0]*uu_avg[0]+ud_avg[3]*uu_avg[3])*(1+ud_avg[2]*uu_avg[2])) ##g[1,1] * np.sqrt(g[0,0]*C1**2 + g[1,1]*C0**2 + g[3,3]*C2**2 + 2*g[0,3]*C1*C2)
  N2 = np.sqrt(gbl[2,2]*(1+uu_avg[2]*ud_avg[2]))
  N3 = np.sqrt(gbl[0,0]*l**2 -2*gbl[0,3]*l + gbl[3,3])
  omega_t = uu_avg
  omega_r = uu_avg*0
  omega_r[0] = s/N1 * ud_avg[1]*uu_avg[0]
  omega_r[1] = s/N1 * - (uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3])
  omega_r[2] = 0
  omega_r[3] = s/N1 * ud_avg[1]*uu_avg[3]
  omega_th = uu_avg*0
  omega_th[0] = 1./N2 * ud_avg[2]*uu_avg[0]
  omega_th[1] = 1./N2 * ud_avg[2]*uu_avg[1]
  omega_th[2] = 1./N2 * (1.0 + ud_avg[2]*uu_avg[2])
  omega_th[3] = 1./N2 * ud_avg[2]*uu_avg[3]
  omega_ph = uu_avg*0
  omega_ph[0] = 1./N3 * (-l)
  omega_ph[3] = 1./N3

  omega_t = bl_vec_to_ks(omega_t)
  omega_r = bl_vec_to_ks(omega_r)
  omega_th = bl_vec_to_ks(omega_th)
  omega_ph = bl_vec_to_ks(omega_ph)
  global reynolds_stress,maxwell_stress,total_stress,alpha,alpha_m,alpha_r

  total_stress=0
  maxwell_stress=0
  reynolds_stress=0
  Tud_calc(is_magnetic=True)
  for mu in range(4):
    for nu in range(4):
      total_stress += Tud[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      maxwell_stress += TudEM[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      reynolds_stress += TudMA[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
  alpha_m = angle_average(maxwell_stress,gr=True)/angle_average(press + bsq/2.0,gr=True)
  alpha_r = angle_average(reynolds_stress,gr=True)/angle_average(press+bsq/2.0,gr=True)
  alpha = alpha_m + alpha_r

def get_total_stress(a=0):
  global omega_t,omega_r,omega_th,omega_ph
  gbl =  bl_metric(r,th)  
  uu_avg = uu*0
  vu = uu*0
  vu[0] = (rho/rho)
  vu[1] = 0
  vu[2] = 0
  vu[3] = 0

  tmp = 0
  for mu in range(4):
    for nu in range(4):
      tmp += g[mu,nu]*vu[mu]*vu[nu]
  norm = np.sqrt(-1./tmp)

  uu_avg = vu * norm


  uu_avg = ks_vec_to_bl(uu_avg)
  ud_avg = Lower(uu_avg,gbl)
  C0 = uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3]
  C1 = uu_avg[1]*ud_avg[0]
  C2 = uu_avg[1]*ud_avg[3]
  l = ud_avg[3]/ud_avg[0]
  s = -C0/abs(C0)
  N1 = np.sqrt(- gbl[1,1] * (ud_avg[0]*uu_avg[0]+ud_avg[3]*uu_avg[3])*(1+ud_avg[2]*uu_avg[2])) ##g[1,1] * np.sqrt(g[0,0]*C1**2 + g[1,1]*C0**2 + g[3,3]*C2**2 + 2*g[0,3]*C1*C2)
  N2 = np.sqrt(gbl[2,2]*(1+uu_avg[2]*ud_avg[2]))
  N3 = np.sqrt(gbl[0,0]*l**2 -2*gbl[0,3]*l + gbl[3,3])
  omega_t = uu_avg
  omega_r = uu_avg*0
  omega_r[0] = s/N1 * ud_avg[1]*uu_avg[0]
  omega_r[1] = s/N1 * - (uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3])
  omega_r[2] = 0
  omega_r[3] = s/N1 * ud_avg[1]*uu_avg[3]
  omega_th = uu_avg*0
  omega_th[0] = 1./N2 * ud_avg[2]*uu_avg[0]
  omega_th[1] = 1./N2 * ud_avg[2]*uu_avg[1]
  omega_th[2] = 1./N2 * (1.0 + ud_avg[2]*uu_avg[2])
  omega_th[3] = 1./N2 * ud_avg[2]*uu_avg[3]
  omega_ph = uu_avg*0
  omega_ph[0] = 1./N3 * (-l)
  omega_ph[3] = 1./N3

  omega_t = bl_vec_to_ks(omega_t)
  omega_r = bl_vec_to_ks(omega_r)
  omega_th = bl_vec_to_ks(omega_th)
  omega_ph = bl_vec_to_ks(omega_ph)
  global reynolds_stress,maxwell_stress,total_stress,alpha,alpha_m,alpha_r

  total_stress=0
  maxwell_stress=0
  reynolds_stress=0
  Tud_calc(is_magnetic=True)
  for mu in range(4):
    for nu in range(4):
      total_stress += Tud[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      maxwell_stress += TudEM[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      reynolds_stress += TudMA[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
  alpha_m = angle_average(maxwell_stress,gr=True)/angle_average(press + bsq/2.0,gr=True)
  alpha_r = angle_average(reynolds_stress,gr=True)/angle_average(press+bsq/2.0,gr=True)
  alpha = alpha_m + alpha_r
def risco(a):
  Z1 = 1.0 + (1.0-a*a)**(1.0/3.0) * ( (1.0+a)**(1.0/3.0) + (1.0-a)**(1.0/3.0) ) 
  Z2 = sqrt(3.0*a*a + Z1*Z1)
  sgn = 1
  if (a>0): sgn = -1
  return 3.0 + Z2 + sgn*sqrt((3.0-Z1) * (3.0+Z1 + 2.0*Z2))


def ks_vec_to_gammie(uu_ks,x1,x2,x3,a=0.0,hslope = 0.3):
  r = np.exp(x1)

  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2

  uu_gammie = uu_ks* 1.0

  #u^\mu_gammie = u^\nu_athena dx^\mu_gammie/dx^\nu_athena
  #u_\mu_gammie = u_\nu_athena dx^\nu_athena/dx^\mu_gammie

  uu_gammie[1] *= dx1_dr
  uu_gammie[2] *= dx2_dtheta

  return uu_gammie




def convert_to_gammie(a=0):
  global hslope, uu_gammie,ud_gammie,bu_gammie,bd_gammie
  hslope = 0.3
  #x1gammie = log(r)  r = exp(x1) 
  # dx1/dr = 1/r -> dx1/dr = r 
  #theta = pi*x2 + 0.5*(1-h)*sin(2*pi*x2)
  # dtheta/dx2 = pi + pi* (1-h) * cos(2*pi*x2)
  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    if (th[0,j,0] ==0):
      x2[j] = 0.0
      continue
    if (th[0,j,0]==np.pi):
      x2[j] = 1.0
      continue
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (r/r)



  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2

  uu_gammie = uu 
  ud_gammie = ud 
  bu_gammie = bu 
  bd_gammie = bd 

  #u^\mu_gammie = u^\nu_athena dx^\mu_gammie/dx^\nu_athena
  #u_\mu_gammie = u_\nu_athena dx^\nu_athena/dx^\mu_gammie

  uu_gammie[1] *= dx1_dr
  uu_gammie[2] *= dx2_dtheta
  bu_gammie[1] *= dx1_dr
  bu_gammie[2] *= dx2_dtheta

  ud_gammie[1] *= dr_dx1 
  ud_gammie[2] *= dtheta_dx2
  bd_gammie[1] *= dr_dx1
  bd_gammie[2] *= dtheta_dx2



  # gprime_mu nu = dx^alpha/dx^mu dx^sigma/dx^nu g_alpha sigma
  # prime^mu^nu = dx^mu/dx^alpha dx^nu/dx^sigma g^alpha sigma

  # so gi_gammie ^mu nu = dx_gammie^mu/dx^sig dx_gammie^nu/dx^alph g^sig alph
  # only nonzero are dx1/dr and dx2/dth

  # gi_gammie00 is unchanged
  # gi_gammie10 = dx1/dr 
  #   g_inv(I00,i) = -(1.0 + 2.0*m*r/sigma);
  #   g_inv(I01,i) = 2.0*m*r/sigma;
  #   g_inv(I11,i) = delta/sigma;
  #   g_inv(I13,i) = a/sigma;
  #   g_inv(I22,i) = 1.0/sigma;
  #   g_inv(I33,i) = 1.0 / (sigma * sin2);
  sigma = r**2.0 + a**2.0 * np.cos(th)**2.0
  m = 1
  gi00 = -(1.0 + 2.0*m*r/sigma)
  gi01 = 2.0*m*r/sigma *dx1_dr
  gi02 = 0
  gi03 = 0
  global v1,v2,v3,B1,B2,B3,gdet_gammie
  v1 = uu_gammie[1] - gi01/gi00 * uu_gammie[0]
  v2 = uu_gammie[2] - gi02/gi00 * uu_gammie[0]
  v3 = uu_gammie[3] - gi03/gi00 * uu_gammie[0]

  B1 = bu_gammie[1]*uu_gammie[0] - bu_gammie[0]*uu_gammie[1]
  B2 = bu_gammie[2]*uu_gammie[0] - bu_gammie[0]*uu_gammie[2]
  B3 = bu_gammie[3]*uu_gammie[0] - bu_gammie[0]*uu_gammie[3]

  gdet_gammie = gdet * dr_dx1 * dtheta_dx2




def gammie_metric(r,th,a=0,hslope = 0.3):
  global gg

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    if (th[0,j,0] ==0):
      x2[j] = 0.0
      continue
    if (th[0,j,0]==np.pi):
      x2[j] = 1.0
      continue
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (r/r)
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  rfac = r*1.0;
  hfac = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * x2);
  gg  = np.zeros((4,4,nx,ny,nz))
  gg[0][0] = -(1.0 - 2.0*m*r/sigma);
  gg[0][1] = 2.0*m*r/sigma * rfac
  gg[1][0] = gg[0][1] 
  gg[0][3] = -2.0*m*a*r/sigma * sin2
  gg[3][0] = gg[0][3]
  gg[1][1] = (1.0 + 2.0*m*r/sigma) * rfac * rfac
  gg[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2 * rfac
  gg[3][1] = gg[1][3] 
  gg[2][2] = sigma * hfac * hfac
  gg[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2

def gammie_gcon(r,th,a=0,hslope=0.3):
  global ggcon

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    if (th[0,j,0] ==0):
      x2[j] = 0.0
      continue
    if (th[0,j,0]==np.pi):
      x2[j] = 1.0
      continue
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (r/r)
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  delta = r**2 - 2.0*m*r + a**2
  rfac = r*1.0;
  hfac = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * x2);
  ggcon  = np.zeros((4,4,nx,ny,nz))
  ggcon[0][0] = -(1.0 + 2.0*m*r/sigma);
  ggcon[0][1] =  2.0*m*r/sigma /(rfac)
  ggcon[1][0] = ggcon[0][1] 
  ggcon[1][1] = delta/sigma /( rfac * rfac)
  ggcon[1][3] =  a/sigma / rfac
  ggcon[3][1] = ggcon[1][3] 
  ggcon[2][2] = 1.0/sigma /( hfac * hfac )
  ggcon[3][3] = 1.0 / (sigma * sin2)




def gammie_grid():
  global ri,thi,phii,x1_grid,x2_grid,x3_grid
  global igrid_new,jgrid_new,kgrid_new
  global igrid,jgrid,kgrid
  global x1,x2,x3
  # dx3grid = 2.0*pi/(nz*1.)
  
  # dx1grid = log(np.amax(r) /np.amin(r))/(nx*1.0)
  # dx2grid = 1.0/(ny*1.0)
  x1_grid_faces = np.linspace(log(np.amin(r)),log(np.amax(r)),nx+1)  ##faces
  x2_grid_faces = np.linspace(0,1,ny+1)       ##faces
  x3_grid_faces = np.linspace(0,2.0*pi,nz+1)  ##faces
  x1_grid = ( (x1_grid_faces) + 0.5*np.diff(x1_grid_faces)[0] )[:-1]
  x2_grid = ( x2_grid_faces + 0.5*np.diff(x2_grid_faces)[0] ) [:-1]
  if (nz==1): x3_grid = x3_grid_faces[0] + np.pi
  else: x3_grid =( x3_grid_faces + 0.5*np.diff(x3_grid_faces)[0] )[:-1]

  # x1_grid = np.linspace(log(np.amin(r)) + 0.5*dx1grid,log(np.amax(r))-0.5*dx1grid,nx)
  # x2_grid = np.linspace(0+0.5*dx2grid,1-0.5*dx2grid,ny)
  # x3_grid = np.linspace(0+0.5*dx3grid,2.0*pi-0.5*dx3grid,nz)
  #x1_grid = x1_grid + 0.5*np.diff(x1_grid)[0]
  #x2_grid = x2_grid + 0.5*np.diff(x2_grid)[0]
  #if (nz==1): x3_grid = x3_grid + np.pi
  #else: x3_grid = x3_grid + 0.5*np.diff(x3_grid)[0]
  ri = np.exp(x1_grid)
  thi = np.pi*x2_grid + 0.5*(1.0-hslope)*np.sin(2.0*pi*x2_grid)
  phii = x3_grid

  ri,thi,phii = np.meshgrid(ri,thi,phii,indexing='ij')


  kgrid,jgrid,igrid = meshgrid(np.arange(0,nz),np.arange(0,ny),np.arange(0,nx),indexing='ij')
  igrid,jgrid,kgrid = meshgrid(np.arange(0,nx),np.arange(0,ny),np.arange(0,nz),indexing='ij')
  mgrid = igrid + jgrid*nx  + kgrid*nx*ny

  mnew = scipy.interpolate.griddata((r.flatten(),th.flatten(),ph.flatten()),mgrid[:,:,:].flatten(),(ri,thi,phii),method='nearest')


  # index = np.arange(nx*ny*nz)

  # new_index  = scipy.interpolate.griddata((r.flatten(),th.flatten(),ph.flatten()),index,(ri,thi,phii),method='nearest')

  igrid_new= mod(mod(mnew,ny*nx),nx)
  jgrid_new = mod(mnew,ny*nx)//nx
  kgrid_new = mnew//(ny*nx)


def cross_product(a,b):
    return [ a[1]*b[2] - b[1]*a[2], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] ]
def dot_product(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def curl(a):
  x_tmp = x[:,0,0]
  y_tmp = y[0,:,0]
  z_tmp = z[0,0,:]
  return [ gradient(a[2],y_tmp,axis=1) - gradient(a[1],z_tmp,axis=2), gradient(a[0],z_tmp,axis=2) - gradient(a[2],x_tmp,axis=0), gradient(a[1],x_tmp,axis=0) - gradient(a[0],y_tmp,axis=1)]

def divergence(a):
  x_tmp = x[:,0,0]
  y_tmp = y[0,:,0]
  z_tmp = z[0,0,:]
  return gradient(a[0],x_tmp,axis=0) + gradient(a[1],y_tmp,axis=1) + gradient(a[2],z_tmp,axis=2)

def curl_spherical(a):
  iphi = 2
  ith = 1
  ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]
  th_tmp = th[0,:,0]
  phi_tmp = ph[0,0,:]
  curl_r   = 1.0/(r*np.sin(th) + small) * (gradient(a[iphi]*np.sin(th),th_tmp,axis=ith)             - gradient(a[ith]   , phi_tmp , axis=iphi) )
  curl_th  = 1.0/r                      * ( 1.0/(sin(th)+small) * gradient(a[ir],phi_tmp,axis=iphi) - gradient(r*a[iphi], r_tmp   , axis=ir)   )
  curl_phi = 1.0/r                      * ( gradient(r*a[ith],r_tmp,axis=ir)                        - gradient(a[ir]    , th_tmp  , axis=ith)  )
  return [ curl_r,curl_th,curl_phi]


def curl_gr(a):
  iphi = 2
  ith = 1
  ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]
  th_tmp = th[0,:,0]
  phi_tmp = ph[0,0,:]
  curl_r   = 1.0/(r**2*np.sin(th) + small) * (gradient(a[iphi],th_tmp,axis=ith)             - gradient(a[ith]   , phi_tmp , axis=iphi) )
  curl_th  = 1.0/(r**2*np.sin(th) + small) * (gradient(a[ir],phi_tmp,axis=iphi)             - gradient(a[iphi]  , r_tmp   , axis=ir)   )
  curl_phi = 1.0/(r**2*np.sin(th) + small) * (gradient(a[ith],r_tmp,axis=ir)                - gradient(a[ir]    , th_tmp  , axis=ith)  )
  return [ curl_r,curl_th,curl_phi]

def div_spherical(a):
  iphi = 2
  ith = 1
  ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]
  th_tmp = th[0,:,0]
  phi_tmp = ph[0,0,:]
  return 1.0/r**2.0 * gradient(r**2.0*a[ir],r_tmp,axis=ir) + 1.0/(r*np.sin(th) + small) * gradient(a[ith]*np.sin(th),th_tmp,axis=ith) + 1.0/(r*np.sin(th)+small) * gradient(a[iphi],phi_tmp,axis=iphi)

def advection_derivative_spherical(a,b):
  iphi = 2 ;ith = 1; ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]; th_tmp = th[0,:,0]; phi_tmp = ph[0,0,:]

  def ddr(c):
    return gradient(c,r_tmp,axis=ir)
  def ddth(c):
    return gradient(c,th_tmp,axis=ith)
  def ddphi(c):
    return gradient(c,phi_tmp,axis=iphi)

  ar = a[ir]; ath = a[ith]; aphi = a[iphi]
  br = b[ir]; bth = b[ith]; bphi = b[iphi]

  r_term = ar * ddr(br) + ath/r*ddth(br) + aphi/(r*np.sin(th)+small)*ddphi(br) - (ath*bth+aphi*bphi)/r
  th_term = ar * ddr(bth) + ath/r * ddth(bth) + aphi/(r*np.sin(th)+small)*ddphi(bth) + ath*br/r - aphi*bphi*np.cos(th)/(r*np.sin(th)+small)
  phi_term = ar * ddr(bphi) + ath/r * ddth(bphi) + aphi/(r*np.sin(th)+small)*ddphi(bphi) + aphi*br/r + aphi*bth*np.cos(th)/(r*np.sin(th)+small)
  return [r_term,th_term,phi_term]

def curl_2d(a):
  x_tmp = x[:,0]
  y_tmp = y[0,:]
  return [ gradient(a[2],y_tmp,axis=1), - gradient(a[2],x_tmp,axis=0), gradient(a[1],x_tmp,axis=0) - gradient(a[0],y_tmp,axis=1)]

def make_grmonty_dump(fname,a=0,gam=5./3.):
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]
  convert_to_gammie(a = a)
  gammie_grid()
  gammie_metric(ri,thi,a=a)
  gammie_gcon(ri,thi,a=a)
  dx1 = np.diff(x1_grid)[0]
  dx2 = np.diff(x2_grid)[0]
  if (nz==1): dx3 = 2.*np.pi
  else: dx3 = np.diff(x3_grid)[0]


  Nprim = 8 
  header = [str(np.array(t)), str(nx), str(ny), str(nz), str(np.amin(x1_grid)-0.5*dx1),str(np.amin(x2_grid)-0.5*dx2),str(np.amin(x3_grid)-0.5*dx3), str(dx1),str(dx2),str(dx3),str(a),str(gam),str(np.amin(ri)),str(hslope),str(Nprim)]

  rhoi =rho[igrid_new,jgrid_new,kgrid_new]
  #uui = uu_gammie[:,igrid_new,jgrid_new,kgrid_new]

  v1i = v1[igrid_new,jgrid_new,kgrid_new]
  v2i = v2[igrid_new,jgrid_new,kgrid_new]
  v3i = v3[igrid_new,jgrid_new,kgrid_new]

  qsq = v1i*v1i*gg[1,1] + v2i*v2i*gg[2,2] + v3i*v3i*gg[3,3] + 2.*v1i*v2i*gg[1,2] + 2.*v1i*v3i*gg[1,3] + 2.*v2i*v3i*gg[2,3]
  alpha = 1./np.sqrt(-ggcon[0,0]) 
  beta = 0*uu_gammie
  for l in range(1,4): beta[l] = ggcon[0][l]*alpha*alpha ;

  qsq[qsq<0] = 1e-10
  
  gamma = np.sqrt(1.0 + qsq)
  uui = 0*uu_gammie
  uui[0] = gamma/alpha 
  uui[1] = v1i - gamma*beta[1]/alpha
  uui[2] = v2i - gamma*beta[2]/alpha
  uui[3] = v3i - gamma*beta[3]/alpha

  udi = Lower(uui,gg)



  B1i = B1[igrid_new,jgrid_new,kgrid_new]
  B2i = B2[igrid_new,jgrid_new,kgrid_new]
  B3i = B3[igrid_new,jgrid_new,kgrid_new]

  bui = bu_gammie*0

  bui[0] = B1i*udi[1] + B2i*udi[2] + B3i*udi[3]
  bui[1] = (B1i + bui[0]*uui[1])/uui[0]
  bui[2] = (B2i + bui[0]*uui[2])/uui[0]
  bui[3] = (B3i + bui[0]*uui[3])/uui[0]

  bdi = Lower(bui,gg)



  #udi = ud_gammie[:,igrid_new,jgrid_new,kgrid_new]
  #bui = bu_gammie[:,igrid_new,jgrid_new,kgrid_new]
  #bdi = bd_gammie[:,igrid_new,jgrid_new,kgrid_new]
  pressi = press[igrid_new,jgrid_new,kgrid_new]
  gdeti = gdet_gammie[igrid_new,jgrid_new,kgrid_new]

  tmp = rhoi*0
  x1_grid,x2_grid,x3_grid = meshgrid(x1_grid,x2_grid,x3_grid,indexing='ij')
  data = [igrid,jgrid,kgrid,x1_grid.astype(float32),x2_grid.astype(float32),x3_grid.astype(float32),ri.astype(float32),thi.astype(float32),phii.astype(float32),
          rhoi.astype(float32),(pressi/(5./3.-1.)).astype(float32),v1i.astype(float32),v2i.astype(float32),v3i.astype(float32),B1i.astype(float32),B2i.astype(float32),
          B3i.astype(float32),(pressi/rhoi**(5./3.)).astype(float32),
          uui[0].astype(float32),uui[1].astype(float32),uui[2].astype(float32),uui[3].astype(float32),udi[0].astype(float32),udi[1].astype(float32),udi[2].astype(float32),
          udi[3].astype(float32), bui[0].astype(float32),bui[1].astype(float32),bui[2].astype(float32),bui[3].astype(float32),bdi[0].astype(float32),bdi[1].astype(float32),
          bdi[2].astype(float32),bdi[3].astype(float32),gdeti.astype(float32)]
  data = np.array(data).astype(float32)
  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()

def make_grmonty_dump_cartesian(fname,idump,a=0,gam=5./3.,hslope = 0.3,high_res = False):

  if (high_res == True):rd_yt_convert_to_gammie(idump,MHD=True,gr=True,a=a,hslope = hslope,low_res=True,nr=356,nth=200,nphi=400)
  else: rd_yt_convert_to_gammie(idump,MHD=True,gr=True,a=a,hslope = hslope)
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]


  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 ) * abs(dr_dx1 * dtheta_dx2)

  gammie_metric(r,th,a=a,hslope=hslope)
  gammie_gcon(r,th,a=a,hslope=hslope)
  dx1 = np.diff(x1[:,0,0])[0]
  dx2 = np.diff(x2[0,:,0])[0]
  if (nz==1): dx3 = 2.*np.pi
  else: dx3 = np.diff(x3[0,0,:])[0]


  Nprim = 8 
  # electron_data = []
  # if ("ke_ent" in globals()):  
  #   Nprim += 1
  #   electron_data.append(ke_ent.astype(float32))
  # if ("ke_ent2" in globals()): 
  #   Nprim += 1
  #   electron_data.append(ke_ent2.astype(float32))
  # if ("ke_ent3" in globals()): 
  #   Nprim += 1
  #   electron_data.append(ke_ent3.astype(float32))
  # if ("ke_ent4" in globals()):
  #   Nprim +=1 
  #   electron_data.append(ke_ent4.astype(float32))


  header = [str(np.array(t)), str(nx), str(ny), str(nz), str(np.amin(x1)-0.5*dx1),str(np.amin(x2)-0.5*dx2),str(np.amin(x3)-0.5*dx3), str(dx1),str(dx2),str(dx3),str(a),str(gam),str(np.amin(r)),str(hslope),str(Nprim)]

  ud = Lower(uu,gg)
  bd = Lower(bu,gg)

  igrid,jgrid,kgrid = meshgrid(np.arange(0,nx),np.arange(0,ny),np.arange(0,nz),indexing='ij')
  data = [igrid,jgrid,kgrid,x1.astype(float32),x2.astype(float32),x3.astype(float32),r.astype(float32),th.astype(float32),ph.astype(float32),
          rho.astype(float32),(press/(5./3.-1.)).astype(float32),vel1.astype(float32),vel2.astype(float32),vel3.astype(float32),Bcc1.astype(float32),Bcc2.astype(float32),
          Bcc3.astype(float32),(press/rho**(5./3.)).astype(float32),
          uu[0].astype(float32),uu[1].astype(float32),uu[2].astype(float32),uu[3].astype(float32),ud[0].astype(float32),ud[1].astype(float32),ud[2].astype(float32),
          ud[3].astype(float32), bu[0].astype(float32),bu[1].astype(float32),bu[2].astype(float32),bu[3].astype(float32),bd[0].astype(float32),bd[1].astype(float32),
          bd[2].astype(float32),bd[3].astype(float32),gdet.astype(float32)]
  # data = data + electron_data
  data = np.array(data).astype(float32)
  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()
def box_limits(a,center_x = 0.0, center_y = 0.0):
  plt.xlim(-a+center_x,a+center_x)
  plt.ylim(-a+center_y,a+center_y)

def subplot_axis_label(fig,x_label,y_label,fontsize=15,bottom=0.15,left=0.15):
  fig.subplots_adjust(bottom=bottom)
  fig.subplots_adjust(left=left)
  fig.text(0.5, 0.04, x_label, fontsize=fontsize,ha='center')
  fig.text(0.02, 0.5,y_label, fontsize=fontsize, va='center', rotation='vertical')

def subplot_draw_cb(fig,c,r_lim=0.83,b_lim = 0.15,height=0.77,width = 0.03):
    global cb
    fig.subplots_adjust(right=r_lim)
    cbar_ax = fig.add_axes([r_lim+0.02, b_lim, width, height])
    cb = plt.colorbar(c, cax=cbar_ax)
  


def r_to_ir_npz(r_input,r_arr):
  dlog10r = np.diff(np.log10(r_arr[:,0,0]))[0]
  r_min = r_arr[0,0,0]
  r_out = r_arr[-1,0,0]
  #r = r_min * 10**(ir*dlog10r)
  return np.int(np.round(np.log10(r_input/r_min)/dlog10r))

def th_to_ith_npz(th_input,th_arr):
  dth = np.diff((th_arr[0,:,0]))[0]
  th_min = th_arr[0,0,0]
  th_out = th_arr[0,-1,0]
  #r = r_min * 10**(ir*dlog10r)
  return np.int(np.round((th_input-th_min)/dth))

def ph_to_iph_npz(ph_input,ph_arr):
  dph = np.diff((ph_arr[0,0,:]))[0]
  ph_min = ph_arr[0,0,0]
  ph_out = ph_arr[0,0,-1]
  #r = r_min * 10**(ir*dlog10r)
  return np.int(np.round((ph_input-ph_min)/dph))

def single_star_accretion_diagram():
  clf()
  plt.style.use('dark_background')  
  from matplotlib.patches import Ellipse
  from matplotlib.patches import FancyArrowPatch
  l_array = np.linspace(0,2,20) #np.array([,0.1,0.3,0.,1])
  th_array = np.linspace(0,np.pi,20)
  cmap = plt.get_cmap('RdBu_r')
  indices = np.linspace(0,cmap.N,len(l_array))
  my_colors = [cmap(int(i)) for i in indices]
  i = 0
  for l in l_array:
    lw = 2 #fabs(l)*2
    #l = 1+cos(th)
    if l<1: color = 'blue'
    else: color = 'red'
    if (l<=np.sqrt(2)):
      ecc = 1-l**2
      # r = a (1 - ecc*cos(E) ) .... cos(E) = -1
      a = 1/(1+ecc)
      b = a*sqrt(1-ecc**2)
      x_cent = 1-a
      r_circ = sqrt(l)
      e1 = Ellipse((x_cent,0),2*a,2*b,fill=False,lw=lw,color = color) #my_colors[i]) 
      e1 = Ellipse((0,0),r_circ*2,r_circ*2,fill=False,lw=lw,color = color) #my_colors[i]) 

      plt.gca().add_patch(e1)
    else:
      continue
      ecc =l**2-1
      a = 1/(ecc-1) #1/(ecc-1) #-1*l/(1-ecc**2)
      # a = 1/(2 - l**2)
      # ecc = 1-1/a
      b = a*sqrt(ecc**2-1)
      x_cent = 1+a 
      x_arr = np.linspace(-10,1,2000)
      y_arr = b*np.sqrt((x_arr-x_cent)**2/a**2 -1)
      plt.plot(x_arr,y_arr,lw=lw,color = color )#my_colors[i])
    i = i + 1
  plt.xlim(-2.25,1.25)
  plt.ylim(-1.5,2)
  plt.xlim(-1.5,1.5)
  plt.ylim(-1.5,1.5)
  plt.plot(1,0,marker='*',ms=20,color = 'gold')
  arrow = FancyArrowPatch((1.15,-.1),(1.15,.4),arrowstyle='simple,head_width=8,head_length=8',color = 'gold', \
    lw=2)#,connectionstyle="arc3,rad=0.4")
  plt.gca().add_patch(arrow)
  bh = Ellipse((0,0),.01,.01,color = 'white',lw=2) 
  star_orbit = Ellipse((0,0),2,2,color = 'gold',lw=2,ls = '--',fill=False)
  plt.gca().add_patch(star_orbit)
  plt.gca().add_patch(bh)
  plt.gca().get_xaxis().set_visible(False)
  plt.gca().get_yaxis().set_visible(False)
  plt.gca().set_aspect('equal')
  plt.savefig('single_star_circ.pdf')


def run_dependency(qsub_file,i_dep = 3,orig_id = None):
  import subprocess
  if (orig_id is None):
    out = subprocess.check_output(['sbatch',qsub_file])
  else:
    out = subprocess.check_output(['sbatch','--dependency=afterany:%d' %orig_id,qsub_file])
  for i in range(i_dep):
     prev_job = [int(s) for s in out.split() if s.isdigit()][0]
     out = subprocess.check_output(['sbatch','--dependency=afterany:%d' %prev_job,qsub_file])
     print (out)
     print ("with dependency: %d" %prev_job)


def get_RM(x=0,y=0,cum=False):
  from scipy.integrate import cumtrapz
  global z_los,ray
  e_charge = 4.803e-10
  me = 9.109e-28
  cl = 2.997924e10
  mp = 1.6726e-24
  pc = 3.086e18
  kyr = 3.154e10
  msun = 1.989e33


  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = 3. * Z_solar
  X = 0.
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)

  Bunit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 

  ray = ds.r[(x,"pc"),(y,"pc"),:]

  ne = np.array(ray['density'].in_cgs())/mue/mp
  z = np.array(ray['z'].in_cgs())

  integrand_RM = e_charge**3/(2.0*np.pi * me**2 * cl**4) * ne *np.array(-ray['Bcc3'].in_cgs())
  integrand_DM = ne
  z_los = z
  if (cum==False):
    RM = np.trapz(integrand_RM[z<0],z[z<0])
    DM = np.trapz(integrand_DM[z<0],z[z<0])
    return RM,DM
  else:
    RM = cumtrapz(integrand_RM,z)
    DM = cumtrapz(integrand_DM,z)
    return RM,DM


def get_Xray_Lum_los(file_prefix,x=0,y=0):
    mp_over_kev = yt.YTQuantity(9.994827,"kyr**2/pc**2")
    Z_o_X_solar = 0.0177
    Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
    X_solar = 0.7491
    Z_solar = 1.0-X_solar - Y_solar

    muH_solar = 1./X_solar
    Z = 3.0 * Z_solar
    X = 0.
    mue = 2. /(1.+X)
    mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
    mp = yt.YTQuantity(8.41175e-58,"Msun")
    kb = yt.YTQuantity(1.380658e-16,"erg/K")

    def Lam_func(TK):
        f1 = file_prefix + "_H_only.dat"
        f2 = file_prefix + "_He_only.dat"
        f3 = file_prefix + "_Metals_only.dat"
        data = np.loadtxt(f1)
        T_tab = data[:,0] * 1.16045e7
        Lam_H = data[:,1]
        data = np.loadtxt(f2)
        Lam_He = data[:,1]
        data = np.loadtxt(f3)
        Lam_metals = data[:,1]
        # T_tab = 10.**data[:,0]
        T_min = np.amin(T_tab)
        T_max = np.amax(T_tab)
        # if isinstance(TK,)
        # TK[TK<T_min] = T_min
        # TK[TK>T_max] = T_max
        # Lam_tab = 10.**data[:,1]

        Lam_tab = (X/X_solar) * Lam_H + ( (1.-X-Z)/Y_solar) * Lam_He + (Z/Z_solar) * Lam_metals
        from scipy.interpolate import InterpolatedUnivariateSpline
        Lam = InterpolatedUnivariateSpline(T_tab,Lam_tab,k = 1,ext =3)  #ext = 3 returns boundary value if outside range of T
        return Lam(TK)* yt.YTQuantity(1,"g * cm**5/s**3")
    def _Lam_chandra(field,data):
        T_K = (data['gas','pressure']/data['gas','density'] * mu_highT*mp/kb).in_units("K")
        #T_K = T_kev*1.16e7
        nH = data['gas','density']/mp/muH_solar
        ne = data['gas','density']/mp/mue
        return Lam_func(T_K).in_units("Msun *pc**5/kyr**3 ") * (ne*nH).in_units("pc**-6")

    ds.add_field(('gas','Lam_chandra'),function = _Lam_chandra,units="Msun/kyr**3/pc",particle_type = False,sampling_type="cell",force_override=True)
    
    ray = ds.r[(x,"pc"),(y,"pc"),:]

    return np.trapz(np.array(ray['Lam_chandra'].in_cgs()),np.array(ray['z'].in_cgs()))



    #res = 300
    #length_im = r_out
#frb = proj.to_frb((length_im,"pc"),[res,res])
      #image = frb['Lam_chandra']*Lz
      #x_im = np.linspace(-length_im/2.,length_im/2.,res)
      #y_im = np.linspace(-length_im/2.,length_im/2.,res)


    return average_value
def PSR_pos():
  global t_yr,ddelta,dalpha,YEAR,MONTH,DAY,valpha,vdelta
  arc_secs = 0.040239562
  dat = np.loadtxt('PSR.dat',usecols=(0,13,16))
  t_mjd = dat[:,0]

  #correct for Errata Bower+ 2016
  dalpha = (dat[:,1] + 18.36) * 1e-3 * arc_secs
  ddelta = (dat[:,2] + 32.00) * 1e-3 * arc_secs

  JD = t_mjd + 2400000.5 
  L= JD+68569
  N= 4*L//146097
  L= L-(146097*N+3)//4
  I= 4000*(L+1)//1461001
  L= L-1461*I//4+31
  J= 80*L//2447
  K= L-2447*J//80
  L= J//11
  J= J+2-12*L
  I= 100*(N-49)+I+L
  YEAR= I
  MONTH= J
  DAY= K

  t_yr = YEAR + MONTH/12.0 + DAY/365.25

  valpha = 2.45 * arc_secs
  vdelta = 5.89 * arc_secs

  #t_yr = (t_mjd + 2400000.5)/365.25 - 4713

def set_Be_units(r_star = 7.0,m_star = 9.2):
  global rho_unit, p_unit,B_unit,L_unit,T_unit,mass_unit,t_unit,v_unit
  global mp,kb
  rho_unit = 1e-11 #Okazaki 2001
  R_sun = 6.96e10
  M_sun = 1.99e33
  L_unit =  r_star * R_sun
  G_newt = 6.67259e-8
  kb = 1.380649e-16
  mp = 1.67e-24
  #GM/L_unit = v_unit
  #M = r_star/G
  mass_unit = m_star * M_sun
  t_unit = L_unit/np.sqrt(G_newt*mass_unit/L_unit)
  v_unit = L_unit/t_unit
  T_unit = mp * v_unit**2.0 / (kb)


  #2 kb T/mp = press/rho

def set_units():
  global UnitB,Unitlength
  UnitDensity = 6.767991e-23; 
  UnitEnergyDensity = 6.479592e-7; 
  UnitTime = 3.154e10;  
  Unitlength = 3.086e+18; 
  UnitB = Unitlength/UnitTime * np.sqrt(4. * np.pi* UnitDensity);



def compare_old_wind_new_wind():
  old_dir = '/global/scratch/smressle/star_cluster/stellar_wind_test_mhd/no_refine_cos_theta'
  new_dir = '/global/scratch/smressle/star_cluster/stellar_wind_test_mhd/3D_B_source_sinr_norm'

  os.chdir(old_dir)
  rdhdf5(100,ndim=3,coord="xy",user_x2=False,gr=False,a=0)

  plt.close()
  plt.figure(figsize=(10,5))
  plt.clf()
  plt.subplot(121)
  pcolormesh(x[:,0,:],z[:,0,:],log10(bsq/np.amax(bsq))[:,ny//2,:],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$z$ ',fontsize = 20)

  plt.subplot(122)
  pcolormesh(x[:,:,0],y[:,:,0],log10(bsq/np.amax(bsq))[:,:,nz//2],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$y$ ',fontsize = 20)
  cb = plt.colorbar()
  cb.set_label(r'$\log_{10}\left(b^2\right)$',fontsize=25)

  plt.suptitle(r'Old (Bad)',fontsize=20)


  os.chdir(new_dir)
  rdhdf5(100,ndim=3,coord="xy",user_x2=False,gr=False,a=0)

  plt.figure(figsize=(10,5))
  plt.clf()
  plt.subplot(121)
  pcolormesh(x[:,0,:],z[:,0,:],log10(bsq/np.amax(bsq))[:,ny//2,:],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$z$ ',fontsize = 20)

  plt.subplot(122)
  pcolormesh(x[:,:,0],y[:,:,0],log10(bsq/np.amax(bsq))[:,:,nz//2],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$y$ ',fontsize = 20)
  cb = plt.colorbar()
  cb.set_label(r'$\log_{10}\left(b^2\right)$',fontsize=25)

  plt.suptitle(r'New (Good)',fontsize=20)

  os.chdir(old_dir)

  rd_yt_convert_to_spherical(100,MHD=True)
  Bcc1 = B1
  Bcc2 = B2
  Bcc3 = B3
  get_mdot(mhd=True)
  
  set_units()
  B_a = 1.0/UnitB
  r_a = 6.957e10/Unitlength
  plt.figure(1)
  plt.clf()
  ir = 200
  B_sol = (B_a * r_a/r * np.sin(theta) ).mean(-1)[ir,ny//2]
  plt.plot(theta[200,:,0],Bphi[ir,:,:].mean(-1)/B_sol,lw=2,label = r'Old')

  os.chdir(new_dir)
  rd_yt_convert_to_spherical(100,MHD=True)
  Bcc1 = B1
  Bcc2 = B2
  Bcc3 = B3
  get_mdot(mhd=True)
  
  plt.figure(1)
  B_a = 30/UnitB
  B_sol = (B_a * r_a/r * np.sin(theta) ).mean(-1)[ir,ny//2]
  plt.plot(theta[200,:,0],Bphi[ir,:,:].mean(-1)/B_sol,lw=2,label = r'New')

  plt.plot(theta[200,:,0],sin(theta[200,:,0]),lw=2,ls='--',label = r'$\sin(\theta))$')

  plt.xlabel(r'$\theta$',fontsize = 20)
  plt.ylabel(r'$B_\varphi$ (Norm.) ',fontsize = 20)
  plt.legend(loc = 'best',frameon=False,fontsize=15)
  plt.setp(plt.gca().get_xticklabels(), fontsize=15)
  plt.setp(plt.gca().get_yticklabels(), fontsize=15)
  plt.tight_layout()


def rd_rst(fname):
  fin = open(fname,'rb')
  while b'par_end' not in fin.readline():
    continue
  dtype = np.float32
  body = np.fromfile(fin,dtype=dtype,count=-1)


def brackett_gamma():
  TK = press/rho * mu_highT*mp_over_kev * keV_to_Kelvin
  TK[TK<1e4] = 1e4
  ne = rho/mue * rho_to_n_cgs
  npro = rho/mu_highT * rho_to_n_cgs
  j = 3.44e-27 * ne*npro *(1e4/TK)**1.09 * ne * npro 

  pc = 3.086e18
  surface_brightness = j.mean(-1) * (amax(z)-amin(z) ) * pc 



def plot_SMR_grid(levels =9):
  plt.clf()
  plt.ylim(-1,1)
  plt.xlim(-1,1)
  for level in np.arange(levels+1):
    xmax = 1.0/2.0**level
    for x in np.linspace(-xmax,xmax,17):
      plt.plot([-xmax,xmax],[x,x],lw=1,color='k')
      plt.plot([x,x],[-xmax,xmax],lw=1,color='k')


def plot_fieldlines_gr(box_radius = 0.003,xbox_radius = None, ybox_radius = None,a=0,density=1,color='black',npz=False,phi_avg = False,arrowstyle='->',lw=1,yz=False):
  global x_stream,z_stream,Bx,Bz
  if (xbox_radius is None and ybox_radius is None): 
    xbox_radius = box_radius
    ybox_radius = box_radius
  elif (ybox_radius is None): ybox_radius = xbox_radius
  elif (xbox_radius is None): xbox_radius = ybox_radius

  x_stream,z_stream = np.meshgrid(np.linspace(0,xbox_radius,128),np.linspace(-ybox_radius,ybox_radius ,128),indexing = 'ij')

  dx_dr = np.sin(th)
  dz_dr = np.cos(th)

  dx_dtheta = r * np.cos(th)
  dz_dtheta = -r * np.sin(th)

  dx_dphi = a *np.cos(ph)
  dz_dphi = 0.0

  if (npz==True):
    Bx = Bcc1 *1.0
    Bz = Bcc3 * 1.0
    if(yz==True): Bx = Bcc2 *1.0 
  else:
    Bx  = dx_dr * Bcc1 + dx_dtheta * Bcc2 + dx_dphi * Bcc3
    Bz  = dz_dr * Bcc1 + dz_dtheta * Bcc2 + dz_dphi * Bcc3

  if (phi_avg==True):
    Bx = Bx.mean(-1)[:,:,None] * (r/r)
    Bz = Bz.mean(-1)[:,:,None] * (r/r)

  if (yz==True): iz = nz//4
  else: iz = 0

  vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bx[:,:,iz].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bz[:,:,iz].flatten(),(x_stream,z_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)
  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bx[:,:,iz+nz//2].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bz[:,:,iz + nz//2].flatten(),(x_stream,z_stream),method = 'nearest')

  if (npz==False): plt.streamplot(-x_stream.transpose(),z_stream.transpose(),-vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)
  else: plt.streamplot(-x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)
  
def plot_fieldlines_gr_midplane(box_radius = 0.003,a = 0,color ='black',npz=False,density = 1,arrowstyle=None,lw=1):
  global x_stream,y_stream,Bx,By
  x_stream,y_stream = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  dx_dr = np.cos(ph)
  dy_dr = np.sin(ph)
  dx_dphi = -r * np.sin(ph) + a * np.cos(ph)
  dy_dphi =  r * np.cos(ph) + a * np.sin(ph)

  if (npz==True):
    Bx = Bcc1
    By = Bcc2
  else:
    Bx  = dx_dr * Bcc1 + dx_dphi * Bcc3 
    By  = dy_dr * Bcc1 + dy_dphi * Bcc3


  vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Bx[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')
  vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),By[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),y_stream.transpose(),vx.transpose(),vy.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)

def plot_streamlines_gr_midplane(box_radius = 0.003,a = 0,color ='black'):
  global x_stream,y_stream,Bx,By
  x_stream,y_stream = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  dx_dr = np.cos(ph)
  dy_dr = np.sin(ph)
  dx_dphi = -r * np.sin(ph) + a * np.cos(ph)
  dy_dphi =  r * np.cos(ph) + a * np.sin(ph)

  Bx  = dx_dr * uu[1]/uu[0] + dx_dphi * uu[3]/uu[0] 
  By  = dy_dr * uu[1]/uu[0] + dy_dphi * uu[3]/uu[0]


  vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Bx[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')
  vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),By[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),y_stream.transpose(),vx.transpose(),vy.transpose(),color = color)

def plot_streamlines_gr(box_radius = 0.003,xbox_radius = None,ybox_radius = None,npz=False,yz=False,lw=1,density = 1,arrowstyle='->',color='black'):
  global x_stream,z_stream,Bx,Bz
  if (xbox_radius is None and ybox_radius is None): 
    xbox_radius = box_radius
    ybox_radius = box_radius
  elif (ybox_radius is None): ybox_radius = xbox_radius
  elif (xbox_radius is None): xbox_radius = ybox_radius

  x_stream,z_stream = np.meshgrid(np.linspace(0,xbox_radius,128),np.linspace(-ybox_radius,ybox_radius ,128),indexing = 'ij')

  # x_stream,z_stream = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  dx_dr = np.sin(th)
  dz_dr = np.cos(th)
  dx_dtheta = r * np.cos(th)
  dz_dtheta = -r * np.sin(th)

  if (npz==True):
    vvx = uu[1]/uu[0]
    vvz = uu[3]/uu[0]
    if (yz==True): vvx = uu[2]/uu[0]
  else: 
    vvx =    dx_dr * uu[1]/uu[0] + dx_dtheta * uu[2]/uu[0]
    vvz =    dz_dr * uu[1]/uu[0] + dz_dtheta * uu[2]/uu[0]
    ##if (yz==True):vvx = dx_dr * uu[1]/uu[0] + dx_dtheta * uu[2]/uu[0]

  if (yz==True): iz = nz//4
  else: iz = 0


  vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvx[:,:,iz].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvz[:,:,iz].flatten(),(x_stream,z_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = color,linewidth=lw,arrowstyle=arrowstyle,density=density)
  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvx[:,:,iz + nz//2].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvz[:,:,iz + nz//2].flatten(),(x_stream,z_stream),method = 'nearest')

  if (npz==False): plt.streamplot(-x_stream.transpose(),z_stream.transpose(),-vx.transpose(),vz.transpose())
  else: plt.streamplot(-x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),linewidth=lw,arrowstyle=arrowstyle,density=density,color = color)
  


def rd_inflow_file(fname,a=0,create_vector_potential = False):
  data = np.loadtxt(fname)
  global r, rho, ur,uphi, Br,Bphi,atheta
  r = data[:,0]
  rho = data[:,1]
  ur = data[:,2]
  uphi = data[:,3]
  Br = data[:,4]
  Bphi = data[:,5]


  th = np.pi/2.0
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )

  #atheta = -(gdet*Br) * 2.0 * np.pi 

  if (create_vector_potential == True):

    dr = np.diff(log(r))[0]*r

    atheta2 = (gdet*Bphi*dr).cumsum()

    data_new = np.array([r,rho,ur,uphi,Br,Bphi,atheta2]).transpose()

    np.savetxt(fname[:-4] + "_vector_potential.dat",data_new)


def convert_inflow_to_cartesian(fname, a=0):

    rd_inflow_file(fname)
    nx = r.shape[0]
    ny = 1
    nz = 1

    g_bl = bl_metric(r[:,None,None],th = pi/2,a=0)
    g_bl = g_bl[:,:,:,0,0]*1.0

    ut = np.sqrt((-ur*ur*g_bl[1][1] - uphi*uphi*g_bl[3][3] -1) / g_bl[0][0] )

    bt = g_bl[0,3]*ut*Bphi + g_bl[1][1]*ur*Br + g_bl[3][3]*uphi*Bphi 
    br = 1.0/ut * (Br + bt * ur);
    bphi = 1.0/ut * (Bphi + bt * uphi)

    phi_vals = np.linspace(0,2*pi,200)

    r_grid,phi_grid = np.meshgrid(r,phi_vals,indexing='ij')

    x_grid = r_grid * np.cos(phi_grid) + a * np.sin(phi_grid)
    y_grid = r_grid * np.sin(phi_grid) - a * np.cos(phi_grid)
    z_grid = 0.0*x_grid

    ut = ut[:,None]*(r_grid/r_grid)
    uphi = uphi[:,None]*(r_grid/r_grid)
    ur = ur[:,None]*(r_grid/r_grid)

    bt = bt[:,None]*(r_grid/r_grid)
    bphi = bphi[:,None]*(r_grid/r_grid)
    br = br[:,None]*(r_grid/r_grid)

    uu_bl = np.array([ut,ur,0*ur,uphi])
    bu_bl = np.array([bt,br,0*br,bphi])


    g_bl = bl_metric(r_grid[:,:,None]*1.0,pi/2*(r_grid/r_grid)[:,:,None],a=0)
    g_bl = g_bl[:,:,:,:,0]*1.0
    ud_bl = Lower(uu_bl,g_bl)
    bd_bl = Lower(bu_bl,g_bl)

    bsq_bl = gr_dot(bu_bl,bd_bl)

    uu_cks = bl_vec_to_cks(x_grid,y_grid,z_grid,np.array([ut,ur,ur*0,uphi]),a=a)
    bu_cks = bl_vec_to_cks(x_grid,y_grid,z_grid,np.array([bt,br,br*0,bphi]),a=a)

    cks_metric(x_grid[:,:,None]*1.0,y_grid[:,:,None]*1.0,z_grid[:,:,None]*1.0,0,0,a)
    g = g[:,:,:,:,0]

    ud_cks = Lower(uu_cks,g)
    bd_cks = Lower(bu_cks,g)

    bsq_cks = gr_dot(bu_cks,bd_cks)

    Bx = bu_cks[1]*uu_cks[0] - bu_cks[0]*uu_cks[1]
    By = bu_cks[2]*uu_cks[0] - bu_cks[0]*uu_cks[2]

    ## d Az/dy = Bx 
    ## d Az/dx = -By 



def psicalc_slice(B1 = None,gr=False,xy=False,iphi = 0):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Bcc1
    if (xy==False):
      _dx2 = np.diff(x2f)
      daphi = -(r*np.sin(th)*B1)[:,:,iphi]*_dx2[None,:]
      if (gr==True): daphi = -(gdet*B1)[:,:,iphi]*_dx2[None,:]
      if (gr== False): aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/np.sin(th)[:,:,0]
      else: aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    else: #Calculate Ay assuming By = 0 (i.e. projecting the magnetic field onto the plane)
      daphi = -B1[:,ny//2,:]
      aphi = daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi


    return(aphi)


def test_cks_functions():
  a =0.95
  nx = 15
  ny = 15
  nz = 15
  x = np.linspace(-5,5,nx)
  y = np.linspace(-5,5,ny)
  z = np.linspace(-5,5,nz)

  x,y,z = np.meshgrid(x,y,z,indexing='ij')


  rand_vec = rand(4,nx,ny,nz)


  cks_metric(x,y,z,0,0,a)

  rand_vec_lower = Lower(rand_vec,g)

  rand_vec_sq_cks = gr_dot(rand_vec,rand_vec_lower)

  rand_vec_ks = cks_vec_to_ks(rand_vec,x,y,z,0,0,a)

  cks_coord_to_ks(x,y,z,a=a)

  ks_metric(r,th,a=a)

  rand_vec_ks_lower = Lower(rand_vec_ks,g)

  rand_vec_sq_ks = gr_dot(rand_vec_ks,rand_vec_ks_lower)



  g = bl_metric(r,th,a=a)
  rand_vec_bl_lower = Lower(rand_vec,g)
  rand_vec_sq_bl = gr_dot(rand_vec_bl_lower,rand_vec)

  rand_vec_cks = bl_vec_to_cks(x,y,z,rand_vec,a=a)

  cks_metric(x,y,z,0,0,a)
  rand_vec_cks_lower = Lower(rand_vec_cks,g)
  rand_vec_sq_cks = gr_dot(rand_vec_cks,rand_vec_cks_lower)


def get_xyz_angular_momentum():
  global lz,ly,lx,Lx,Ly,Lz,Ltot
  uu_bl = ks_vec_to_bl(uu,0)  
  uu_cks = bl_vec_to_cks(x,y,z,uu_bl,0)

  cks_metric(x,y,z,0,0,0)
  ud_cks = Lower(uu_cks,g)

  ##ud_phi_ks = u_mu dx^mu/dphi_ks = u_x dx/dphi + u_y dy/dphi + ..
  ##ud_phi_ks =  u_x (-y) + u_y (x) + ..


  lz = ud_cks[1] * (-y) + ud_cks[2] * x 
  ly = ud_cks[1] * z - ud_cks[3] * x
  lx = ud_cks[3] * (y) - ud_cks[2] * z       

  Lx = angle_average(rho*lx,gr=True)
  Ly = angle_average(rho*ly,gr=True)
  Lz = angle_average(rho*lz,gr=True)

  Ltot = np.sqrt(Lx**2 + Ly**2 + Lz**2)


def test_restart_file():

  yr = 31556926.0
  pc = 3.09e18;    
  cl = 2.99792458e10 * (1e3 * yr)/pc 
  gm_ = 0.019
  L_unit = gm_/cl**2;
  r_in = 2.0*2.0/128.0/(2.0**9.0)/L_unit
  rs = 2.0
  rho_unit = 1960.53
  B_unit = cl * np.sqrt(rho_unit);
  
  rdhdf5(00,ndim=3,coord='spherical',gr=True,a=0,user_x2=True)

  r_orig=r 

  vr = angle_average(uu[1]/uu[0],weight=rho,gr=True)
  vth = angle_average(uu[2]/uu[0] * r,weight=rho,gr=True)
  vphi = angle_average(uu[3]/uu[0] * r * np.sin(th),weight=rho,gr=True)

  vsq_restart = angle_average((uu[1]/uu[0])**2 + (uu[2]/uu[0])**2*r**2 + (uu[3]/uu[0])**2*r**2*sin(th)**2,weight=rho,gr=True)

  Br = angle_average(Bcc1,gr=True)
  Bphi = angle_average(Bcc3 * r * np.sin(th),gr=True)
  Bth = angle_average(Bcc2 * r ,gr=True)

  bsq = angle_average(bsq,gr=True)

  clf()
  # loglog(r[:,0,0],-vr,ls='-',color='blue')
  # loglog(r[:,0,0],vphi,ls='-',color='red')

  loglog(r[:,0,0],Br,ls='-',color='blue')
  loglog(r[:,0,0],Bphi,ls='-',color='red')
  loglog(r[:,0,0],Bth,ls='-',color='green')


  # loglog(r[:,0,0],bsq,ls='-',color='red')

  rdnpz("/global/scratch/smressle/star_cluster/test_new_code/without_S2_9_levels_beta_1e2_v3_orbits_comet/dump_spher_120_th_1.3_phi_-1.8.npz")

  get_mdot(True)

  vsq_orig = angle_average_npz(vr**2 + vphi**2 + vth**2,weight=rho)
  bsq = angle_average_npz(Br**2 + Bphi**2 + Bth**2)

  vr = angle_average_npz(vr,weight=rho)
  vphi = angle_average_npz(vphi,weight=rho)

  Br = angle_average_npz(Br)
  Bphi = angle_average_npz(Bphi)
  Bth = angle_average_npz(Bth)



  # loglog(r[:,0,0]/L_unit * (rs/r_in),-vr/cl * sqrt(r_in/rs),ls='--',color='blue')
  # loglog(r[:,0,0]/L_unit * (rs/r_in),vphi/cl * sqrt(r_in/rs),ls='--',color='red')

  loglog(r[:,0,0]/L_unit * (rs/r_in),Br/B_unit * (r_in/rs),ls='--',color='blue')
  loglog(r[:,0,0]/L_unit * (rs/r_in),Bphi/B_unit * (r_in/rs),ls='--',color='red')
  loglog(r[:,0,0]/L_unit * (rs/r_in),Bth/B_unit * (r_in/rs),ls='--',color='green')

def cks_metric_code(x1,x2,x3,a):
  global g
  x = x1;
  y =x2;
  z = x3;
  def SQR(q):
    return q**2.0
  R = np.sqrt(SQR(x) + SQR(y) + SQR(z));
  r = SQR(R) - SQR(a) + np.sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = np.sqrt(r/2.0);


  eta = [0,0,0,0]
  l_lower = [0,0,0,0]
  l_upper = [0,0,0,0]

  f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  l_upper[0] = -1.0;
  l_upper[1] = (r*x + a*y)/( SQR(r) + SQR(a) );
  l_upper[2] = (r*y - a*x)/( SQR(r) + SQR(a) );
  l_upper[3] = z/r;

  l_lower[0] = 1.0;
  l_lower[1] = l_upper[1];
  l_lower[2] = l_upper[2];
  l_lower[3] = l_upper[3];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;


  g = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
  g[0][0] = eta[0] + f * l_lower[0]*l_lower[0];
  g[0][1] = f * l_lower[0]*l_lower[1];
  g[0][2] = f * l_lower[0]*l_lower[2];
  g[0][3] = f * l_lower[0]*l_lower[3];
  g[1][0] = g[0][1]
  g[2][0] = g[0][2]
  g[3][0] = g[0][3]
  g[1][1] = eta[1] + f * l_lower[1]*l_lower[1];
  g[1][2] = f * l_lower[1]*l_lower[2];
  g[2][1] = g[1][2]
  g[1][3] = f * l_lower[1]*l_lower[3];
  g[3][1] = g[1][3]
  g[2][2] = eta[2] + f * l_lower[2]*l_lower[2];
  g[2][3] = f * l_lower[2]*l_lower[3];
  g[3][2] = g[2][3]
  g[3][3] = eta[3] + f * l_lower[3]*l_lower[3];
  ##loglog(r[:,0,0]/L_unit * (rs/r_in),bsq/B_unit**2 * (r_in/rs)**2,ls='--',color='red')


def plot_mass_temp_hist():

  plt.clf()
  low_res = "/global/scratch/smressle/star_cluster/binary_wind_shock/symmetric_128_tcool_off_axis"
  high_res = "/global/scratch/smressle/star_cluster/binary_wind_shock/symmetric_512_tcool_off_axis"

  hist_avg_high_res = 0.0
  hist_avg_low_res = 0.0
  set_constants()
  dump_range = arange(8,14)
  for i in dump_range:
    for res_dir in [low_res,high_res]:
      os.chdir(res_dir)
      rdhdf5(i,ndim=3,coord='xy')
      box_length = 2.0*1.018e-3
      TK = press/rho * mu_highT*mp_over_kev*keV_to_Kelvin
      if (res_dir==low_res): res = 128.0
      else: res = 512.0
      dx = box_length/res
      cell_vol = dx**3.0 
      cell_mass = rho * cell_vol
      if (res_dir==low_res): nbins = 30
      else: 200
      T_bins = np.logspace(3.7,7.2,nbins)

      index_array = (y<(-x*1.75+.0006)) * (y>(-x*1.75-.0006))
      T_array = TK[index_array].flatten()
      cell_mass_array = cell_mass[index_array].flatten()
      if (res_dir==low_res): line_style = '--'
      else: line_style = '-'
      hist,bins = np.histogram(T_array,bins = T_bins, weights = cell_mass_array,density=True)

      if (res_dir==high_res): bins_high_res = bins 
      else: bins_low_res = bins

      if (res_dir==high_res): hist_avg_high_res += hist/(len(dump_range))
      else: hist_avg_low_res  += hist/(len(dump_range))

  plt.loglog(bins_low_res[1:],hist_avg_low_res,lw=2,label = r"$N=%d$" %(np.int(128)),ls="--")
  plt.loglog(bins_high_res[1:],hist_avg_high_res,lw=2,label = r"$N=%d$" %(np.int(512)),ls="--")

  plt.ylabel(r'$dM/dT$',fontsize = 20)
  plt.xlabel(r'$T$ (K)',fontsize=20)

  plt.legend(loc='best',frameon=0,fontsize=15)

  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
      label.set_fontsize(10)
  plt.tight_layout()
  plt.savefig("dM_dT_binary_wind.png")

  os.chdir(high_res)
  rdhdf5(13,ndim=3,coord='xy')
  plt.figure(2)
  plt.clf()

  c1 = pcolormesh(x[:,:,0]*206265,y[:,:,0]*206265,log10(rho[:,:,nz//2]),vmin=3,vmax=5)
  cb = plt.colorbar(c1,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
  cb.set_label(r"$\log_{10}(\int \rho dz)$ ($T > 9 \times 10^4 $K) ",fontsize=17)
  plt.xlabel(r'$x$ (Au)',fontsize = 20)
  plt.ylabel(r'$y$ (Au)',fontsize = 20)
  plt.tight_layout()
  plt.savefig("binary_wind_512.png")


  os.chdir(low_res)
  rdhdf5(13,ndim=3,coord='xy')
  plt.figure(3)
  plt.clf()

  c1 = pcolormesh(x[:,:,0]*206265,y[:,:,0]*206265,log10(rho[:,:,nz//2]),vmin=3,vmax=5)
  cb = plt.colorbar(c1,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
  cb.set_label(r"$\log_{10}( \rho)$ ",fontsize=17)
  plt.xlabel(r'$x$ (Au)',fontsize = 20)
  plt.ylabel(r'$y$ (Au)',fontsize = 20)
  plt.tight_layout()
  plt.savefig("binary_wind_128.png")



def plot_1d_grmhd():
  rd_1d_avg()
  plt.figure(1)
  plt.clf()
  ir = 10
  plt.plot(t,(((EdotEM)/abs(mdot)))[:,ir]*100,lw=2,ls='-',label = r'$\frac{P_{\rm jet}}{\dot M c^2} \times 100 $')
  plt.plot(t,(Phibh/sqrt(-mdot))[:,ir],lw=2,ls = '--',label = r'$\phi_{\rm BH}$')
  plt.xlabel(r'$t$ (M)',fontsize = 20)
  plt.legend(loc='best',frameon=False,fontsize=15)
  plt.ylim(0,150)
  plt.tight_layout()
  plt.figure(2)
  plt.clf()
  ir = 10
  p0 = 5.9880242e-03
  rho0 = 1
  gm_ = 1

  plt.plot(t,((-mdot))[:,ir]/mdot_bondi(p0,rho0),lw=2,ls='-')
  plt.xlabel(r'$t$ (M)',fontsize = 20)
  plt.ylabel(r'$|\dot M/\dot M_{\rm Bondi}|$',fontsize = 20)

  plt.ylim(0,0.4)
  plt.tight_layout()

  def plot_restart_mhd():
    os.chdir("/global/scratch/smressle/star_cluster/restart_mhd/beta_1e2_cooling_125_comet")
    rd_hst("star_wind.hst",True)
    plt.figure(1)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,-mdot_avg[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\dot M$ ($10^{-8} M_\odot/$yr)',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,3)
    plt.tight_layout()

    plt.savefig("mdot_t_restart_mhd.png")
    plt.figure(2)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,phibh[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\phi_{BH}$',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig("phibh_restart_mhd.png")

    plt.savefig("mdot_t_restart_mhd.png")
    plt.figure(2)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,phibh[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\phi_{BH}$',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig("phibh_restart_mhd.png")

    plt.figure(3)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,phibh[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\phi_{BH}$',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig("phibh_restart_mhd.png")

def get_Te(Rhigh,Rlow=1):
  global Te
  mue = 2
  mu = 1.351
  mui = mue * mu / (mue - mu)
  beta = press/bsq*2.0
  Ti_o_Te = (Rhigh*beta**2 + Rlow)/(1.0 + beta**2.0)

  T_tot = mu * press/(rho)

  Te = T_tot/ (mu/mue + mu/mui *Ti_o_Te)


def EHT_comp(a=0.9375):

  thmin = pi/3.0
  thmax = 2.0*pi/3.0


  get_mdot(mhd=True,gr=True,az=a)
  plt.figure(1)
  plt.clf()
  plt.subplot(231)

  loglog(r[:,0,0],angle_average_npz(rho,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))
  plt.xlim(1,50)
  plt.ylim(1e-2,1e0)

  plt.subplot(232)
  loglog(r[:,0,0],angle_average_npz(press,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-6,1e-2)

  plt.subplot(233)

  loglog(r[:,0,0],angle_average_npz(np.sqrt(bsq),weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-4,1e-1)

  plt.subplot(234)

  loglog(r[:,0,0],angle_average_npz(uu_ks[3],weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-3,1e1)

  plt.subplot(235)

  loglog(r[:,0,0],angle_average_npz(press+bsq/2.0,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-6,1e-2)

  plt.subplot(236)

  loglog(r[:,0,0],angle_average_npz(bsq/2.0/press,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-2,1e0)


  Edotmean = (Edot/abs(mdot))[:,50][500:].mean()
  plt.figure(2)

  ir = 50

  plt.subplot()


def Chris_EHT_comp():



  rd_1d_avg()
  dic = np.load("code_comparison_192.npz")
  a = 0.9375
  rh = ( 1.0 + np.sqrt(1.0-a**2) )
  ir = r_to_ir(rh)
  plt.figure(1)
  plt.clf()
  plt.subplot(411)



  plt.plot(t,-mdot[:,ir],lw=2,ls='--',label='Cartesian')
  plt.plot(dic['t'],dic['Mdot_horizon'],lw=2,ls='-',label='Spherical')
  plt.ylabel(r'$\dot M$')
  plt.ylim(0,1)
  plt.xlim(0,1e4)
  plt.legend(loc='best',frameon=False)


  plt.subplot(412)

  plt.plot(t,abs(Jdot/mdot)[:,ir],lw=2,ls='--')
  plt.plot(dic['t'],abs(dic['Jdot_horizon']/dic['Mdot_horizon']),lw=2,ls='-')
  plt.ylabel(r'$\dot J/ \dot M$')
  plt.ylim(1.5,2.5)
  plt.xlim(0,1e4)


  plt.subplot(413)

  plt.plot(t,abs(Edot/mdot)[:,ir],lw=2,ls='--')
  plt.plot(dic['t'],abs((dic['Edot_horizon']-dic['Mdot_horizon'])/dic['Mdot_horizon']),lw=2,ls='-')
  plt.ylabel(r'$|\dot E - \dot M|/ \dot M$')

  plt.ylim(0,.2)
  plt.xlim(0,1e4)

  plt.subplot(414)

  plt.plot(t,abs(Phibh/sqrt(-mdot)/sqrt(4*pi))[:,ir],lw=2,ls='--')
  plt.plot(dic['t'],abs((dic['Phi_horizon'])/sqrt(dic['Mdot_horizon'])/sqrt(4*pi)),lw=2,ls='-')
  plt.ylabel(r'$\phi_{\rm BH}$')

  plt.ylim(0,2)
  plt.xlim(0,1e4)

  plt.xlabel(r'$t$ $[M]$')

  plt.tight_layout()


  rd_1d_torus_avg()

  plt.figure(2)
  plt.clf()

  plt.subplot(221)
  plt.loglog(r[-1,:],rho[500:,:].mean(0),lw =2,ls='--',label='Cartesiann')
  plt.loglog(dic['r'],dic['rho_ave'],lw=2,ls='-',label = 'Spherical')
  plt.legend(loc='best',frameon=False)
  plt.ylabel(r'$\rho$')
  plt.ylim(1e-2,1e0)
  plt.xlim(1,50)

  subplot(222)
  plt.loglog(r[-1,:],press[500:,:].mean(0),lw=2,ls='--')
  plt.loglog(dic['r'],dic['pgas_ave'],lw=2,ls='-')
  plt.ylabel(r'$P_{\rm g}$')
  plt.ylim(1e-6,1e-2)
  plt.xlim(1,50)


  subplot(223)
  plt.loglog(r[-1,:],beta_inv[500:,:].mean(0),lw=2,ls='--')
  plt.loglog(dic['r'],1/dic['beta_inv_ave'],lw=2,ls='-')
  plt.xlabel(r'$r$ $[r_{\rm g}]$')

  plt.ylabel(r'$\beta^{-1}$')
  plt.ylim(1e-2,1e0)
  plt.xlim(1,50)

  subplot(224)
  plt.loglog(r[-1,:],pmag[500:,:].mean(0),lw=2,ls='--')
  plt.loglog(dic['r'],dic['pmag_ave'],lw=2,ls='-')

  plt.ylabel(r'$P_{B}$')
  #plt.ylim()
  plt.xlim(1,50)
  plt.xlabel(r'$r$ $[r_{\rm g}]$')

  plt.tight_layout()



  Edotmean = (Edot/abs(mdot))[:,ir][500:].mean()
  Edotstd = (Edot/abs(mdot))[:,ir][500:].std()

  phibhmean = abs(Phibh/sqrt(-mdot)/sqrt(4*pi))[:,ir][500:].mean()
  phibhstd = abs(Phibh/sqrt(-mdot)/sqrt(4*pi))[:,ir][500:].std()

  print("Edot/Mdot: ", Edotmean,"+/- ",Edotstd)
  print("phibh: ", phibhmean,"+/- ",phibhstd)





def Harrison_plots():

  c = 2.99792458e10
  gg_msun = 1.3271244e26
  kb = 1.380649e-16
  kpc = 3.085677581491368e21
  distance_kpc = 8.178
  nu = 230.0e9
  dic = np.load("star_wind_1e6_select.npz")


  I1 = dic['iquv_nu_select'][0,0] * c**2 / (2.0 * nu**2 * kb)
  I2 = dic['iquv_nu_select'][1,0] * c**2 / (2.0 * nu**2 * kb)
  I3 = dic['iquv_nu_select'][2,0] * c**2 / (2.0 * nu**2 * kb)
  I4 = dic['iquv_nu_select'][3,0] * c**2 / (2.0 * nu**2 * kb)

  Imax = 1.2
  Imin = 0.0

  n =1
  for I in [I1,I2,I3,I4]:
    plt.figure(figsize=(8,8))
    plt.clf()
    contourf(I/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("I%d.png" %n)
    n = n + 1

  plt.figure(figsize=(8,8))
  plt.clf()
  plt.subplot(221)
  contourf(I1/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')
  plt.subplot(222)
  contourf(I2/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')
  plt.subplot(223)
  contourf(I4/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')
  plt.subplot(224)
  contourf(I3/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')

  plt.tight_layout()
  plt.savefig("I_panel.png")

def gamma_rel(theta):
  return (10.0 + 20.0*theta)/(6.0 + 15.0*theta)
def ue_to_kappae(ue,rho,mue=2.0):
  mp_over_me = 1836.15267507
  cl = 306.4

  rhoe = rho/mp_over_me/mue 
  urat = ue/(rhoe *cl**2);
  theta_e = 1.0/30.0 * (-6.0 + 5.0 * urat + np.sqrt(36.0 + 180.0*urat +25.0*(urat**2)) ) ;
  return theta_e**(3.0/2.0) *  (theta_e + 2.0/5.0)**(3.0/2.0) / rhoe;

def get_Te_Tg(kappa,rho,press,gr=False,mue=2.0,mu_tot = None):
  global Te, Tg, Ti, mu_highT
  mp_over_kev = 9.994827
  mp_over_me = 1836.15267507
  set_constants()
  if (mu_tot is not None): mu_highT = mu_tot*1.0
  mui = 1.0/(1.0/mu_highT - 1.0/mue)
  cl = 306.4
  rhoe = rho/mp_over_me/mue;
  theta_e = 1.0/5.0 * (np.sqrt(1.0 + 25.0*(rhoe*kappa)**(2.0/3.0)) -1.0 )
  Te = theta_e * cl**2 /mp_over_me * mp_over_kev 
  Tg = press/rho * mp_over_kev * mu_highT

  pe = theta_e/mp_over_me *rho /mue
  pi = press-pe
  Ti = pi/rho * mui
  if (gr==True):
    Te = theta_e/mp_over_me 
    Tg = press/rho * mu_highT
def kappa_to_ue(kappa,rho,gr=False,mue=2.0):
  global theta_e
  mp_over_me = 1836.15267507
  cl = 306.4
  if (gr==True): cl = 1.0

  rhoe = rho/mp_over_me/mue;
  theta_e = 1.0/5.0 * (np.sqrt(1.0 + 25.0*(rhoe*kappa)**(2.0/3.0)) -1.0 )
  pe_ = rhoe * theta_e * (cl**2.0);
  return pe_ / (gamma_rel(theta_e) - 1.0); 

  
def gravity_term_gr(r,th,a,m=1):
  global aterm,pressterm,EMterm, advection_term,massterm,centrifugal_term
  global pressterm_EM,tensionterm_EM
  get_mdot(mhd=True,gr=True,az=a)
  ks_metric(r,th,a)
  ks_Gamma_ud(r,th,a)

  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(bu_ks,g))
  Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic = True,gam=5.0/3.0)
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )

  TudKinetic = 0 * Tud
  gam = 5./3.
  Tudpress = 0*Tud
  Tudpress_EM = 0*Tud
  Tudtension_EM = 0 * Tud
  for kapa in np.arange(4):
    for nu in np.arange(4):
      TudKinetic[kapa,nu] = (rho+press*(gam/(gam-1.0) ))*uu_ks[kapa]*ud_ks[nu]
      Tudpress[kapa,nu] = press*(kapa==nu)
      Tudpress_EM[kapa,nu]  = bsq/2.0 * (kapa==nu)
      Tudtension_EM[kapa,nu]  = - bu_ks[kapa]*bd_ks[nu]
  aterm = 0
  pressterm = 0
  pressterm_EM = 0
  advection_term= 0
  EMterm = 0
  centrifugal_term = 0
  massterm = 0
  tensionterm_EM = 0
  for i in arange(4):
    for j in arange(4):
      if(i==0 and j==0): aterm+= (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      else: advection_term += (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      EMterm += (gdet * TudEM[j][i]) * gammaud[i][1][j][:,:,None]
      pressterm += (gdet * Tudpress[j][i] * gammaud[i][1][j][:,:,None])
      pressterm_EM += (gdet * Tudpress_EM[j][i] * gammaud[i][1][j][:,:,None])
      tensionterm_EM += (gdet * Tudtension_EM[j][i] * gammaud[i][1][j][:,:,None])
      if ((i==3 and j==3) or (i==2 and j==2)): centrifugal_term += (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]

      massterm += (gdet * Tud[j][i]) * gammaud[i][1][j][:,:,None]
  #centrifugal_term += (gdet * TudKinetic[3][3]) * gammaud[3][1][3][:,:,None]
  EMterm -= np.gradient(gdet * TudEM[1][1],r[:,0,0],axis=0) + np.gradient(gdet * TudEM[2][1],th[0,:,0],axis=1)
  EMterm -= np.gradient(gdet * TudEM[3][1],ph[0,0,:],axis=2)
  tensionterm_EM -= np.gradient(gdet * Tudtension_EM[1][1],r[:,0,0],axis=0) + np.gradient(gdet * Tudtension_EM[2][1],th[0,:,0],axis=1)
  tensionterm_EM -= np.gradient(gdet * Tudtension_EM[3][1],ph[0,0,:],axis=2)
  pressterm -= np.gradient(press*gdet,r[:,0,0],axis=0)
  pressterm_EM -= np.gradient(bsq/2.0*gdet,r[:,0,0],axis=0)
  advection_term -= np.gradient(gdet * TudKinetic[1][1],r[:,0,0],axis=0) + np.gradient(gdet * TudKinetic[2][1],th[0,:,0],axis=1)
  advection_term -= np.gradient(gdet * TudKinetic[3][1],ph[0,0,:],axis=2)
  massterm -= np.gradient(gdet * Tud[1][1],r[:,0,0],axis=0) + np.gradient(gdet * Tud[2][1],th[0,:,0],axis=1)
  massterm -= np.gradient(gdet * Tud[3][1],ph[0,0,:],axis=2)
def Edot_terms_gr(r,th,a,m=1):
  global Edot,EdotEM,EdotMA
  get_mdot(mhd=True,gr=True,az=a)
  ks_metric(r,th,a)

  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(bu_ks,g))
  Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic = True,gam=5.0/3.0)
  mdot =rho*uu_ks[1]*4.0 * np.pi/3.0 * (3.0*r**2 + a**2)
  Edot = - (Tud[1][0]*4.0 * np.pi/3.0 * (3.0*r**2 + a**2) +mdot )
  EdotEM = -(TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*r**2 + a**2))
  EdotMA = Edot - EdotEM

def force_terms_gr(r,th,a):
  global EMforce, press_force
  get_mdot(mhd=True,gr=True,az=a)
  ks_metric(r,th,a)
  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(bu_ks,g))
  Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic = True,gam=5.0/3.0)
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
  EMforce = np.gradient(gdet * Tud[1][1],r[:,0,0],axis=1)

def get_uu_ff(r,th,a):
  global uu_ff_ks, ud_ff_ks

  uu_ff_bl = 0*uu
  uu_ff_bl[0] = (r**2+a**2+2.0*a**2/r)/(r**2+a**2-2.0*r)
  uu_ff_bl[1] = - sqrt(2.0*(1.0/r)+a**2/r**3) 
  uu_ff_bl[3] = (2*a/r)/(r**2+a**2-2.0*r)


  ut_m1_bl =2.0/r * (a**2+r**2)/(a**2+r**2-2.0*r) 
  Delta = r**2 -2.0*r + a**2
  ut_m1_ks = ut_m1_bl + uu_ff_bl[1] * 2.*r/Delta


  uu_ff_ks = bl_vec_to_ks(uu_ff_bl,a=a)
  ks_metric(r,th,a=a)
  ud_ff_ks = Lower(uu_ff_ks,g)



def udm1_approx(r,th,a):
  global ud_t_p1
  ks_inverse_metric(r,th,a=a)
  v1 = uu_ks[1] - gi[0,1]/gi[0,0] * uu_ks[0]
  v2 = uu_ks[2] - gi[0,2]/gi[0,0] * uu_ks[0]
  v3 = uu_ks[3] - gi[0,3]/gi[0,0] * uu_ks[0]


  ks_metric(r,th,a)
  tmp = g[1,1]*v1*v1 + 2.0*g[1,2]*v1*v2 + 2.0*g[1,3]*v1*v3+ g[2,2]*v2*v2 + 2.0*g[2,3]*v2*v3+ g[3,3]*v3*v3
  gamma = np.sqrt(1.0 + tmp);

  sigma = r**2 + a**2.*np.cos(th)**2
  gi_tt = -(1.0 + 2.0*r/sigma);
  g_tt = -(1.0 - 2.0*r/sigma);

  alpha = np.sqrt(-1.0/gi[0,0])

  A = 2.0*r/sigma
  alphai_approx = 1.0 + A/2.0 - A**2.0 /8.0 
  gamma_approx =  1.0 + tmp/2.0 - tmp**2.0/8.0 

  term_1 = A/2.0 - A**2.0 /8.0 
  term_2 = tmp/2.0 - tmp**2.0/8.0 
  gamma_over_alpha_approx = 1.0  + term_1 + term_2 +A*tmp/4.0

  term_3 = - 2.0*r/sigma
  ud_term_1_approx = -(1.0 +term_3 + term_1 + term_2 + A*tmp/4.0 + term_3*A/2.0 + term_3*tmp/2.0 )



  ud_t_p1  = -(term_3 + term_1 + term_2 + A*tmp/4.0 + term_3*A/2.0 + term_3*tmp/2.0 )

  for i in arange(1,4):
    ud_t_p1 += g[i,0]*uu_ks[i]


def get_conv_flux():


  Bx_dir = "/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/Bx_192_stampede"
  Bz_dir  = "/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/Bz_192_stampede"
  sixty_degree_dir = "/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/60_degree_192_stampede"
  thirty_degree_dir = '/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/30_degree_192_stampede'

  dir_list = [Bz_dir,thirty_degree_dir, sixty_degree_dir, Bx_dir ]
  for i in arange(1,5):
    plt.figure(i)
    plt.clf()
    os.chdir(dir_list[i-1])
    rdnpz('dump_spher_600_th_0_phi_0.npz')
    a = 0.9375
    get_mdot(mhd=True,gr=True,az=0.9375)
    ks_metric(r,th,a=0.9375)
    ud_ks = Lower(uu_ks,g)
    bd_ks = Lower(bu_ks,g)
    gam = 5.0/3.0
    w = 1.0 + press/rho * gam/(gam-1.0)
    Be = -ud_ks[0]*w  - 1.0

    Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic=True,gam =gam)

    FMA = -(TudMA[1][0] + rho * uu_ks[1]) * 4.0*np.pi/3.0 * (3.0*r**2+a**2)
    FEM = -(TudEM[1][0] ) * 4.0*np.pi/3.0 * (3.0*r**2+a**2)

    angle_limit = abs(th-pi/2) < 30.0/180.0*pi
    scale_height = angle_average_npz(abs(th-pi/2),weight=rho,gr=True,a=0.9375)
    scale_height_3D = scale_height[:,None,None] * (rho/rho)
    angle_limit = abs(th-pi/2) < scale_height_3D
    Edot_ma = angle_average_npz(FMA,weight=angle_limit,gr=True,a=a)
    mdot = angle_average_npz(rho * uu_ks[1] * 4.0*np.pi/3.0 * (3.0*r**2+a**2),weight=angle_limit,gr=True,a=a)
    Be_avg = angle_average_npz(Be,weight=angle_limit,gr=True,a=a)
    Be_avg_rho = angle_average_npz(Be*angle_limit,weight=rho,gr=True,a=a)

    Edot_EM = angle_average_npz(FEM,weight=angle_limit,gr=True,a=a)

    Edot_conv = Edot_ma - Be_avg* mdot 

    plt.loglog(r[:,0,0],-Edot_ma,lw=2,ls='-',label = r'$-F_{\rm M}$')
    plt.loglog(r[:,0,0],Edot_conv,lw=2,ls='--',label = r'$F_{\rm conv}$')
    plt.loglog(r[:,0,0],-Be_avg*mdot,lw=2,ls=':',label = r'$-F_{\rm adv}$')
    plt.loglog(r[:,0,0],Edot_EM,lw=2,ls='-.',label = r'$F_{\rm EM}$')
    plt.xlim(1e0,1e3)
    plt.ylim(1e0,1e4)

    plt.legend(loc='best',frameon=0,fontsize=12)


def get_gr_bondi_sol(r_bondi,gam=5.0/3.0):
  #r_b = 2 GM/ a_inf^2
  global mdotgr_bondi,rho,T,cs,ur,r,rc,ut,grr,gtt
  n = 1.0/(gam-1.0)
  a_inf = sqrt(2/r_bondi)
  T_inf = a_inf**2.0/gam

  def eqn_(rc):
    Tc = - n/(n**2 -2.0*(n+1.0)*rc +4.0*n+3.0)
    uc = sqrt(1.0/(2.0*rc))
    C1 = uc * Tc**n * rc**2
    C2 = (1.0 + (1.0+n)*Tc)**2.0 * (1.0-2.0/rc + C1**2/(rc**4*Tc**(2.0*n)))
    C2_prime = (1.0+(1.0+n)*T_inf)**2.0
    return C2-C2_prime

  rc = scipy.optimize.fsolve(eqn_,5.0)

  uc = sqrt(1.0/(2.0*rc))
  Tc = - n/(n**2 -2.0*(n+1.0)*rc +4.0*n+3.0)
  C1 = uc * Tc**n * rc**2
  C2 = (1.0 + (1.0+n)*Tc)**2.0 * (1.0-2.0/rc + C1**2/(rc**4*Tc**(2.0*n)))
  r = np.logspace(np.log10(2.0),4,1000)

  def eqn_(T,r):
      return (((1 + (1+n)*T)**2*(1 -2.0/r + C1**2/r**4/T**(2*n)) - C2))[0]
  T_guess = 2*T_inf
  T = []
  for r_val in r[::-1]:
      T_new = scipy.optimize.fsolve(eqn_,T_guess,args=(r_val))[0]
      T.append(T_new)
      T_guess = T_new
  T = np.array(T)[::-1]
  ur = C1/(T**n*r**2)
  grr = 1./(1.0-2.0/r)
  gtt = -(1.0-2.0/r)
  ut = np.sqrt((-1 - ur**2.0*grr)/gtt)

  rho = (T/T_inf)**n
  cs = sqrt(gam*T*rho/(gam/(gam-1.0)*rho*T + rho))
  mdotgr_bondi = rho * ur * r**2.0 * 4.0*pi
  v_loc = sqrt(grr/abs(gtt))*ur/ut



def get_analytic_bondi_gr(r_array,gamma_adi = 5.0/3.0,r_crit = 8.0):
  rb = r_array[:,0,0]
  
  def SQR(a): return a**2.0

  temp_min = 1.0e-2  ## lesser temperature root must be greater than this
  temp_max = 1.0e1 
  k_adi = 1.0
  
  n_adi = 1.0/(gamma_adi-1.0)
  u_crit_sq = 1.0/(2.0*r_crit);                                          ## (HSW 71)
  u_crit = -np.sqrt(u_crit_sq);
  t_crit = n_adi/(n_adi+1.0) * u_crit_sq/(1.0-(n_adi+3.0)*u_crit_sq);  ## (HSW 74)
  c1 = t_crit**n_adi * u_crit * SQR(r_crit);                      ## (HSW 68)
  c2 = SQR(1.0 + (n_adi+1.0) * t_crit) * (1.0 - 3.0/(2.0*r_crit));        ## (HSW 69)
  def TemperatureResidual(t,r):
    return SQR(1.0 + (n_adi+1.0) * t)* (1.0 - 2.0/r + SQR(c1) / (SQR(SQR(r)) * t**(2.0*n_adi))) - c2

  def TemperatureMin(r,t_min,t_max):
    ## Parameters
    ratio = 0.3819660112501051;  ## (3+\sqrt{5})/2
    max_iterations = 100;          ## maximum number of iterations

    ## Initialize values
    t_mid = t_min + ratio * (t_max - t_min);
    res_mid = TemperatureResidual(t_mid, r);

    ## Apply golden section method
    larger_to_right = True  ## flag indicating larger subinterval is on right
    for n in arange(0,max_iterations):
      if (res_mid < 0.0): return t_mid
      if (larger_to_right):
        t_new = t_mid + ratio * (t_max - t_mid)
        res_new = TemperatureResidual(t_new, r)
        if (res_new < res_mid):
          t_min = t_mid;
          t_mid = t_new;
          res_mid = res_new;
        else:
          t_max = t_new;
          larger_to_right = False;
      else:
        t_new = t_mid - ratio * (t_mid - t_min);
        res_new = TemperatureResidual(t_new, r);
        if (res_new < res_mid):
          t_max = t_mid;
          t_mid = t_new;
          res_mid = res_new;
        else:
          t_min = t_new;
          larger_to_right = True;
    return nan

  def TemperatureBisect(r, t_min, t_max):
      # Parameters
      max_iterations = 20
      tol_residual = 1.0e-6
      tol_temperature = 1.0e-6
      
      # Find initial residuals
      res_min = TemperatureResidual(t_min, r)
      res_max = TemperatureResidual(t_max, r)
      if np.fabs(res_min) < tol_residual:
          return t_min
      if np.fabs(res_max) < tol_residual:
          return t_max
      if (res_min < 0.0 and res_max < 0.0) or (res_min > 0.0 and res_max > 0.0):
          return float('nan')
      
      # Iterate to find root
      t_mid = None
      for i in arange(max_iterations):
          t_mid = (t_min + t_max) / 2.0
          if t_max - t_min < tol_temperature:
              return t_mid
          res_mid = TemperatureResidual(t_mid, r)
          if np.fabs(res_mid) < tol_residual:
              return t_mid
          if (res_mid < 0.0 and res_min < 0.0) or (res_mid > 0.0 and res_min > 0.0):
              t_min = t_mid
              res_min = res_mid
          else:
              t_max = t_mid
              res_max = res_mid
      return t_mid

  global prho,ppgas,put,pur

  ## Calculate solution to (HSW 76)
  prho = rb*0 
  ppgas = rb*0
  put = rb*0
  pur = rb*0 

  for ir in np.arange(len(rb)):
    rb_ = rb[ir]
    temp_neg_res = TemperatureMin(rb_, temp_min, temp_max);
    if (rb_ <= r_crit):  ## use lesser of two roots
      temp = TemperatureBisect(rb_, temp_min, temp_neg_res)
    else:  ## user greater of two roots
      temp = TemperatureBisect(rb_, temp_neg_res, temp_max)
    

    # Calculate primitives
    rho = (temp/k_adi)**n_adi             # not same K as HSW
    pgas = temp * rho;
    ur = c1 / (SQR(rb_) * temp**n_adi);    ## (HSW 75)
    ut = np.sqrt(1.0/SQR(1.0-2.0/rb_) * SQR(ur) + 1.0/(1.0-2.0/rb_));

    ## Set primitives
    prho[ir] = rho;
    ppgas[ir] = pgas;
    put[ir] = ut;
    pur[ir] = ur;

  prho = prho[:,None,None]*(r_array/r_array)
  ppgas = prho[:,None,None]*(r_array/r_array)
  put = put[:,None,None]*(r_array/r_array)
  pur = pur[:,None,None]*(r_array/r_array)

def pcolormesh_corner(r,th,myvar,coords = 'xz',flip_x = False,**kwargs):
  r_face = np.logspace(log10(np.amin(r)),log10(np.amax(r)),nx+1)
  th_face = np.linspace(0,pi,ny+1) 
  ph_face = np.linspace(0,2.0*pi,nz+1)
  if (coords =='xz'):
    x_corner = r_face[:,None] * np.sin(th_face[None,:])
    y_corner = r_face[:,None] * np.cos(th_face[None,:])
  else:
    x_corner = r_face[:,None] * np.cos(ph_face[None,:])
    y_corner = r_face[:,None] * np.sin(ph_face[None,:])
  if (flip_x==True): x_corner = x_corner * -1.0

  return plt.pcolormesh(x_corner,y_corner,myvar,**kwargs)


# def contourf_corner(r,th,myvar,coords = 'xz',flip_x = False,**kwargs):
#   r_face = np.logspace(log10(np.amin(r)),log10(np.amax(r)),nx+1)
#   th_face = np.linspace(0,pi,ny+1) 
#   ph_face = np.linspace(0,2.0*pi,nz+1)
#   if (coords =='xz'):
#     x_corner = r_face[:,None] * np.sin(th_face[None,:])
#     y_corner = r_face[:,None] * np.cos(th_face[None,:])
#   else:
#     x_corner = r_face[:,None] * np.cos(ph_face[None,:])
#     y_corner = r_face[:,None] * np.sin(ph_face[None,:])
#   if (flip_x==True): x_corner = x_corner * -1.0

#   return plt.pcolormesh(x_corner,y_corner,myvar,**kwargs)


def gravity_term_bondi(rb,a,m=1):
  global aterm,pressterm, advection_term,massterm,centrifugal_term
  get_gr_bondi_sol(rb)
  global r,th,ph  
  th = np.linspace(0,pi,200)
  ph = np.linspace(0,2*pi,200)
  r,th,ph = np.meshgrid(r,th,ph,indexing='ij')
  global nx,ny,nz
  nx = r.shape[0]
  ny = r.shape[1]
  nz = r.shape[2]
  ks_metric(r,th,a)
  ks_Gamma_ud(r,th,a)

  global rho,T,ut,ur, press

  press = (rho*T)[:,None,None]*(r/r)
  rho = rho[:,None,None]*(r/r)

  global uu_bl,uu_ks,ud_ks,bd_ks
  uu_bl = np.array([ut[:,None,None]*(r/r),-ur[:,None,None]*(r/r),0*r,0*r])
  uu_ks = bl_vec_to_ks(uu_bl,a)

  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(uu_ks,g))
  global Tud,Tudpress,Tudkinetic

  Tud_calc(uu_ks,ud_ks,uu_ks,bd_ks,is_magnetic = False,gam=5.0/3.0)
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )

  TudKinetic = 0 * Tud
  gam = 5./3.
  Tudpress = 0*Tud
  for kapa in np.arange(4):
    for nu in np.arange(4):
      TudKinetic[kapa,nu] = (rho+press*(gam/(gam-1.0)))*uu_ks[kapa]*ud_ks[nu]
      Tudpress[kapa,nu] = press*(kapa==nu)
  aterm = 0
  pressterm = 0
  advection_term= 0
  centrifugal_term = 0
  massterm = 0
  for i in arange(4):
    for j in arange(4):
      if(i==0 and j==0): aterm+= (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      else: advection_term += (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      pressterm += (gdet * Tudpress[j][i] * gammaud[i][1][j][:,:,None])
      if ((i==3 and j==3) or (i==2 and j==2)): centrifugal_term += (gdet * TudKinetic[i][j]) * gammaud[i][1][j][:,:,None]
      massterm += (gdet * Tud[j][i]) * gammaud[i][1][j][:,:,None]
  #centrifugal_term += (gdet * TudKinetic[3][3]) * gammaud[3][1][3][:,:,None]


  pressterm -= np.gradient(press*gdet,r[:,0,0],axis=0)
  advection_term -= np.gradient(gdet * TudKinetic[1][1],r[:,0,0],axis=0) + np.gradient(gdet * TudKinetic[2][1],th[0,:,0],axis=1)
  advection_term -= np.gradient(gdet * TudKinetic[3][1],ph[0,0,:],axis=2)
  massterm -= np.gradient(gdet * Tud[1][1],r[:,0,0],axis=0) + np.gradient(gdet * Tud[2][1],th[0,:,0],axis=1)
  massterm -= np.gradient(gdet * Tud[3][1],ph[0,0,:],axis=2)


def make_Veronica_npz():

  f1 = "/global/scratch/smressle/star_cluster/cuadra_comp/without_S2_9_levels_new_orbits_Z_3_no_H_solar_intel/star_wind.hst"
  t_i = -0.1
  t_f = 0.00

  k_green = '#4cbb17'
  silver = '#8D9093'
  midnight_green = '#004851'
  charcoal = '#54585A'
  dark_gray = '#A9A9A9'
  light_blue = "#ADD8E6"

  arc_secs = 0.03878512
  r_in = 2.*2./128./2.**8.
  r_in = 2.*2./128./2.**9.
  #r_in = 2.*2./128./2.**10.

  simulation_start_time =  - 1.1

  t_i = t_i - simulation_start_time
  t_f = t_f - simulation_start_time

  rd_hst(f1)
  rho_unit = 6.768e-23 #g/cm^3  
  v_unit = 9.785e7
  L_unit = 3.086e18
  t_unit = 3.154e10
  iti1,itf1 = t_to_it(t_i),t_to_it(t_f)
  def tavg(var):
    return var[iti1:itf1,:].mean(0)
  rho_1d = rho_avg[iti1:itf1,:-3].mean(0) * rho_unit
  cs_1d = np.sqrt(5./3.*p_avg[iti1:itf1,:-3].mean(0)/rho_avg[iti1:itf1,:-3].mean(0)) * v_unit
  vr_abs_1d = abs((mdot_avg[iti1:itf1,:-3].mean(0)/rho_avg[iti1:itf1,:-3].mean(0)/(r**2.)[iti1:itf1,:-3].mean(0)))/np.pi/4. * v_unit
  mdot_1d = abs((mdot_avg[iti1:itf1,:-3].mean(0))) * rho_unit*L_unit**3.0 /t_unit
  press_1d = p_avg[iti1:itf1,:-3].mean(0) * rho_unit * v_unit**2.0
  L_1d = np.sqrt( tavg(Lx_avg)**2. + tavg(Ly_avg)**2. + tavg(Lz_avg)**2.)[:-3] * rho_unit * v_unit * L_unit
  Ldot_1d = np.sqrt( tavg(Lxdot_avg)**2. + tavg(Lydot_avg)**2. + tavg(Lzdot_avg)**2.)[:-3] * rho_unit * v_unit * L_unit/t_unit * L_unit**3.0
  Edot_1d = Edot_avg[iti1:itf1,:-3].mean(0) * rho_unit * v_unit**2/t_unit * L_unit**3.0


  dic = {"r":r[iti1:itf1,:-3].mean(0)*L_unit,
          "rho":rho_1d,
          "cs":cs_1d,
          "vr_abs":vr_abs_1d,
          "mdot":-mdot_1d,
          "press":press_1d,
          "L":L_1d,
          "Jdot":Ldot_1d,
          "Edot":Edot_1d }

  np.savez("hydro_wind_1d.npz",**dic)



def two_fes():
  global fe_tot, fe_ion
  set_constants()
  mrat = 1836.152672
  beta = press/bsq*2.0
  ue = kappa_to_ue(ke_ent,rho,gr=True)
  Te = theta_e/mrat
  Ttot = press/rho * mu_highT
  fe_tot = fe_howes(beta,Te,Ttot)
  mue = 2.0
  mui = 1.0/(1.0/mu_highT - 1.0/mue)

  pe = theta_e/mrat *rho /mue;
  pi = press - pe
  pi[pi<0] = press[pi<0] * 0.01
  Ti = pi/rho * mui
  betai = 2.0 * pi/(bsq + 1e-15)
  fe_ion = fe_howes(betai,Te,Ti)

def fe_howes(beta,Te, Ttot):
  mrat = 1836.152672;
  Te[Te<1e-15] = 1e-15
  Trat = abs(Ttot/Te)
  c1 = .92 
  
  beta[isnan(beta)] = 1e20
  beta[isinf(beta)] = 1e20 
  beta[beta>1e20] = 1e20
  mbeta = 2.-.2*log10(Trat)

  c2 = 1.6 / Trat * (Trat<=1) + 1.2/ Trat *(Trat>1)
  c3 = (18. + 5.*log10(Trat)) * (Trat<=1) + (18.0) * (Trat>1)
  
  c22 = c2**2.0
  c32 = c3**2.0
  
  Qp_over_Qe = c1 * (c22+beta**mbeta)/(c32 + beta**mbeta) * exp(-1./beta)*(mrat*Trat)**0.5 ;
  

  
  return 1./(1.+Qp_over_Qe);


def fe_rowan(beta,sigma_w,Ttot,Te):

  #sigma_w = bsq/(rho + gam/gm1 * press);
  beta_max = 1.0/(4.0*sigma_w)
  beta_max[beta_max>1.e20] =1e20
  beta_max[isnan(beta_max)] =1e20
  beta[isnan(beta)] = 1e20
  beta[isinf(beta)] = 1e20 
  beta[beta>1e20] = 1e20

  beta[beta>beta_max] = beta_max[beta>beta_max]

  arg_num = (1.0-beta/beta_max)**3.3
  arg_den = (1.0 + 1.2 * (sigma_w)**0.7)
  arg = arg_num/arg_den;
  return 0.5 * np.exp(-arg)


def ud_zamo(a):
  ks_inverse_metric(r,th,a)
  return np.array([-1.0/sqrt(-gi[0][0]),0.0*r,0.0*r,0.0*r])

def wald_solution():
  r2 = r**2.
  a2 = a**2.
  s2 = np.sin(th)**2.
  c2 = np.cos(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2
  Delta = r**2 -2.0*r + a**2
  Ex = -a * r *s2/sigma - a*(r2-a2*c2)*(1+c2)/sigma**2 
  Ey = -sqrt(Delta)*a*sin(th)*cos(th)/sigma
  Ez = -0*r
  Bx = -cos(th)/sigma * (r2+a2-2*r*a2*(1+c2)/sigma) 
  By = sqrt(Delta)*r*sin(th)/sigma
  Bz = 0*r

  bsq = Bx**2+By**2+Bz**2 - Ex**2 - Ey**2 -Ez**2


def energy_spectrum():
  gr = False
  yt_extract_box(125,box_radius = 1,gr=gr)
  ##yt_extract_box(1900,box_radius = 40,gr=gr)
  Lx = np.amax(x)-np.amin(x)
  Ly = np.amax(y)-np.amin(y)
  Lz = np.amax(z)-np.amin(z)
  dx = np.diff(x[:,0,0])[0]
  dy = dx*1.0
  dz = dx*1.0
  if (gr==False):
    vel1_tilde = np.fft.fftn(sqrt(rho)*vel1) * dx**3.0
    vel2_tilde = np.fft.fftn(sqrt(rho)*vel2) * dx**3.0
    vel3_tilde = np.fft.fftn(sqrt(rho)*vel3) * dx**3.0
  else:
    def SQR(var):
      return var**2.0
    R = np.sqrt(SQR(x) + SQR(y) + SQR(z));
    r = SQR(R) - SQR(a) + np.sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
    r = np.sqrt(r/2.0);
    th = np.arccos(z/r)
    vel1_tilde = np.fft.fftn(sqrt(rho)*uu[1]/uu[0]) * dx**3.0
    vel2_tilde = np.fft.fftn(sqrt(rho)*uu[2]/uu[0] * r) * dx**3.0
    vel3_tilde = np.fft.fftn(sqrt(rho)*uu[3]/uu[0] * r * sin(th)) * dx**3.0
  # vel1_tilde = np.fft.fftn(vel1)
  # vel2_tilde = np.fft.fftn(vel2)
  # vel3_tilde = np.fft.fftn(vel3)
  E_tilde_vec = 0.5*(vel1_tilde*np.conj(vel1_tilde) + vel2_tilde*np.conj(vel2_tilde)+ vel3_tilde*np.conj(vel3_tilde))

  k1 = np.fft.fftfreq(nx,d=dx) #np.linspace(0,kmax,nx)#/Lx
  k2 = np.fft.fftfreq(ny,d=dy) #np.linspace(0,kmax,ny)#/Ly
  k3 = np.fft.fftfreq(nz,d=dz) #np.linspace(0,kmax,nz)#/Lz
  dk = np.diff(k1)[0]
  k1,k2,k3 = np.meshgrid(k1,k2,k3,indexing='ij')
  ksq = k1**2 + k2**2 + k3**2
  kr = np.sqrt(ksq)
  kph = np.arctan2(k2,k1)
  th_k = np.arccos(k3/sqrt(ksq))

  kmax = np.amax(sqrt(ksq))
  kr_new = np.linspace(k1[1,0,0],kmax,nx*4+1)[:-1] + np.diff(np.linspace(k1[1,0,0],kmax,nx*4+1))[0]
  kth_new = np.linspace(0,pi,ny+1)[:-1] + np.diff(np.linspace(0,pi,ny+1))[0]
  if(nz==1): kth_new = np.array([np.pi/2])
  if (nz>1): kph_new = np.linspace(0,pi*2.0,nz+1)[:-1] + np.diff(np.linspace(0,pi*2.0,nz+1))[0]
  else: kph_new = np.linspace(0,pi*2.0,ny+1)[:-1] + np.diff(np.linspace(0,pi*2.0,ny+1))[0]


  kr_new,kth_new,kph_new = np.meshgrid(kr_new,kth_new,kph_new,indexing='ij')

  k1_new = kr_new * np.sin(kth_new) * np.cos(kph_new)
  k2_new = kr_new * np.sin(kth_new) * np.sin(kph_new)
  k3_new = kr_new * np.cos(kth_new)

  E_tilde_vec_spher = scipy.interpolate.griddata((k1.flatten(),k2.flatten(),k3.flatten()),E_tilde_vec.flatten(),
    (k1_new,k2_new,k3_new), method='nearest',fill_value = 0.0)


  dr_k = np.diff(kr_new[:,0,0])[0]
  if (nz>1): dth_k = np.diff(kth_new[0,:,0])[0]
  else: dth_k = 1.0
  dph_k = np.diff(kph_new[0,0,:])[0]

  E_tilde = (E_tilde_vec_spher * kr_new**2 * sin(kth_new) * dth_k * dph_k).sum(-1).sum(-1) 


def Okazaki(alpha=0.2,a=0.1):
  #R_crit = 
  #vphi_crit = np.sqrt(1.0/R_crit - 5.0/2.0*a**2) 
  vr_crit = a
  vr_0_guess = 8.7e-3*a #a/(R_crit - (1-vphi_crit)/(alpha*a))
  vphi_0 = 1.0




  R_crit_max = 2.0/5.0 * 1/a**2.0 
  R_crit_min = 15.0


  def get_sol(R_crit,full_range=False):
    global sol
    vphi_crit = np.sqrt(1.0/R_crit - 5.0/2.0*a**2) 
    l_crit = vphi_crit * R_crit 

    v_r0 = - ( (1.0 - l_crit)/(alpha * a**2) - R_crit/a )**-1
    if (full_range==False): r_vec = np.logspace(0,log10(R_crit),200)
    else: r_vec = np.logspace(0,log10(500),200)
    def dvr_dr(v_r,v_phi,r):
      return (-1.0/r**2 + v_phi**2/r + 5.0/2.0*a**2/r)/(v_r - a**2.0/v_r)
    def func(r,v):
      v_r = v
      #v_phi = v[1]
      #v_phi = vphi_crit*R_crit/r + alpha * a**2/r  * (R_crit/a - r/v_r)
      v_phi = ( 1.0/r + alpha * a**2/r * (1.0/v_r0 - r/v_r) ) * (r<=R_crit) + ( 1.0/R_crit + alpha * a**2/R_crit * (1.0/v_r0 - R_crit/a) )*R_crit/r * (r>R_crit)

      #dvr_dr = (-1.0/r**2 + v_phi**2/r + 5.0/2.0*a**2/r)/(v_r - a**2.0/v_r)
      dvrdr = -1/2*(5*a**2*r + 2*r*v_phi**2 - 2)*v_r/(a**2*r**2 - r**2*v_r**2)
      #dvphi_dr = -1/2*(7*a**4*alpha*r + 2*a**2*alpha*r*v_phi**2 - 2*a**2*alpha*r*v_r**2 + 2*a**2*r*v_phi*v_r - 2*r*v_phi*v_r**3 - 2*a**2*alpha)/(a**2*r**2*v_r - r**2*v_r**3)
      return dvrdr #dvr_dr(v_r,v_phi,r)
    y_0 = [v_r0]

    from scipy.integrate import solve_ivp

    def dfunc(r,v):
      v_r = v
      return 2*(a**2*alpha*(r/v_r - 1/v_r0)/r - 1/r)*a**2*alpha*r/((a**2*r**2 - r**2*v_r**2)*v_r) - (2*(a**2*alpha*(r/v_r - 1/v_r0)/r - 1/r)**2*r + 5*a**2*r - 2)*r**2*v_r**2/(a**2*r**2 - r**2*v_r**2)**2 - 1/2*(2*(a**2*alpha*(r/v_r - 1/v_r0)/r - 1/r)**2*r + 5*a**2*r - 2)/(a**2*r**2 - r**2*v_r**2)


    sol = solve_ivp(func,(r_vec[0],r_vec[-1]),y_0,t_eval=r_vec) #Dfun=dfunc# ,Dfun=dfunc)
    #vphi_sol = vphi_crit*R_crit/r_vec + alpha * a**2/r_vec  * (R_crit/a - r_vec/sol[0])
    vr_sol = sol['y'][0,:]
    vphi_sol = (1.0/sol['t'] + alpha * a**2/sol['t'] * (1.0/v_r0 - sol['t']/sol['y'][0,:]) ) * (sol['t']<=R_crit) + ( 1.0/R_crit + alpha * a**2/R_crit * (1.0/v_r0 - R_crit/a) )*R_crit/sol['t'] * (sol['t']>R_crit)
    return [sol['t'],vr_sol,vphi_sol]

  def objective(R_crit):
    r,vr_sol,vphi_sol = get_sol(R_crit)
    return vr_sol[-1]/a - 1

  import scipy

  root = []
  r_crit_arr = np.logspace(log10(R_crit_min),log10(R_crit_max*0.99),100)
  for r_crit in r_crit_arr:
    root.append(objective(r_crit))
  root = np.array(root)
  for i in arange(len(root)):
    if (root[i] >0 ): break
  R_crit = scipy.optimize.bisect(objective, R_crit_min, r_crit_arr[i])

  r,vr,vphi = get_sol(R_crit,full_range=True)
  Sigma = 1.0/(vr * r)
  Sigma = Sigma/Sigma[0]
  vk = sqrt(1.0/r)
  H = r * (a/vk)
  rho = Sigma/H
  rho = rho/rho[0]

  return [r,vr,vphi,rho]


def render_jet(i_dump,box_radius=100,a=0.0,res=128,th_=0,ph_=0):


  yt_extract_box(i_dump,box_radius = box_radius,mhd=True,gr=True,a=a,res=res)


  def transferFunction(x):
    """Transfer Function returns r,g,b,a values as a function of density x"""
    peak_1 = 1
    peak_2 = 0
    peak_3 = -2
    r = 1.0*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.5 )
    g = 1.0*np.exp( -(x - peak_1)**2/1.0 ) +  1.0*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.5 )
    b = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  1.0*np.exp( -(x - peak_3)**2/0.5 )
    a = 0.6*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) + 0.01*np.exp( -(x - peak_3)**2/0.5 )

    return r,g,b,a

  camera_grid = bsq/rho #np.transpose(bsq/rho,axes=(0,2,1))

  # Do Volume Rendering
  image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

  for dataslice in camera_grid:
    r,g,b,a_ = transferFunction(np.log10(dataslice))
    image[:,:,0] = a_*r + (1-a_)*image[:,:,0]
    image[:,:,1] = a_*g + (1-a_)*image[:,:,1]
    image[:,:,2] = a_*b + (1-a_)*image[:,:,2]

  image = np.clip(image,0.0,1.0)
  plt.imshow(np.transpose(image,axes=(1,0,2)))


def get_stellar_boundary(r_=1,rho_0 = 1.0, P=4.11,phi_0=0.0,rho_pert = 1.1,gm_=1.0,r_star=1.0,cs_ratio=0.1,
                         Omega_over_Omega_crit=0.95,delta_v_frac=2.0,m_phi=-2,mhd=False,va_ratio =0.01 ):

  th_ = th[0,:,:]*1.0
  ph_ = ph[0,:,:]*1.0
  s = r_star * np.sin(th_);

  vkep_star = np.sqrt(gm_/r_star);
  cs = cs_ratio * vkep_star


  rho_star = rho_0;
  press_star = cs**2.0*rho
  vphi_star = Omega_over_Omega_crit * s * np.sqrt(gm_/r_star**3.0)
  rho_max = rho_0 * rho_pert
  delta_v = np.sqrt(gm_/r_star) * (1.0 - Omega_over_Omega_crit);
  vphi_pert_amp = delta_v_frac*delta_v;

  Plm = np.sin(th_)**(abs(m_phi*1.0)) 

  rho_pert_arg = np.log10(rho_max/rho_0) * np.sin( 2.0 * np.pi * t/P + m_phi*ph_) * Plm
  phi_pert = vphi_pert_amp * np.sin( 2.0 * np.pi * t/P + m_phi*ph_) * Plm

  rho_sol = rho_0 * 10.0**rho_pert_arg
  vphi_sol = vphi_star + phi_pert

  if (mhd==False): return rho_sol,vphi_sol

  B_constant = va_ratio * r_star**3.0/(2.0*np.pi)*np.sqrt(rho_0);
  Br_sol = 2.0 * B_constant * np.cos(th_) * np.sin(2.0*np.pi * r_/r_star)/r_**3.0
  Bth_sol = - (B_constant *(2.0 *np.pi* r_ *np.cos((2.0*np.pi*r_)/r_star) - r_star*np.sin((2.0*np.pi*r_)/r_star))*np.sin(th_))/(r_star*r_**3.0)

  return rho_sol,vphi_sol,Br_sol,Bth_sol
def get_boundary_flux(dl,dr,vel1l,vel1r,vel2l,vel2r,vel3l,vel3r,cs=0.1,dir='x1'):
  global al,vx_roe,a,cl
  gam = 5.0/3.0
  gm1 = gam-1.0


  Pr = dr * cs**2.0
  Pl = dl * cs**2.0


  vx_roe = (np.sqrt(dl)*vel1l + np.sqrt(dr)*vel1r)/(np.sqrt(dl) + np.sqrt(dr))
  vy_roe = (np.sqrt(dl)*vel2l + np.sqrt(dr)*vel2r)/(np.sqrt(dl) + np.sqrt(dr))
  vz_roe = (np.sqrt(dl)*vel3l + np.sqrt(dr)*vel3r)/(np.sqrt(dl) + np.sqrt(dr))

  el = Pl/(gm1) + 0.5*dl*(vel1l**2+vel2l**2 + vel3l**2)
  er = Pr/(gm1) + 0.5*dr*(vel1r**2+vel2r**2 + vel3r**2)

  hroe = ((el + Pl)/np.sqrt(dl) + (er + Pr)/np.sqrt(dr))/(np.sqrt(dl) + np.sqrt(dr))

  q = hroe - 0.5*(vx_roe**2.0 + vy_roe**2.0 + vz_roe**2.0)
  a =  np.sqrt(gm1*q) 
  a[q<0] = 0.0

  cl = np.sqrt(gam)*cs
  cr = np.sqrt(gam)*cs

  if (dir=='x1'):
    al = vx_roe -a   #std::min((vx_roe - a),(vel1l - cl));
    al[(vel1l - cl)<(vx_roe -a)] = (vel1l - cl)[(vel1l - cl)<(vx_roe -a)]
    ar = vx_roe +a  #std::max((vx_roe + a),(vel1r + cr))
    ar[(vel1r+ cr)>(vx_roe +a)] = (vel1r + cr)[(vel1r+ cr)>(vx_roe +a)]
  elif (dir=='x2'):
    al = vy_roe -a   #std::min((vx_roe - a),(vel1l - cl));
    al[(vel2l - cl)<(vy_roe -a)] = (vel2l - cl)[(vel2l - cl)<(vy_roe -a)]
    ar = vx_roe +a  #std::max((vx_roe + a),(vel1r + cr))
    ar[(vel2r+ cr)>(vy_roe +a)] = (vel2r + cr)[(vel2r+ cr)>(vy_roe +a)]
  else: 
    print(dir, "is not a valid direction")
    return None

  bp = ar * 1.0
  bp[bp<0]= 0.0
  bm = al * 1.0
  bm[bm>0] = 0.0

  # bp = ar > 0.0 ? ar : 0.0;
  # bm = al < 0.0 ? al : 0.0;

  tmp = nan_to_num(0.5*(bp + bm)/(bp - bm))

  if (dir=='x1'):
    vxl = vel1l - bm
    vxr = vel1r - bp
    Fr = dr*vxr
    Fl = dl*vxl
  elif(dir=='x2'):
    vyl = vel2l - bm
    vyr = vel2r - bp
    Fr = dr*vyr
    Fl = dl*vyl   

  return  0.5*(Fr + Fl) + (Fl-Fr)*tmp;


def get_dV():
  r_term = np.zeros(len(x1f)-1)
  th_term = np.zeros(len(x2f)-1)
  for i in arange(len(x1f)-1):
    r_term[i] = 1.0/3.0 * (x1f[i+1]**3.0 - x1f[i]**3.0)
  for j in arange(len(x2f)-1):
    th_term[j] = np.abs(np.cos(x2f[j])-np.cos(x2f[j+1]))
  phi_term = np.diff(x3f)

  return (r_term[:,None,None]*(r/r)) * (th_term[None,:,None]*r/r) * (phi_term[None,None,:]*r/r)


def FastMagnetosonicSpeed(den,pr,B_par,B_perp_sq,gam=5.0/3.0):
    asq = gam*pr;
    vaxsq = B_par*B_par;
    ct2 = B_perp_sq
    qsq = vaxsq + ct2 + asq;
    tmp = vaxsq + ct2 - asq;
    return np.sqrt(0.5*(qsq + np.sqrt(tmp*tmp + 4.0*asq*ct2))/den);
def calc_ppm_quantities(x1f,x1v,x2f,x2v):
  global c1i, c2i,c3i,c4i,c5i,c6i
  global c1j,c2j,c3j,c4j,c5j,c6j
  global hplus_ratio_i,hminus_ratio_i
  global hplus_ratio_j,hminus_ratio_j

  dx1f = diff(x1f)
  dx2f = diff(x2f)
  c1i = dx1f*0
  c2i = dx1f*0
  c3i = dx1f*0
  c4i = dx1f*0
  c5i = dx1f*0
  c6i = dx1f*0
  hplus_ratio_i = dx1f*0
  hminus_ratio_i = dx1f*0

  c1j = dx2f*0
  c2j = dx2f*0
  c3j = dx2f*0
  c4j = dx2f*0
  c5j = dx2f*0
  c6j = dx2f*0
  hplus_ratio_j = dx2f*0
  hminus_ratio_j = dx2f*0
  def SQR(a):
    return a**2.0
  for i in arange(1,len(dx1f)-1):
    dx_im1 = dx1f[i-1];
    dx_i   = dx1f[i  ];
    dx_ip1 = dx1f[i+1];
    qe = dx_i/(dx_im1 + dx_i + dx_ip1);       ## Outermost coeff in CW eq 1.7
    c1i[i] = qe*(2.0*dx_im1+dx_i)/(dx_ip1 + dx_i); ## First term in CW eq 1.7
    c2i[i] = qe*(2.0*dx_ip1+dx_i)/(dx_im1 + dx_i); ## Second term in CW eq 1.7

    if (i > 1):  ## c3-c6 are not computed in first iteration
      dx_im2 = dx1f[i-2];
      qa = dx_im2 + dx_im1 + dx_i + dx_ip1;
      qb = dx_im1/(dx_im1 + dx_i);
      qc = (dx_im2 + dx_im1)/(2.0*dx_im1 + dx_i);
      qd = (dx_ip1 + dx_i)/(2.0*dx_i + dx_im1);
      qb = qb + 2.0*dx_i*qb/qa*(qc-qd);
      c3i[i] = 1.0 - qb;
      c4i[i] = qb;
      c5i[i] = dx_i/qa*qd;
      c6i[i] = -dx_im1/qa*qc;

    dx_i   = dx1f[i];
    xv_i   = x1v[i];
    #spherical radial coordinate

    h_plus = 3.0 + (2.0*dx_i*(10.0*xv_i + dx_i))/(20.0*SQR(xv_i) + SQR(dx_i));
    h_minus = 3.0 + (2.0*dx_i*(-10.0*xv_i + dx_i))/(20.0*SQR(xv_i) + SQR(dx_i));
    hplus_ratio_i[i] = (h_plus + 1.0)/(h_minus - 1.0);
    hminus_ratio_i[i] = (h_minus + 1.0)/(h_plus - 1.0);

  for j in arange(1,len(dx2f)-1):
      dx_jm1 = dx2f[j-1];
      dx_j   = dx2f[j  ];
      dx_jp1 = dx2f[j+1];
      qe = dx_j/(dx_jm1 + dx_j + dx_jp1);       ## Outermost coeff in CW eq 1.7
      c1j[j] = qe*(2.0*dx_jm1+dx_j)/(dx_jp1 + dx_j); ## First term in CW eq 1.7
      c2j[j] = qe*(2.0*dx_jp1+dx_j)/(dx_jm1 + dx_j); ## Second term in CW eq 1.7

      if (j > 1):  ## c3-c6 are not computed in first iteration
        dx_jm2 = dx2f[j-2];
        qa = dx_jm2 + dx_jm1 + dx_j + dx_jp1;
        qb = dx_jm1/(dx_jm1 + dx_j);
        qc = (dx_jm2 + dx_jm1)/(2.0*dx_jm1 + dx_j);
        qd = (dx_jp1 + dx_j)/(2.0*dx_j + dx_jm1);
        qb = qb + 2.0*dx_j*qb/qa*(qc-qd);
        c3j[j] = 1.0 - qb;
        c4j[j] = qb;
        c5j[j] = dx_j/qa*qd;
        c6j[j] = -dx_jm1/qa*qc;
            ## x2 = theta polar coordinate adjustment
      h_plus, h_minus;
      dx_j   = dx2f[j];
      xf_j   = x2f[j];
      xf_jp1   = x2f[j+1];
      dmu = cos(xf_j) - cos(xf_jp1);
      dmu_tilde = sin(xf_j) - sin(xf_jp1);
      h_plus = (dx_j*(dmu_tilde + dx_j*cos(xf_jp1)))/(
          dx_j*(sin(xf_j) + sin(xf_jp1)) - 2.0*dmu);
      h_minus = -(dx_j*(dmu_tilde + dx_j*cos(xf_j)))/(
          dx_j*(sin(xf_j) + sin(xf_jp1)) - 2.0*dmu);
      hplus_ratio_j[j] = (h_plus + 1.0)/(h_minus - 1.0);
      hminus_ratio_j[j] = (h_minus + 1.0)/(h_plus - 1.0);
def ppm_r(i,q,q_im2,q_im1,q_ip1,q_ip2):
  wc = q
  ##--- Step 1. ----------------------------------------------------------------------------
  # Reconstruct interface averages <a>_{i-1/2} and <a>_{i+1/2}
  ## Compute average slope in i-1, i, i+1 zones
  qa = (q - q_im1);
  qb = (q_ip1 - q);
  dd_im1 = c1i[i-1]*qa + c2i[i-1]*(q_im1 - q_im2);
  dd     = c1i[i  ]*qb + c2i[i  ]*qa;
  dd_ip1 = c1i[i+1]*(q_ip2 - q_ip1) + c2i[i+1]*qb;

  # Approximate interface average at i-1/2 and i+1/2 using PPM (CW eq 1.6)
  # KGF: group the biased stencil quantities to preserve FP symmetry
  dph= (c3i[i]*q_im1 + c4i[i]*q) +(c5i[i]*dd_im1 + c6i[i]*dd);
  dph_ip1= (c3i[i+1]*q + c4i[i+1]*q_ip1) +(c5i[i+1]*dd + c6i[i+1]*dd_ip1 );

  #dph     = min(dph    , max(q,q_im1));
  max_tmp = q*0
  max_tmp[q>=q_im1] = q[q>=q_im1]
  max_tmp[q<q_im1] =q_im1[q<q_im1]

  dph[dph>=max_tmp] = max_tmp[dph>=max_tmp]

  #dph_ip1 = min(dph_ip1, max(q,q_ip1));

  max_tmp = q*0
  max_tmp[q>=q_ip1] = q[q>=q_ip1]
  max_tmp[q<q_ip1] =q_ip1[q<q_ip1]

  dph_ip1[dph_ip1>=max_tmp] = max_tmp[dph_ip1>=max_tmp]

  #dph     = max(dph    , min(q,q_im1));

  min_tmp = q*0
  min_tmp[q<=q_im1] = q[q<=q_im1]
  min_tmp[q>q_im1] =q_im1[q>q_im1]

  dph[dph>=min_tmp] = dph[dph>=min_tmp]
  dph[dph<min_tmp] = min_tmp[dph<min_tmp]

  #dph_ip1 = max(dph_ip1, min(q,q_ip1));

  min_tmp = q*0
  min_tmp[q<=q_ip1] = q[q<=q_ip1]
  min_tmp[q>q_ip1] =q_ip1[q>q_ip1]

  dph_ip1[dph_ip1>=min_tmp] = dph_ip1[dph_ip1>=min_tmp]
  dph_ip1[dph_ip1<min_tmp] = min_tmp[dph_ip1<min_tmp]
    

  ## Cache Riemann states for both non-/uniform limiters
  qminus = dph;
  qplus =  dph_ip1

  ##--- Step 3. ----------------------------------------------------------------------------
  ## Compute cell-centered difference stencils (MC section 2.4.1)
  dqf_minus = q - qminus; ## (CS eq 25)
  dqf_plus  = qplus - q;

  ##--- Step 4b. ---------------------------------------------------------------------------
      ## Non-uniform/curvilinear mesh: apply Mignone limiters to parabolic interpolant
      ## Note, the Mignone limiter does not check for cell-averaged extrema:
  qa = dqf_minus*dqf_plus;

  # ## Local extrema detected
  qminus[(qa<=0)] = q[(qa<=0)]
  qplus[(qa<=0)] =q[(qa<=0)]

  index = (qa>0)*((np.abs(dqf_minus) >= hplus_ratio_i[i]*np.abs(dqf_plus)))
  qminus[index] = ( q - hplus_ratio_i[i]*dqf_plus )[index]

  index = (qa>0)*(np.abs(dqf_plus) >= hminus_ratio_i[i]*np.abs(dqf_minus))
  qplus[index] = ( q + hminus_ratio_i[i]*dqf_minus )[index]
  # # No extrema detected
  # if (qa <= 0.0): ## Local extrema detected
  #   qminus = q;
  #   qplus = q;
  # else: # No extrema detected
  #   # Overshoot i-1/2,R / i,(-) state
  #   if ( (np.abs(dqf_minus) >= hplus_ratio_i[i]*np.abs(dqf_plus[i])) ):
  #     qminus = q - hplus_ratio_i[i]*dqf_plus[i];
  #   ## Overshoot i+1/2,L / i,(+) state
  #   if (np.abs(dqf_plus) >= hminus_ratio_i[i]*np.abs(dqf_minus[i])):
  #     qplus = q + hminus_ratio_i[i]*dqf_minus[i];

  

  ##--- Step 5. ----------------------------------------------------------------------------
  ## Convert limited cell-centered values to interface-centered L/R Riemann states
  ## both L/R values defined over [il,iu]
  ql_iph = qplus;
  qr_imh = qminus;

  ## compute ql_(i+1/2) and qr_(i-1/2)
  return [qr_imh, ql_iph]
  # wl(n,k,j,i+1) = ql_iph;
  # wr(n,k,j,i  ) = qr_imh;
        # ## Reapply EOS floors to both L/R reconstructed primitive states
        # pmb->peos->ApplyPrimitiveFloors(wl, k, j, i+1);
        # pmb->peos->ApplyPrimitiveFloors(wr, k, j, i);

def ppm_th(j,q,q_jm2,q_jm1,q_jp1,q_jp2):
  ##--- Step 1. ----------------------------------------------------------------------------
  ## Reconstruct interface averages <a>_{j-1/2} and <a>_{j+1/2}

  ## Compute average slope in j-1, j, j+1 zones
  #pragma omp simd simdlen(SIMD_WIDTH)
  qa = (q - q_jm1);
  qb = (q_jp1 - q);
  dd_jm1 = c1j[j-1]*qa + c2j[j-1]*(q_jm1 - q_jm2);
  dd     = c1j[j  ]*qb + c2j[j  ]*qa;
  dd_jp1 = c1j[j+1]*(q_jp2 - q_jp1) + c2j[j+1]*qb;

  ## Approximate interface average at j-1/2 and j+1/2 using PPM (CW eq 1.6)
  ## KGF: group the biased stencil quantities to preserve FP symmetry
  dph= (c3j[j]*q_jm1 + c4j[j]*q) + (c5j[j]*dd_jm1 + c6j[j]*dd);
  dph_jp1= (c3j[j+1]*q + c4j[j+1]*q_jp1) + (c5j[j+1]*dd + c6j[j+1]*dd_jp1);

  # dph     = std::min(dph    , std::max(q,q_jm1));
  # dph_jp1 = std::min(dph_jp1, std::max(q,q_jp1));

  # dph     = std::max(dph    , std::min(q,q_jm1));
  # dph_jp1 = std::max(dph_jp1, std::min(q,q_jp1));

  max_tmp = q*0
  max_tmp[q>=q_jm1] = q[q>=q_jm1]
  max_tmp[q<q_jm1] =q_jm1[q<q_jm1]

  dph[dph>=max_tmp] = max_tmp[dph>=max_tmp]

  #dph_jp1 = min(dph_jp1, max(q,q_jp1));

  max_tmp = q*0
  max_tmp[q>=q_jp1] = q[q>=q_jp1]
  max_tmp[q<q_jp1] =q_jp1[q<q_jp1]

  dph_jp1[dph_jp1>=max_tmp] = max_tmp[dph_jp1>=max_tmp]

  #dph     = max(dph    , min(q,q_im1));

  min_tmp = q*0
  min_tmp[q<=q_jm1] = q[q<=q_jm1]
  min_tmp[q>q_jm1] =q_jm1[q>q_jm1]

  dph[dph>=min_tmp] = dph[dph>=min_tmp]
  dph[dph<min_tmp] = min_tmp[dph<min_tmp]

  #dph_jp1 = max(dph_jp1, min(q,q_jp1));

  min_tmp = q*0
  min_tmp[q<=q_jp1] = q[q<=q_jp1]
  min_tmp[q>q_jp1] =q_jp1[q>q_jp1]

  dph_jp1[dph_jp1>=min_tmp] = dph_jp1[dph_jp1>=min_tmp]
  dph_jp1[dph_jp1<min_tmp] = min_tmp[dph_jp1<min_tmp]



  ## Cache Riemann states for both non-/uniform limiters
  qminus = dph;
  qplus =  dph_jp1;

  ##--- Step 3. ----------------------------------------------------------------------------
  ## Compute cell-centered difference stencils (MC section 2.4.1)
  dqf_minus = q - qminus; ## (CS eq 25)
  dqf_plus  = qplus - q;

  qa = dqf_minus*dqf_plus;

    # ## Local extrema detected
  qminus[(qa<=0)] = q[(qa<=0)]
  qplus[(qa<=0)] =q[(qa<=0)]

  index = (qa>0)*((np.abs(dqf_minus) >= hplus_ratio_j[j]*np.abs(dqf_plus)))
  qminus[index] = ( q - hplus_ratio_j[j]*dqf_plus )[index]

  index = (qa>0)*(np.abs(dqf_plus) >= hminus_ratio_j[j]*np.abs(dqf_minus))
  qplus[index] = ( q + hminus_ratio_j[j]*dqf_minus )[index]
  # if (qa <= 0.0) { ## Local extrema detected
  #   qminus = q;
  #   qplus = q;
  # } else { ## No extrema detected
  #   ## Overshoot j-1/2,R / j,(-) state
  #   if (fabs(dqf_minus) >= hplus_ratio_j[j]*fabs(dqf_plus)) {
  #     qminus = q - hplus_ratio_j[j]*dqf_plus;
  #   }
  #   ## Overshoot j+1/2,L / j,(+) state
  #   if (fabs(dqf_plus) >= hminus_ratio_j[j]*fabs(dqf_minus)) {
  #     qplus = q + hminus_ratio_j[j]*dqf_minus;
  #   }


  ##--- Step 5. ----------------------------------------------------------------------------
  ## Convert limited cell-centered values to interface-centered L/R Riemann states
  ## both L/R values defined over [il,iu]
  #pragma omp simd
  ql_jph = qplus;
  qr_jmh = qminus;

  ## compute ql_(j+1/2) and qr_(j-1/2)
  # wl(n,k,j+1,i) = ql_jph;
  # wr(n,k,j  ,i) = qr_jmh;

  return [qr_jmh,ql_jph]
  ## Reapply EOS floors to both L/R reconstructed primitive states
  # pmb->peos->ApplyPrimitiveFloors(wl, k, j+1, i);
  # pmb->peos->ApplyPrimitiveFloors(wr, k, j, i);

def compute_boundary_fluxes_Be(mhd=False):

  rho_star, vphi_star = get_stellar_boundary(r_=1.0,cs_ratio=0.1)
  NGHOST =4
  dlogx = np.diff(log(x1f))[0]
  x0 = x1f[0]
  x1f_tmp = np.array([np.exp(log(x0)-4*dlogx),np.exp(log(x0)-3*dlogx),np.exp(log(x0)-2*dlogx),np.exp(log(x0)-dlogx)]+x1f.tolist())
  x0 = x1v[0]
  x1v_tmp = np.array([np.exp(log(x0)-4*dlogx),np.exp(log(x0)-3*dlogx),np.exp(log(x0)-2*dlogx),np.exp(log(x0)-dlogx)]+x1v.tolist())

  def mks_theta_func(x,th_min,th_max,h=0.2,mhd=False):
    return (th_max-th_min) * x + (th_min) + 0.5*(1.0-h)*np.sin(2.0*((th_max-th_min) * x + (th_min)))
    ##return np.pi*x_vals + 0.5*(1.0-h) * np.sin(2.0*np.pi*x_vals)

  thmin = 0.31415926535
  thmax = 2.82743338823
  dx_th = 1.0/(ny*1.0)
  x0 = 0.0
  #x2f[0] = mks_theta_func(0.0,thmin,thmax)
  x2f_tmp = np.array([mks_theta_func(x0-4*dx_th,thmin,thmax),mks_theta_func(x0-3*dx_th,thmin,thmax),mks_theta_func(x0-2*dx_th,thmin,thmax),mks_theta_func(x0-dx_th,thmin,thmax)] + x2f.tolist() )
  x2v_tmp = x2f_tmp*0
  for j in arange(len(x2f_tmp)-1):
    x2v_tmp[j] = ((sin(x2f_tmp[j+1]) - x2f_tmp[j+1]*cos(x2f_tmp[j+1])) - (sin(x2f_tmp[j]) - x2f_tmp[j]*cos(x2f_tmp[j])))/(cos(x2f_tmp[j]) - cos(x2f_tmp[j+1]));

  calc_ppm_quantities(x1f_tmp,x1v_tmp,x2f_tmp,x2v_tmp)


  [rhor,tmp ] = ppm_r(4,rho[0],rho_star,rho_star,rho[1],rho[2])
  [tmp,rhol] =  ppm_r(3,rho_star,rho_star,rho_star,rho[0],rho[1])

  [vel1r,tmp ] = ppm_r(4,vel1[0],0*rho[0],0*rho[0],vel1[1],vel1[2])
  [tmp,vel1l] =  ppm_r(3,0*rho[0],0*rho[0],0*rho[0],vel1[0],vel1[1])

  [vel2r,tmp ] = ppm_r(4,vel2[0],0*rho[0],0*rho[0],vel2[1],vel2[2])
  [tmp,vel2l] =  ppm_r(3,0*rho[0],0*rho[0],0*rho[0],vel2[0],vel2[1])

  [vel3r,tmp ] = ppm_r(4,vel3[0],vphi_star,vphi_star,vel3[1],vel3[2])
  [tmp,vel3l] =  ppm_r(3,vphi_star,vphi_star,vphi_star,vel3[0],vel3[1])

  if (mhd==True):
    Br_star = [th[0,:,:]*0.0,th[0,:,:]*0.0,th[0,:,:]*0.0,th[0,:,:]*0.0]
    Bth_star = [th[0,:,:]*0.0,th[0,:,:]*0.0,th[0,:,:]*0.0,th[0,:,:]*0.0]

    for i in arange(NGHOST):
      rho_star, vphi_star,Br_star[NGHOST-i-1], Bth_star[NGHOST-i-1]  = get_stellar_boundary(r_=x1v[NGHOST-i-1],cs_ratio=0.1,mhd=True)

    [Bcc1r,tmp ] = ppm_r(4,Bcc1[0],Br_star[-2],Br_star[-1],Bcc1[1],Bcc1[2])
    [tmp,Bcc1l] =  ppm_r(3,Br_star[-1],Br_star[-3],Br_star[-2],Bcc1[0],Bcc1[1])


    [Bcc2r,tmp ] = ppm_r(4,Bcc2[0],Bth_star[-2],Bth_star[-1],Bcc2[1],Bcc2[2])
    [tmp,Bcc2l] =  ppm_r(3,Bth_star[-1],Bth_star[-3],Bth_star[-2],Bcc2[0],Bcc2[1])


    [Bcc3r,tmp ] = ppm_r(4,Bcc3[0],0.0*Bcc3[0],0.0*Bcc3[0],Bcc3[1],Bcc3[2])
    [tmp,Bcc3l] =  ppm_r(3,0.0*Bcc3[0],0.0*Bcc3[0],0.0*Bcc3[0],Bcc3[0],Bcc3[1])

    tmp, tmp2,Bcc1_face,tmp3 = get_stellar_boundary(r_=1.0,cs_ratio=0.1,mhd=True)
  # if (mhd==True):
  #   [rhor,tmp ] = ppm_r(4,rho[0],rho_star,rho_star,rho[1],rho[2])
  #   [tmp,rhol] =  ppm_r(3,rho_star,rho_star,rho_star,rho[0],rho[1])

  if (mhd==False): flux_r = get_boundary_flux(rhol,rhor,vel1l,vel1r,vel2l,vel2r,vel3l,vel3r,cs=0.1,dir='x1')
  else: flux_r = hlle_mhd(rhol,rhor,vel1l,vel1r,vel2l,vel2r,vel3l,vel3r,Bcc1_face,Bcc2l,Bcc2r,Bcc3l,Bcc3r,dir='x1')


  ##inner theta
  [rhor,tmp ] = ppm_th(4,rho[:,0,:],rho[:,0,:],rho[:,0,:],rho[:,1,:],rho[:,2,:])
  [tmp,rhol] =  ppm_th(3,rho[:,0,:],rho[:,0,:],rho[:,0,:],rho[:,0,:],rho[:,1,:])

  [vel1r,tmp ] = ppm_th(4,vel1[:,0,:],vel1[:,0,:],vel1[:,0,:],vel1[:,1,:],vel1[:,2,:])
  [tmp,vel1l] =  ppm_r(3,0*rho[:,0,:],vel1[:,0,:],vel1[:,0,:],vel1[:,0,:],vel1[:,1,:])

  vel2_bound = vel2*(vel2<0)
  [vel2r,tmp ] = ppm_th(4,vel2[:,0,:],vel2_bound[:,0,:],vel2_bound[:,0,:],vel2[:,1,:],vel2[:,2,:])
  [tmp,vel2l] =  ppm_th(3,vel2_bound[:,0,:],vel2_bound[:,0,:],vel2_bound[:,0,:],vel2[:,0,:],vel2[:,1,:])

  [vel3r,tmp ] = ppm_th(4,vel3[:,0,:],vel3[:,0,:],vel3[:,0,:],vel3[:,1,:],vel3[:,2,:])
  [tmp,vel3l] =  ppm_th(3,vel3[:,0,:],vel3[:,0,:],vel3[:,0,:],vel3[:,0,:],vel3[:,1,:])


  flux_th_0 = get_boundary_flux(rhol,rhor,vel1l,vel1r,vel2l,vel2r,vel3l,vel3r,cs=0.1,dir='x2')

  ##outer inner theta


  thmin = 0.31415926535
  thmax = 2.82743338823
  dx_th = 1.0/(ny*1.0)
  x0 = 1.0
  x2f_tmp = np.array(x2f.tolist() + [mks_theta_func(x0+dx_th,thmin,thmax),mks_theta_func(x0+2*dx_th,thmin,thmax),mks_theta_func(x0+3*dx_th,thmin,thmax),mks_theta_func(x0+4*dx_th,thmin,thmax)] )
  x2v_tmp = x2f_tmp*0
  for j in arange(len(x2f_tmp)-1):
    x2v_tmp[j] = ((sin(x2f_tmp[j+1]) - x2f_tmp[j+1]*cos(x2f_tmp[j+1])) - (sin(x2f_tmp[j]) - x2f_tmp[j]*cos(x2f_tmp[j])))/(cos(x2f_tmp[j]) - cos(x2f_tmp[j+1]));

  calc_ppm_quantities(x1f_tmp,x1v_tmp,x2f_tmp,x2v_tmp)
  [tmp,rhol] =  ppm_th(-5,rho[:,-1,:],rho[:,-3,:],rho[:,-2,:],rho[:,-1,:],rho[:,-1,:])
  [rhor,tmp ] = ppm_th(-4,rho[:,-1,:],rho[:,-2,:],rho[:,-1,:],rho[:,-1,:],rho[:,-1,:])

  [tmp,vel1l] =  ppm_th(-5,vel1[:,-1,:],vel1[:,-3,:],vel1[:,-2,:],vel1[:,-1,:],vel1[:,-1,:])
  [vel1r,tmp ] = ppm_th(-4,vel1[:,-1,:],vel1[:,-2,:],vel1[:,-1,:],vel1[:,-1,:],vel1[:,-1,:])

  vel2_bound = vel2*(vel2>0)

  [tmp,vel2l] =  ppm_th(-5,vel2[:,-1,:],vel2[:,-3,:],vel2[:,-2,:],vel2_bound[:,-1,:],vel2_bound[:,-1,:])
  [vel2r,tmp ] = ppm_th(-4,vel2_bound[:,-1,:],vel2[:,-2,:],vel2[:,-1,:],vel2_bound[:,-1,:],vel2_bound[:,-1,:])

  [tmp,vel3l] =  ppm_th(-5,vel3[:,-1,:],vel3[:,-3,:],vel3[:,-2,:],vel3[:,-1,:],vel3[:,-1,:])
  [vel3r,tmp ] = ppm_th(-4,vel3[:,-1,:],vel3[:,-2,:],vel3[:,-1,:],vel3[:,-1,:],vel3[:,-1,:])


  flux_th_plus = get_boundary_flux(rhol,rhor,vel1l,vel1r,vel2l,vel2r,vel3l,vel3r,cs=0.1,dir='x2')


  flux_th = flux_th_0 - flux_th_plus

  return [flux_r, flux_th]


def hlle_mhd(dl,dr,vel1l,vel1r,vel2l,vel2r,vel3l,vel3r,B1,B2l,B2r,B3l,B3r,cs=0.1,dir='x1',gam=5.0/3.0):

  def SQR(b):
    return b**2.0

  gm1 = gam-1.0
  iso_cs = cs;

  ##--- Step 1.  Load L/R states into local variables

  bxi = B1;
  Pr = dr * cs**2.0
  Pl = dl * cs**2.0
  ##--- Step 2.  Compute Roe-averaged state

  sqrtdl = np.sqrt(dl);
  sqrtdr = np.sqrt(dr);
  isdlpdr = 1.0/(sqrtdl + sqrtdr);

  rho_roe = sqrtdl*sqrtdr;
  vx_roe = (sqrtdl*vel1l + sqrtdr*vel1r)*isdlpdr;
  vy_roe = (sqrtdl*vel2l + sqrtdr*vel2r)*isdlpdr;
  vz_roe = (sqrtdl*vel3l + sqrtdr*vel3r)*isdlpdr;
  ## Note Roe average of magnetic field is different
  B2_roe = (sqrtdr*B2l + sqrtdl*B2r)*isdlpdr;
  B3_roe = (sqrtdr*B3l + sqrtdl*B3r)*isdlpdr;
  x = 0.5*(SQR(B2l-B2r) + SQR(B3l-B3r))/(SQR(sqrtdl+sqrtdr));
  y = 0.5*(dl + dr)/rho_roe;

  ## Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
  ## rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
  pbl = 0.5*(bxi*bxi + SQR(B2l) + SQR(B3l));
  pbr = 0.5*(bxi*bxi + SQR(B2r) + SQR(B3r));
  el = Pl/gm1 + 0.5*rhol*(SQR(vel1l)+SQR(vel2l)+SQR(vel3l)) +pbl;
  er = Pr/gm1 + 0.5*rhor*(SQR(vel1r)+SQR(vel2r)+SQR(vel3r)) +pbr;
  hroe = ((el + Pl + pbl)/sqrtdl + (er + Pr + pbr)/sqrtdr)*isdlpdr;

  ##--- Step 3.  Compute fast magnetosonic speed in L,R, and Roe-averaged states

  cl = FastMagnetosonicSpeed(dl,Pl,bxi,SQR(B2l)+SQR(B3l));
  cr = FastMagnetosonicSpeed(dr,Pr,bxi,SQR(B2r)+SQR(B3r));

  ## Compute fast-magnetosonic speed using eq. B18 (adiabatic) or B39 (isothermal)
  btsq = SQR(B2_roe) + SQR(B3_roe);
  vaxsq = bxi*bxi/rho_roe;

  bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
  hp = hroe - (vaxsq + btsq/rho_roe);
  vsq = SQR(vx_roe) + SQR(vy_roe) + SQR(vz_roe);
  #twid_asq = std::max((gm1*(hp-0.5*vsq)-(gm1-1.0)*x), 0.0);
  twid_asq = (gm1*(hp-0.5*vsq)-(gm1-1.0)*x)
  twid_asq[twid_asq>0.0] = 0.0
  ct2 = bt_starsq/rho_roe;
  tsum = vaxsq + ct2 + twid_asq;
  tdif = vaxsq + ct2 - twid_asq;
  cf2_cs2 = np.sqrt(tdif*tdif + 4.0*twid_asq*ct2);

  cfsq = 0.5*(tsum + cf2_cs2);
  a = np.sqrt(cfsq);

  ##--- Step 4.  Compute the max/min wave speeds based on L/R and Roe-averaged values

  # al = std::min((vx_roe - a),(vel1l - cl));
  # ar = std::max((vx_roe + a),(vel1r + cr));


  if (dir=='x1'):
    al = vx_roe -a   #std::min((vx_roe - a),(vel1l - cl));
    al[(vel1l - cl)<(vx_roe -a)] = (vel1l - cl)[(vel1l - cl)<(vx_roe -a)]
    ar = vx_roe +a  #std::max((vx_roe + a),(vel1r + cr))
    ar[(vel1r+ cr)>(vx_roe +a)] = (vel1r + cr)[(vel1r+ cr)>(vx_roe +a)]
  elif (dir=='x2'):
    al = vy_roe -a   #std::min((vx_roe - a),(vel1l - cl));
    al[(vel2l - cl)<(vy_roe -a)] = (vel2l - cl)[(vel2l - cl)<(vy_roe -a)]
    ar = vx_roe +a  #std::max((vx_roe + a),(vel1r + cr))
    ar[(vel2r+ cr)>(vy_roe +a)] = (vel2r + cr)[(vel2r+ cr)>(vy_roe +a)]
  else: 
    print(dir, "is not a valid direction")
    return None

  bp = ar * 1.0
  bp[bp<0]= 0.0
  bm = al * 1.0
  bm[bm>0] = 0.0

  ##--- Step 5.  Compute L/R fluxes along the lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R

  vxl = vel1l - bm;
  vxr = vel1r - bp;

  fl = rhol*vxl;
  fr = rhor*vxr;


  ##--- Step 6.  Compute the HLLE flux at interface.

  tmp=0.0;
  tmp = nan_to_num( 0.5*(bp + bm)/(bp - bm));

  return 0.5*(fl+fr) + (fl-fr)*tmp;



def get_mks_pole_fix_coords(th_min = 0.1,h_slope=0.2,ny=300,x2min=0,x2max=pi):
  global m,b
  def func(x_lower):
    return np.pi * x_lower + (1.0-h_slope)/2.0 * np.sin(np.pi * 2.0 * x_lower) -th_min ;
  x_l = scipy.optimize.bisect(func,0,0.5)
  x_p = 1.0-x_l

  y = np.linspace(0,1,ny+1)
  dy = 1.0/(ny)

  ##dy = 0


  #map (dy,1-dy) to x_l,x_p

  # (x_p-x_l)*(y-dy) + x_l

  # x_l = m*dy + b 
  # x_p = m*(1-dy) + b 
  # m = (x_l -b)/dy
  # x_p = (x_l-b)/dy * (1-dy) + b

  # x_p = x_l/dy *(1-dy) - b*(1-dy)/dy + b = x_l/dy *(1-dy)  + b(1-(1-dy)/dy)
  b =1.0 / (2*dy-1) * (dy*(x_p+x_l) - x_l)
  m = (x_p-x_l)/(1.0-2.0*dy)

  var = m*y+b 

  th_f = pi*var + (1-h_slope)/2.0*sin(2*pi*var)
  th_f[0] = x2min
  th_f[-1] = x2max
  return th_f

def Te_Tg_plot_for_conference():
  clf()
  os.chdir("/global/scratch/smressle/star_cluster/restart_grmhd/restart_grmhd_beta_1e2_121_a9_electrons/test")
  rd_yt_convert_to_spherical(1000,True,gr=True,az=0.9375)
  get_Te_Tg(ke_ent,rho,True)
  r_ = r[:,0,0]
  semilogx(r_[r_<1e3],angle_average(Te/Tg,weight=rho)[r_<1e3],lw=2,color='blue')

  os.chdir("/global/scratch/smressle/star_cluster/restart_mhd/beta_1e2_cooling_125_comet_electrons")
  rd_yt_convert_to_spherical(121,True) 
  get_Te_Tg(ke_ent,rho,False)
  set_constants()
  rg = gm_/(cl/pc*kyr)**2
  r_ = r[:,0,0]/rg
  semilogx(r_[r_>.9e3],angle_average(Te/Tg,weight=rho)[r_>0.9e3],lw=2,color='green')


  xlim(1,1e4)
  ylim(0,1)

  plt.ylabel(r'$T_{\rm e}/T_{\rm g}$',fontsize=20)
  plt.xlabel(r'$r$ [$r_{\rm g}$]',fontsize=20)



def get_dr_smr(max_level=5,ny_block = 4,r_max=500.0):
  global dth_grid,dph_grid,dr_grid,rc_grid,thc_grid

  def r_center(xm,xp):
    return 3.0/4.0 * (xp**4-xm**4) / (xp**3-xm**3)
  def th_center(xm, xp):
    sm = np.sin(xm)
    cm = np.cos(xm)
    sp = np.sin(xp)
    cp = np.cos(xp)
    return (sp-xp*cp - sm+xm*cm) / (cm-cp)
  nx_base = nx//2**max_level
  ny_base = ny//2**max_level
  nz_base = nz//2**max_level

  x_ = np.linspace(0,1.0,nx_base+1)
  x1rat = np.exp(np.log(r_max/1.0)/nx_base)
  ratn=x1rat**nx_base
  rnx=x1rat**(x_*nx_base)
  lw=(rnx-ratn)/(1.0-ratn)
  rw=1.0-lw

  r_face = 1.0*lw+r_max*rw

  rc_grids = []
  for n in arange(1,max_level+1):
    rf = x1f[::nx//2**n]
    rc = r_center(rf[:-1],rf[1:])
    #np.repeat(,nx//2**n)

  dlogr = np.diff(r_face)

  dlogr_base = diff(log(r_face))[0]
  dth_base = np.pi/ny_base
  dph_base = 2.0*np.pi/nz_base

  level_th_bound = []
  level_r_bound = []
  for n in arange(0,max_level):
    if (n==max_level-1): level_th_bound += [(ny_block * dth_base , np.pi - ny_block * dth_base)]
    elif (n<=3): level_th_bound += [(np.pi/2-np.pi/32*2**n,np.pi/2+np.pi/32*2**n)]
    else: level_th_bound += [(ny_block * dth_base , np.pi - ny_block * dth_base)]

    if (4**n<= r_max): level_r_bound += [(1.0,4**(n+1))]
    else: level_r_bound += [(1.0,r_max)]

  level_th_bound += [(0.0,np.pi)]
  level_r_bound += [(1.0,r_max)]


  dlogr_level = [dlogr_base/2**max_level]
  dth_level = [dth_base/2**max_level]
  dph_level = [dph_base/2**max_level]

  for n in arange(1,max_level+1):
    dth_level += [dth_base/2**(max_level-n)]
    dph_level += [dph_base/2**(max_level-n)] 
    dlogr_level += [dlogr_base/2**(max_level-n)] 


  dth_grid = dth_base * r/r 
  dph_grid = dph_base *r/r
  dr_grid  = dlogr_base * r
  thc_grid = 1.0*th 
  rc_grid = 1.0*r
  for i_ in arange(len(level_th_bound))[::-1]:
    thb_min,thb_max = level_th_bound[i_]
    rb_min,rb_max = level_r_bound[i_]
    dth_grid[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)] = dth_level[i_]
    dph_grid[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)] = dph_level[i_]

    rf = x1f[::2**i_]
    thf = x2f[::2**i_]
    rc = r_center(rf[:-1],rf[1:])
    thc = th_center(thf[:-1],thf[1:])
    rc = np.repeat(rc,2**i_)
    thc = np.repeat(thc,2**i_)

    dr_grid[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)]  = dlogr_level[i_] * (rc[:,None,None] *r/r)[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)]

    rc_grid[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)]  = (rc[:,None,None] *r/r)[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)]
    thc_grid[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)]  = (thc[None,:,None] *r/r)[(th>thb_min)*(th<thb_max)*(r>rb_min)*(r<rb_max)]





def plot_spherical_mesh():
  # Extract inputs
  input_file = "mesh_structure.dat" ##kwargs['input']
  output_file = "show" #kwargs['output']

  # Load Python plotting modules

  import matplotlib.pyplot as plt

  # Read and plot block edges
  plt.clf()
  ax = plt.gca() ##projection='3d')
  r = []
  th = []
  ph = []
  with open(input_file) as f:
      for line in f:
          if line[0] != '\n' and line[0] != '#':
              numbers_str = line.split()
              r.append(float(numbers_str[0]))
              th.append(float(numbers_str[1]))
              # append zero if 2D
              if(len(numbers_str) > 2):
                  ph.append(float(numbers_str[2]))
              else:
                  ph.append(0.0)
          if line[0] == '\n' and len(r) != 0:
              ax.plot(np.array(r)*np.sin(np.array(th)), np.array(r)*np.cos(np.array(th)),'k-')
              r = []
              th = []
              ph = []
  if output_file == 'show':
      plt.show()
  else:
      plt.savefig(output_file, bbox_inches='tight')

def plot_cartesian_mesh(lw=2,color='black'):
  # Extract inputs
  input_file = "mesh_structure.dat" ##kwargs['input']
  output_file = "show" #kwargs['output']

  # Load Python plotting modules

  import matplotlib.pyplot as plt

  # Read and plot block edges
  plt.clf()
  ax = plt.gca() ##projection='3d')
  x = []
  y = []
  z = []
  with open(input_file) as f:
      for line in f:
          if line[0] != '\n' and line[0] != '#':
            numbers_str = line.split()
            if np.isclose(float(numbers_str[0]),0):
              x.append(float(numbers_str[0]))
              y.append(float(numbers_str[1]))
              # append zero if 2D
              if(len(numbers_str) > 2):
                  z.append(float(numbers_str[2]))
              else:
                  z.append(0.0)
          if line[0] == '\n' and len(r) != 0 and len(y)!=0:
            ymin = np.amin(np.array(y))
            ymax = np.amax(np.array(y))
            zmin = np.amin(np.array(z))
            zmax = np.amax(np.array(z))
            ax.plot((ymin,ymin),(zmin,zmax),ls='-',color=color,lw=lw)
            ax.plot((ymin,ymax),(zmin,zmin),ls='-',color=color,lw=lw)
            ax.plot((ymax,ymax),(zmin,zmax),ls='-',color=color,lw=lw)
            ax.plot((ymin,ymax),(zmax,zmax),ls='-',color=color,lw=lw)


            ax.plot((ymin,ymax),((zmin+zmax)/2.0,(zmin+zmax)/2.0),ls='-',color=color,lw=lw)

            # ax.plot(np.array(y), np.array(z),ls='-',color=color,lw=lw)
            x = []
            y = []
            z = []
  if output_file == 'show':
      plt.show()
  else:
      plt.savefig(output_file, bbox_inches='tight')

def check_dt():
  def r_center(xm,xp):
    return 3.0/4.0 * (xp**4-xm**4) / (xp**3-xm**3)
  def th_center(xm, xp):
    sm = np.sin(xm)
    cm = np.cos(xm)
    sp = np.sin(xp)
    cp = np.cos(xp)
    return (sp-xp*cp - sm+xm*cm) / (cm-cp)

  max_level=5
  #level 4 
  level = 4
  rf = x1f[::2**max_level//2**level]
  thf = x2f[::2**max_level//2**level]
  rc = r_center(rf[:-1],rf[1:])    
  thc = th_center(thf[:-1],thf[1:])    
  dth = np.diff(thf)
  dr = np.diff(rf)
  dphi = np.diff(x3f[::2**max_level//2**level])[0]

  va = np.sqrt(bsq/rho)[::2**max_level//2**level,::2**max_level//2**level,::2**max_level//2**level]
  v3 = vel3[::2**max_level//2**level,::2**max_level//2**level,::2**max_level//2**level]
  dt_min_va_phi = 0.3*dphi*(rc[:,None,None]*(va*0+1.0)) * (sin(thc)[None,:,None] *(va*0+1.0))/va
  dt_min_v3_phi = 0.3*dphi*(rc[:,None,None]*(va*0+1.0)) * (sin(thc)[None,:,None] *(va*0+1.0))/abs(v3)

  rmin = 1.0
  rmax = rf[16]


def get_hdf5_data(filename,mhd=True,user_file=False):

  global r_,th_,ph_
  global rho_,press_,vel1_,vel2_,vel3_,Bcc1_,Bcc2_,Bcc3_
  global dr_,dth_,dph_,t_
  global flux_th_, flux_r_,dM_floor_, dM_r_outer,dM_r_inner, dM_th_outer,dM_th_inner
  global dM_r_inner_pos
  import h5py
  with h5py.File(filename, 'r') as f:
    if (user_file==False):
      shape_array = np.array(f['prim'])[0]*0+1.0
      r_ = (np.array(f['x1v'])[:,None,None,:]*shape_array).flatten()
      th_ = (np.array(f['x2v'])[:,None,:,None]*shape_array).flatten()
      ph_ = (np.array(f['x3v'])[:,:,None,None]*shape_array).flatten()

      dr_ = (np.diff(np.array(f['x1f']),axis=-1)[:,None,None,:]*shape_array).flatten()
      dth_ = (np.diff(np.array(f['x2f']),axis=-1)[:,None,:,None]*shape_array).flatten()
      dph_ = (np.diff(np.array(f['x3f']),axis=-1)[:,:,None,None]*shape_array).flatten()

      rho_ = np.array(f['prim'])[0].flatten()
      press_ = np.array(f['prim'])[1].flatten()

      vel1_ = np.array(f['prim'])[2].flatten()
      vel2_ = np.array(f['prim'])[3].flatten()
      vel3_ = np.array(f['prim'])[4].flatten()

      if (mhd==True):
        Bcc1_ = np.array(f['B'])[0].flatten()
        Bcc2_ = np.array(f['B'])[1].flatten()
        Bcc3_ = np.array(f['B'])[2].flatten()

      t_ = f.attrs['Time']
    else:
      shape_array = np.array(f['hydro'])[0]*0+1.0
      r_facem = (np.array(f['x1f'])[:,None,None,:-1]*shape_array).flatten()
      r_facep = (np.array(f['x1f'])[:,None,None,1:]*shape_array).flatten()

      th_facem = (np.array(f['x2f'])[:,None,:-1,None]*shape_array).flatten()
      th_facep = (np.array(f['x2f'])[:,None,1:,None]*shape_array).flatten()

      flux_r_ = np.array(f['hydro'])[0].flatten()
      flux_th_ = np.array(f['hydro'])[1].flatten()
      dM_floor_ = np.array(f['hydro'])[2].flatten()
      
      dM_r_inner = flux_r_[r_facem==np.amin(np.array(f['x1f']))].sum()
      dM_r_outer = flux_r_[r_facep==np.amax(np.array(f['x1f']))].sum()

      dM_r_inner_pos = (flux_r_[r_facem==np.amin(np.array(f['x1f']))]*(flux_r_[r_facem==np.amin(np.array(f['x1f']))]>0) ).sum()

      dM_th_inner = flux_th_[th_facem==np.amin(np.array(f['x2f']))].sum()
      dM_th_outer = flux_th_[th_facep==np.amax(np.array(f['x2f']))].sum()


def anna_plot():

  beta_1e6_dir = '/global/scratch/users/smressle/star_cluster/test_new_code/without_S2_9_levels_beta_1e6_v3_orbits_cooling/'
  beta_1e2_dir = '/global/scratch/users/smressle/star_cluster/test_new_code/without_S2_9_levels_beta_1e2_v3_orbits_cooling/'
  #hydro_dir = 

  dir_list = [beta_1e6_dir,beta_1e2_dir]
  beta_label_list = [r'$\beta_{\rm w}=10^6$',r'$\beta_{\rm w}=10^2$']
  time_label_list = ['1997','2007','2017']
  i_list = [108,109,110]
  beta_list = ['1e6','1e2']

  i_t = 0
  for i_ in i_list:
    i_dir = 0
    for dir_ in dir_list:

      clf()
      set_constants()
      os.chdir(dir_)
      # rd_yt_convert_to_spherical(i_,True,th=1.1170150415666702,ph=0.7505609576204696,nr=356,nth=128,nphi=256)
      # th_ =  1.1170150415666702
      # ph_ =  0.7505609576204696

      yt_extract_box(i_,box_radius = 0.04,mhd=True,gr=False,a=0.0,res=128)

      z1 = 0.045*arc_secs
      z2 = -0.258*arc_secs

      def z_to_iz(z_input):
        if (z_input <=np.amin(z)):
          return 0
        if (z_input >=np.amax(z)):
          return z[0,0].shape[0]-1
        for iz in range(z[0,0].shape[0]-1):
          if ( (z[0,0,iz] <z_input) and (z_input<=z[0,0,iz+1]) ):
            if (z_input/z[0,0,iz]-1. < 1. - z_input/z[0,0,iz+1]):
              return iz
            else:
              return iz+1

      iz1 = z_to_iz(z1)
      iz2 = z_to_iz(z2)
      # z_hat = np.array([sin(th_)*cos(ph_),sin(th_)*sin(ph_),cos(th_)])   #r
      # x_hat = np.array([cos(th_)*cos(ph_),cos(th_)*sin(ph_),-sin(th_)])  #theta
      # y_hat = np.array([-sin(ph_),cos(ph_),0])                        #phi

      # r1 =   [0.496*arc_secs, -0.564*arc_secs, 0.045*arc_secs] 
      # r2 =   [0.353*arc_secs, -0.194*arc_secs, -0.258*arc_secs] 

      # def dot(vec1,vec2):
      #   return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
      # r1new = [dot(r1,x_hat),dot(r1,y_hat),dot(r1,z_hat)]
      # r2new = [dot(r2,x_hat),dot(r2,y_hat),dot(r2,z_hat)]

      box_limits(0.04/arc_secs)
      plt.xlim(0.04/arc_secs,-0.04/arc_secs)

      # c1 = pcolormesh(x[:,ny//2,:],y[:,ny//2,:],log10(rho)[:,ny//2,:],vmax=2,vmin=-1)
      c1 = pcolormesh(x[:,:,0]/arc_secs,y[:,:,0]/arc_secs,log10(rho[:,:,iz2:iz1].mean(-1)),vmax=2,vmin=0)

      # Bcc1 = vel1
      # Bcc2 = vel2
      # Bcc3 = vel3
      # plot_fieldlines_midplane_slice(box_radius = 0.04,spherical_coords=False,iphi=0,arrowstyle='->',lw=.2,density=5,color='white')

      # scatter(x=r1new[0],y=r1new[1],marker='o',color='red',s=20)
      # scatter(x=r2new[0],y=r2new[1],marker='s',color='red',s=20)


      vx_weighted = (rho*vel1)[:,:,iz2:iz1].mean(-1)/(rho)[:,:,iz2:iz1].mean(-1)
      vy_weighted = (rho*vel2)[:,:,iz2:iz1].mean(-1)/(rho)[:,:,iz2:iz1].mean(-1)
      plt.streamplot(x.mean(-1).transpose()/arc_secs,y.mean(-1).transpose()/arc_secs,vx_weighted.transpose(),vy_weighted.transpose(),color = 'white',density=5,linewidth=0.2,arrowstyle='->')


      scatter(x=0.496,y=-0.564,marker='o',color='red',s=20)
      scatter(x=0.353,y=-0.194,marker='s',color='red',s=20)

      plt.xlabel(r'RA [arcsec]',fontsize=15)
      plt.ylabel(r'dec [arcsec]',fontsize=15)


      cb = plt.colorbar(c1)


      cb.set_label(r'$\log_{10}(\rho)$', fontsize=15)
      plt.title(r'%s year = %s' % (beta_label_list[i_dir],time_label_list[i_t]),fontsize=15)
      plt.tight_layout()

      plt.savefig('%s_%s.png' %(beta_list[i_dir],i_))

      i_dir = i_dir +1 
    i_t = i_t + 1

def EHT_mad_comp_plots():

  os.chdir("/global/scratch/users/smressle/star_cluster/gr_torus_cartesian/mad_case_a_0.9_128_ppm")

  rd_1d_avg()

  def plot_label_resize_1d(fontsize = 10):
     for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontsize(fontsize)
  a = 0.9375
  rh = rhor = ( 1.0 + np.sqrt(1.0-a**2) )
  irh = r_to_ir(rh)
  ir2 = r_to_ir(2)
  ir5 = r_to_ir(5)
  ir10 = r_to_ir(10)


  #mdot

  plt.figure(1)
  plt.clf()

  plt.plot(t,-mdot[:,irh],lw=2,ls='-',label = r'$r=r_{\rm H}$')
  plt.plot(t,-mdot[:,ir2],lw=2,ls='--',label = r'$r=2r_{\rm g}$')
  plt.plot(t,-mdot[:,ir5],lw=2,ls=':',label = r'$r=5r_{\rm g}$')
  plt.plot(t,-mdot[:,ir10],lw=2,ls='-.',label = r'$r=10r_{\rm g}$')

  plt.legend(loc='best',fontsize=15,frameon=0)
  plt.ylim(0,60)
  plt.xlabel(r'$t$ [M]',fontsize=15)
  plt.ylabel(r'$\dot M$',fontsize=15)
  plot_label_resize_1d(fontsize = 15)
  plt.grid()
  plt.tight_layout()
  plt.savefig("athena_mad_cart_mdot.pdf")



  #phibh
  plt.figure(1)
  plt.clf()

  plt.plot(t,(Phibh/sqrt(-mdot*4*pi))[:,irh],lw=2,ls='-',label = r'$r=r_{\rm H}$')
  plt.plot(t,(Phibh/sqrt(-mdot*4*pi))[:,ir2],lw=2,ls='--',label = r'$r=2r_{\rm g}$')
  plt.plot(t,(Phibh/sqrt(-mdot*4*pi))[:,ir5],lw=2,ls=':',label = r'$r=5r_{\rm g}$')
  plt.plot(t,(Phibh/sqrt(-mdot*4*pi))[:,ir10],lw=2,ls='-.',label = r'$r=10r_{\rm g}$')

  plt.legend(loc='best',fontsize=15,frameon=0)
  plt.ylim(0,20)
  plt.xlabel(r'$t$ [M]',fontsize=15)
  plt.ylabel(r'$\Phi_{\rm BH}/\sqrt{\dot M}$',fontsize=15)
  plot_label_resize_1d(fontsize = 15)
  plt.grid()
  plt.tight_layout()
  plt.savefig("athena_mad_cart_phibh.pdf")
  plt.tight_layout()
  plt.savefig("athena_mad_cart_phibh.pdf")
  #Edot
  plt.figure(1)
  plt.clf()

  plt.plot(t,-((Edot+mdot)/mdot)[:,irh],lw=2,ls='-',label = r'$r=r_{\rm H}$')
  plt.plot(t,-((Edot+mdot)/mdot)[:,ir2],lw=2,ls='--',label = r'$r=2r_{\rm g}$')
  plt.plot(t,-((Edot+mdot)/mdot)[:,ir5],lw=2,ls=':',label = r'$r=5r_{\rm g}$')
  plt.plot(t,-((Edot+mdot)/mdot)[:,ir10],lw=2,ls='-.',label = r'$r=10r_{\rm g}$')

  plt.legend(loc='best',fontsize=15,frameon=0)
  plt.ylim(-1,2)
  plt.xlabel(r'$t$ [M]',fontsize=15)
  plt.ylabel(r'$\dot E/\dot M$',fontsize=15)
  plot_label_resize_1d(fontsize = 15)
  plt.grid()
  plt.tight_layout()
  plt.savefig("athena_mad_cart_Edot.pdf")
  plt.tight_layout()
  plt.savefig("athena_mad_cart_Edot.pdf")

    #Ldot
  plt.figure(1)
  plt.clf()

  plt.plot(t,-((Jdot)/mdot)[:,irh],lw=2,ls='-',label = r'$r=r_{\rm H}$')
  plt.plot(t,-((Jdot)/mdot)[:,ir2],lw=2,ls='--',label = r'$r=2r_{\rm g}$')
  plt.plot(t,-((Jdot)/mdot)[:,ir5],lw=2,ls=':',label = r'$r=5r_{\rm g}$')
  plt.plot(t,-((Jdot)/mdot)[:,ir10],lw=2,ls='-.',label = r'$r=10r_{\rm g}$')

  plt.legend(loc='best',fontsize=15,frameon=0)
  plt.ylim(-3,15)
  plt.xlabel(r'$t$ [M]',fontsize=15)
  plt.ylabel(r'$\dot L/\dot M$',fontsize=15)
  plt.grid()
  plot_label_resize_1d(fontsize = 15)

  plt.tight_layout()

  plt.savefig("athena_mad_cart_Ldot.pdf")


  yt_extract_box(500,box_radius=10,mhd=True,gr=True,a=0.9375)
  plt.clf()
  c1 = pcolormesh(x[:,ny//2,:],z[:,ny//2,:],log10(rho)[:,ny//2,:],vmax=1,vmin=-4)
  sig = bsq/rho

  plt.contour(x[:,ny//2,:],z[:,ny//2,:],sig[:,ny//2,:],levels=np.linspace(.99,1),colors='k')

  plt.xlabel(r'$x$ [$r_{\rm g}$]',fontsize=15)
  plt.ylabel(r'$z$ [$r_{\rm g}$]',fontsize=15)

  cb = plt.colorbar(c1)
  cb.set_label(r'$\log_{10}(\rho)$', fontsize=15)
  bhole(1.0,facecolor='black')
  gca().set_aspect('equal')

  plt.tight_layout()
  plt.savefig("athena_mad_rho_contour_10.png")


  yt_extract_box(500,box_radius=50,mhd=True,gr=True,a=0.9375)
  plt.clf()
  c1 = pcolormesh(x[:,ny//2,:],z[:,ny//2,:],log10(rho)[:,ny//2,:],vmax=1,vmin=-4)
  sig = bsq/rho

  plt.contour(x[:,ny//2,:],z[:,ny//2,:],sig[:,ny//2,:],levels=np.linspace(.99,1),colors='k')

  plt.xlabel(r'$x$ [$r_{\rm g}$]',fontsize=15)
  plt.ylabel(r'$z$ [$r_{\rm g}$]',fontsize=15)

  cb = plt.colorbar(c1)
  cb.set_label(r'$\log_{10}(\rho)$', fontsize=15)
  bhole(1.0,facecolor='black')
  gca().set_aspect('equal')

  plt.tight_layout()
  plt.savefig("athena_mad_rho_contour_50.png")


def print_refinement_levels(n_levels,box_radius,center_x=0.0,center_y=0.0,center_z=0.0):

  for n in arange(n_levels):
    box_radius_n = box_radius/(2**(n+1)*1.0)

    print("<refinement%d>" %(n+1))
    print("x1min = %g" %(center_x-box_radius_n))
    print("x1max = %g" %(center_x+box_radius_n))
    print("x2min = %g" %(center_y-box_radius_n))
    print("x2max = %g" %(center_y+box_radius_n))
    print("x3min = %g" %(center_z-box_radius_n))
    print("x3max = %g" %(center_z+box_radius_n))
    print("level = %d \n" % (n+1))


def get_emission_proxy(type="SANE",degrees=5,beta=1e2):

  if (type=="SANE"):
    if (degrees==5):  rho_unit = 2.7340608669743466e-16  ##g/cm^3
    if (degrees==45): rho_unit =  2.814947043721345e-16
    Rhigh = 40.0
    mue = 1.0
    gam = 4.0/3.0
    mui = 1.0
    mu = 0.5
  if (type=="MAD"):
    if (degrees==5):  rho_unit = 9.446233271373403e-19
    if (degrees==45): rho_unit = 8.590499578849372e-19
    mue = 1.0
    mu = 0.5
    mui = 1.0
    Rhigh = 40.0
    gam = 13.0/9.0
  if (type=="Wind"):
    if (beta==1e2):   
      rho_unit = 3.6506779166993816e-18
      Rhigh = 23
    if (beta==1e6):   
      rho_unit = 2.01265993837633e-18
      Rhigh = 9.2
    mu = 1.351
    mui = 4.16
    mue = 2.0
    gam = 5.0/3.0


  #n_tot = rho/(mu m_p)
  #T_tot = P_tot/(n_tot k_b)
  #T_e = T_tot/ (mu/mu_e + mu/mui T_i/T_e) 
  mp_over_me = 1836.15267507
  cl_cgs = 3e10 
  e_cgs = 4.8032068e-10
  mp_cgs = 1.672623099e-24
  me_cgs =  9.1093897e-28

  B_unit = cl_cgs * np.sqrt(4.0 * np.pi * rho_unit);
  B_cgs = np.sqrt(bsq) * B_unit
  beta_p = press/bsq*2.0


  R = Rhigh * beta_p**2/(1.0+beta_p**2.0) + 1.0/(1.0+beta_p)

  Thetae =  mp_over_me *press/(rho ) / (2.0+R)
  rho_cgs = rho * rho_unit 
  n_e = rho_cgs / mue/mp_cgs

  nu_c = e_cgs * B_cgs /(2.0*np.pi * me_cgs * cl_cgs)
  nu_s = 2.0/9.0 * nu_c * Thetae**2.0



  C = 0.2
  nu = 230e9
  X = nu/nu_s
  return n_e * np.sqrt(2.0)*np.pi * e_cgs**2.0*nu_s/(6.0*Thetae**2.0*cl_cgs) * X * np.exp(-X**0.333)
  #exp_arg = rho**2.0/(press**2.0 * np.sqrt(bsq))
  #return rho**3.0/press**2.0 * np.exp( -C * exp_arg**0.333 )


def variability_paper_plot():

  sane_dir = '/global/scratch/users/smressle/star_cluster/gr_torus_cartesian/sane_structure_function'
  mad_dir = '/global/scratch/users/smressle/star_cluster/gr_torus_cartesian/mad_case_a_0.9_128_ppm'
  wind_dir = '/global/scratch/users/smressle/star_cluster/restart_grmhd/restart_grmhd_beta_1e2_121_comet'
  import os 

  plt.figure(1)
  plt.clf()
  os.chdir(mad_dir)

  rd_yt_convert_to_spherical(8000,True,gr=True,az=0.9375) 
  j =get_emission_proxy(type="MAD",degrees=45)
  sig = bsq/rho
  pcolormesh_corner(r,th,log10((j*(sig<1)+1e-35)[:,:,nz//2]),vmax=-16,vmin=-20,cmap='inferno',flip_x=True)
  c = pcolormesh_corner(r,th,log10((j*(sig<1)+1e-35)[:,:,0]),vmax=-16,vmin=-20,cmap='inferno')

  cb = plt.colorbar(c)
  cb.set_label(r"$\log_{10}(j_\nu)  $ [cgs]",fontsize=17)
  #cb.set_ticks(np.arange(-5.0,2.5,.5))
  box_limits(20)
  plt.xlabel(r'$r$ $\sin(\theta)$ [$r_{\rm g}$]',fontsize=20)
  plt.ylabel(r'$r$ $\cos(\theta )$ [$r_{\rm g}$]',fontsize=20)
  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels(): # + cb2.ax.get_yticklabels():
    label.set_fontsize(10)
  bhole(az = 0.9375,facecolor='black')

  plt.title("MAD Torus",fontsize=15)
  plt.tight_layout()
  plt.savefig("mad_torus_jnu.png")

  plt.figure(2)
  plt.clf()
  os.chdir(sane_dir)

  rdhdf5(1000,ndim=3,coord='spherical',gr=True,a=0.9375,block_level=3,vol_params=[0.9])
  n_scale = 2
  r = r[::n_scale,::n_scale,:]*1.0
  th = th[::n_scale,::n_scale,:]*1.0
  ph = ph[::n_scale,::n_scale,:]*1.0

  j =get_emission_proxy(type="SANE",degrees=45)
  sig = bsq/rho

  j = j[::n_scale,::n_scale,:]*1.0
  sig = sig[::n_scale,::n_scale,:]*1.0

  nx = nx//n_scale
  ny = ny//n_scale
  pcolormesh_corner(r,th,log10((j*(sig<1)+1e-35)[:,:,nz//2]),vmax=-16,vmin=-20,cmap='inferno',flip_x=True)
  c = pcolormesh_corner(r,th,log10((j*(sig<1)+1e-35)[:,:,0]),vmax=-16,vmin=-20,cmap='inferno')
  bhole(az = 0.9375,facecolor='black')

  cb = plt.colorbar(c)
  cb.set_label(r"$\log_{10}(j_\nu)  $ [cgs]",fontsize=17)
  #cb.set_ticks(np.arange(-5.0,2.5,.5))
  box_limits(20)
  plt.xlabel(r'$r$ $\sin(\theta)$ [$r_{\rm g}$]',fontsize=20)
  plt.ylabel(r'$r$ $\cos(\theta )$ [$r_{\rm g}$]',fontsize=20)
  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels(): # + cb2.ax.get_yticklabels():
    label.set_fontsize(10)

  plt.title("SANE Torus",fontsize=15)
  plt.tight_layout()
  plt.savefig("sane_torus_jnu.png")

  plt.figure(3)
  plt.clf()
  os.chdir(wind_dir)

  rd_yt_convert_to_spherical(1000,True,gr=True,a=0.0,th =98.72815/180.0*np.pi,ph= -71.97593/180.0*np.pi) 
  j =get_emission_proxy(type="Wind",degrees=45,beta=1e2)
  sig = bsq/rho
  pcolormesh_corner(r,th,log10((j*(sig<1)+1e-35)[:,:,nz//2]),vmax=-16,vmin=-20,cmap='inferno',flip_x=True)
  c = pcolormesh_corner(r,th,log10((j*(sig<1)+1e-35)[:,:,0]),vmax=-16,vmin=-20,cmap='inferno')

  cb = plt.colorbar(c)
  cb.set_label(r"$\log_{10}(j_\nu)  $ [cgs]",fontsize=17)
  #cb.set_ticks(np.arange(-5.0,2.5,.5))
  box_limits(20)
  plt.xlabel(r'$r$ $\sin(\theta^\prime)$ [$r_{\rm g}$]',fontsize=20)
  plt.ylabel(r'$r$ $\cos(\theta^\prime )$ [$r_{\rm g}$]',fontsize=20)
  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels(): # + cb2.ax.get_yticklabels():
    label.set_fontsize(10)
  bhole(az = 0.0,facecolor='black')

  plt.title("MAD Wind",fontsize=15)
  plt.tight_layout()
  plt.savefig("mad_wind_jnu.png")


def monopole_problem(a=0.9375,m=1.0):
  global uu_cks,bu_cks,Bx,By,Bz

  def SQR(blah):
    return blah**2.0
  delta = SQR(r) - 2.0*m*r + SQR(a);
  sigma = SQR(r) + SQR(a*np.cos(theta));

  Br = 1.0/sigma;

  gitt_KS = -(1.0 + 2.0*m*r/sigma);
  gitr_KS = 2.0*m*r/sigma;

  grt_KS = 2.0*m*r/sigma;
  grr_KS = 1.0 + 2.0*m*r/sigma;

  ut_KS = np.sqrt(-gitt_KS);
  ur_KS = -1.0/np.sqrt(-gitt_KS) * gitr_KS;

  bt_KS = Br * (ut_KS * grt_KS + ur_KS * grr_KS);

  br_KS = (Br + bt_KS*ur_KS)/ut_KS;

  uu_cks = ks_vec_to_cks(np.array([ut_KS,ur_KS,0,0]),x,y,z,a)
  bu_cks = ks_vec_to_cks(np.array([bt_KS,br_KS,0,0]),x,y,z,a)
  Bx = (bu_cks[1] * uu_cks[0] - bu_cks[0]*uu_cks[1]) 
  By = (bu_cks[2] * uu_cks[0] - bu_cks[0]*uu_cks[2]) 
  Bz = (bu_cks[3] * uu_cks[0] - bu_cks[0]*uu_cks[3]) 



def get_four_vectors(a=0.0):
  global uu,bu,bsq
  uu = [0,0,0,0]
  bu = [0,0,0,0]

  ks_metric(r,th,a)
  tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
  gamma = np.sqrt(1.0 + tmp);

  # Calculate 4-velocity
  ks_inverse_metric(r,th,a)
  alpha = np.sqrt(-1.0/gi[0,0]);
  uu[0] = gamma/alpha;
  uu[1] = vel1 - alpha * gamma * gi[0,1];
  uu[2] = vel2 - alpha * gamma * gi[0,2];
  uu[3] = vel3 - alpha * gamma * gi[0,3];
  uu = np.array(uu)


  # bsq = ds2.r['user_out_var5'][new_index]*2.0
  B_vec = np.zeros(uu.shape)
  B_vec[1] = Bcc1 
  B_vec[2] = Bcc2 
  B_vec[3] = Bcc3
  ks_metric(r,th,a)
  for i in range(1,4):
    for mu in range(0,4):
      bu[0] += g[i,mu]*uu[mu]*B_vec[i]
  bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
  bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
  bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
  bu = np.array(bu)
  bu_tmp = bu* 1.0

  bsq = 0
  for i in range(4):
    for j in range(4):
      bsq += g[i,j] * bu[i] * bu[j]


def plot_ipole_image(fname):
  import h5py

  # load image data (unpolarized intensity and scaling factors) from hdf5 image
  hfp = h5py.File(fname,'r')
  dx_cgs = hfp['header']['camera']['dx'][()] * hfp['header']['units']['L_unit'][()]
  fovMuas = dx_cgs / hfp['header']['dsource'][()] * 2.06265e11
  cgsToJy = hfp['header']['scale'][()]
  unpol = np.copy(hfp['unpol']).transpose((1,0)) * cgsToJy
  hfp.close()

  # make plot and show
  ext = [ -fovMuas/2, fovMuas/2, -fovMuas/2, fovMuas/2 ]
  plt.imshow(unpol, origin='lower', extent=ext)
  plt.show()

def run_ipole_lambda_sq(dump_file,outfile):
  lambda_sq_vals = np.linspace(0.15e-5,0.2e-5,6,endpoint=True)  # in m
  #lambda_sq_vals = np.linspace(0.01e-5,0.25e-5,10,endpoint=True)  # in m
  lambda_sq_vals *= 100**2.0
  cl = 2.99792458e10
  freq_vals = cl/np.sqrt(lambda_sq_vals)

  n = 0 
  for freq in freq_vals:
    print (freq)
    command_ = "./ipole --freqcgs=%ge11 --MBH=4.3e6 --M_unit=1.0 --thetacam=1.0 --dsource=8127.0" \
              " --fov=200 --rmax_geo=500 --dump=%s --outfile=%s_%d.h5" %  \
              (freq/1e11,dump_file,outfile,n)
    print (command_)
    os.system(command_)
    n = n+1


def rd_1d_image():
  fname = "1d_image.npz"
  if (os.path.isfile(fname)): 
    rdnpz(fname)
    return
  dump_list = glob.glob("image_dump_*_0.h5")
  dump_list.sort()
  i_dump_max = len(dump_list)*4
  global flux_array,t_array,CP_array,LP_array,RM_array
  t_array = []
  flux_array = []
  CP_array = []
  LP_array = []
  RM_array = []
  for i in arange(0,i_dump_max):
    if os.path.isfile("image_dump_%d_freq_0.h5" %i) and os.path.isfile("image_dump_%d_freq_1.h5" %i):
      rd_image("image_dump_%d_freq_0.h5" %i)
      flux_array.append(fluxtot)
      t_array.append(t)
      CP_array.append(CP)
      LP_array.append(LP)
      RM = rd_ipole_rm(i)
      RM_array.append(RM)
  t_array = np.array(t_array)
  LP_array = np.array(LP_array)
  CP_array = np.array(CP_array)
  RM_array = np.array(RM_array)
  flux_array = np.array(flux_array)

  dic = {}
  dic["t"] = t_array
  dic["LP"] = LP_array
  dic["CP"] = CP_array
  dic['RM'] = RM_array
  dic['flux'] = flux_array
  np.savez("1d_image.npz",**dic)


def rd_image(fname):
  global t, evpatot, freq_cgs, fluxtot,LP,CP
  ## EVPA_CONV not yet implemented! this will only work for observer convention!
  EVPA_CONV = "EofN"  # can be set fo "EofN" or "NofW" 
  import h5py
  hfp = h5py.File(fname,'r')    
  evpa_0 = 'W'
  if 'evpa_0' in hfp['header']:
    evpa_0 = hfp['header']['evpa_0'][()]
  t = hfp['header']['t'][()]
  freq_cgs = hfp['header']['freqcgs'][()]
  imagep = np.copy(hfp['pol']).transpose((1,0,2))
  I = imagep[:,:,0]
  Q = imagep[:,:,1]
  U = imagep[:,:,2]
  V = imagep[:,:,3]
  scale = hfp['header']['scale'][()]
  hfp.close()
  evpatot = 180./3.14159*0.5*np.arctan2(U.sum(),Q.sum())
  if evpa_0 == "W":
    evpatot += 90. 
    if evpatot > 90.:
      evpatot -= 180
  if EVPA_CONV == "NofW":
    evpatot += 90.
    if evpatot > 90.:
      evpatot -= 180

  fluxtot = I.sum()*scale
  LP = np.sqrt(Q.sum()**2+U.sum()**2)/I.sum()
  CP = V.sum()/I.sum()
def get_evpa(fname):
  ## EVPA_CONV not yet implemented! this will only work for observer convention!
  EVPA_CONV = "EofN"  # can be set fo "EofN" or "NofW" 
  import h5py
  hfp = h5py.File(fname,'r')    
  evpa_0 = 'W'
  if 'evpa_0' in hfp['header']:
    evpa_0 = hfp['header']['evpa_0'][()]
  freq_cgs = hfp['header']['freqcgs'][()]
  imagep = np.copy(hfp['pol']).transpose((1,0,2))
  I = imagep[:,:,0]
  Q = imagep[:,:,1]
  U = imagep[:,:,2]
  V = imagep[:,:,3]
  hfp.close()
  evpatot = 180./3.14159*0.5*np.arctan2(U.sum(),Q.sum())
  if evpa_0 == "W":
    evpatot += 90. 
    if evpatot > 90.:
      evpatot -= 180
  if EVPA_CONV == "NofW":
    evpatot += 90.
    if evpatot > 90.:
      evpatot -= 180
  # print("EVPA [deg]:   {0:g}".format(evpatot))
  return (freq_cgs,evpatot)

def get_evpa_array(n_start,n_end):
  global freq_vals,evpa_vals, lamsq_vals_m
  freq_vals = []
  evpa_vals = []
  for i in arange(n_start,n_end):
    freq,evpa = get_evpa('image_freq_%d.h5' % i)
    freq_vals.append(freq)
    evpa_vals.append(evpa)
  evpa_vals = np.array(evpa_vals)/180*np.pi
  freq_vals = np.array(freq_vals)
  lamsq_vals_m = (299792458/np.array(freq_vals))**2


def rd_ipole_rm(idump):
  dump_file_1 = "image_dump_%d_freq_0.h5" %(idump)
  dump_file_2 = "image_dump_%d_freq_1.h5" %(idump)

  freq1, evpa1 = get_evpa(dump_file_1)
  freq2, evpa2 = get_evpa(dump_file_2)

  cl = 2.997924e8

  lam1 = cl/freq1  # in meters
  lam2 = cl/freq2  # in meters

  evpa1_rad = evpa1 /180.*np.pi 
  evpa2_rad = evpa2 /180.*np.pi 

  RM = (evpa2_rad-evpa1_rad)/(lam2**2.0-lam1**2.0)

  return RM



# import os
# import os
# import numpy as np

# dir_list = ["/scratch/03496/smressle/chris_archived/jet_resolution/jet_0",
#             "/scratch/03496/smressle/chris_archived/jet_resolution/jet_1",
#             "/scratch/03496/smressle/chris_archived/jet_resolution/jet_2",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_0",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_1",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_2",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_3",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_0",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_1",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_2",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_3",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_base",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_loop",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_loop_old",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_gamma",
#             "/scratch/03496/smressle/chris_archived/tilted_disks/s90_t00_hight"]

# chris_dir_list = ["/scratch/04939/cjwhite/archived/jet_resolution/jet_0",
#             "/scratch/04939/cjwhite/archived/jet_resolution/jet_1",
#             "/scratch/04939/cjwhite/archived/jet_resolution/jet_2",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_0",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_1",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_2",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_3",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_0",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_1",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_2",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_3",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_base",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_loop",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_loop_old",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_gamma",
#             "/scratch/04939/cjwhite/archived/tilted_disks/s90_t00_hight"]

# for idir in np.arange(len(dir_list)):
#   dir_ = dir_list[idir]
#   chris_dir = chris_dir_list[idir]

#   os.chdir(dir_)
#   os.system("ls %s/*.tar | xargs -i tar -xf {}" %chris_dir)




# DX = 1600.0*2./192.
# dx_list = np.array([DX,DX/2,DX/2**2,DX/2**3,DX/2**4,DX/2**5,DX/2**6,DX/2**7,DX/2**8 ,DX/2**9])#,DX/2**10])
# r_lim_list = np.array([1.0,1.0/2.0,1.0/2.0**2,1.0/2.0**3,1.0/2.0**4,1.0/2.0**5,1.0/2.0**6,1.0/2.0**7,1.0/2.0**8,1.0/2.0**9]) #,1.0/2.0**10]
# r_lim_list = r_lim_list * 1600.

# dx_arr = rho*0 + DX

# for i in range(len(dx_list)):
#     r_lim = r_lim_list[i]
#     ind = (x<r_lim)*(x>-r_lim) * (y<r_lim)*(y>-r_lim) * (z<r_lim)*(z>-r_lim)
#     dx_arr[ind] = dx_list[i]

# plt.figure(figsize=(8,8*4))
# ir = r_to_ir(5.0)
# subplot(411)
# plot(t,-mdot[:,ir])
# ylim(0,60)
# ylabel(r'$|\dot M|$',fontsize=20)

# subplot(412)
# plot(t,(Phibh/sqrt(-mdot*4*pi))[:,ir])
# ylim(0,20)
# ylabel(r'$\phi$',fontsize=20)

# subplot(413)

# plot(t,((Edot+mdot)/(-mdot))[:,ir])
# ylim(-1,2)
# ylabel(r'$\dot E/|\dot M|$',fontsize=20)

# subplot(414)
# plot(t,(Jdot/(-mdot))[:,ir])
# ylabel(r'$\dot J/|\dot M|$',fontsize=20)
# ylim(-2,15)

# xlabel(r'$t$',fontsize=20)


# yt_load(120)
# from yt.visualization.volume_rendering.api import Scene, VolumeSource 
# import numpy as np
# sc  = Scene()
# vol = VolumeSource(ds, field="density")
# bounds = (10**-1, 10.**2)
# tf = yt.ColorTransferFunction(np.log10(bounds))
# def linramp(vals, minval, maxval):
#     return (vals - vals.min())/(vals.max() - vals.min())
# #tf.add_layers(8, colormap='ocean')
# tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='Blues_r',scale_func=linramp)
# #tf.add_layers(8, colormap='ocean')
# tf.grey_opacity = False
# vol.transfer_function = tf
# vol.tfh.tf = tf
# vol.tfh.bounds = bounds
# vol.tfh.plot('transfer_function.png', profile_field="density")
# cam = sc.add_camera(ds, lens_type='plane-parallel')
# cam.resolution = [512,512]
# # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
# # cam.switch_orientation(normal_vector=normal_vector,
# #                        north_vector=north_vector)
# cam.set_width(ds.domain_width*0.2)

# cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')
# normal_vector = [0,0,-1]  #camera to focus
# north_vector = [0,1,0]  #up direction
# cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
# sc.add_source(vol)
# sc.render()
# # sc.save('tmp2.png',sigma_clip = 6.0)
# # sc = yt.create_scene(ds,lens_type = 'perspective')
# # sc.camera.zoom(2.0)
# # sc[0].tfh.set_bounds([1e-4,1e2])
# os.system("mkdir -p frames")
# sc.save(fname,sigma_clip = 4.0)
# plt.imshow(imread(fname))

def Hector_plot():
  
  os.chdir('/global/scratch/users/smressle/star_cluster/gr_torus_cartesian/mad_case_a_0.9_128_ppm')
  plt.figure(1)
  plt.clf()

  rd_1d_torus_avg()

  semilogx(r[-1,:],PEM_jet[-1,:],lw=2,ls='-',label = r'$P_{\rm EM}$')
  semilogx(r[-1,:],PPAKE_jet[-1,:],lw=2,ls='--',label = r'$P_{\rm PAKE}$')
  semilogx(r[-1,:],PEN_jet[-1,:],lw=2,ls=':',label = r'$P_{\rm EN}$')
  semilogx(r[-1,:],Mdot_jet[-1,:],lw=2,ls=':',label = r'$\dot M_{\rm jet}$')

  plt.ylim(-10,50)
  plt.xlim(1,3000)

  plt.legend(loc='best',fontsize=15,frameon=0)
  plt.xlabel(r'$r$ [$r_{\rm g}$]',fontsize=15)
  # plot_label_resize_1d(fontsize = 15)
  plt.grid()
  plt.tight_layout()

def bh2_pos_constant_velocity(t,vbh,z0):
  return (0.0,0.0, vbh*t +z0)

def bh2_pos(t,r_bh2,t0=0,inclination=0.0):
  v_bh2 = 1.0/np.sqrt(r_bh2);
  Omega_bh2 = v_bh2/r_bh2;

  xbh_ = 0.0;
  ybh_ = r_bh2 * np.sin(Omega_bh2 * (t-t0))
  zbh_ = r_bh2 * np.cos(Omega_bh2 * (t-t0))


  # *xbh = std::sin(orbit_inclination) * zbh_;
  # *ybh = ybh_;
  # *zbh = std::cos(orbit_inclination) * zbh_;

  x_bh2 = np.sin(inclination) * zbh_
  y_bh2 = ybh_
  z_bh2 = np.cos(inclination) * zbh_


  return (x_bh2,y_bh2,z_bh2)


def bh2_vel(t,r_bh2,t0=0,inclination=0.0):
  v_bh2 = 1.0/np.sqrt(r_bh2);
  Omega_bh2 = v_bh2/r_bh2;


  vxbh_ = 0.0;
  vybh_ =  Omega_bh2 * r_bh2 * np.cos(Omega_bh2 * (t-t0));
  vzbh_ = -Omega_bh2 * r_bh2 * np.sin(Omega_bh2 * (t-t0));


  # *vxbh = std::sin(orbit_inclination) * vzbh_;
  # *vybh = vybh_;
  # *vzbh = std::cos(orbit_inclination) * vzbh_;

  vx_bh2 = np.sin(inclination) * vzbh_;
  vy_bh2 = vybh_;
  vz_bh2 = np.cos(inclination) * vzbh_;

  return (vx_bh2,vy_bh2,vz_bh2)

def cks_binary_metric(t,x_,y_,z_,a1x=0,a1y=0,a1z=0,a2x_=0,a2y_=0,a2z_=0,q=0,r_bh2=10.0,t0=1e5,inclination=0.0):
  global g
  m = 1.0
  x,y,z = x_*1.0,y_*1.0,z_*1.0
  def SQR(arg_):
    return arg_*arg_

  x_bh2, y_bh2, z_bh2 = bh2_pos(t,r_bh2=r_bh2,t0=t0,inclination=inclination)
  # x_bh2,y_bh2,z_bh2 = x2,y2,z2

  v2x, v2y, v2z = bh2_vel(t,r_bh2=r_bh2,t0=t0,inclination=inclination)


  a1 = np.sqrt(a1x**2+a1y**2+a1z**2)
  a2 = np.sqrt(a2x_**2+a2y_**2+a2z_**2)

  # v1 = np.sqrt(v1x**2+v1y**2+v1z**2)
  v2 = np.sqrt(v2x**2+v2y**2+v2z**2)

  Lorentz = np.sqrt(1.0/(1.0 - v2**2))


  a_dot_x = a1x * x + a1y * y + a1z * z;

  small = 1e-5
  diff =  (abs(a_dot_x)<small) * ( small * (a_dot_x>=0) - small * (a_dot_x<0)  -  a_dot_x/(a1+small)  )

  x = x + diff*a1x/(a1+small);
  y = y + diff*a1y/(a1+small);
  z = z + diff*a1z/(a1+small);

  a_dot_x = a1x * x + a1y * y + a1z * z;

  x[abs(x)<0.1] = 0.1
  y[abs(y)<0.1] = 0.1
  z[abs(z)<0.1] = 0.1

  [r,th,phi] = GetBoyerLindquistCoordinates(x,y,z,a1x,a1y,a1z);

  rh =  ( m + np.sqrt(SQR(m)-SQR(a1)) );
  r[r<rh*0.5] = rh * 0.5
  [x,y,z] = convert_spherical_to_cartesian_ks(r,th,phi, a1x,a1y,a1z)
  
  a_dot_x = a1x * x + a1y * y + a1z * z;


  a_cross_x = [a1y * z - a1z * y,
               a1z * x - a1x * z,
               a1x * y - a1y * x]


  rsq_p_asq = SQR(r) + SQR(a1);



  f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a_dot_x));

  l0 = 1.0;
  l1 = (r * x - a_cross_x[0] + a_dot_x * a1x/r)/(rsq_p_asq);
  l2 = (r * y - a_cross_x[1] + a_dot_x * a1y/r)/(rsq_p_asq);
  l3 = (r * z - a_cross_x[2] + a_dot_x * a1z/r)/(rsq_p_asq);


  nx_ = x.shape[0]
  ny_ = x.shape[1]
  nz_ = x.shape[2]
  g  = np.zeros((4,4,nx_,ny_,nz_))
  g[0][0] = -1.0 + f * l0*l0
  g[0][1] = f * l0*l1
  g[0][2] = f * l0*l2
  g[0][3] = f * l0*l3
  g[1][1] = 1.0 + f * l1*l1
  g[1][3] = f * l1*l3
  g[2][2] = 1.0 + f * l2*l2
  g[2][3] = f * l2*l3 
  g[1][2] = f * l1*l2
  g[3][3] = 1.0 + f * l3*l3

  [xprime,yprime,zprime,rprime,Rprime] = get_prime_coords(x,y,z,x_bh2,y_bh2,z_bh2,v2x,v2y,v2z,a2x_,a2y_,a2z_)
  a_dot_x_prime = a2x_ * xprime + a2y_ * yprime + a2z_ * zprime;


  small = 1e-5
  diff =  (abs(a_dot_x_prime)<small) * ( small * (a_dot_x_prime>=0) - small * (a_dot_x_prime<0)  -  a_dot_x_prime/(a2+small)  )

  xprime = xprime + diff*a2x_/(a2+small);
  yprime = yprime + diff*a2y_/(a2+small);
  zprime = zprime + diff*a2z_/(a2+small);

  [rprime,thprime,phiprime] = GetBoyerLindquistCoordinates(xprime,yprime,zprime,a2x_,a2y_,a2z_)


  rhprime = ( q + np.sqrt(SQR(q)-SQR(a2)) );
  rprime[rhprime<0.8*rhprime] = 0.8*rhprime
  [xprime,yprime,zprime] = convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a2x_,a2y_,a2z_)

  a_dot_x_prime = a2x_ * xprime + a2y_ * yprime + a2z_ * zprime;



  a_cross_x_prime = [a2y_ * zprime - a2z_ * yprime,
                     a2z_ * xprime - a2x_ * zprime,
                     a2x_ * yprime - a2y_ * xprime];


  rsq_p_asq_prime = SQR(rprime) + SQR(a2);


  fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(a_dot_x_prime));

  l0prime = 1.0;
  l1prime =  (rprime * xprime - a_cross_x_prime[0] + a_dot_x_prime * a2x_/rprime)/(rsq_p_asq_prime);
  l2prime =  (rprime * yprime - a_cross_x_prime[1] + a_dot_x_prime * a2y_/rprime)/(rsq_p_asq_prime);
  l3prime =  (rprime * zprime - a_cross_x_prime[2] + a_dot_x_prime * a2z_/rprime)/(rsq_p_asq_prime);

  vsq = SQR(v2x) + SQR(v2y) + SQR(v2z);
  beta_mag = np.sqrt(vsq);
  Lorentz = np.sqrt(1.0/(1.0 - vsq));
  nx = v2x/beta_mag;
  ny = v2y/beta_mag;
  nz = v2z/beta_mag;


  Lambda  = np.zeros((4,4))
  Lambda[0,0] =  Lorentz;
  Lambda[0,1] = -Lorentz * v2x;
  Lambda[1,0] = Lambda[0,1]
  Lambda[0,2] = -Lorentz * v2y;
  Lambda[2,0] = Lambda[0,2]
  Lambda[0,3] = -Lorentz * v2z;
  Lambda[3,0] = Lambda[0,3]
  Lambda[1,1] = ( 1.0 + (Lorentz - 1.0) * nx * nx );
  Lambda[1,2] = (       (Lorentz - 1.0) * nx * ny ); 
  Lambda[2,1] = Lambda[1,2]
  Lambda[1,3] = (       (Lorentz - 1.0) * nx * nz );
  Lambda[3,1] = Lambda[1,3]
  Lambda[2,2] = ( 1.0 + (Lorentz - 1.0) * ny * ny ); 
  Lambda[2,3] = (       (Lorentz - 1.0) * ny * nz );
  Lambda[3,2] = Lambda[2,3]
  Lambda[3,3] = ( 1.0 + (Lorentz - 1.0) * nz * nz );



  def matrix_multiply_vector_lefthandside(A,b):
    result = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            result[i] += A[i,j]*b[j]
    return result

  [l0prime_transformed,l1prime_transformed,l2prime_transformed,l3prime_transformed] =  matrix_multiply_vector_lefthandside(Lambda,[l0prime,l1prime,l2prime,l3prime])


  # matrix_multiply_vector_lefthandside(Lambda,l_lowerprime,l_lowerprime_transformed);



  g[0][0] += fprime * l0prime_transformed*l0prime_transformed
  g[0][1] += fprime * l0prime_transformed*l1prime_transformed
  g[0][2] += fprime * l0prime_transformed*l2prime_transformed
  g[0][3] += fprime * l0prime_transformed*l3prime_transformed
  g[1][1] += fprime * l1prime_transformed*l1prime_transformed
  g[1][3] += fprime * l1prime_transformed*l3prime_transformed
  g[2][2] += fprime * l2prime_transformed*l2prime_transformed
  g[2][3] += fprime * l2prime_transformed*l3prime_transformed
  g[1][2] += fprime * l1prime_transformed*l2prime_transformed
  g[3][3] += fprime * l3prime_transformed*l3prime_transformed


  g[1][0] = g[0][1]
  g[2][0] = g[0][2]
  g[3][0] = g[0][3]
  g[3][1] = g[1][3]
  g[3][2] = g[2][3]
  g[2][1] = g[1][2]

def cks_boosted_metric(t,x,y,z,a=0.0,vbh=0.1,z0=-80.0):
  global g
  m = 1
  x_bh2,y_bh2,z_bh2 = bh2_pos_constant_velocity(t,z0=z0,vbh=vbh)

  Lorentz = 1.0/np.sqrt(1.0 - vbh**2.0)
  aprime = a*1.0

  xprime = (x-x_bh2)
  yprime = (y-y_bh2)
  zprime = (z-z_bh2)
  Rprime = np.sqrt(xprime**2+yprime**2+zprime**2)
  rprime = np.sqrt(Rprime**2-aprime**2 + np.sqrt( (Rprime**2-aprime**2)**2 + 4*aprime**2*zprime**2 ))/np.sqrt(2.0)

  fprime = 2.0*rprime**3/(rprime**4 + aprime**2*zprime**2)
  l0 = 1.0
  l1 = (rprime*xprime+aprime*yprime)/(rprime**2+aprime**2)
  l2 = (rprime*yprime-aprime*xprime)/(rprime**2+aprime**2)
  l3 = zprime/rprime


  l0_ = l0*1.0;
  l3_ = l3*1.0;

  l0 = Lorentz * (l0_ - vbh * l3_);
  l3 = Lorentz * (l3_ - vbh * l0_);

  nx = x.shape[0]
  ny = x.shape[1]
  nz = x.shape[2]
  g_  = np.zeros((4,4,nx,ny,nz))
  g_[0][0] = -1.0 + fprime * l0*l0
  g_[0][1] = fprime * l0*l1
  g_[1][0] = g_[0][1]
  g_[0][2] = fprime * l0*l2
  g_[2][0] = g_[0][2]
  g_[0][3] = fprime * l0*l3
  g_[3][0] = g_[0][3]
  g_[1][1] = 1.0 + fprime * l1*l1
  g_[1][3] = fprime * l1*l3
  g_[3][1] = g_[1][3]
  g_[2][2] = 1.0 + fprime * l2*l2
  g_[2][3] = fprime * l2*l3 
  g_[3][2] = g_[2][3]
  g_[1][2] = fprime * l1*l2
  g_[2][1] = g_[1][2]
  g_[3][3] = 1.0 + fprime * l3*l3

  global g

  g =g_*1.0

def convert_spherical_to_cartesian_ks(r,th,phi,ax,ay,az):

  x_ = r * np.sin(th) * np.cos(phi) + ay * np.cos(th)                 - az*np.sin(th) * np.sin(phi);
  y_ = r * np.sin(th) * np.sin(phi) + az * np.sin(th) * np.cos(phi)   - ax*np.cos(th)                ;
  z_ = r * np.cos(th)               + ax * np.sin(th) * np.sin(phi)   - ay*np.sin(th) * np.cos(phi);

  return [x_,y_,z_]
def GetBoyerLindquistCoordinates(x1,x2,x3,ax,ay,az):
  x,y,z = x1*1.0,x2*1.0,x3*1.0
  def SQR(var):
    return var**2.0
  a =  np.sqrt( SQR(ax) + SQR(ay) + SQR(az) );

  a_dot_x = ax * x + ay * y + az * z;

  a_cross_x = [ay * z - az * y,
               az * x - ax * z,
               ax * y - ay * x];


  small = 1e-5
  diff =  (abs(a_dot_x)<small) * ( small * (a_dot_x>=0) - small * (a_dot_x<0)  -  a_dot_x/(a+small)  )

  x = x + diff*ax/(a+small);
  y = y + diff*ay/(a+small);
  z = z + diff*az/(a+small);

  a_dot_x = ax * x + ay * y + az * z;



  R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
  r = np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a_dot_x) )  )/np.sqrt(2.0);

  rsq_p_asq = SQR(r) + SQR(a);

  lx = (r * x - a_cross_x[0] + a_dot_x * ax/r)/(rsq_p_asq);
  ly = (r * y - a_cross_x[1] + a_dot_x * ay/r)/(rsq_p_asq);
  lz = (r * z - a_cross_x[2] + a_dot_x * az/r)/(rsq_p_asq);

  lz[lz>1.0] = 1.0
  lz[lz<-1.0] = -1.0

  pr = r;
  ptheta = np.arccos(lz); 
  pphi = np.arctan2(ly,lx); 

  return [pr,ptheta,pphi]

def get_prime_coords(x1,y1,z1,x2,y2,z2,v2x,v2y,v2z,a2x,a2y,a2z):

  def SQR(var):
    return var**2.0
  xbh = x2*1.0
  ybh = y2*1.0
  zbh = z2*1.0


  ax = a2x*1.0
  ay = a2y*1.0
  az = a2z*1.0

  a_mag = np.sqrt( SQR(ax) + SQR(ay) + SQR(az) );

  vxbh = v2x*1.0
  vybh = v2y*1.0
  vzbh = v2z*1.0


  vsq = SQR(vxbh) + SQR(vybh) + SQR(vzbh);
  beta_mag = np.sqrt(vsq);
  Lorentz = np.sqrt(1.0/(1.0 - vsq));

  nx = vxbh/beta_mag;
  ny = vybh/beta_mag;
  nz = vzbh/beta_mag;

  xprime = (1.0 + (Lorentz - 1.0) * nx * nx) * ( x1 - xbh ) +  \
            (      (Lorentz - 1.0) * nx * ny) * ( y1 - ybh ) +  \
            (      (Lorentz - 1.0) * nx * nz) * ( z1 - zbh );
  
  yprime = (      (Lorentz - 1.0) * ny * nx) * ( x1 - xbh ) +  \
            (1.0 + (Lorentz - 1.0) * ny * ny) * ( y1 - ybh ) + \
            (      (Lorentz - 1.0) * ny * nz) * ( z1 - zbh );  
 
  zprime = (      (Lorentz - 1.0) * nz * nx) * ( x1 - xbh ) +  \
            (      (Lorentz - 1.0) * nz * ny) * ( y1 - ybh ) + \
            (1.0 + (Lorentz - 1.0) * nz * nz) * ( z1 - zbh );  


  a_dot_x_prime = ax * (xprime) + ay * (yprime) + az * (zprime);


  small = 1e-5
  diff =  (abs(a_dot_x_prime)<small) * ( small * (a_dot_x_prime>=0) - small * (a_dot_x_prime<0)  -  a_dot_x_prime/(a_mag+small)  )

  xprime = xprime + diff*ax/(a_mag+small);
  yprime = yprime + diff*ay/(a_mag+small);
  zprime = zprime + diff*az/(a_mag+small);

  a_dot_x_prime = ax * (xprime) + ay * (yprime) + az * (zprime);



  Rprime = np.sqrt(SQR(xprime) + SQR(yprime) + SQR(zprime));
  rprime = SQR(Rprime) - SQR(a_mag) + np.sqrt( SQR( SQR(Rprime) - SQR(a_mag) ) + 4.0*SQR(a_dot_x_prime) );
  rprime = np.sqrt(rprime/2.0);


  return [xprime,yprime,zprime,rprime,Rprime];


def interp_orbits(t,arr,t0):

  t_arr = t_array + t0
  t0_orbits = t_arr[0]
  dt_orbits = t_arr[1]-t_arr[0]

  it = (int) ((t - t_arr[0]) / dt_orbits + 1000) - 1000;

  if (it<= 0): it = 0;
  if (it>=nt-1): it = nt-1;


  if (t<t0_orbits):
      slope = (arr[it+1]-arr[it])/dt_orbits;
      result = (t - t_arr[it] ) * slope + arr[it];
  elif (it==nt-1):
      slope = (arr[it]-arr[it-1])/dt_orbits;
      result = (t - t_arr[it] ) * slope + arr[it]
  else:
      slope = (arr[it+1]-arr[it])/dt_orbits;
      result = (t - t_arr[it] ) * slope + arr[it];

  return result

def get_binary_quantities(t,t0):
  global x1,y1,z1,x2,y2,z2,a1x,a1y,a1z,a2x,a2y,a2z,v1x,v1y,v1z,v2x,v2y,v2z

  x1 = interp_orbits(t,x1_array,t0)
  y1 = interp_orbits(t,y1_array,t0)
  z1 = interp_orbits(t,z1_array,t0)

  x2 = interp_orbits(t,x2_array,t0)
  y2 = interp_orbits(t,y2_array,t0)
  z2 = interp_orbits(t,z2_array,t0)


  v1x = interp_orbits(t,v1x_array,t0)
  v1y = interp_orbits(t,v1y_array,t0)
  v1z = interp_orbits(t,v1z_array,t0)


  v2x = interp_orbits(t,v2x_array,t0)
  v2y = interp_orbits(t,v2y_array,t0)
  v2z = interp_orbits(t,v2z_array,t0)

  a1x = interp_orbits(t,a1x_array,t0)
  a1y = interp_orbits(t,a1y_array,t0)
  a1z = interp_orbits(t,a1z_array,t0)


  a2x = interp_orbits(t,a2x_array,t0)
  a2y = interp_orbits(t,a2y_array,t0)
  a2z = interp_orbits(t,a2z_array,t0)

  
def cks_full_binary_metric(fname,t,x_,y_,z_,m=1.0,t0=1e5):
  global g

  x,y,z = x_*1.0,y_*1.0,z_*1.0
  def SQR(arg_):
    return arg_*arg_
  rd_binary_orbits(fname)
  get_binary_quantities(t,t0)
  x_bh2,y_bh2,z_bh2 = x2,y2,z2

  a2x_ = a2x * q
  a2y_ = a2y * q
  a2z_ = a2z * q

  a1 = np.sqrt(a1x**2+a1y**2+a1z**2)
  a2 = np.sqrt(a2x_**2+a2y_**2+a2z_**2)

  v1 = np.sqrt(v1x**2+v1y**2+v1z**2)
  v2 = np.sqrt(v2x**2+v2y**2+v2z**2)

  Lorentz = np.sqrt(1.0/(1.0 - v2**2))


  a_dot_x = a1x * x + a1y * y + a1z * z;

  small = 1e-5
  diff =  (abs(a_dot_x)<small) * ( small * (a_dot_x>=0) - small * (a_dot_x<0)  -  a_dot_x/(a1+small)  )

  x = x + diff*a1x/(a1+small);
  y = y + diff*a1y/(a1+small);
  z = z + diff*a1z/(a1+small);

  a_dot_x = a1x * x + a1y * y + a1z * z;

  x[abs(x)<0.1] = 0.1
  y[abs(y)<0.1] = 0.1
  z[abs(z)<0.1] = 0.1

  [r,th,phi] = GetBoyerLindquistCoordinates(x,y,z,a1x,a1y,a1z);

  rh =  ( m + np.sqrt(SQR(m)-SQR(a1)) );
  r[r<rh*0.5] = rh * 0.5
  [x,y,z] = convert_spherical_to_cartesian_ks(r,th,phi, a1x,a1y,a1z)
  
  a_dot_x = a1x * x + a1y * y + a1z * z;


  a_cross_x = [a1y * z - a1z * y,
               a1z * x - a1x * z,
               a1x * y - a1y * x]


  rsq_p_asq = SQR(r) + SQR(a1);



  f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a_dot_x));

  l0 = 1.0;
  l1 = (r * x - a_cross_x[0] + a_dot_x * a1x/r)/(rsq_p_asq);
  l2 = (r * y - a_cross_x[1] + a_dot_x * a1y/r)/(rsq_p_asq);
  l3 = (r * z - a_cross_x[2] + a_dot_x * a1z/r)/(rsq_p_asq);


  nx_ = x.shape[0]
  ny_ = x.shape[1]
  nz_ = x.shape[2]
  g  = np.zeros((4,4,nx_,ny_,nz_))
  g[0][0] = -1.0 + f * l0*l0
  g[0][1] = f * l0*l1
  g[0][2] = f * l0*l2
  g[0][3] = f * l0*l3
  g[1][1] = 1.0 + f * l1*l1
  g[1][3] = f * l1*l3
  g[2][2] = 1.0 + f * l2*l2
  g[2][3] = f * l2*l3 
  g[1][2] = f * l1*l2
  g[3][3] = 1.0 + f * l3*l3

  [xprime,yprime,zprime,rprime,Rprime] = get_prime_coords(x,y,z,x2,y2,z2,v2x,v2y,v2z,a2x_,a2y_,a2z_)
  a_dot_x_prime = a2x_ * xprime + a2y_ * yprime + a2z_ * zprime;


  small = 1e-5
  diff =  (abs(a_dot_x_prime)<small) * ( small * (a_dot_x_prime>=0) - small * (a_dot_x_prime<0)  -  a_dot_x_prime/(a2+small)  )

  xprime = xprime + diff*a2x_/(a2+small);
  yprime = yprime + diff*a2y_/(a2+small);
  zprime = zprime + diff*a2z_/(a2+small);

  [rprime,thprime,phiprime] = GetBoyerLindquistCoordinates(xprime,yprime,zprime,a2x_,a2y_,a2z_)


  rhprime = ( q + np.sqrt(SQR(q)-SQR(a2)) );
  rprime[rhprime<0.8*rhprime] = 0.8*rhprime
  [xprime,yprime,zprime] = convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a2x_,a2y_,a2z_)

  a_dot_x_prime = a2x_ * xprime + a2y_ * yprime + a2z_ * zprime;



  a_cross_x_prime = [a2y_ * zprime - a2z_ * yprime,
                     a2z_ * xprime - a2x_ * zprime,
                     a2x_ * yprime - a2y_ * xprime];


  rsq_p_asq_prime = SQR(rprime) + SQR(a2);


  fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(a_dot_x_prime));

  l0prime = 1.0;
  l1prime =  (rprime * xprime - a_cross_x_prime[0] + a_dot_x_prime * a2x_/rprime)/(rsq_p_asq_prime);
  l2prime =  (rprime * yprime - a_cross_x_prime[1] + a_dot_x_prime * a2y_/rprime)/(rsq_p_asq_prime);
  l3prime =  (rprime * zprime - a_cross_x_prime[2] + a_dot_x_prime * a2z_/rprime)/(rsq_p_asq_prime);

  vsq = SQR(v2x) + SQR(v2y) + SQR(v2z);
  beta_mag = np.sqrt(vsq);
  Lorentz = np.sqrt(1.0/(1.0 - vsq));
  nx = v2x/beta_mag;
  ny = v2y/beta_mag;
  nz = v2z/beta_mag;


  Lambda  = np.zeros((4,4))
  Lambda[0,0] =  Lorentz;
  Lambda[0,1] = -Lorentz * v2x;
  Lambda[1,0] = Lambda[0,1]
  Lambda[0,2] = -Lorentz * v2y;
  Lambda[2,0] = Lambda[0,2]
  Lambda[0,3] = -Lorentz * v2z;
  Lambda[3,0] = Lambda[0,3]
  Lambda[1,1] = ( 1.0 + (Lorentz - 1.0) * nx * nx );
  Lambda[1,2] = (       (Lorentz - 1.0) * nx * ny ); 
  Lambda[2,1] = Lambda[1,2]
  Lambda[1,3] = (       (Lorentz - 1.0) * nx * nz );
  Lambda[3,1] = Lambda[1,3]
  Lambda[2,2] = ( 1.0 + (Lorentz - 1.0) * ny * ny ); 
  Lambda[2,3] = (       (Lorentz - 1.0) * ny * nz );
  Lambda[3,2] = Lambda[2,3]
  Lambda[3,3] = ( 1.0 + (Lorentz - 1.0) * nz * nz );



  def matrix_multiply_vector_lefthandside(A,b):
    result = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            result[i] += A[i,j]*b[j]
    return result

  [l0prime_transformed,l1prime_transformed,l2prime_transformed,l3prime_transformed] =  matrix_multiply_vector_lefthandside(Lambda,[l0prime,l1prime,l2prime,l3prime])


  # matrix_multiply_vector_lefthandside(Lambda,l_lowerprime,l_lowerprime_transformed);



  g[0][0] += fprime * l0prime_transformed*l0prime_transformed
  g[0][1] += fprime * l0prime_transformed*l1prime_transformed
  g[0][2] += fprime * l0prime_transformed*l2prime_transformed
  g[0][3] += fprime * l0prime_transformed*l3prime_transformed
  g[1][1] += fprime * l1prime_transformed*l1prime_transformed
  g[1][3] += fprime * l1prime_transformed*l3prime_transformed
  g[2][2] += fprime * l2prime_transformed*l2prime_transformed
  g[2][3] += fprime * l2prime_transformed*l3prime_transformed
  g[1][2] += fprime * l1prime_transformed*l2prime_transformed
  g[3][3] += fprime * l3prime_transformed*l3prime_transformed


  g[1][0] = g[0][1]
  g[2][0] = g[0][2]
  g[3][0] = g[0][3]
  g[3][1] = g[1][3]
  g[3][2] = g[2][3]
  g[2][1] = g[1][2]



# def cks_binary_inverse_metric(t,x,y,z,aprime_=0.0,ONED = False,q=0,r_bh2=10.0,a=0.0,t0=0.0,inclination=0.0):
#   global gi_
#   m = 1
#   x_bh2,y_bh2,z_bh2 = bh2_pos(t,r_bh2,t0=t0,inclination=inclination)


#   aprime = aprime_ * q
#   xprime = x-x_bh2
#   yprime = y-y_bh2
#   zprime = z-z_bh2
#   Rprime = np.sqrt(xprime**2+yprime**2+zprime**2)
#   rprime = np.sqrt(Rprime**2-aprime**2 + np.sqrt( (Rprime**2-aprime**2)**2 + 4*aprime**2*zprime**2 ))/np.sqrt(2.0)

#   fprime = q * 2.0*rprime**3/(rprime**4 + aprime**2*zprime**2)
#   l0 = -1.0
#   l1 = (rprime*xprime+aprime*yprime)/(rprime**2+aprime**2)
#   l2 = (rprime*yprime-aprime*xprime)/(rprime**2+aprime**2)
#   l3 = zprime/rprime

#   if (ONED==False):
#     nx = x.shape[0]
#     ny = x.shape[1]
#     nz = x.shape[2]
#     gi_  = np.zeros((4,4,nx,ny,nz))
#   else: gi_ = np.zeros((4,4,x.shape[0]))
#   gi_[0][0] = - fprime * l0*l0;
#   gi_[0][1] = - fprime * l0*l1;
#   gi_[1][0] = gi_[0][1]
#   gi_[0][2] = - fprime * l0*l2
#   gi_[2][0] = gi_[0][2]
#   gi_[0][3] = - fprime * l0*l3
#   gi_[3][0] = gi_[0][3]
#   gi_[1][1] = - fprime * l1*l1
#   gi_[1][3] = - fprime * l1*l3
#   gi_[3][1] = gi_[1][3]
#   gi_[2][2] = - fprime * l2*l2
#   gi_[2][3] = - fprime * l2*l3 
#   gi_[3][2] = gi_[2][3]
#   gi_[1][2] = - fprime * l1*l2
#   gi_[2][1] = gi_[1][2]
#   gi_[3][3] = - fprime * l3*l3



#   global gi
#   cks_inverse_metric(x,y,z,0,0,a)

#   gi += gi_


def minkowski_metric(x,y,z):
  global eta 

  eta = np.zeros((4,4,nx,ny,nz))
  eta[0][0] = -1.0 + 0.0*x 
  eta[1][1] =  1.0 + 0.0*x 
  eta[2][2] =  1.0 + 0.0*x
  eta[3][3] =  1.0 + 0.0*x

  eta = np.zeros((4,4,nx,ny,nz))
def Lower_tensor_right(Auu,g):
  Aud = 0*Auu
  #A^mu_nu = g_{alpha nu}A^{mu alpha}
  for mu in range(4):
    for nu in range(4):
      for i in range(4):
        Aud[mu,nu] += g[i,nu]*Auu[mu,i]
  return Aud
def Lower_tensor_left(Auu,g):
  Adu = 0*Auu
  #A^mu_nu = g_{alpha nu}A^{mu alpha}
  for mu in range(4):
    for nu in range(4):
      for i in range(4):
        Adu[mu,nu] += g[i,mu]*Auu[nu,i]
  return Adu


def Primitive_to_Conserved(gdd,guu,gam=1.44444444):
  global M1,M2,M3,E,rho_cons
  uu1 = vel1 * 1.0
  uu2 = vel2 * 1.0
  uu3 = vel3 * 1.0

  alpha = np.sqrt(-1.0/guu[0,0]);
  tmp = gdd[1,1]*uu1*uu1 + 2.0*gdd[1,2]*uu1*uu2 + 2.0*gdd[1,3]*uu1*uu3 + gdd[2,2]*uu2*uu2 + 2.0*gdd[2,3]*uu2*uu3+ gdd[3,3]*uu3*uu3;
  gamma = np.sqrt(1.0 + tmp);
  u0 = gamma/alpha;
  u1 = uu1 - alpha * gamma * guu[0,1];
  u2 = uu2 - alpha * gamma * guu[0,2];
  u3 = uu3 - alpha * gamma * guu[0,3];

  uu_ = np.array([u0,u1,u2,u3])
  ud_ = Lower(uu_,gdd)

  # // Calculate 4-magnetic field
  # b0 = gdd(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
  #           + gdd(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
  #           + gdd(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
  #           + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
  # b1 = (bb1 + b0 * u1) / u0;
  # b2 = (bb2 + b0 * u2) / u0;
  # b3 = (bb3 + b0 * u3) / u0;
  # b_0, b_1, b_2, b_3;
  # pco->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);
  # b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  # // Set conserved quantities
  wtot = rho + gam/(gam-1.0) * press ##+ b_sq;
  ptot = press ##+ 0.5 * b_sq;
  rho_cons = rho * uu_[0]
  E = wtot * uu_[0] * ud_[0] + ptot ##- b0 * b_0 ;
  M1 = wtot * uu_[0] * ud_[1] ##- b0 * b_1;
  M2 = wtot * uu_[0] * ud_[2] ##- b0 * b_2;
  M3 = wtot * uu_[0] * ud_[3] ##- b0 * b_3;

def Calculate_Normal_Conserved(gdd,guu):
  global mm1,mm2,mm3,ee,dd,mm0

  def SQR(var):
    return var**2.0
  # // Calculate unit timelike normal
  alpha = np.sqrt(-1.0/guu[0,0])
  n0 = -alpha * guu[0,0]
  n1 = -alpha * guu[0,1]
  n2 = -alpha * guu[0,2]
  n3 = -alpha * guu[0,3]

  # // Calculate projection operator
  j10 = guu[1,0] + n1*n0
  j20 = guu[2,0] + n2*n0 
  j30 = guu[3,0] + n3*n0
  j11 = guu[1,1] + n1*n1
  j21 = guu[2,1] + n2*n1
  j31 = guu[3,1] + n3*n1
  j12 = guu[1,2] + n1*n2
  j22 = guu[2,2] + n2*n2
  j32 = guu[3,2] + n3*n2
  j13 = guu[1,3] + n1*n3
  j23 = guu[2,3] + n2*n3
  j33 = guu[3,3] + n3*n3

  # // Calculate projected momentum densities Q_\mu = -n_\nu T^\nu_\mu (N 17)
  qq_0 = alpha * E;
  qq_1 = alpha * M1;
  qq_2 = alpha * M2;
  qq_3 = alpha * M3;
  qq_n = qq_0*n0 + qq_1*n1 + qq_2*n2 + qq_3*n3;

  # // Calculate projected momentum M^i = j^{i\mu} Q_\mu
  mm1 = j10*qq_0 + j11*qq_1 + j12*qq_2 + j13*qq_3;
  mm2 = j20*qq_0 + j21*qq_1 + j22*qq_2 + j23*qq_3;
  mm3 = j30*qq_0 + j31*qq_1 + j32*qq_2 + j33*qq_3;

  # // Calculate projected field \mathcal{B}^i = alpha B^i (N 5)
  # bbb1 = alpha * bb1;
  # bbb2 = alpha * bb2;
  # bbb3 = alpha * bb3;

  # // Set normal conserved quantities
  dd = alpha * rho_cons;  ##// (N 21)
  ee = -qq_n;
  mm0 = gdd[1,1]*SQR(mm1) + 2.0*gdd[1,2]*mm1*mm2 + 2.0*gdd[1,3]*mm1*mm3+ gdd[2,2]*SQR(mm2) + 2.0*gdd[2,3]*mm2*mm3 + gdd[3,3]*SQR(mm3);
  mm1 = mm1;
  mm2 = mm2;
  mm3 = mm3;
  # bbb(0,i) = g_11*SQR(bbb1) + 2.0*g_12*bbb1*bbb2 + 2.0*g_13*bbb1*bbb3
  #            + g_22*SQR(bbb2) + 2.0*g_23*bbb2*bbb3
  #            + g_33*SQR(bbb3);
  # bbb(1,i) = bbb1;
  # bbb(2,i) = bbb2;
  # bbb(3,i) = bbb3;
  # tt(i) = g_11*mm1*bbb1 + g_12*mm1*bbb2 + g_13*mm1*bbb3
  #         + g_21*mm2*bbb1 + g_22*mm2*bbb2 + g_23*mm2*bbb3
  #         + g_31*mm3*bbb1 + g_32*mm3*bbb2 + g_33*mm3*bbb3;
def Conserved_to_Primitive_Normal(dd,ee,mm0,mm1,mm2,mm3,pgas_old,gamma_adi=1.4444444):
  global prim_rho,prim_press,prim_vel1,prim_vel2,prim_vel3
  def SQR(var):
    return var**2.0
  ## Parameters
  max_iterations = 15;
  tol = 1.0e-12;
  pgas_uniform_min = 1.0e-12;
  a_min = 1.0e-12
  v_sq_max = 1.0 - 1.0e-12
  rr_max = 1.0 - 1.0e-12

  ## Extract conserved values
  mm_sq = mm0*1.0
  bb_sq = 0.0
  bb1 = 0.0
  bb2 = 0.0
  bb3 = 0.0
  tt =  0.0

  ## Calculate functions of conserved quantities
  d = 0.5 * (mm_sq * bb_sq - SQR(tt))                  ## (NH 5.7)
  d = np.fmax(d,0.0)
  ##d[d<0] = 0.0*d[d<0]
  pgas_min = np.cbrt(27.0/4.0 * d) - ee - 0.5*bb_sq;
  pgas_min = np.fmax(pgas_min,pgas_uniform_min)
  ##pgas_min[pgas_min<pgas_uniform_min] = pgas_uniform_min + 0.0*pgas_min[pgas_min<pgas_uniform_min]

  ## Iterate until convergence
  pgas = [0,0,0]
  pgas[0] = np.fmax(pgas_old,pgas_min) ##pgas_old*(pgas_old>=pgas_min) + pgas_min*(pgas_min>pgas_min)
  for n in arange(0,max_iterations):
    ## Step 1: Calculate cubic coefficients
    if (n%3 != 2):
      a = ee + pgas[n%3] + 0.5*bb_sq;  ## (NH 5.7)
      a = np.fmax(a, a_min);

    ## Step 2: Calculate correct root of cubic equation
    if (n%3 != 2):
      phi = np.arccos(1.0/a * np.sqrt(27.0*d/(4.0*a)));                     ## (NH 5.10)
      eee = a/3.0 - 2.0/3.0 * a * np.cos(2.0/3.0 * (phi+np.pi));               ## (NH 5.11)
      ll = eee - bb_sq;                                                       ## (NH 5.5)
      v_sq = (mm_sq*SQR(ll) + SQR(tt)*(bb_sq+2.0*ll)) / SQR(ll * (bb_sq+ll)); ## (NH 5.2)
      v_sq = np.fmin(np.fmax(v_sq, 0.0), v_sq_max);
      gamma_sq = 1.0/(1.0-v_sq);                                         ## (NH 3.1)
      gamma = np.sqrt(gamma_sq);                                       ## (NH 3.1)
      wgas = ll/gamma_sq;                                                ## (NH 5.1)
      rho = dd/gamma;                                                    ## (NH 4.5)
      pgas[(n+1)%3] = (gamma_adi-1.0)/gamma_adi * (wgas - rho);               ## (NH 4.1)
      pgas[(n+1)%3] = np.fmax(pgas[(n+1)%3], pgas_min);

    ## Step 3: Check for convergence
    if (n%3 != 2):
      if (pgas[(n+1)%3] > pgas_min and np.abs(pgas[(n+1)%3]-pgas[n%3]) < tol):
        break


    ## Step 4: Calculate Aitken accelerant and check for convergence
    if (n%3 == 2):
      rr = (pgas[2] - pgas[1]) / (pgas[1] - pgas[0])  ## (NH 7.1)
      if (not np.isfinite(rr) or np.abs(rr) > rr_max):
        continue

      pgas[0] = pgas[1] + (pgas[2] - pgas[1]) / (1.0 - rr);  ## (NH 7.2)
      pgas[0] = np.fmax(pgas[0], pgas_min);
      if (pgas[0] > pgas_min and np.abs(pgas[0]-pgas[2]) < tol):
        break;


  # ## Step 5: Set primitives
  if (n == max_iterations):
    print("reached iteration limit")
    return false;
  prim_press = pgas[(n+1)%3];
  if (not np.isfinite(prim_press)):
    return false
  a = ee + prim_press + 0.5*bb_sq;                      ## (NH 5.7)
  a =np.fmax(a, a_min);
  phi = np.arccos(1.0/a * np.sqrt(27.0*d/(4.0*a)));        ## (NH 5.10)
  eee = a/3.0 - 2.0/3.0 * a * np.cos(2.0/3.0 * (phi+np.pi));  ## (NH 5.11)
  ll = eee - bb_sq;                                          ## (NH 5.5)
  v_sq = (mm_sq*SQR(ll) + SQR(tt)*(bb_sq+2.0*ll)) / SQR(ll * (bb_sq+ll));                             ## (NH 5.2)
  v_sq = np.fmin(np.fmax(v_sq, 0.0), v_sq_max);
  gamma_sq = 1.0/(1.0-v_sq);                                 ## (NH 3.1)
  gamma = np.sqrt(gamma_sq);                               ## (NH 3.1)
  prim_rho = dd/gamma;                                     ## (NH 4.5)
  if (not np.isfinite(prim_rho)):
    return false;
  ss = tt/ll;                          ## (NH 4.8)
  v1 = (mm1 + ss*bb1) / (ll + bb_sq);  ## (NH 4.6)
  v2 = (mm2 + ss*bb2) / (ll + bb_sq);  ## (NH 4.6)
  v3 = (mm3 + ss*bb3) / (ll + bb_sq);  ## (NH 4.6)
  prim_vel1 = gamma*v1;               ## (NH 3.3)
  prim_vel2 = gamma*v2;               ## (NH 3.3)
  prim_vel3 = gamma*v3;               ## (NH 3.3)
  if (not np.isfinite(prim_vel1) or not np.isfinite(prim_vel2) or not np.isfinite(prim_vel3)):
    return false
  p_gamma_lor = gamma;
  p_pmag = 0.5 * (bb_sq/gamma_sq + SQR(ss));  ## (NH 3.7, 3.11)


def xyz_to_ijk(x_,y_,z_,a=0.0):
  def SQR(var): 
    return var**2.0

  R_ = np.sqrt( SQR(x_) + SQR(y_) + SQR(z_) );
  r_ = np.sqrt( SQR(R_) - SQR(a) + np.sqrt( SQR(SQR(R_) - SQR(a)) + 4.0*SQR(a)*SQR(z_) )  )/np.sqrt(2.0);

  th_ = np.arccos(z_/r_)

  ph_ = np.arctan2((r_*y_-a*x_), (a*y_+r_*x_) )
  ir = r_to_ir_npz(r_,r)
  ith = th_to_ith_npz(th_,th)
  iph = ph_to_iph_npz(ph_,ph)

  return ir,ith,iph


def get_l_direction_1d_avg(t_restart = 0.000242):
  it = t_to_it(t_restart)
  nx = Lx/sqrt(Lx**2+Ly**2+Lz**2)
  ny = Ly/sqrt(Lx**2+Ly**2+Lz**2)
  nz = Lz/sqrt(Lx**2+Ly**2+Lz**2)
  ir1 = r_to_ir(1e-5)
  ir2 = r_to_ir(1e-3)

  nx_avg = nx[it,ir1:ir2].mean()
  ny_avg = ny[it,ir1:ir2].mean()
  nz_avg = nz[it,ir1:ir2].mean()

  return (nx_avg,ny_avg,nz_avg)

def get_l_direction_npz():
  ir1 = r_to_ir_npz(1e-5,r)
  ir2 = r_to_ir_npz(1e-3,r)

  Lx = angle_average_npz(rho * (y*vel3 - z*vel2))
  Ly = angle_average_npz(rho * (z*vel1 - x*vel3))
  Lz = angle_average_npz(rho * (x*vel2 - y*vel1))

  nx = Lx/sqrt(Lx**2+Ly**2+Lz**2)
  ny = Ly/sqrt(Lx**2+Ly**2+Lz**2)
  nz = Lz/sqrt(Lx**2+Ly**2+Lz**2)

  nx_avg = nx[ir1:ir2].mean()
  ny_avg = ny[ir1:ir2].mean()
  nz_avg = nz[ir1:ir2].mean()

  return (nx_avg,ny_avg,nz_avg)

def Gillessen_plot():
  x_ = np.linspace(4.0,8.0)
  curve_1 = x_*0.0 + 1e6
  curve_2 = x_*0.0 + 1e7

  plt.fill_between(x_, curve_1, curve_2, color='gray',
                 alpha=0.7)


  plt.plot(1.0,1e6,marker='s',color='gray',ms=10,mfc='None') ## Quataert&Gruzinov
  plt.plot(1.0,1e6,marker='o',color='gray',ms=5) ## might be wrong

  plt.plot(10.0,6e6,marker='s',color='gray',ms=10,mfc='None') ## Agol
  plt.plot(10.0,6e6,marker='o',color='gray',ms=5) 

  plt.plot(5.0,1e6,marker='s',color='gray',ms=10,mfc='None') ## EHT
  plt.plot(5.0,1e6,marker='o',color='gray',ms=5) 


  plt.plot((8.0, 8.0),(2e6,5e6),color='gray',lw=4) ##Bower 2019



  ##plt.plot((1.0, 10.0),(1e6,6e6),color='black',lw=2,ls='--')


  x_ = np.linspace(20.0,200.0)
  pow1 = np.log(8e4/1e4)/np.log(200.0/20.0)
  const1 = 8e4 * (20.0)**pow1
  curve_1 = const1 * x_**-pow1 

  pow2 = np.log(6e5/8e4)/np.log(200.0/20.0)
  const2 = 6e5 * (20.0)**pow2
  curve_2 = const2 * x_**-pow2 

  plt.fill_between(x_, curve_1, curve_2, color='gray',
                 alpha=0.7)

  plt.plot((842*2, 842*2),(2000,8000),color='gray',lw=4)

  x_ = np.linspace(1e4*2.0,2*1e5)
  pow1 = np.log(10**2.5/1e2)/np.log(2e5/2e4)
  const1 = 10**2.5 * (1e4*2.0)**pow1
  curve_1 = const1 * x_**-pow1 

  plt.plot(x_,curve_1,color='gray',lw=4)

def check_vf_orbits():
  def Omega_i_degrees_to_n(Omega,i):
    Omega_ = Omega*np.pi/180.0
    i_ = i*np.pi/180.0
    return sin(i_)*cos(Omega_),-sin(i_)*sin(Omega_),-cos(i_)

  #E20
  Omega = 115.64
  i = 126.14

  nx,ny,nz = -0.38311806103404766,-0.6818594268239478,0.6231278146262039
  nx_,ny_,nz_ = Omega_i_degrees_to_n(Omega,i)
  costheta = dot_product([nx,ny,nz],[nx_,ny_,nz_])
  print("angle between old and new orbits for E20: ", arccos(costheta)/np.pi*180)

  #E23
  Omega = 112.67
  i = 113.6

  nx,ny,nz = -0.3626254397567929,-0.8460685407640985,0.3907311284892737
  nx_,ny_,nz_ = Omega_i_degrees_to_n(Omega,i)
  costheta = dot_product([nx,ny,nz],[nx_,ny_,nz_])
  print("angle between old and new orbits for E23: ", arccos(costheta)/np.pi*180)

  #E32
  Omega = 94.51
  i =  113.02

  nx,ny,nz = -0.20546850597947194,-0.8844097602869284,0.41904900543943713
  nx_,ny_,nz_ = Omega_i_degrees_to_n(Omega,i)
  costheta = dot_product([nx,ny,nz],[nx_,ny_,nz_])
  print("angle between old and new orbits for E32: ", arccos(costheta)/np.pi*180)

  #E40
  Omega = 107.05
  i = 122.23

  nx,ny,nz = -0.12535297871765352,-0.8407822867008401,0.5266609697867479
  nx_,ny_,nz_ = Omega_i_degrees_to_n(Omega,i)
  costheta = dot_product([nx,ny,nz],[nx_,ny_,nz_])
  print("angle between old and new orbits for E40: ", arccos(costheta)/np.pi*180)

  #E56
  Omega = 110.77
  i =  128.27

  nx,ny,nz = -0.12436941778596514,-0.8367099954824774,0.5333372585703093
  nx_,ny_,nz_ = Omega_i_degrees_to_n(Omega,i)
  costheta = dot_product([nx,ny,nz],[nx_,ny_,nz_])
  print("angle between old and new orbits for E56: ", arccos(costheta)/np.pi*180)


def test_mk_frame():


  i_dump = 1 
  a = 0.9375 
  th_tilt =0
  phi_tilt = 0
  plt.figure(1)
  plt.clf()
  rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,az=a,th=th_tilt,ph=phi_tilt)
  nz = rho.shape[-1]
  ny = rho.shape[1]
  # c1 = plt.pcolormesh((x)[:,:,0], (z)[:,:,0],np.log10(rho)[:,:,0],vmin=-3,vmax=0,cmap="ocean")
  # plt.pcolormesh(-(x)[:,:,0], (z)[:,:,0],np.log10(rho)[:,:,nz//2],vmin=-3,vmax=0,cmap="ocean")
  c1 = pcolormesh_corner(r,th,np.log10(rho)[:,:,0],vmin=-3,vmax=0,cmap="ocean")
  pcolormesh_corner(r,th,np.log10(rho)[:,:,nz//2],vmin=-3,vmax=0,cmap="ocean",flip_x=True)

  cb1 = plt.colorbar(c1)
  # if (is_magnetic == True): plt.contour((r*np.sin(th))[:,:,0],(r*np.cos(th))[:,:,0],np.log10(psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
  # if (is_magnetic == True): plt.contour((r*np.sin(th))[:,:,0],(r*np.cos(th))[:,:,0],np.log10(-psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
  # if (is_magnetic == True): plt.contour(-(r*np.sin(th))[:,:,0],(r*np.cos(th))[:,:,0],np.log10(psicalc_slice(gr=True,iphi=nz//2)),30,linestyles='-',colors='white')
  # if (is_magnetic == True): plt.contour(-(r*np.sin(th))[:,:,0],(r*np.cos(th))[:,:,0],np.log10(-psicalc_slice(gr=True,iphi=nz//2)),30,linestyles='--',colors='white')

  plt.ylim(-50,50)
  plt.xlim(-50,50)
  plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
  plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
  plt.title(r'$t = %d M$' %np.int(np.array(t)),fontsize = 20)

  cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
    label.set_fontsize(10)

  bhole()
  plt.axis('off')

  plt.axes().set_aspect('equal')

  #plt.tight_layout()

  os.system("mkdir -p frames")
  ##plt.savefig(fname)

  cks_coord_to_ks(x,y,z,a=a)
  plot_fieldlines_gr(50,a=a,npz=True)
  fname = "frames/frame_fieldlines_%04d.png" % (i_dump)

  plt.clf()
  c1 = pcolormesh_corner(r,th,np.log10(rho)[:,ny//2,:],coords='xy',vmin=-2,vmax=0.5,cmap="ocean")
  ##def pcolormesh_corner(r,th,myvar,coords = 'xz',flip_x = False,**kwargs)

  cb1 = plt.colorbar(c1)

  plt.ylim(-50,50)
  plt.xlim(-50,50)
  plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
  plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
  plt.title(r'$t = %d M$' %np.int(np.array(t)),fontsize = 20)

  cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
    label.set_fontsize(10)

  bhole()
  #plt.axis('off')

  plt.axes().set_aspect('equal')

def rd_yt_convert_to_spherical_2nd_black_hole(idump,MHD=False,dump_name=None,th=0,ph=0,
  method='nearest',fill_value = 0.0,a=0.0, nr = 200, nth = 128,nphi=128,
  double_precision=False,rmin=None,rmax=None,q=0.0,rbh2=60.0,aprime=0.0,t0=0.0,uov=False,inclination=0.0, orbit_file = None):
  if (dump_name is None): 
    dump_name = "dump_spher_2nd_black_hole_%d_th_%.2g_phi_%.2g.npz" %(idump,th,ph)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    yt_load(idump,gr=True,a=a)

    global r,phi,theta,xi,yi,zi
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t 
    global k_ent,ke_ent,ke_ent2,ke_ent3
    global divb
    #unit vectors for the new coordinate system in the frame of the old coordinates

    t = np.array(ds.current_time)

    if (orbit_file==None):

      xbh,ybh,zbh = bh2_pos(t,rbh2,t0=t0,inclination=inclination)
      dxbh_dt,dybh_dt,dzbh_dt =  bh2_vel(t,rbh2,t0=t0,inclination=inclination)

      a1x = 0.0
      a1y = 0.0
      a1z = a*1.0
      a2x_ = 0.0;
      a2y_ = 0.0;
      a2z_ = q* aprime 


      # aprime_ = q * aprime
    else:
      rd_binary_orbits(orbit_file)
      get_binary_quantities(t,t0)

      xbh,ybh,zbh = x2,y2,z2 
      dxbh_dt,dybh_dt,dzbh_dt = v2x, v2y, v2z


      a2x_ = q * a2x
      a2y_ = q * a2y 
      a2z_ = q * a2z

    
    

    a2 = np.sqrt(a2x_**2 + a2y_**2 + a2z_**2)
    vsq = np.sqrt( dxbh_dt**2.0 + dybh_dt**2.0 + dzbh_dt**2.0)
    beta_mag = np.sqrt(vsq)

    rh2 = ( q + np.sqrt(q**2.0-(a2)**2) )
    Lorentz = np.sqrt(1.0/(1.0 - vsq));

    nx_beta = dxbh_dt/beta_mag;
    ny_beta = dybh_dt/beta_mag;
    nz_beta = dzbh_dt/beta_mag;



    z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])   #r
    x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  #theta
    y_hat = np.array([-sin(ph),cos(ph),0])                        #phi

    #unit vectors for original coordinate system in the frame of the new coordinates
    #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
    x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #in terms of [theta_hat, phi_hat, r_hat] 
    y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    z_hat_prime = [-np.sin(th),0,np.cos(th)]

    index = np.arange(ds.r['density'].shape[0])
    if (rmin is None): rmin =rh2*0.8 
    if (rmax is None): rmax = rh2*100.0

    #faces
    theta = np.linspace(0.,np.pi,nth+1)
    phi = np.linspace(0,2.*np.pi,nphi+1)

    #centers
    r = np.logspace(log10(rmin),log10(rmax),nr)

    dth = np.diff(theta)[0]
    theta = (theta + dth/2.0)[:-1]
    dphi = np.diff(phi)[0]
    phi = (phi + dphi/2.0)[:-1]
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')



    ##new x,y,z coords in terms of new r,th,phi coords
    # xi_prime = r*np.cos(phi)*np.sin(theta) - aprime_*np.sin(phi)*np.sin(theta)
    # yi_prime = r*np.sin(phi)*np.sin(theta) + aprime_*np.cos(phi)*np.sin(theta)
    # zi_prime = r*np.cos(theta)


    xi_prime = r * np.sin(theta) *  np.cos(phi) + a2y_ * np.cos(theta)                 - a2z_*np.sin(theta) * np.sin(phi);
    yi_prime = r * np.sin(theta) *  np.sin(phi) + a2z_ * np.sin(theta) * np.cos(phi)   - a2x_*np.cos(theta)                ;
    zi_prime = r * np.cos(theta)                + a2x_ * np.sin(theta) * np.sin(phi)   - a2y_*np.sin(theta) * np.cos(phi);

    xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
    yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
    zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2] 

    xi = xbh + (1.0 + (1.0/Lorentz-1.0)*nx_beta*nx_beta )*xi + (1.0/Lorentz-1.0) *nx_beta*ny_beta*yi + (1.0/Lorentz-1.0) *nx_beta*nz_beta*zi 
    yi = ybh + (1.0 + (1.0/Lorentz-1.0)*ny_beta*ny_beta )*yi + (1.0/Lorentz-1.0) *ny_beta*nx_beta*xi + (1.0/Lorentz-1.0) *ny_beta*nz_beta*zi
    zi = zbh + (1.0 + (1.0/Lorentz-1.0)*nz_beta*nz_beta )*zi + (1.0/Lorentz-1.0) *nx_beta*nz_beta*xi + (1.0/Lorentz-1.0) *ny_beta*nz_beta*yi


    #r_vec = x_new x_hat_new + y_new y_hat_new + z_new z_hat_new  
    # x_old = r_vec dot x_hat_old 
    #original x,y,z coords in terms of new coords
    # xi = xi * x_hat[0] + yi * y_hat[0] + zi * z_hat[0]
    # yi = xi * x_hat[1] + yi * y_hat[1] + zi * z_hat[1]
    # zi = xi * x_hat[2] + yi * y_hat[2] + zi * z_hat[2] 
    new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = method,fill_value = fill_value)


    from yt.units import pc, msun,kyr
    rho = ds.r['density'][new_index] * pc**3/msun
    press = ds.r['press'][new_index] * pc**3/msun * kyr**2/pc**2
    vel1 = ds.r['vel1'][new_index] * kyr/pc
    vel2 = ds.r['vel2'][new_index] * kyr/pc
    vel3 = ds.r['vel3'][new_index] * kyr/pc

    global uu,bu,bsq
    uu = [0,0,0,0]
    uu_ = [0,0,0,0]
    bu = [0,0,0,0]
    bu_ = [0,0,0,0]


    ##first: calculate gamma and alpha in original (lab) frame using full binary metric.  
    ##second: calculate uu in original (lab) frame using the same
    ##third: boost uu to BH frame
    ##fourth recalculate vel (if needed)

    ## then do same for bu and bcc 

    ##cks_metric(xi,yi,zi,a2,ONED=False)

    if (orbit_file is None): cks_binary_metric(t,xi,yi,zi,t0=t0,q=q,a1x=a1x,a1y=a1y,a1z=a1z,a2x_=a2x_,a2y_=a2y_,a2z_=a2z_,inclination=inclination,r_bh2=rbh2)
    else: cks_full_binary_metric(orbit_file,t,xi,yi,zi,m=1.0,t0=t0)
    # cks_binary_metric(t,xi+xbh,yi+ybh,zi+zbh,a=a,aprime_=aprime,q=q,r_bh2=rbh2,t0=t0,midplane=midplane)
    ##(t,x,y,z,aprime_=0.0,ONED=False,q=0,r_bh2=10.0,a=0.0,t0=0.0,midplane=False)
    tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
    gamma = np.sqrt(1.0 + tmp);

    # Calculate 4-velocity
    invert_metric(g)
    # cks_inverse_metric(xi,yi,zi,a2,ONED=False)
    # cks_binary_inverse_metric(t,xi+xbh,yi+ybh,zi+zbh,a=a,aprime_=aprime,q=q,r_bh2=rbh2,t0=t0,midplane=midplane)
    alpha = np.sqrt(-1.0/gi[0,0]);
    uu_[0] = gamma/alpha;
    uu_[1] = vel1 - alpha * gamma * gi[0,1] ##- uu[0] * dxbh_dt 
    uu_[2] = vel2 - alpha * gamma * gi[0,2] ##- uu[0] * dybh_dt
    uu_[3] = vel3 - alpha * gamma * gi[0,3] ##- uu[0] * dzbh_dt

    uu_ = np.array(uu_)

    ##check sign!!
    uu[0] = Lorentz * (uu_[0] - dxbh_dt * uu_[1] - dybh_dt * uu_[2] - dzbh_dt * uu_[3])
    uu[1] = -Lorentz * dxbh_dt * uu_[0] + (1.0 + (Lorentz-1.0)*nx_beta*nx_beta )*uu_[1] + \
            (Lorentz-1.0)*nx_beta*ny_beta*uu_[2]  + (Lorentz-1.0)*nx_beta*nz_beta*uu_[3]
    uu[2] = -Lorentz * dybh_dt * uu_[0] + (1.0 + (Lorentz-1.0)*ny_beta*ny_beta )*uu_[2] + \
            (Lorentz-1.0)*nx_beta*ny_beta*uu_[1]  + (Lorentz-1.0)*ny_beta*nz_beta*uu_[3]

    uu[3] = -Lorentz * dzbh_dt * uu_[0] + (1.0 + (Lorentz-1.0)*nz_beta*nz_beta )*uu_[3] + \
            (Lorentz-1.0)*nx_beta*nz_beta*uu_[1]  + (Lorentz-1.0)*ny_beta*nz_beta*uu_[2]

    uu = np.array(uu)


    ###now boost:

    uu_tmp = uu*1.0

    
    vx_tmp = vel1
    vy_tmp = vel2
    vz_tmp = vel3 

    #new coords in terms of new r,th,phi
    x = xi_prime * 1.0; ##r*np.cos(phi)*np.sin(theta) - aprime_*np.sin(phi)*np.sin(theta)
    y = yi_prime * 1.0; ## r*np.sin(phi)*np.sin(theta) + aprime_*np.cos(phi)*np.sin(theta)
    z = zi_prime * 1.0; ##r*np.cos(theta)

    if (MHD==True):
        B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
        Bcc1 = ds.r['Bcc1'][new_index]/B_unit
        Bcc2 = ds.r['Bcc2'][new_index]/B_unit
        Bcc3 = ds.r['Bcc3'][new_index]/B_unit

        # bsq = ds2.r['user_out_var5'][new_index]*2.0
        B_vec = np.zeros(uu.shape)
        B_vec[1] = Bcc1 
        B_vec[2] = Bcc2 
        B_vec[3] = Bcc3
        # cks_binary_metric(t,xi+xbh,yi+ybh,zi+zbh,a=a,aprime_=aprime,q=q,r_bh2=rbh2,t0=t0,midplane=midplane)
        for i in range(1,4):
          for mu in range(0,4):
            ##use unboosted uu first
            bu_[0] += g[i,mu]*uu_[mu]*B_vec[i]
        bu_[1] = 1.0/uu_[0] * (B_vec[1] + bu_[0]*uu_[1])  ##- bu[0] * dxbh_dt 
        bu_[2] = 1.0/uu_[0] * (B_vec[2] + bu_[0]*uu_[2]) ##- bu[0] * dybh_dt 
        bu_[3] = 1.0/uu_[0] * (B_vec[3] + bu_[0]*uu_[3]) ##- bu[0] * dzbh_dt 
        bu_ = np.array(bu_)

        bsq = 0
        for i in range(4):
          for j in range(4):
            bsq += g[i,j] * bu_[i] * bu_[j]
        if (uov==True): divb = ds2.r['user_out_var6'][new_index]*2.0

        ## now boost b

        bu[0] = Lorentz * (bu_[0] - dxbh_dt * bu_[1] - dybh_dt * bu_[2] - dzbh_dt * bu_[3])
        bu[1] = -Lorentz * dxbh_dt * bu_[0] + (1.0 + (Lorentz-1.0)*nx_beta*nx_beta )*bu_[1] + \
              (Lorentz-1.0)*nx_beta*ny_beta*bu_[2]  + (Lorentz-1.0)*nx_beta*nz_beta*bu_[3]
        bu[2] = -Lorentz * dybh_dt * bu_[0] + (1.0 + (Lorentz-1.0)*ny_beta*ny_beta )*bu_[2] + \
              (Lorentz-1.0)*nx_beta*ny_beta*bu_[1]  + (Lorentz-1.0)*ny_beta*nz_beta*bu_[3]

        bu[3] = -Lorentz * dzbh_dt * bu_[0] + (1.0 + (Lorentz-1.0)*nz_beta*nz_beta )*bu_[3] + \
              (Lorentz-1.0)*nx_beta*nz_beta*bu_[1]  + (Lorentz-1.0)*ny_beta*nz_beta*bu_[2]

        bu = np.array(bu)
        bu_tmp = bu* 1.0

    vel1 = vx_tmp*1.0
    vel2 = vy_tmp*1.0
    vel3 = vz_tmp*1.0
      # x_old = x \cos(\theta)\cos(\varphi) + y (-\sin(\varphi)) + z \sin(\theta)\cos(\varphi)\\
      # y_old = x \cos(\theta)\sin(\varphi) + y \cos(\varphi) + z \sin(\theta)\sin(\varphi) \\
      # z_old = x (-\sin(\theta)) + z \cos(\theta)
      # x_new = x_old * xhat_prime[0] + y_old * yhat_prime[0] + z_old * zhat_prime[0]
      # y_new = x_old * xhat_prime[1] + y_old * yhat_prime[1] + z_old * zhat_prime[1]
      # z_new = x_old * xhat_prime[2] + y_old * yhat_prime[2] + z_old * zhat_prime[2]
        # dxnew_dxold = np.cos(th) * np.cos(ph)
        # dxnew_dyold = - np.sin(ph)
        # dxnew_dzold = np.sin(th) * np.cos(ph)

        # dynew_dxold = np.cos(th) * np.sin(ph)
        # dynew_dyold = np.cos(ph)
        # dynew_dzold = np.sin(th)*np.sin(ph)

        # dznew_dxold = -np.sin(th)
        # dznew_dyold = 0.0
        # dznew_dzold = np.cos(th)
        # uu[1] = uu_tmp[1] * dxnew_dxold + uu_tmp[2] * dxnew_dyold + uu_tmp[3] * dxnew_dzold
        # uu[2] = uu_tmp[1] * dynew_dxold + uu_tmp[2] * dynew_dyold + uu_tmp[3] * dynew_dzold
        # uu[3] = uu_tmp[1] * dznew_dxold + uu_tmp[2] * dznew_dyold + uu_tmp[3] * dznew_dzold

    uu[1] = uu_tmp[1] * x_hat_prime[0] + uu_tmp[2] * y_hat_prime[0]  + uu_tmp[3] * z_hat_prime[0]
    uu[2] = uu_tmp[1] * x_hat_prime[1] + uu_tmp[2] * y_hat_prime[1]  + uu_tmp[3] * z_hat_prime[1]
    uu[3] = uu_tmp[1] * x_hat_prime[2] + uu_tmp[2] * y_hat_prime[2]  + uu_tmp[3] * z_hat_prime[2]

    if (MHD==True):
      bu[1] = bu_tmp[1] * x_hat_prime[0] + bu_tmp[2] * y_hat_prime[0]  + bu_tmp[3] * z_hat_prime[0]
      bu[2] = bu_tmp[1] * x_hat_prime[1] + bu_tmp[2] * y_hat_prime[1]  + bu_tmp[3] * z_hat_prime[1]
      bu[3] = bu_tmp[1] * x_hat_prime[2] + bu_tmp[2] * y_hat_prime[2]  + bu_tmp[3] * z_hat_prime[2]

      # bu[1] = bu_tmp[1] * dxnew_dxold + bu_tmp[2] * dxnew_dyold + bu_tmp[3] * dxnew_dzold
      # bu[2] = bu_tmp[1] * dynew_dxold + bu_tmp[2] * dynew_dyold + bu_tmp[3] * dynew_dzold
      # bu[3] = bu_tmp[1] * dznew_dxold + bu_tmp[2] * dznew_dyold + bu_tmp[3] * dznew_dzold

      Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
      Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
      Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])


    global nx,ny,nz 
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]

    if (double_precision==False):
      rho = np.array(np.float32(rho))
      press = np.array(np.float32(press))
      vel1 = np.array(np.float32(vel1))
      vel2 = np.array(np.float32(vel2))
      vel3 = np.array(np.float32(vel3))
      r = np.array(np.float32(r))
      theta = np.array(np.float32(theta))
      x = np.array(np.float32(x))
      y = np.array(np.float32(y))
      z = np.array(np.float32(z))
      phi = np.array(np.float32(phi))

      if (MHD==True):
        Bcc1 = np.array(np.float32(Bcc1))
        Bcc2 = np.array(np.float32(Bcc2))
        Bcc3 = np.array(np.float32(Bcc3))
      uu = np.array(np.float32(uu))
      if (MHD==True):
        bu = np.array(np.float32(bu))
        bsq = np.array(np.float32(bsq))

    dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz,"th_tilt":th,"phi_tilt":ph, "t":ds.current_time, "r": r,"th": theta, "ph": phi  }
    dic['uu'] = uu
    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        dic["bu"] = bu
        dic['bsq'] = bsq
    if (('athena_pp','r0') in ds.field_list ): 
      k_ent = ds.r['r0'][new_index]
    if (('athena_pp','r1') in ds.field_list ): 
      ke_ent = ds.r['r1'][new_index]
      dic["ke_ent"] = ke_ent
    if (('athena_pp','r2') in ds.field_list ): 
      ke_ent2 = ds.r['r2'][new_index]
      dic["ke_ent2"] = ke_ent2
    if (('athena_pp','r3') in ds.field_list ): 
      ke_ent3 = ds.r['r3'][new_index]
      dic["ke_ent3"] = ke_ent3




    t = ds.current_time

    np.savez(dump_name,**dic)


def rd_yt_convert_to_spherical_moving_bh(idump,MHD=False,dump_name=None,vbh=0.0,z0=-80.0,th=0,ph=0,
  method='nearest',fill_value = 0.0,gr=True,a=0.0, nr = 200, nth = 128,nphi=128,
  double_precision=False,rmin=None,rmax=None,aprime=0.0,t0=0.0,uov=False,midplane=False):
  if (dump_name is None): 
    dump_name = "dump_spher_2nd_black_hole_%d_th_%.2g_phi_%.2g.npz" %(idump,th,ph)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    yt_load(idump,gr=gr,a=a)

    q=1.0

    global r,phi,theta,xi,yi,zi
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t 
    global k_ent,ke_ent,ke_ent2,ke_ent3
    global divb
    #unit vectors for the new coordinate system in the frame of the old coordinates

    t = np.array(ds.current_time)

    xbh,ybh,zbh = bh2_pos_constant_velocity(t,z0=z0,vbh=vbh)
    # dxbh_dt,dybh_dt,dzbh_dt =  bh2_vel(t,rbh2,t0=t0,midplane=midplane)

    aprime_ = q * aprime

    Lorentz = np.sqrt(1.0/(1.0-vbh**2.0))

    rh2  =  ( q + np.sqrt(q**2.0-aprime_**2) )


    z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])   #r
    x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  #theta
    y_hat = np.array([-sin(ph),cos(ph),0])                        #phi

    #unit vectors for original coordinate system in the frame of the new coordinates
    #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
    x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #in terms of [theta_hat, phi_hat, r_hat] 
    y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    z_hat_prime = [-np.sin(th),0,np.cos(th)]

    index = np.arange(ds.r['density'].shape[0])
    if (rmin is None): rmin =rh2*0.8 
    if (rmax is None): rmax = rh2*100.0

    #faces
    theta = np.linspace(0.,np.pi,nth+1)
    phi = np.linspace(0,2.*np.pi,nphi+1)

    #centers
    r = np.logspace(log10(rmin),log10(rmax),nr)

    dth = np.diff(theta)[0]
    theta = (theta + dth/2.0)[:-1]
    dphi = np.diff(phi)[0]
    phi = (phi + dphi/2.0)[:-1]
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')



    ##new x,y,z coords in terms of new r,th,phi coords
    if (gr==True):
      xi_prime = r*np.cos(phi)*np.sin(theta) - aprime_*np.sin(phi)*np.sin(theta)
      yi_prime = r*np.sin(phi)*np.sin(theta) + aprime_*np.cos(phi)*np.sin(theta)
      zi_prime = r*np.cos(theta)
    else:
      xi_prime = r*np.cos(phi)*np.sin(theta) 
      yi_prime = r*np.sin(phi)*np.sin(theta) 
      zi_prime = r*np.cos(theta)

    #r_vec = x_new x_hat_new + y_new y_hat_new + z_new z_hat_new  
    # x_old = r_vec dot x_hat_old 
    #original x,y,z coords in terms of new coords
    if (gr==False):
      xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
      yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
      zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2]
    else: #assume cooordinate system aligned with spin of black hole
      xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
      yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
      zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2] 
    new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi+xbh,yi+ybh,(zi/Lorentz+zbh)),method = method,fill_value = fill_value)
    #new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi+xbh,yi+ybh,(zi+zbh)),method = method,fill_value = fill_value)


    from yt.units import pc, msun,kyr
    rho = ds.r['density'][new_index] * pc**3/msun
    press = ds.r['press'][new_index] * pc**3/msun * kyr**2/pc**2
    vel1 = ds.r['vel1'][new_index] * kyr/pc
    vel2 = ds.r['vel2'][new_index] * kyr/pc
    vel3 = ds.r['vel3'][new_index] * kyr/pc

    global uu,bu,bsq
    uu = [0,0,0,0]
    bu = [0,0,0,0]

    if (gr==True):
        # cks_metric(xi,yi,zi,a=aprime)
        cks_boosted_metric(t,xi+xbh,yi+ybh,zi+zbh,a=aprime,vbh=vbh,z0=z0)
        ##(t,x,y,z,aprime_=0.0,ONED=False,q=0,r_bh2=10.0,a=0.0,t0=0.0,midplane=False)
        tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
        gamma = np.sqrt(1.0 + tmp);

        # Calculate 4-velocity
        invert_metric(g)
        # cks_inverse_metric(xi,yi,zi,a=aprime)
        alpha = np.sqrt(-1.0/gi[0,0]);
        uu[0] = gamma/alpha;
        Lorentz = np.sqrt(1.0/(1.0-vbh**2.0))
        uu[1] = vel1 - alpha * gamma * gi[0,1] ## - uu[0] * 0.0
        uu[2] = vel2 - alpha * gamma * gi[0,2] ##- uu[0] * 0.0
        uu[3] = vel3 - alpha * gamma * gi[0,3] ##- uu[0] * vbh

        uu_tmp = np.array(uu)*1.0

        uu[0] = Lorentz * (uu_tmp[0]*1.0 - vbh * uu_tmp[3])
        uu[3] = Lorentz * (uu_tmp[3]*1.0 - vbh * uu_tmp[0])

      # uu[0] = ds2.r['user_out_var1'][new_index]
      # uu[1] = ds2.r['user_out_var2'][new_index]
      # uu[2] = ds2.r['user_out_var3'][new_index]
      # uu[3] = ds2.r['user_out_var4'][new_index]
        uu = np.array(uu)

        # uu_tmp = uu*1.0

    
    vx_tmp = vel1
    vy_tmp = vel2
    vz_tmp = vel3 


    #new coords in terms of new r,th,phi
    if (gr==True):
      x = r*np.cos(phi)*np.sin(theta) - aprime_*np.sin(phi)*np.sin(theta)
      y = r*np.sin(phi)*np.sin(theta) + aprime_*np.cos(phi)*np.sin(theta)
      z = r*np.cos(theta)
    else:
      x = r*np.cos(phi)*np.sin(theta) 
      y = r*np.sin(phi)*np.sin(theta) 
      z = r*np.cos(theta)

    if (MHD==True):
        B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
        Bcc1 = ds.r['Bcc1'][new_index]/B_unit
        Bcc2 = ds.r['Bcc2'][new_index]/B_unit
        Bcc3 = ds.r['Bcc3'][new_index]/B_unit

        if (gr==False):
          Bx_tmp = Bcc1
          By_tmp = Bcc2
          Bz_tmp = Bcc3
          Bcc1 = Bx_tmp*x_hat_prime[0] + By_tmp*y_hat_prime[0] + Bz_tmp*z_hat_prime[0]
          Bcc2 = Bx_tmp*x_hat_prime[1] + By_tmp*y_hat_prime[1] + Bz_tmp*z_hat_prime[1]   
          Bcc3 = Bx_tmp*x_hat_prime[2] + By_tmp*y_hat_prime[2] + Bz_tmp*z_hat_prime[2]

        if (gr==True):
          # bsq = ds2.r['user_out_var5'][new_index]*2.0
          B_vec = np.zeros(uu.shape)
          B_vec[1] = Bcc1 
          B_vec[2] = Bcc2 
          B_vec[3] = Bcc3
          # cks_metric(xi,yi,zi,a=aprime)
          cks_boosted_metric(t,xi+xbh,yi+ybh,zi+zbh,a=aprime,vbh=vbh,z0=z0)
          for i in range(1,4):
            for mu in range(0,4):
              bu[0] += g[i,mu]*uu_tmp[mu]*B_vec[i]
          bu[1] = 1.0/uu_tmp[0] * (B_vec[1] + bu[0]*uu_tmp[1]) 
          bu[2] = 1.0/uu_tmp[0] * (B_vec[2] + bu[0]*uu_tmp[2]) 
          bu[3] = 1.0/uu_tmp[0] * (B_vec[3] + bu[0]*uu_tmp[3]) 
          bu = np.array(bu)

          bu_tmp = bu* 1.0

          bu[0] = Lorentz * (bu_tmp[0]*1.0 - vbh * bu_tmp[3])
          bu[3] = Lorentz * (bu_tmp[3]*1.0 - vbh * bu_tmp[0])
          bu_tmp = bu* 1.0

          cks_metric(xi,yi,zi,0,0,aprime)


          bsq = 0
          for i in range(4):
            for j in range(4):
              bsq += g[i,j] * bu[i] * bu[j]
    #       if (uov==True): divb = ds2.r['user_out_var6'][new_index]*2.0

    uu_tmp = np.array(uu)*1.0

    if (gr==False):
        vel1 = vx_tmp*x_hat_prime[0] + vy_tmp*y_hat_prime[0] + vz_tmp*z_hat_prime[0]
        vel2 = vx_tmp*x_hat_prime[1] + vy_tmp*y_hat_prime[1] + vz_tmp*z_hat_prime[1]
        vel3 = vx_tmp*x_hat_prime[2] + vy_tmp*y_hat_prime[2] + vz_tmp*z_hat_prime[2]
    else:
        vel1 = vx_tmp*1.0
        vel2 = vy_tmp*1.0
        vel3 = vz_tmp*1.0
        # x_old = x \cos(\theta)\cos(\varphi) + y (-\sin(\varphi)) + z \sin(\theta)\cos(\varphi)\\
        # y_old = x \cos(\theta)\sin(\varphi) + y \cos(\varphi) + z \sin(\theta)\sin(\varphi) \\
        # z_old = x (-\sin(\theta)) + z \cos(\theta)
        # x_new = x_old * xhat_prime[0] + y_old * yhat_prime[0] + z_old * zhat_prime[0]
        # y_new = x_old * xhat_prime[1] + y_old * yhat_prime[1] + z_old * zhat_prime[1]
        # z_new = x_old * xhat_prime[2] + y_old * yhat_prime[2] + z_old * zhat_prime[2]
          # dxnew_dxold = np.cos(th) * np.cos(ph)
          # dxnew_dyold = - np.sin(ph)
          # dxnew_dzold = np.sin(th) * np.cos(ph)

          # dynew_dxold = np.cos(th) * np.sin(ph)
          # dynew_dyold = np.cos(ph)
          # dynew_dzold = np.sin(th)*np.sin(ph)

          # dznew_dxold = -np.sin(th)
          # dznew_dyold = 0.0
          # dznew_dzold = np.cos(th)
          # uu[1] = uu_tmp[1] * dxnew_dxold + uu_tmp[2] * dxnew_dyold + uu_tmp[3] * dxnew_dzold
          # uu[2] = uu_tmp[1] * dynew_dxold + uu_tmp[2] * dynew_dyold + uu_tmp[3] * dynew_dzold
          # uu[3] = uu_tmp[1] * dznew_dxold + uu_tmp[2] * dznew_dyold + uu_tmp[3] * dznew_dzold

        uu[1] = uu_tmp[1] * x_hat_prime[0] + uu_tmp[2] * y_hat_prime[0]  + uu_tmp[3] * z_hat_prime[0]
        uu[2] = uu_tmp[1] * x_hat_prime[1] + uu_tmp[2] * y_hat_prime[1]  + uu_tmp[3] * z_hat_prime[1]
        uu[3] = uu_tmp[1] * x_hat_prime[2] + uu_tmp[2] * y_hat_prime[2]  + uu_tmp[3] * z_hat_prime[2]

        if (MHD==True):
          bu[1] = bu_tmp[1] * x_hat_prime[0] + bu_tmp[2] * y_hat_prime[0]  + bu_tmp[3] * z_hat_prime[0]
          bu[2] = bu_tmp[1] * x_hat_prime[1] + bu_tmp[2] * y_hat_prime[1]  + bu_tmp[3] * z_hat_prime[1]
          bu[3] = bu_tmp[1] * x_hat_prime[2] + bu_tmp[2] * y_hat_prime[2]  + bu_tmp[3] * z_hat_prime[2]

          # bu[1] = bu_tmp[1] * dxnew_dxold + bu_tmp[2] * dxnew_dyold + bu_tmp[3] * dxnew_dzold
          # bu[2] = bu_tmp[1] * dynew_dxold + bu_tmp[2] * dynew_dyold + bu_tmp[3] * dynew_dzold
          # bu[3] = bu_tmp[1] * dznew_dxold + bu_tmp[2] * dznew_dyold + bu_tmp[3] * dznew_dzold

          Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
          Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
          Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])


    global nx,ny,nz 
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]

    if (double_precision==False):
      rho = np.array(np.float32(rho))
      press = np.array(np.float32(press))
      vel1 = np.array(np.float32(vel1))
      vel2 = np.array(np.float32(vel2))
      vel3 = np.array(np.float32(vel3))
      r = np.array(np.float32(r))
      theta = np.array(np.float32(theta))
      x = np.array(np.float32(x))
      y = np.array(np.float32(y))
      z = np.array(np.float32(z))
      phi = np.array(np.float32(phi))

      if (MHD==True):
        Bcc1 = np.array(np.float32(Bcc1))
        Bcc2 = np.array(np.float32(Bcc2))
        Bcc3 = np.array(np.float32(Bcc3))
      if (gr==True):
        uu = np.array(np.float32(uu))
        if (MHD==True):
          bu = np.array(np.float32(bu))
          bsq = np.array(np.float32(bsq))

    dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz,"th_tilt":th,"phi_tilt":ph, "t":ds.current_time, "r": r,"th": theta, "ph": phi  }
    if (gr==True):
      dic['uu'] = uu
    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        if (gr==True): dic["bu"] = bu
        if (gr==True): dic['bsq'] = bsq
    if (('athena_pp','r0') in ds.field_list ): 
      k_ent = ds.r['r0'][new_index]
    if (('athena_pp','r1') in ds.field_list ): 
      ke_ent = ds.r['r1'][new_index]
      dic["ke_ent"] = ke_ent
    if (('athena_pp','r2') in ds.field_list ): 
      ke_ent2 = ds.r['r2'][new_index]
      dic["ke_ent2"] = ke_ent2
    if (('athena_pp','r3') in ds.field_list ): 
      ke_ent3 = ds.r['r3'][new_index]
      dic["ke_ent3"] = ke_ent3


    global uov0,uov1,uov2,uov3,uov4,uov5,uov6,uov7,uov8,uov9,uov10
    if (uov==True):
      if (('athena_pp','user_out_var0') in ds2.field_list ): 
        uov0 = ds2.r['user_out_var0'][new_index]
        dic["uov0"] = np.array(np.float32(uov0))
      if (('athena_pp','user_out_var1') in ds2.field_list ): 
        uov1 = ds2.r['user_out_var1'][new_index]
        dic["uov1"] = np.array(np.float32(uov1))
      if (('athena_pp','user_out_var2') in ds2.field_list ): 
        uov2 = ds2.r['user_out_var2'][new_index]
        dic["uov2"] = np.array(np.float32(uov2))
      if (('athena_pp','user_out_var3') in ds2.field_list ): 
        uov3 = ds2.r['user_out_var3'][new_index]
        dic["uov3"] = np.array(np.float32(uov3))
      if (('athena_pp','user_out_var4') in ds2.field_list ): 
        uov4 = ds2.r['user_out_var4'][new_index]
        dic["uov4"] = np.array(np.float32(uov4))
      if (('athena_pp','user_out_var5') in ds2.field_list ): 
        uov5 = ds2.r['user_out_var5'][new_index]
        dic["uov5"] = np.array(np.float32(uov5))
      if (('athena_pp','user_out_var6') in ds2.field_list ): 
        uov6 = ds2.r['user_out_var6'][new_index]
        dic["uov6"] = np.array(np.float32(uov6))
      if (('athena_pp','user_out_var7') in ds2.field_list ): 
        uov7 = ds2.r['user_out_var7'][new_index]
        dic["uov7"] = np.array(np.float32(uov7))
      if (('athena_pp','user_out_var8') in ds2.field_list ): 
        uov8 = ds2.r['user_out_var8'][new_index]
        dic["uov8"] = np.array(np.float32(uov8))
      if (('athena_pp','user_out_var9') in ds2.field_list ): 
        uov9 = ds2.r['user_out_var9'][new_index]
        dic["uov9"] = np.array(np.float32(uov9))
      if (('athena_pp','user_out_var10') in ds2.field_list ): 
        uov10 = ds2.r['user_out_var10'][new_index]
        dic["uov10"] = np.array(np.float32(uov10))



    t = ds.current_time

    np.savez(dump_name,**dic)
  # def compute_det():

  #   g[0,0], g[0,1],g[0,2],g[0,3], g[1,1],g[1,2],g[1,3],g[2,2],g[2,3],g[3,3] = (4.00172, -1.47426, 4.73599, 0.373575, 1.43574, -1.40222, -0.115107, 5.51729, 0.379883, 1.04869)

  #   g[1,0] = g[0,1]
  #   g[2,0] = g[0,2]
  #   g[3,0] = g[0,3]
  #   g[3,2] = g[2,3]
  #   g[2,1] = g[1,2]
  #   g[3,1] = g[1,3]
  #   g(I00,i),g(I01,i),g(I02,i),g(I03,i),
  #           g(I11,i),g(I12,i),g(I13,i),g(I22,i),g(I23,i),g(I33,i)



def invert_metric(m):
  global gi
  gi = g * 1.0

  gi[0,0] = m[1,1]  * m[2,2] * m[3,3] - \
         m[1,1]  * m[2,3] * m[2,3] - \
         m[1,2]  * m[1,2]  * m[3,3] + \
         m[1,2]  * m[1,3]  * m[2,3] +\
         m[1,3] * m[1,2]  * m[2,3] - \
         m[1,3] * m[1,3]  * m[2,2];

  gi[0,1] = -m[0,1]  * m[2,2] * m[3,3] + \
          m[0,1]  * m[2,3] * m[2,3] + \
          m[0,2]  * m[1,2]  * m[3,3] - \
          m[0,2]  * m[1,3]  * m[2,3] - \
          m[0,3] * m[1,2]  * m[2,3] + \
          m[0,3] * m[1,3]  * m[2,2];

  gi[1,0] = gi[1,0]


  gi[0,2] = m[0,1]  * m[1,2] * m[3,3] - \
         m[0,1]  * m[2,3] * m[1,3] - \
         m[0,2]  * m[1,1] * m[3,3] + \
         m[0,2]  * m[1,3] * m[1,3] + \
         m[0,3] * m[1,1] * m[2,3] - \
         m[0,3] * m[1,3] * m[1,2];

  gi[2,0] = gi[0,2]


  gi[0,3] = -m[0,1]  * m[1,2] * m[2,3] + \
           m[0,1]  * m[2,2] * m[1,3] + \
           m[0,2]  * m[1,1] * m[2,3] - \
           m[0,2]  * m[1,2] * m[1,3] - \
           m[0,3] * m[1,1] * m[2,2] + \
           m[0,3] * m[1,2] * m[1,2];

  gi[3,0] = gi[0,3]


  gi[1,1] = m[0,0]  * m[2,2] * m[3,3] - \
         m[0,0]  * m[2,3] * m[2,3] - \
         m[0,2]  * m[0,2] * m[3,3] + \
         m[0,2]  * m[0,3] * m[2,3] + \
         m[0,3] * m[0,2] * m[2,3] - \
         m[0,3] * m[0,3] * m[2,2];

  gi[1,2] = -m[0,0]  * m[1,2] * m[3,3] + \
          m[0,0]  * m[2,3] * m[1,3] + \
          m[0,2]  * m[0,1] * m[3,3] - \
          m[0,2]  * m[0,3] * m[1,3] - \
          m[0,3] * m[0,1] * m[2,3] + \
          m[0,3] * m[0,3] * m[1,2];

  gi[2,1] = gi[1,2]

  gi[1,3] = m[0,0]  * m[1,2] * m[2,3] - \
          m[0,0]  * m[2,2] * m[1,3] - \
          m[0,2]  * m[0,1] * m[2,3] + \
          m[0,2]  * m[0,2] * m[1,3] + \
          m[0,3] * m[0,1] * m[2,2] - \
          m[0,3] * m[0,2] * m[1,2];
  gi[3,1] = gi[1,3]

  gi[2,2] = m[0,0]  * m[1,1] * m[3,3] - \
          m[0,0]  * m[1,3] * m[1,3] - \
          m[0,1]  * m[0,1] * m[3,3] + \
          m[0,1]  * m[0,3] * m[1,3] + \
          m[0,3] * m[0,1] * m[1,3] - \
          m[0,3] * m[0,3] * m[1,1];

  gi[2,3] = -m[0,0]  * m[1,1] * m[2,3] + \
           m[0,0]  * m[1,2] * m[1,3] + \
           m[0,1]  * m[0,1] * m[2,3] - \
           m[0,1]  * m[0,2] * m[1,3] - \
           m[0,3] * m[0,1] * m[1,2] + \
           m[0,3] * m[0,2] * m[1,1];

  gi[3,2] = gi[2,3]


  gi[3,3] = m[0,0] * m[1,1] * m[2,2] - \
          m[0,0] * m[1,2] * m[1,2] - \
          m[0,1] * m[0,1] * m[2,2] + \
          m[0,1] * m[0,2] * m[1,2] + \
          m[0,2] * m[0,1] * m[1,2] - \
          m[0,2] * m[0,2] * m[1,1];

  det = m[0,0] * gi[0,0] + m[0,1] * gi[0,1] + m[0,2] * gi[0,2] + m[0,3] * gi[0,3];

  det = 1.0 / det


  gi = gi * det

def get_Xprime_analytic(t_,r_bh=20.0,logrmax=4.0,oned=False):

  global t_tau_array,beta_t_tau_array,s_t_tau_array,X_array,Y_array,Z_array,X_approx_array,Y_approx_array,Z_approx_array
  global R_array,R_approx_array
  global X_approx_array_order_1,Y_approx_array_order_1,Z_approx_array_order_1
  global R_approx_array_order_1
  global nx,ny,nz, residual_array
  nx = 128
  ny = 64
  nz = 32

  if (oned==False): r_arr, th_arr,ph_arr = np.meshgrid(np.logspace(-1,logrmax,nx),np.linspace(0,np.pi,ny),np.linspace(0,2.0*np.pi,nz),indexing='ij')
  else: 
    ny=1
    nz=1
    r_arr, th_arr,ph_arr = np.meshgrid(np.logspace(-1,logrmax,nx),np.linspace(np.pi/2.0,np.pi,ny),np.linspace(np.pi/2.0,2.0*np.pi,nz),indexing='ij')
    # r_arr = np.logspace(-1,logrmax,nx)
    # th_arr = np.array([np.pi/2.0])
    # ph_arr = np.array([np.pi/2.0])



  v = 1.0/np.sqrt(r_bh);

  omega = v/r_bh;

  global r,th,ph
  r = r_arr * 1.0
  th = th_arr * 1.0
  ph = ph_arr * 1.0
  def s(t):
    return (r_bh * np.cos(omega*t),r_bh * np.sin(omega*t),0)
  def beta(t):
    return (-omega * r_bh * np.sin(omega*t),r_bh * omega * np.cos(omega*t),0)

  def a(t):
    return (-omega**2.0 * r_bh * np.cos(omega*t),-r_bh * omega**2.0 * np.sin(omega*t),0)
  def get_t_tau(t,x,y,z):
    def eqn_(t_tau):
      s_ = s(t_tau)
      beta_ = beta(t_tau)
      R = np.sqrt((x-s_[0])**2.0 + (y-s_[1]) **2.0 + (z-s_[2])**2.0)
      return (t-t_tau - beta_[0]*(x-s_[0]) - beta_[1]*(y-s_[1]) - beta_[2]*(z-s_[2])) #/R

    st = s(t)
    betat = beta(t)
    # betatmin = np.amin((beta(t)))  ## np.sqrt(beta(t)[0]**2.0 + beta(t)[1]**2.0 + beta(t)[2]**2.0)
    # betatmax = np.amax((beta(t)))
    # if (abs(betatmax)>abs(betatmin)): betat = betatmax
    # else: betat = betatmin
    #r_est = np.sqrt( (x-st[0])**2.0 + (y-st[1])**2.0  + (z-st[2])**2.0 )
    r_est = np.sqrt(x**2.0 + y**2.0 + z**2.0)

    est = t - betat[0]*(x-st[0]) - betat[1]*(y-st[1]) - betat[2]*(z-st[2])

    lower_limit = t - (omega*r_bh * (r_est + r_bh))*1.2
    upper_limit = t + (omega*r_bh * (r_est + r_bh))*1.2

    t_tau = scipy.optimize.bisect(eqn_,lower_limit,upper_limit)
    ##t_tau = t
    s_t_tau = s(t_tau)
    beta_t_tau = beta(t_tau)
    beta_mag_t_tau = np.sqrt(beta_t_tau[0]**2.0 + beta_t_tau[1]**2.0  + beta_t_tau[2]**2.0 )
    Gamma_t_tau = 1.0/np.sqrt(1.0 - beta_mag_t_tau**2.0)

    dir_x = beta_t_tau[0]/beta_mag_t_tau
    dir_y = beta_t_tau[1]/beta_mag_t_tau
    dir_z = beta_t_tau[2]/beta_mag_t_tau

    X = (1.0             + (1.0/Gamma_t_tau-1)*dir_x*dir_x ) * (x - s_t_tau[0]) + \
        (                  (1.0/Gamma_t_tau-1)*dir_y*dir_x ) * (y - s_t_tau[1]) + \
        (                  (1.0/Gamma_t_tau-1)*dir_z*dir_x ) * (z - s_t_tau[2])  ##- Gamma_t_tau * beta_t_tau[0] * (t-t_tau)

    Y = (                  (1.0/Gamma_t_tau-1)*dir_x*dir_y ) * (x - s_t_tau[0]) + \
        (1.0             + (1.0/Gamma_t_tau-1)*dir_y*dir_y ) * (y - s_t_tau[1]) + \
        (                  (1.0/Gamma_t_tau-1)*dir_z*dir_y ) * (z - s_t_tau[2]) ##- Gamma_t_tau * beta_t_tau[1] * (t-t_tau)

    Z = (                  (1.0/Gamma_t_tau-1)*dir_x*dir_z ) * (x - s_t_tau[0]) + \
        (                  (1.0/Gamma_t_tau-1)*dir_y*dir_z ) * (y - s_t_tau[1]) + \
        (1.0             + (1.0/Gamma_t_tau-1)*dir_z*dir_z ) * (z - s_t_tau[2]) ##- Gamma_t_tau * beta_t_tau[2] * (t-t_tau)

    R = np.sqrt(X**2+Y**2+Z**2)


    s_t = s(t)
    beta_t = beta(t)
    beta_mag_t = np.sqrt(beta_t[0]**2.0 + beta_t[1]**2.0  + beta_t[2]**2.0 )
    Gamma_t = 1.0/np.sqrt(1.0 - beta_mag_t**2.0)
    dir_x = beta_t[0]/beta_mag_t
    dir_y = beta_t[1]/beta_mag_t
    dir_z = beta_t[2]/beta_mag_t
    X_approx_order_1 = (1.0 + (Gamma_t-1)*dir_x*dir_x ) * (x - s_t[0]) + \
                       (      (Gamma_t-1)*dir_y*dir_x ) * (y - s_t[1]) + \
                       (      (Gamma_t-1)*dir_z*dir_x ) * (z - s_t[2]) 

    Y_approx_order_1 = (      (Gamma_t-1)*dir_x*dir_y ) * (x - s_t[0]) + \
                       (1.0 + (Gamma_t-1)*dir_y*dir_y ) * (y - s_t[1]) + \
                       (      (Gamma_t-1)*dir_z*dir_y ) * (z - s_t[2]) 

    Z_approx_order_1 = (      (Gamma_t-1)*dir_x*dir_z ) * (x - s_t[0]) + \
                       (      (Gamma_t-1)*dir_y*dir_z ) * (y - s_t[1]) + \
                       (1.0 + (Gamma_t-1)*dir_z*dir_z ) * (z - s_t[2]) 



    X_approx = (1.0      + (1.0/Gamma_t-1)*dir_x*dir_x ) * (x - s_t[0]) + \
        (                  (1.0/Gamma_t-1)*dir_y*dir_x ) * (y - s_t[1]) + \
        (                  (1.0/Gamma_t-1)*dir_z*dir_x ) * (z - s_t[2])  ##- Gamma_t_tau * beta_t_tau[0] * (t-t_tau)

    Y_approx = (           (1.0/Gamma_t-1)*dir_x*dir_y ) * (x - s_t[0]) + \
        (1.0             + (1.0/Gamma_t-1)*dir_y*dir_y ) * (y - s_t[1]) + \
        (                  (1.0/Gamma_t-1)*dir_z*dir_y ) * (z - s_t[2]) ##- Gamma_t_tau * beta_t_tau[1] * (t-t_tau)

    Z_approx = (           (1.0/Gamma_t-1)*dir_x*dir_z ) * (x - s_t[0]) + \
        (                  (1.0/Gamma_t-1)*dir_y*dir_z ) * (y - s_t[1]) + \
        (1.0             + (1.0/Gamma_t-1)*dir_z*dir_z ) * (z - s_t[2]) ##- Gamma_t_tau * beta_t_tau[2] * (t-t_tau)


    R_approx = np.sqrt(X_approx**2 + Y_approx**2.0 + Z_approx**2.0)
    R_approx_order_1 = np.sqrt(X_approx_order_1**2.0 + Y_approx_order_1**2.0 + Z_approx_order_1**2.0)

    residual = eqn_(t_tau)

    return  (t_tau,s_t_tau,beta_t_tau,X,Y,Z,X_approx,Y_approx, Z_approx,R,R_approx,X_approx_order_1,Y_approx_order_1,Z_approx_order_1,R_approx_order_1,residual)


  t_tau_array = np.zeros((nx,ny,nz))
  s_t_tau_array = np.zeros((3,nx,ny,nz))
  beta_t_tau_array = np.zeros((3,nx,ny,nz))
  X_array,Y_array,Z_array = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
  X_approx_array,Y_approx_array,Z_approx_array = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
  R_array = np.zeros((nx,ny,nz))
  R_approx_array = np.zeros((nx,ny,nz))
  R_approx_array_order_1 = np.zeros((nx,ny,nz))
  X_approx_array_order_1, Y_approx_array_order_1, Z_approx_array_order_1 = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
  residual_array = np.zeros((nx,ny,nz))
  for i in np.arange(nx):
    for j in np.arange(ny):
      for k in np.arange(nz):
        r_,th_,ph_ = r_arr[i,0,0],th_arr[0,j,0],ph_arr[0,0,k]
        x_ = r_*np.sin(th_)*np.cos(ph_) + s(t_)[0]
        y_ = r_*np.sin(th_)*np.sin(ph_) + s(t_)[1]
        z_ = r_*np.cos(th_) + s(t_)[2]


        (t_tau,s_t_tau,beta_t_tau,X,Y,Z,X_approx,Y_approx, Z_approx,R,R_approx,X_approx_order_1,Y_approx_order_1,Z_approx_order_1,R_approx_order_1,residual) = get_t_tau(t_,x_,y_,z_)

        t_tau_array[i,j,k] = t_tau
        s_t_tau_array[0,i,j,k] =s_t_tau[0]
        s_t_tau_array[1,i,j,k] =s_t_tau[1]
        s_t_tau_array[2,i,j,k] =s_t_tau[2]
        beta_t_tau_array[0,i,j,k] =beta_t_tau[0]
        beta_t_tau_array[1,i,j,k] =beta_t_tau[1]
        beta_t_tau_array[2,i,j,k] =beta_t_tau[2]
        X_array[i,j,k] =X
        Y_array[i,j,k] =Y
        Z_array[i,j,k] =Z
        X_approx_array[i,j,k] =X_approx
        Y_approx_array[i,j,k] =Y_approx
        Z_approx_array[i,j,k] =Z_approx
        X_approx_array_order_1[i,j,k] =X_approx_order_1
        Y_approx_array_order_1[i,j,k] =Y_approx_order_1
        Z_approx_array_order_1[i,j,k] =Z_approx_order_1
        R_array[i,j,k] =R
        R_approx_array[i,j,k] =R_approx
        R_approx_array_order_1[i,j,k] = R_approx_order_1
        residual_array[i,j,k] = residual

def save_dic_r_approx(filename):
  dic = {"r": r, "R":R_array,"R_approx":R_approx_array_order_1}

  np.savez(filename,**dic)

def rd_error_1d_metric():
  global res_array,err_array
  res_array = [32,64,128,256,512,1024]
  ##res_array = [128,256,512,1024]

  err_array = []

  for res in res_array:
    os.chdir("N_%d" %res)
    rdhdf5(0,coord='xy',uov=True)

    rho0 = rho*1.0
    rdhdf5(10,coord='xy',uov=True)

    err = abs(rho-rho0).sum()/(res*1.0)
    err_array.append(err)
    os.chdir("../")

  res_array = np.array(res_array)
  err_array = np.array(err_array)
  
def set_up_runs():
  res_array = [32, 64,128,256,512,1024]
  for res in res_array:
    os.system("mkdir -p N_%d" %res)
    os.chdir("N_%d" %res)
    # os.system("cp /global/scratch/users/smressle/star_cluster/test_current_amr/1d_metric_test/B_field/N_%d/athinput* ./" %res)
    os.system("cp /global/scratch/users/smressle/star_cluster/test_current_amr/1d_metric_test/for_paper/2nd_order/N_%d/athinput* ./" %res)
    os.system(" cp ../athena ./")
    os.system("sed -i 's_^cfl.*_cfl\_number = %g_' athinput.fm\_torus" %0.1)
    os.system("sed -i 's_^x1min.*_x1min = %g_' athinput.fm\_torus" %(-3.5))
    os.system("sed -i 's_^x1max.*_x1max = %g_' athinput.fm\_torus" %(3.5))
    # os.system("sed -i 's_^nx1.*_nx1 = %d_1' athinput.fm\_torus" %res)
    # os.system("sed -i 's_^nx1.*_nx1 = %d_2' athinput.fm\_torus" %32)
    os.chdir("..")

  for res in res_array:
    os.chdir("N_%d" %res)
    if (res<128): n_cores = 1
    elif (res==128): n_cores = 4
    elif (res==256): n_cores = 8
    else: n_cores = 16
    os.system("mpirun -np %d ./athena -i athinput.fm_torus" %n_cores)
    os.chdir("..")


def get_analytic_bondi_gr_single(r,gamma_adi = 5.0/3.0,r_crit = 8.0):
  
  def SQR(a): return a**2.0

  temp_min = 1.0e-2  ## lesser temperature root must be greater than this
  temp_max = 1.0e1 
  k_adi = 1.0
  
  n_adi = 1.0/(gamma_adi-1.0)
  u_crit_sq = 1.0/(2.0*r_crit);                                          ## (HSW 71)
  u_crit = -np.sqrt(u_crit_sq);
  t_crit = n_adi/(n_adi+1.0) * u_crit_sq/(1.0-(n_adi+3.0)*u_crit_sq);  ## (HSW 74)
  c1 = t_crit**n_adi * u_crit * SQR(r_crit);                      ## (HSW 68)
  c2 = SQR(1.0 + (n_adi+1.0) * t_crit) * (1.0 - 3.0/(2.0*r_crit));        ## (HSW 69)
  def TemperatureResidual(t,r):
    return SQR(1.0 + (n_adi+1.0) * t)* (1.0 - 2.0/r + SQR(c1) / (SQR(SQR(r)) * t**(2.0*n_adi))) - c2

  def TemperatureMin(r,t_min,t_max):
    ## Parameters
    ratio = 0.3819660112501051;  ## (3+\sqrt{5})/2
    max_iterations = 100;          ## maximum number of iterations

    ## Initialize values
    t_mid = t_min + ratio * (t_max - t_min);
    res_mid = TemperatureResidual(t_mid, r);

    ## Apply golden section method
    larger_to_right = True  ## flag indicating larger subinterval is on right
    for n in arange(0,max_iterations):
      if (res_mid < 0.0): return t_mid
      if (larger_to_right):
        t_new = t_mid + ratio * (t_max - t_mid)
        res_new = TemperatureResidual(t_new, r)
        if (res_new < res_mid):
          t_min = t_mid;
          t_mid = t_new;
          res_mid = res_new;
        else:
          t_max = t_new;
          larger_to_right = False;
      else:
        t_new = t_mid - ratio * (t_mid - t_min);
        res_new = TemperatureResidual(t_new, r);
        if (res_new < res_mid):
          t_max = t_mid;
          t_mid = t_new;
          res_mid = res_new;
        else:
          t_min = t_new;
          larger_to_right = True;
    return nan

  def TemperatureBisect(r, t_min, t_max):
      # Parameters
      max_iterations = 20
      tol_residual = 1.0e-6
      tol_temperature = 1.0e-6
      
      # Find initial residuals
      res_min = TemperatureResidual(t_min, r)
      res_max = TemperatureResidual(t_max, r)
      if np.fabs(res_min) < tol_residual:
          return t_min
      if np.fabs(res_max) < tol_residual:
          return t_max
      if (res_min < 0.0 and res_max < 0.0) or (res_min > 0.0 and res_max > 0.0):
          return float('nan')
      
      # Iterate to find root
      t_mid = None
      for i in arange(max_iterations):
          t_mid = (t_min + t_max) / 2.0
          if t_max - t_min < tol_temperature:
              return t_mid
          res_mid = TemperatureResidual(t_mid, r)
          if np.fabs(res_mid) < tol_residual:
              return t_mid
          if (res_mid < 0.0 and res_min < 0.0) or (res_mid > 0.0 and res_min > 0.0):
              t_min = t_mid
              res_min = res_mid
          else:
              t_max = t_mid
              res_max = res_mid
      return t_mid


  temp_neg_res = TemperatureMin(r, temp_min, temp_max);
  if (r <= r_crit):  ## use lesser of two roots
    temp = TemperatureBisect(r, temp_min, temp_neg_res)
  else:  ## user greater of two roots
    temp = TemperatureBisect(r, temp_neg_res, temp_max)
  

  # Calculate primitives
  rho = (temp/k_adi)**n_adi             # not same K as HSW
  pgas = temp * rho;
  ur = c1 / (SQR(r) * temp**n_adi);    ## (HSW 75)
  ut = np.sqrt(1.0/SQR(1.0-2.0/r) * SQR(ur) + 1.0/(1.0-2.0/r));

  return rho,pgas,ur,ut



def calc_error_bondi_boosted(i_dump,vbh=0.9,z0=-80):
  yt_load(i_dump,gr=True)

  x,y,z = np.array(ds.r['x']),np.array(ds.r['y']),np.array(ds.r['z'])

  z_bh = z0 + vbh * np.array(ds.current_time)
  Lorentz = np.sqrt(1.0/(1.0-vbh**2.0))

  xprime = x*1.0
  yprime = y*1.0
  zprime = Lorentz * (z - z_bh)

  rprime = np.sqrt( xprime**2.0 + yprime**2.0 + zprime**2.0 )

  in_domain = (rprime>3.0)*(rprime<10.0)

  x_prime_ = xprime[in_domain]

  # rho_in_domain = np.array(ds.r['rho'])[in_domain]

  err = ds2.r['user_out_var0'][in_domain].sum()/np.sum(in_domain)

  # rho_analytic = []
  # for rprime_ in rprime[in_domain]:
  #   rho_sol, pgas_sol,ur_sol,ut_sol = get_analytic_bondi_gr_single(rprime_,gamma_adi = 4.0/3.0,r_crit = 8.0)
  #   rho_analytic.append(rho_sol)
  # rho_analytic = np.array(rho_analytic)

  # err = np.sum(np.abs(rho_in_domain - rho_analytic))/np.sum(in_domain)

  return np.array(err)

def calc_area_contour_in_3D():
  sigma = bsq/rho 
  indices = np.where(np.isclose(sigma, 1.0,rtol=0.1))
  x_contour = x[indices]                              
  y_contour = y[indices]
  z_contour = z[indices]
  points = np.column_stack((x_contour, y_contour, z_contour))

  tri = Delaunay(points)

  def triangle_area(vertices):
      # Ensure vertices is a 2D array
      vertices = np.atleast_2d(vertices)
      
      # Calculate the area of a triangle in 3D space using its vertices
      a, b, c = vertices[0], vertices[1], vertices[2]
      return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

  triangle_areas = np.array([triangle_area(points[simplex]) for simplex in tri.simplices])
  total_area = np.sum(triangle_areas)


#####################################################################################################################################################
#####################################################################################################################################################


# Function to compute the slice of a 3D field along a plane defined by the direction (vx, vy) and the length L, using interpolation on the given field and its corresponding coordinates
def slice(field, vx, vy, L):
  v = np.array([vx, vy])
  v = v / np.linalg.norm(v)

  n = 128
  r_val = np.linspace(0, L, n)
  z_val = np.linspace(-L/2, L/2, n)
  r_grid, z_grid = np.meshgrid(r_val, z_val, indexing='ij')

  x = r_grid * v[0]
  y = r_grid * v[1]
  z = z_grid

  def to_index(coord):
    return (coord + L) * (n - 1) / (2 * L)

  x_idx = to_index(x)
  y_idx = to_index(y)
  z_idx = to_index(z)

  coords = np.array([x_idx, y_idx, z_idx])
  slice_data = map_coordinates(field, coords, order=1, mode='nearest')

  return slice_data, coords, r_grid, z_grid

# Function to project the 3D fields Fx, Fy, Fz onto a 2D plane defined by the direction (vx, vy) and the coordinates of the slice, using interpolation on the given fields and their corresponding coordinates
def project(Fx, Fy, Fz, coords, vx, vy):
  v = np.array([vx, vy])
  v = v / np.linalg.norm(v)
  F_e1 = Fx * v[0] + Fy * v[1]
  F1_slice = map_coordinates(F_e1, coords, order=1, mode='nearest')
  F2_slice = map_coordinates(Fz, coords, order=1, mode='nearest')
  return F1_slice, F2_slice

# Function to compute the profile of a field along a line defined by the direction (dx, dz) and length l, centered at (x0, z0), using interpolation on the given field and its corresponding x and z axes
def profile(field, dx, dz, l, x0, z0, x_axis, z_axis):
  if dz==0:
    vx = 0
    vz = 1
  elif dx==0:
    vx = 1
    vz = 0
  else:
    vx = dz / np.sqrt(dx**2 + dz**2)
    vz = - dx / np.sqrt(dx**2 + dz**2)
  s = np.linspace(-l/2, l/2, 100)
  x_l = x0 + s*vx
  z_l = z0 + s*vz
  points = np.stack([z_l, x_l], axis=-1)
  interp = RegularGridInterpolator((z_axis, x_axis), field.T, bounds_error=False, fill_value=np.nan)
  return interp(points), x_l, z_l, -(x_l-x0)*vx-(z_l-z0)*vz

# Function to compute the slice of a 3D field along a cylindrical surface defined by the radius r0, the z-axis, and the length L, using interpolation on the given field and its corresponding coordinates  
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
  return slice_data, coords, theta_grid, z_gri

# Function to compute the numerical derivative of a field f with respect to the coordinate specified by n (0 for time, 1 for x, 2 for y, 3 for z), using numpy's gradient function
def der(f,n,x,y,z):
  if (n==0):
      return 0
  elif (n==1):
      return np.gradient(f,x[:,0,0],axis=n-1, edge_order=2)
  elif (n==2):
      return np.gradient(f,y[0,:,0],axis=n-1, edge_order=2)
  elif (n==3):
      return np.gradient(f,z[0,0,:],axis=n-1, edge_order=2)

# Function to compute the Levi Civita symbol
def symbol(i,j,k,l):
  if ((i==j) or (i==k) or (l==i) or (k==j) or (l==j) or (l==k)):
      return 0
  else:
      return (j - i) * (k - i) * (l - i) * (k - j) * (l - j) * (l - k) / (np.abs(j - i) * np.abs(k - i) * np.abs(l - i) * np.abs(k - j) * np.abs(l - j) * np.abs(l - k))

# Function to compute the Faraday tensor F from the metric g, its inverse gu, the 4-velocity ud, and the magnetic field bd
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

# Function to compute the Christoffel symbols and the current density J from the metric g, its inverse gu, the Faraday tensor F, and its time derivative F_post
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
  return cur, np.einsum('mnxyz,mxyz,nxyz->xyz', cur, cur, gd)

# Function to clean up the mask by removing regions that are likely to be noise, based on the sigma and da values, and the distance from the origin
def cleanup(sigma, da, rr, rplus):
  sigma[rr < rplus] = sigma.max()
  da[rr < rplus] = da.min()
  mask = (sigma < 2) & (np.log10(da) > -1)
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

# Function to compute the skeleton of a mask and find the best path from the point closest to the origin to a terminal node, minimizing the average sigma along the path
def sheet(mask, sigma_slice, r_slice, z_slice):
  skel_img = skeletonize(mask).astype(bool)
  if np.count_nonzero(skel_img) < 2:
      return np.array([]), np.array([]), np.array([])
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

# Function to compute tangent vectors along a curve defined by (x, y) with a smoothing parameter b
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
  delx = x[1]-x[0]
  dely = y[1]-y[0]
  dx[0] = delx/np.sqrt(delx**2+dely**2)
  dy[0] = dely/np.sqrt(delx**2+dely**2)
  delx2 = x[-1]-x[-2]
  dely2 = y[-1]-y[-2]
  dx[-1] = delx2/np.sqrt(delx2**2+dely2**2)
  dy[-1] = dely2/np.sqrt(delx2**2+dely2**2)
  if (x.size>2):
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
  return dx,dy

# Function to compute the eruption data from the simulation runs
def eruption_data():
    data = {
        'standard': {
            'a9': {}
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
        data['standard']['a9']['rflux'].append((Phibh[:,ir2][peaksm[i]]-Phibh[:,ir2][troughsm[i]])/Phibh[:,ir2][peaksm[i]])
        data['standard']['a9']['dt'].append(t[troughsm[i]]-t[peaksm[i]])
        data['standard']['a9']['jetmax'].append(((rjet_max_p[peaksm[i]:troughsm[i]+1]+rjet_max_m[peaksm[i]:troughsm[i]+1])/2).max())
        data['standard']['a9']['njetmax'].append(njetm[peaksm[i]:troughsm[i]+1].max())
        data['standard']['a9']['phimax'].append(Phibh[:,ir2][peaksm[i]])
        if i==0:
            continue
        else:
            data['standard']['a9']['del_t'].append(t[peaksm[i]]-t[troughsm[i-1]])
    return data   

# Function to create histograms and scatter plots for the specified data, x and y variables, and state, with appropriate labels, bins, colors, and legends
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
      'beta6_a0': r'$\beta=10^6,a=0$',
      'beta6_a9': r'$\beta=10^6,a=0.9375$',
      'beta2_a9': r'$\beta=10^2,a=0.9375$',
      'beta2_a0': r'$\beta=10^2,a=0$',
      'a9': 'a=0.9375',
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
          ax[1,0].scatter(data[state][run][xdata], data[state][run][ydata], label=name[run])
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