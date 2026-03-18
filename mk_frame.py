# Python modules
import numpy as np
from numpy import *
import glob
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--a',
                    default='0',
                    help='spin of black hole for GR')

parser.add_argument('--gamma',
                    default='1.666667',
                    help='adiabatic index')


parser.add_argument('--th',
                    default='0',
                    help='theta coordinate of new axes')

parser.add_argument('--phi',
                    default='0',
                    help='phi coordinate of new axes')

parser.add_argument('--i_glob',
                    default='',
                    help='index of processor')
parser.add_argument('--n_processors',
                    default='',
                    help='number of processors')

parser.add_argument('--max_level',
                    default='0',
                    help='number of processors')

parser.add_argument('--rmin_rm',
                    default='2',
                    help='rmin for RM calc in rg')


parser.add_argument('--mtype',
                    default='',
                    choices =[
                    	"mk_frame_inner",
						"mk_frame_outer",
						"mk_frame_outer_slice",
						"mk_frame_outer_slice_mhd",
						"mk_frame_inner_slice_mhd",
						"mk_frame_inner_slice",
						"convert_dumps",
						"convert_dumps_mhd",
						"convert_dumps_disk_frame",
						"convert_dumps_disk_frame_mhd",
						"Lx_calc",
						"mk_3Dframe",
						"mk_3D_jet_frame",
						"mk_grmhdframe",
						"mk_grframe",
						"mk_grframe_cartesian",
                        "mk_angle",
                        "mk_frame_single",
                        "mk_frame_sheet",
                        "mk_frame_double",
                        "mk_frame_triple",
                        "mk_frame_triple_rot",
                        "mk_frame_total",
						"mk_frame_grmhd_restart_cartesian",
						"mk_grframe_magnetically_frustrated",
						"mk_1d",
						"mk_1d_cartesian",
						"mk_1d_cartesian_newtonian",
						"mk_1d_Be",
						"mk_1d_Be_smr",
						'mk_RM',
						'mk_RM_rel',
						'mk_RM_moving',
						"RM_movie",
						"mk_frame_outer_cold",
						"mk_frame_inner_cold",
						"mk_frame_disk",
						"mk_frame_L_aligned",
						"mk_frame_L_aligned_fieldlines",
						"mk_grframe_magnetically_frustrated_cartesian",
						"mk_frame_Be_star","mk_frame_Be_star_smr",
						"mk_ipole_input", 
						"run_ipole", "run_ipole_frequency",
						"mk_images", "mk_frame_binary","mk_1d_quantities_binary",
						"mk_1d_quantities_binary_spin_orbit",
						"mk_frame_boosted","mk_frame_bhl","mk_3D_jet_and_disk_frame",
						"mk_1d_cartesian_spin_orbit"],
                    help='type of analysis')

parser.add_argument('--i_start',
                    default='0',
                    help='index of first dump to process')
parser.add_argument('--i_end',
                    default='',
                    help='index of last dump to process')
parser.add_argument('--i_increment',
                    default='1',
                    help='spacing between dumps to process')
parser.add_argument('-mhd',
                    action='store_true',
                    default=False,
                    help='read in magnetic fields')

parser.add_argument('--sheet_type',
                    default='bphi',
                    help='algorithm for sheet detection')

parser.add_argument('--q',
                    default=0.0,
                    help='mass ratio for binary simulations')
parser.add_argument('--rbh2',
                    default=20.0,
                    help='distance of companion bh')
parser.add_argument('--t0',
                    default=1e4,
                    help='time when companion bh turns on')
parser.add_argument('--aprime',
                    default=0.0,
                    help='spin of companion bh in units of its mass')
parser.add_argument('-midplane',
                    action='store_true',
                    default=False,
                    help='secondary black hole is orbiting in midplane')
parser.add_argument('--inclination',
                    default=0.0,
                    help='secondary black hole orbital inclination')

args = vars(parser.parse_args())

if (args['mtype'] == ''):
	raise SystemExit('### No analysis selected...please specify using --mtype')
if (args['n_processors'] == ''):
	raise SystemExit('### Need to know number of processors...set using --n_processors')	
if (args['i_glob'] == ''):
	raise SystemExit('### Need to know index of processor...set using --i_glob')	

mhd_switch = args['mhd']

midplane = args['midplane']

a = np.double(args['a'])
aprime = np.double(args['aprime'])
q = np.double(args['q'])
inclination= np.double(args['inclination'])
t0 = np.double(args['t0'])
rbh2 = np.double(args['rbh2'])
gam = np.double(args['gamma'])
th_tilt = np.double(args['th'])
phi_tilt = np.double(args['phi'])
i_glob = int(args['i_glob'])
n_processors = int(args['n_processors'])
m_type = args['mtype']
i_increment = int(args['i_increment'])
i_start_glob = int(args['i_start'])
sheet_type = args['sheet_type']

max_level_ = int(args['max_level'])

rmin_rm = np.double(args["rmin_rm"])

n_freqs = 50  ## number of frequencies for RM runs



sys.path.append("/global/scratch/users/smressle/star_cluster/restart_grmhd/vis/python")
sys.path.append("/dodrio/scratch/projects/2022_057/SeanRessler/athenapp_1.1.1/vis/python")
sys.path.append("/fs/lustre/project/plasmaastro/sressler/athenapp_1.1.1/vis/python")

from athena_script import *
import athena_script as asc


Z_o_X_solar = 0.0177
Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
X_solar = 0.7491
Z_solar = 1.0-X_solar - Y_solar

muH_solar = 1./X_solar
Z = 3. * Z_solar
X = 0.
mue = 2. /(1.+X)
mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
mp_over_kev = 9.994827
keV_to_Kelvin = 1.16045e7
box_radius_inner = 0.05
box_radius_outer = 0.5
def set_limits(isslice = False):
	global den_min_inner,den_min_outer,den_max_inner,den_max_outer,T_min_inner,T_min_outer,T_max_outer,T_max_inner
	if (isslice==False):
		den_min_inner = -1
		den_max_inner = 1.5
		den_min_outer = -2
		den_max_outer = 1
		T_min_inner = -3
		T_max_inner =  0
		T_min_outer = -3
		T_max_outer = 0
	else:
		den_min_outer = -1.5 -0.5 +0.75
		den_max_outer= 0.5 +1.0 -0.25
		den_min_inner = den_min_outer + 1.5 +0.5 - 0.75
		den_max_inner = den_max_outer + 1.5 - 1.0 +0.25
		T_min_outer = -1.5- 1.0+1.0
		T_max_outer = -0.25 + 0.5-0.25
		T_min_inner = T_min_outer + 0.5
		T_max_inner = T_max_outer + 0.5

# box_radius_outer = 2e-3
# box_radius_inner = 2e-4

# den_min_inner = 2
# den_max_inner = 3.5
# den_min_outer = 1.
# den_max_outer = 3.
# T_min_inner = 1.5
# T_max_inner =  3.
# T_min_outer = 0.5
# T_max_outer = 2.5


def mk_frame(var_num,var_den=None,projection_axis = 2,file_suffix ='rho',cb_label = r"$\log_{10}\left(\rho\right)$",i_frame = 0,var_min = 0,var_max = 2.5,cmap = 'ocean' ,isslice=False,fieldlines=False,
	B1=None,B2=None,B3=None,box_radius = 1.0):
    plt.figure(1,figsize = (6,6))
    plt.clf()
    plt.style.use('dark_background')

    nx = var_num.shape[0]
    ny = var_num.shape[1]
    nz = var_num.shape[2]

    if (projection_axis ==2):
        x_plot = region['x'][:,:,nz//2]
        y_plot = region['y'][:,:,nz//2]
        proj_label = 'z'
        if (isslice ==True):
            var_num = var_num[:,:,nz//2]
            if (var_den != None): var_den = var_den[:,:,nz//2]

    elif (projection_axis==1):
        x_plot = region['z'][:,ny//2,:]
        y_plot = region['x'][:,ny//2,:]
        proj_label = 'y'
        if (isslice ==True):
            var_num = var_num[:,ny//2,:]
            if (var_den !=None): var_den = var_den[:,ny//2,:]
    else:
        x_plot = region['y'][nx//2,:,:]
        y_plot = region['z'][nx//2,:,:]
        proj_label = 'x'
        if (isslice ==True):
            var_num = var_num[nx//2,:,:]
            if (var_den != None): var_den = var_den[nx//2,:,:]

    if (var_den ==None):
        if (isslice==False):
            c = plt.contourf(x_plot,y_plot,np.log10(var_num.mean(projection_axis)),levels = np.linspace(var_min,var_max,200),cmap = cmap,extend = 'both')
        else:
            c = plt.contourf(x_plot,y_plot,np.log10(var_num),levels = np.linspace(var_min,var_max,200),cmap = cmap,extend = 'both')
    else:
        if (isslice==False):
            c = plt.contourf(x_plot,y_plot,np.log10(var_num.mean(projection_axis)/var_den.mean(projection_axis)),levels = np.linspace(var_min,var_max,200),
            cmap = cmap,extend = 'both')
        else:
            c = plt.contourf(x_plot,y_plot,np.log10(var_num/var_den),levels = np.linspace(var_min,var_max,200),
            cmap = cmap,extend = 'both')

    plt.xlim(box_radius,-box_radius)
    plt.ylim(-box_radius,box_radius)
    if (fieldlines==True and projection_axis==2): plt.streamplot(np.array(x_plot.transpose()),np.array(y_plot.transpose()),np.array(B1[:,:,nz//2].transpose()),np.array(B2[:,:,nz//2].transpose()),color = 'white')

    # if (projection_axis ==2):
    # 	plt.xlabel(r'$x$ (pc)',fontsize = 25)
    # 	plt.ylabel(r'$y$ (pc)',fontsize = 25)
    # elif (projection_axis==1):
    # 	plt.xlabel(r'$z$ (pc)',fontsize = 25)
    # 	plt.ylabel(r'$x$ (pc)',fontsize = 25)
    # else:
    # 	plt.xlabel(r'$y$ (pc)',fontsize = 25)
    # 	plt.ylabel(r'$z$ (pc)',fontsize = 25)

    # cb = plt.colorbar(c,ax=plt.gca())
    # cb.set_ticks(np.arange(var_min,var_max+.5,.5))
    # cb.set_label(cb_label,fontsize=25)

    # ax = plt.gca()
    # for label in ax.get_xticklabels() + ax.get_yticklabels()+cb.ax.get_yticklabels():
    #     label.set_fontsize(20)

    plt.axis('off')
    plt.tight_layout()

    os.system("mkdir -p frames")
    if (isslice == False):
        plt.savefig("frames/frame_%s_%s_%d.png" % (file_suffix,proj_label,i_frame))
    else: 
        plt.savefig("frames/frame_slice_%s_%s_%d.png" % (file_suffix,proj_label,i_frame))



def set_dump_range(ipole_files=False):
	global i_start,i_end
	# if len(sys.argv[2:])>=2 and sys.argv[2].isdigit() and sys.argv[3].isdigit():
	# 	i_glob = int(sys.argv[2])
	# 	n_processors = int(sys.argv[3])
	# 	print("i_glob, n_processors:",i_glob,n_processors)
	# else:
	# 	print("Syntax error")
	# 	exit()


	if (ipole_files==False): dump_list = glob.glob("*.athdf")
	else: dump_list = glob.glob("*freq_0*")
	dump_list.sort()
	n_dumps = len(dump_list)

	if (args['i_end']==''): i_end_glob = n_dumps-1
	else: i_end_glob = int(args['i_end'])

	n_dumps = len(arange(i_start_glob,i_end_glob+1,i_increment)) 


	i_0 = i_start_glob #int(dump_list[0][15:-6])

	n_dumps_per_processor = int ( np.round((n_dumps*1.)/(n_processors*1.)+0.5) )
	i_start = i_0 + i_glob*n_dumps_per_processor
	i_end = i_start + n_dumps_per_processor

	if (i_end>(i_0 + n_dumps-1)):
		i_end = i_0 + n_dumps -1 

	print("i_start, i_end",i_start,i_end)
def set_dump_range_gr():
	global i_start,i_end
	# if len(sys.argv[2:])>=2 and sys.argv[2].isdigit() and sys.argv[3].isdigit():
	#     i_glob = int(sys.argv[2])
	#     n_processors = int(sys.argv[3])
	# else:
	# 	print("Syntax error")
	# 	exit()

	dump_list = glob.glob("*.athdf")
	dump_list.sort()
	n_dumps = len(dump_list) ##//2

	i_0 = i_start_glob #int(dump_list[0][15:-6])

	if (args['i_end']==''): i_end_glob = n_dumps-1
	else: i_end_glob = int(args['i_end'])

	n_dumps = len(arange(i_start_glob,i_end_glob+1,i_increment)) 


	n_dumps_per_processor = int ( (n_dumps*1.)/(n_processors*1.)+0.5 )
	i_start = i_0 + i_glob*n_dumps_per_processor
	i_end = i_start + n_dumps_per_processor

	if (i_end>(n_dumps-1+i_0)):
		i_end = n_dumps -1 +i_0

def set_freq_range(): 
	global i_start,i_end
	n_freqs_per_processor = int ( (n_freqs*1.)/(n_processors*1.)+0.5 )
	i_start = 0 + i_glob*n_freqs_per_processor
	i_end = i_start + n_freqs_per_processor

	if (i_end>(n_freqs-1+0)):
		i_end = n_freqs -1 +0

def mk_frame_inner(isslice=False,iscold=False,mhd=False):
	set_dump_range()
	set_limits(isslice= isslice)
	global region

	for i_dump in range(i_start,i_end):
		fcheck = "frames/frame_T_inner_z_%d.png" %i_dump
		if (isslice==True): fcheck = "frames/frame_slice_T_inner_z_%d.png" %i_dump
		if not os.path.isfile(fcheck): #asc.rdhdf5(i_dump,ndim=3,block_level=6,x1min=-.05,x1max=.05,x2min=-.05,x2max=.05,x3min=-.05,x3max=.05)
			asc.yt_load(i_dump)
			region = asc.ds.r[(-box_radius_inner,'pc'):(box_radius_inner,'pc'):128j,(-box_radius_inner,'pc'):(box_radius_inner,'pc'):128j,
			(-box_radius_inner,'pc'):(box_radius_inner,'pc'):128j]
			T_kelvin = (region['press']/region['rho']*mu_highT*mp_over_kev*keV_to_Kelvin)
			#from athena_script import *
			for i_proj in [2]: #range(3):
				if (mhd==False): mk_frame(var_num = region['rho'],var_min=den_min_inner,var_max=den_max_inner,file_suffix="rho_inner",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,box_radius = box_radius_inner)
				else:  mk_frame(var_num = region['rho'],var_min=den_min_inner,var_max=den_max_inner,file_suffix="rho_inner",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=True, B1=region['Bcc1'],B2=region['Bcc2'],B3=region['Bcc3'],box_radius = box_radius_inner)
				mk_frame(var_num = region['press'],var_den = region['rho'],file_suffix="T_inner",cb_label = r"$\log_{10}\left(T\right)$",
					i_frame=i_dump,var_min=T_min_inner,var_max=T_max_inner,cmap = 'gist_heat',projection_axis = i_proj,isslice=isslice,box_radius = box_radius_inner)
				if (iscold==True): mk_frame(var_num = region['rho']*(T_kelvin<12.5e3),var_min =den_min_inner,var_max = den_max_inner,file_suffix ="rho_cold_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=False,box_radius = box_radius_inner)

def mk_frame_outer(isslice=False,mhd=False,iscold=False):
	set_dump_range()
	set_limits(isslice= isslice)
	global region
	for i_dump in range(i_start,i_end):
		if not os.path.isfile("frames/frame_T_outer_z_%d.png" %i_dump): #asc.rdhdf5(i_dump,ndim=3,block_level=3,x1min=-.5,x1max=.5,x2min=-.5,x2max=.5,x3min=-.5,x3max=.5)
			asc.yt_load(i_dump)
			region = asc.ds.r[(-box_radius_outer,'pc'):(box_radius_outer,'pc'):128j,(-box_radius_outer,'pc'):(box_radius_outer,'pc'):128j,
			(-box_radius_outer,'pc'):(box_radius_outer,'pc'):128j]
			T_kelvin = (region['press']/region['rho']*mu_highT*mp_over_kev*keV_to_Kelvin)
			#from athena_script import *
			for i_proj in [2]: #range(3):
			    if (mhd==False): mk_frame(var_num = region['rho'],var_min =den_min_outer,var_max = den_max_outer,file_suffix ="rho_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,box_radius = box_radius_outer)
			    else: mk_frame(var_num = region['rho'],var_min =den_min_outer,var_max = den_max_outer,file_suffix ="rho_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=True,B1=region['Bcc1'],B2=region['Bcc2'],B3=region['Bcc3'],box_radius = box_radius_outer)
			    if (iscold==True): mk_frame(var_num = region['rho']*(T_kelvin<12.5e3),var_min =den_min_outer,var_max = den_max_outer,file_suffix ="rho_cold_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=False,box_radius = box_radius_outer)
			    mk_frame(var_num = region['press'],var_den = region['rho'],file_suffix="T_outer",cb_label = r"$\log_{10}\left(T\right)$",
			        i_frame=i_dump,var_min=T_min_inner,var_max=T_max_outer,cmap = 'gist_heat',projection_axis = i_proj,isslice=isslice,box_radius = box_radius_outer)


def convert_dumps_to_spher(MHD=False):
	set_dump_range()
	omega_phi = None
	# if len(sys.argv)>6:
	# 	omega_phi = np.float(sys.argv[6])
	# else:
	# 	omega_phi = None
	for idump in range(i_start,i_end):
		asc.rd_yt_convert_to_spherical(idump,th=th_tilt,ph=phi_tilt,omega_phi = omega_phi,MHD=MHD)

def convert_dumps_disk_frame(mhd=False):
	set_dump_range()
	asc.set_constants()


	def get_l_angles_slice(idump,levels = 8):
		global th_l,phi_l
		asc.rd_hst('star_wind.hst',is_magnetic=mhd)
		L_tot = np.sqrt(asc.Lx_avg**2. + asc.Ly_avg**2. + asc.Lz_avg**2.) + 1e-15
		r_in = 2.*2./2.**levels/128.

		x_rat = (asc.Lx_avg/L_tot)[:,asc.r_to_ir(r_in*10):r_to_ir(0.8*asc.arc_secs)].mean(-1)[idump]
		y_rat = (asc.Ly_avg/L_tot)[:,asc.r_to_ir(r_in*10):r_to_ir(0.8*asc.arc_secs)].mean(-1)[idump]
		z_rat = (asc.Lz_avg/L_tot)[:,asc.r_to_ir(r_in*10):r_to_ir(0.8*asc.arc_secs)].mean(-1)[idump]
		th_l = np.arccos(z_rat)
		phi_l = np.arctan2(y_rat,x_rat)

	# def get_l_angles_slice(idump,levels = 8):
	# 	global th_l,phi_l
	# 	rd_hst('star_wind.hst',is_magnetic=mhd)
	# 	L_tot = np.sqrt(Lx_avg**2. + Ly_avg**2. + Lz_avg**2.) + 1e-15
	# 	r_in = 2.*2./2.**levels/128.

	# 	x_rat = (Lx_avg/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)[idump]
	# 	y_rat = (Ly_avg/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)[idump]
	# 	z_rat = (Lz_avg/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)[idump]
	# 	th_l = np.arccos(z_rat)
	# 	phi_l = np.arctan2(y_rat,x_rat)

	for idump in range(i_start,i_end):
		get_l_angles_slice(idump,levels = 8)
		dump_name="dump_spher_disk_frame_%04d.npz" %idump
		asc.rd_yt_convert_to_spherical(idump,th=th_l,ph=phi_l,MHD=mhd,dump_name=dump_name)
def merge_frames():
    n_frames = len(glob.glob("frame_rho_outer_x_*.png"))
    for iframe in range(n_frames):
        plt.figure(figsize=(8,2))
        plt.subplot(131)
        plt.imshow(plt.imread('frame_rho_outer_x_%d.png' %iframe))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(plt.imread('frame_rho_outer_y_%d.png' %iframe))
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(plt.imread('frame_rho_outer_z_%d.png' %iframe))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('frame_rho_outer_all_%d.png' %iframe,bbox_inches='tight')
        plt.close()

def calculate_L_X():
	set_dump_range()
	D_BH = 8.3e3 #in parsecs
	#tan(theta) ~ theta ~ x_pc /D_BH
	arc_secs = 4.84814e-6 * D_BH
	for i_dump in range(i_start,i_end):
		asc.yt_load(i_dump)
		L_x_1_5 = asc.get_Xray_Lum("Lam_spex_Z_solar_2_10_kev",1.5*arc_secs)
		L_x_10 = asc.get_Xray_Lum("Lam_spex_Z_solar_2_10_kev",10.0*arc_secs)
		dic = {"t": asc.ds.current_time, "Lx_1_5": L_x_1_5,"Lx_10": L_x_10}
		np.savez("Lx_%04d.npz" %i_dump,**dic)

def column_density():
	set_dump_range()
	global region
	box_radius = 1.0 #pc 
	for i_dump in range(i_start,i_end):
		fname = "npz_files/column_density_z_%d.npz" % (i_dump)
		if not os.path.isfile(fname):
			asc.yt_load(i_dump)
			Lz =(asc.ds.domain_right_edge-asc.ds.domain_left_edge)[2]  #pc 

			region = asc.ds.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
			(-Lz/2.0,'pc'):(Lz/2.0,'pc'):128j ]

			column_density = region['rho'].mean(-1)*Lz

			dic = {"column_density": column_density, "x": region['x'].mean(-1), "y": region['y'].mean(-1), "t":asc.ds.current_time}
			os.system("mkdir -p npz_files")
			np.savez(fname,**dic)

def Xray_image():
	set_dump_range()
	D_BH = 8.3e3 #in parsecs
	#tan(theta) ~ theta ~ x_pc /D_BH
	arc_secs = 4.84814e-6 * D_BH

def mk_frame_3D():
    set_dump_range()
    for i_dump in range(i_start,i_end):
        fname = "frames/3Dframe_smooth_%d.png" % (i_dump)
        if isfile(fname):
            continue
        else:
            asc.yt_load(i_dump)
            from yt.visualization.volume_rendering.api import Scene, VolumeSource 
            import numpy as np
            sc  = Scene()
            vol = VolumeSource(asc.ds, field="density")
            bounds = (1e-2, 10.**1.5)
            tf = yt.ColorTransferFunction(np.log10(bounds))
            def linramp(vals, minval, maxval):
                return (vals - vals.min())/(vals.max() - vals.min())
            #tf.add_layers(8, colormap='ocean')
            tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
            #tf.add_layers(8, colormap='ocean')
            tf.grey_opacity = False
            vol.transfer_function = tf
            vol.tfh.tf = tf
            vol.tfh.bounds = bounds
            vol.tfh.plot('transfer_function.png', profile_field="density")
            cam = sc.add_camera(asc.ds, lens_type='plane-parallel')
            cam.resolution = [512,512]
            # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
            # cam.switch_orientation(normal_vector=normal_vector,
            #                        north_vector=north_vector)
            cam.set_width(asc.ds.domain_width*0.25)

            cam.position = asc.ds.arr(np.array([0,0,-0.5]), 'code_length')
            normal_vector = [0,0,-1]  #camera to focus
            north_vector = [0,1,0]  #up direction
            cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
            sc.add_source(vol)
            sc.render()
            # sc.save('tmp2.png',sigma_clip = 6.0)
            # sc = yt.create_scene(asc.ds,lens_type = 'perspective')
            # sc.camera.zoom(2.0)
            # sc[0].tfh.set_bounds([1e-4,1e2])
            os.system("mkdir -p frames")
            sc.save(fname,sigma_clip = 6.0)

def mk_frame_3D_uniform_grid():
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
			fname_rho = "frames/3Dframe_uniform_grid_rho_%d.png" % (i_dump)
			fname_T = "frames/3Dframe_uniform_grid_T_%d.png" % (i_dump)
			if os.path.isfile(fname_rho):
				continue
			else:
				asc.yt_load(i_dump)
				box_radius = 1.0 
				region = asc.ds.r[(-box_radius,'pc'):(box_radius,'pc'):512j,
				(-box_radius,'pc'):(box_radius,'pc'):512j,
				(-box_radius,'pc'):(box_radius,'pc'):512j ]
				x,y,z = region['x'],region['y'],region['z']
				import numpy as np

				bbox = np.array([[-box_radius,box_radius],[-box_radius,box_radius],[-box_radius,box_radius]])
				rho = region['density']
				press = region['press']
				T  = press/rho *  mu_highT*mp_over_kev*keV_to_Kelvin
				data =  dict(density = (np.array(rho),"Msun/pc**3"),temperature = (np.array(T),"K"),x = (np.array(x),"pc"), y = (np.array(y),"pc"),z = (np.array(z),"pc"))
				ds = yt.load_uniform_grid(data,rho.shape,length_unit="pc",bbox=bbox)

				#phi = np.linspace(0,2*pi,100)
				#		for iphi in range(100):
				from yt.visualization.volume_rendering.api import Scene, VolumeSource
				sc  = Scene()
				vol = VolumeSource(ds, field="density")
				bound_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
				bound_max = ds.arr(10.**1.5,"Msun/pc**3.").in_cgs()
				tf_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
				tf_max = ds.arr(10.**0.5,"Msun/pc**3.").in_cgs()
				bounds = (bound_min, bound_max)

				tf = yt.ColorTransferFunction(np.log10(bounds))
				def linramp(vals, minval, maxval):
					return (vals - vals.min())/(vals.max() - vals.min())
				#tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
				tf.add_layers(8, colormap='ocean',mi = np.log10(tf_min),ma = np.log10(tf_max),col_bounds=([np.log10(tf_min),np.log10(tf_max)])) #,w = 0.01,alpha = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])  #ds_highcontrast
				tf.add_step(np.log10(tf_max*2),np.log10(bound_max), [0.5,0.5,0.5,1.0])
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
				normal_vector = [0,0,-1]  #camera to focus  #CHANGE WITH PHI
				north_vector = [0,1,0]  #up direction   
				cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
				sc.add_source(vol)
				sc.render()
				# sc.save('tmp2.png',sigma_clip = 6.0)
				# sc = yt.create_scene(asc.ds,lens_type = 'perspective')
				# sc.camera.zoom(2.0)
				# sc[0].tfh.set_bounds([1e-4,1e2])
				os.system("mkdir -p frames")
				sc.save(fname_rho,sigma_clip = 6.0)


				############  TEMPERATURE ##############
				sc  = Scene()
				vol = VolumeSource(ds, field="temperature")
				bound_min = ds.arr(1e5,"K").in_cgs()
				bound_max = ds.arr(1e9,"K").in_cgs()
				bounds = (bound_min, bound_max)

				tf = yt.ColorTransferFunction(np.log10(bounds))
				def linramp(vals, minval, maxval):
					return (vals - vals.min())/(vals.max() - vals.min())
				#tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
				tf.add_step(np.log10(3e7),np.log10(1e9),[1.0,0.0,0.0,0.5])
				tf.add_step(np.log10(1e6),np.log10(3e7),[0.5,0.0,0.5,0.2])
				tf.add_step(np.log10(1e5),np.log10(1e6),[0.0,0.0,1.0,1.0])
				tf.grey_opacity = False
				vol.transfer_function = tf
				vol.tfh.tf = tf
				vol.tfh.bounds = bounds
				vol.tfh.plot('transfer_function.png', profile_field="temperature")
				cam = sc.add_camera(ds, lens_type='perspective')
				cam.resolution = [512,512]
				# cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
				# cam.switch_orientation(normal_vector=normal_vector,
				#                        north_vector=north_vector)
				cam.set_width(ds.domain_width*0.25)

				#cam.position = ds.arr(np.array([0.5*np.sin(phi),,0.5*np.cos(phi)]),'code_length')
				cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')   #CHANGE WITH PHI
				normal_vector = [0,0,-1]  #camera to focus  #CHANGE WITH PHI
				north_vector = [0,1,0]  #up direction   
				cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
				sc.add_source(vol)
				sc.render()
				# sc.save('tmp2.png',sigma_clip = 6.0)
				# sc = yt.create_scene(asc.ds,lens_type = 'perspective')
				# sc.camera.zoom(2.0)
				# sc[0].tfh.set_bounds([1e-4,1e2])
				os.system("mkdir -p frames")
				sc.save(fname_T,sigma_clip = 6.0)


def mk_frame_3D_jet():
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)

	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_3D_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()

			res = 256
			box_radius = 500
			asc.yt_extract_box(i_dump,box_radius = box_radius,mhd=True,gr=True,a=a,res=res)

			camera_grid = asc.bsq/asc.rho #np.transpose(bsq/rho,axes=(0,2,1))


			def transferFunction_sigma(x):
				"""Transfer Function returns r,g,b,a values as a function of density x"""
				peak_1 = 1
				peak_2 = 0
				peak_3 = -2
				r_ = 1.0*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.5 )
				g = 1.0*np.exp( -(x - peak_1)**2/1.0 ) +  1.0*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.5 )
				b = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  1.0*np.exp( -(x - peak_3)**2/0.5 )
				a_ = 0.6*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) + 0.01*np.exp( -(x - peak_3)**2/0.5 )

				return r_,g,b,a_

			def transferFunction_rhor(x):
				"""Transfer Function returns r,g,b,a values as a function of density x"""
				peak_1 = 3
				peak_2 = 1
				peak_3 = -1
				r_ = 1.0*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.5 )
				g = 1.0*np.exp( -(x - peak_1)**2/1.0 ) +  1.0*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.5 )
				b = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  1.0*np.exp( -(x - peak_3)**2/0.5 )
				a_ = 0.6*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) + 0.01*np.exp( -(x - peak_3)**2/0.5 )

				return r_,g,b,a_


			# Do Volume Rendering
			image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

			for dataslice in camera_grid:
				r_,g,b,a_ = transferFunction_sigma(np.log10(dataslice))
				image[:,:,0] = a_*r_ + (1-a_)*image[:,:,0]
				image[:,:,1] = a_*g + (1-a_)*image[:,:,1]
				image[:,:,2] = a_*b + (1-a_)*image[:,:,2]

			image = np.clip(image,0.0,1.0)
			plt.imshow(np.transpose(image,axes=(1,0,2)))
			plt.axis("off")
			os.system("mkdir -p frames")
			plt.savefig(fname)

def mk_frame_3D_jet_and_disk():
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)

	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_3D_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()

			res = 256
			box_radius = 50
			asc.yt_extract_box(i_dump,box_radius = box_radius,mhd=True,gr=True,a=a,res=res)

			orbit_file = glob.glob("*orbit*.dat")[0]
			asc.rd_binary_orbits(orbit_file)
			t0=1e5
			asc.get_binary_quantities(np.array(asc.ds.current_time),t0)


			##need 2
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,ax=asc.a1x,ay=asc.a1y,az=asc.a1z,rmin=0.5,rmax=1e3)
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,ax=asc.a1x,ay=asc.a1y,az=asc.a1z,rmin=0.5,rmax=1e3)

			ax = asc.a1x 
			ay = asc.a1y 
			az = asc.a1z

			a_dot_x = ax * asc.x + ay * asc.y + az * asc.z

			gdet = np.sqrt( np.sin(asc.th)**2.0 * ( asc.r**2.0 + a_dot_x**2.0/asc.r**2.0)**2.0 )
			rho_ref =  asc.angle_average_npz(asc.rho,gr=True,gdet=gdet)
			r_ref = asc.r[:,0,0]


			asc.yt_extract_box(i_dump,box_radius = box_radius,mhd=True,gr=True,a=a,res=res)


			def r_func(x,y,z,ax,ay,az):
				def SQR(var):
					return var**2.0

				a = np.sqrt(ax**2.0+ay**2.0+az**2.0)
				a_dot_x = ax * x + ay * y + az * z;

				R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
				return np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a_dot_x) )  )/np.sqrt(2.0);


			asc.r = r_func(asc.x,asc.y,asc.z,ax,ay,az)

			from scipy.interpolate import interp1d
			interpolated_rho_1d = interp1d(r_ref, rho_ref, kind='linear', fill_value="extrapolate")

			# Get values of rho_1d at radial distances r_xyz
			rho_ref_3D = interpolated_rho_1d(asc.r)

			# Optionally, reshape rho_1d_at_xyz back to the shape of x, y, z
			rho_ref_3D = np.reshape(rho_ref_3D, asc.x.shape)

			camera_grid = asc.bsq/asc.rho #np.transpose(bsq/rho,axes=(0,2,1))
			camera_grid = asc.bsq/asc.rho #np.transpose(bsq/rho,axes=(0,2,1))
			camera_grid2 = asc.rho/rho_ref_3D

			def transferFunction_sigma(x):
				"""Transfer Function returns r,g,b,a values as a function of density x"""
				peak_1 = 1
				peak_2 = 0
				peak_3 = -100
				r_ = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  1.0*np.exp( -(x - peak_3)**2/2.0 )
				g =  1.0*np.exp( -(x - peak_1)**2/1.0 )  +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/2.0 )
				b =  0.1*np.exp( -(x - peak_1)**2/1.0 )  +  1.0*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/2.0 )
				a_ = (0.3*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) + 0.05*np.exp( -(x - peak_3)**2/2.0 ) )/16.0

				return r_,g,b,a_

			def transferFunction_rho(x):
				"""Transfer Function returns r,g,b,a values as a function of density x"""
				peak_1 = -10
				peak_2 = -10
				peak_3 = 0.5
				r_ = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  1.0*np.exp( -(x - peak_3)**2/0.1 )
				g = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 )  +  0.1*np.exp( -(x - peak_3)**2/0.1 )
				b = 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 )  +  0.1*np.exp( -(x - peak_3)**2/0.1 )
				a_ = ( 0.1*np.exp( -(x - peak_1)**2/1.0 ) +  0.1*np.exp( -(x - peak_2)**2/0.1 ) +  0.1*np.exp( -(x - peak_3)**2/0.1 ) )

				return r_,g,b,a_

			def alpha_blend(color1, color2):
				# Normalize alpha values
				A1 = color1[3] / 1.0
				A2 = color2[3] / 1.0

				# Calculate resulting alpha
				A_result = A1 + A2 * (1 - A1)

				# Calculate resulting RGB values
				R_result = (color1[0] * A1 + color2[0] * A2 * (1 - A1)) / A_result
				G_result = (color1[1] * A1 + color2[1] * A2 * (1 - A1)) / A_result
				B_result = (color1[2] * A1 + color2[2] * A2 * (1 - A1)) / A_result

				# # Round and clamp the resulting RGB values
				# R_result = round(max(0, min(1.0, R_result)))
				# G_result = round(max(0, min(1.0, G_result)))
				# B_result = round(max(0, min(1.0, B_result)))

				# Return the superimposed color
				return (R_result, G_result, B_result, (A_result * 1.0))
			# Do Volume Rendering
			image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))


			distance_from_secondary = np.sqrt( (asc.x-asc.x2)**2 + (asc.y-asc.y2)**2 + (asc.z-asc.z2)**2 )
			distance_from_primary = np.sqrt( (asc.x)**2 + (asc.y)**2 + (asc.z)**2 )
			for iy in np.arange(asc.ny):
				dataslice = camera_grid[:,iy,:]
				dataslice2 = camera_grid2[:,iy,:]
				r_,g,b,a_ = transferFunction_sigma(np.log10(dataslice))
				r_2,g2,b2,a_2 = transferFunction_rho(np.log10(dataslice2))

				index_inside_secondary = distance_from_secondary[:,iy,:]<2.0

				index_inside_primary = distance_from_primary[:,iy,:]<4.0

				(r_,g,b,a_) = alpha_blend([r_,g,b,a_],[r_2,g2,b2,a_2])

				r_[index_inside_secondary] = 1.0 + 0.0* a_[index_inside_secondary]
				g[index_inside_secondary] = 1.0 + 0.0* a_[index_inside_secondary]
				b[index_inside_secondary] = 0.0* a_[index_inside_secondary]
				a_[index_inside_secondary] = 1.0 + 0.0* a_[index_inside_secondary]

				r_[index_inside_primary] = 1.0 + 0.0* a_[index_inside_primary]
				g[index_inside_primary] = 1.0 + 0.0* a_[index_inside_primary]
				b[index_inside_primary] = 0.0* a_[index_inside_primary]
				a_[index_inside_primary] = 1.0 + 0.0* a_[index_inside_primary]

				# a_ += a_2
				# r_ += r_2 
				# g  += g2 
				# b  += b2
				image[:,:,0] = a_*r_ + (1-a_)*image[:,:,0]
				image[:,:,1] = a_*g + (1-a_)*image[:,:,1]
				image[:,:,2] = a_*b + (1-a_)*image[:,:,2]


			image = np.clip(image,0.0,1.0)
			plt.imshow(np.transpose(image,axes=(1,0,2)))
			plt.axis("off")
			os.system("mkdir -p frames")
			plt.savefig(fname)


def mk_sheet_angle(is_magnetic=True):
        set_dump_range_gr()
        print("Processing dumps ",i_start," to ",i_end)
        for i_dump in range(i_start,i_end):
            print('Processing frame %i'%i_dump)
            fname = "angles2.csv"
            a = 0.9375
            rplus = 1.+ np.sqrt(1-a**2)
            th = np.linspace(0.,2*np.pi,1000)
            xh = rplus*np.sin(th)
            yh = rplus*np.cos(th)
            asc.rd_1d_avg()
            ir = asc.r_to_ir(2)
            asc.yt_extract_box(i_dump=i_dump+1, box_radius=7.5, mhd=True, gr=True, a=0.9375)
            far_post = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
            asc.yt_extract_box(i_dump=i_dump, box_radius=7.5, mhd=True, gr=True, a=0.9375)
            far = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
            cur, cur_mag = J(asc.g,asc.gi,far,far_post,asc.x,asc.y,asc.z)
            gamma=5/3
            sigma = asc.bsq / (asc.rho + gamma/(gamma-1) * asc.press)
            temp = asc.press/asc.rho
            uu_ks = asc.cks_vec_to_ks(asc.uu,asc.x,asc.y,asc.z,0,0,a)
            bu_ks = asc.cks_vec_to_ks(asc.bu,asc.x,asc.y,asc.z,0,0,a)
            Bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
            zoom_bphi = zoom(Bphi, zoom=2, order=1)
            zoom_sigma = zoom(sigma, zoom=2, order=1)
            zoom_temp = zoom(temp, zoom=2, order=1)
            zoom_cur = zoom(cur_mag, zoom=2, order=1)
            vals = np.zeros((4,360))
            for j in range(360):
                temp_slice, coords, r_slice, z_slice = slice(zoom_temp, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                sigma_slice, coords, r_slice, z_slice = slice(zoom_sigma, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                bphi_slice, coords, r_slice, z_slice = slice(zoom_bphi, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                cur_slice, coords, r_slice, z_slice = slice(zoom_cur, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                rr = np.sqrt(r_slice**2+z_slice**2)
                maskbh = rr<rplus
                sigma_slice[maskbh] = sigma_slice.max()
                dbp_x, dbp_y = np.gradient(np.sign(bphi_slice))
                bp_gradmag = np.sqrt(dbp_x**2 + dbp_y**2)
                mask = cleanup(sigma_slice, bp_gradmag, rr, rplus)
                if np.any(mask):
                    t_m = temp_slice[mask]
                    dbp_x, dbp_y = np.gradient(bphi_slice)
                    bp_gradmag = np.sqrt(dbp_x**2 + dbp_y**2)
                    dbp = bp_gradmag[mask]
                    j_m = cur_slice[mask]
                    rr_m = rr[mask]
                    r_c = np.sum(j_m*rr_m)/np.sum(j_m)
                    vals[0,j] = t_m.max()
                    vals[1,j] = dbp.max()
                    vals[2,j] = j_m.max()
                    vals[3,j] = r_c
            import pandas as pd
            df = pd.read_csv(fname, delimiter=',', header=None)
            angles = df.to_numpy()
            for k in range(4):
                angles[i_dump-2807,k] = np.where(vals[k,:]==vals[k,:].max())[0][0]
            np.savetxt(fname, angles, delimiter=',')
                    

               
def mk_grframe_single(is_magnetic=True):
        set_dump_range_gr()
        print("Processing dumps ",i_start," to ",i_end)
        for i_dump in range(i_start,i_end):
                print("framing dump %d" %i_dump)
                fname = "frames_mayank/frame_%04d.png" % (i_dump)
                if os.path.isfile(fname):
                        dummy = i_dump
                else:
                        plt.figure(1)
                        plt.clf()
                        a = 0.9375
                        rplus = 1.+ np.sqrt(1.-a**2)
                        th = np.linspace(0.,2*np.pi,1000)
                        xh = rplus*np.sin(th)
                        yh = rplus*np.cos(th)
                        rerg = 1.+np.sqrt(1.-a**2*np.cos(th)**2)
                        xe = rerg*np.sin(th)
                        ye = rerg*np.cos(th)
                        plt.rcParams.update({"font.size": 14})
                        asc.rd_1d_avg()
                        asc.yt_extract_box(i_dump=i_dump, box_radius=15, mhd=True, gr=True, a=0.9375)
                        ir = asc.r_to_ir(2)
                        fig,ax = plt.subplots(2,1,figsize=(6,6.62), gridspec_kw={'height_ratios': [4, 1]})
                        f = ax[0].pcolormesh(asc.x[:,64,:], asc.z[:,64,:], np.log10(asc.rho[:,64,:]), cmap='plasma', vmin=-4, vmax=1)
                        ax[0].fill(xh,yh,'k')
                        ax[0].plot(xe,ye,'w')
                        ax[0].set_xlabel(r'$x$ ($r_G$)')
                        ax[0].set_ylabel(r'$z$ ($r_G$)')
                        ax[0].set_title(r'$t = %iM$'%(i_dump*10))
                        ax[1].plot(asc.t,asc.Phibh[:, ir],color='black')
                        ax[1].axvline(i_dump*10,color='black',ls='--')
                        ax[1].scatter(i_dump*10,asc.Phibh[i_dump-1,ir],color='red',alpha=0.5)
                        ax[1].set_xlabel('Time t (in M)')
                        ax[1].set_ylabel(r'$\Phi_{BH}$')
                        fig.colorbar(f,ax=ax[0],label=r'$\log_{10}(\rho)$')
                        fig.tight_layout()
                        os.system("mkdir -p frames_mayank")
                        plt.savefig(fname)


def mk_grframe_double(is_magnetic=True):
        set_dump_range_gr()
        print("Processing dumps ",i_start," to ",i_end)
        for i_dump in range(i_start,i_end):
                print("framing dump %d" %i_dump)
                fname = "frames_mayank/tracer_frame_%04d.png" % (i_dump)
                if os.path.isfile(fname):
                        dummy = i_dump
                else:
                        plt.figure(1)
                        plt.clf()
                        a = 0.9375
                        rplus = 1.+ np.sqrt(1.-a**2)
                        th = np.linspace(0.,2*np.pi,1000)
                        xh = rplus*np.sin(th)
                        yh = rplus*np.cos(th)
                        rerg = 1.+np.sqrt(1.-a**2*np.cos(th)**2)
                        xe = rerg*np.sin(th)
                        ye = rerg*np.cos(th)
                        asc.rd_1d_avg()
                        asc.yt_extract_box(i_dump=i_dump, box_radius=7.5, mhd=True, gr=True, a=0.9375)
                        sigma = asc.bsq / (asc.rho + asc.gamma/(asc.gamma-1) * asc.press)
                        tracer = asc.Bcc3**2/asc.bsq
                        zoom_x = zoom(asc.x, zoom=2, order=1)
                        zoom_y = zoom(asc.y, zoom=2, order=1)
                        #zoom_z = zoom(asc.z, zoom=2, order=1)
                        zoom_sigma = zoom(sigma, zoom=2, order=1)
                        zoom_bx = zoom(asc.Bcc1, zoom=2, order=1)
                        zoom_by = zoom(asc.Bcc2, zoom=2, order=1)
                        #zoom_bz = zoom(asc.Bcc3, zoom=2, order=1)
                        zoom_tracer = zoom(tracer, zoom=2, order=1)
                        ir = asc.r_to_ir(2)                   
                        fig = plt.figure(figsize=(9.1,6), tight_layout=True)
                        gs1 = grd.GridSpec(2,2,height_ratios=[2, 1])
                        gs1.update(wspace=0.4, right=0.92, hspace=0.4, left=0.05)
                        ax1 = fig.add_subplot(gs1[0,0])
                        ax2 = fig.add_subplot(gs1[0,1])
                        gs2 = grd.GridSpec(2,2,height_ratios=[2, 1])
                        ax3 = plt.subplot(gs2[1,:])
                        ax1.set_aspect('equal', adjustable='box')
                        ax2.set_aspect('equal', adjustable='box')
                        ax1.set_xlim(zoom_x.min(),zoom_x.max())
                        ax1.set_ylim(zoom_y.min(),zoom_y.max())
                        ax2.set_xlim(zoom_x.min(),zoom_x.max())
                        ax2.set_ylim(zoom_y.min(),zoom_y.max())
                        f = ax1.pcolormesh(zoom_x[:,:,128], zoom_y[:,:,128], np.log10(zoom_sigma[:,:,128]), cmap='plasma', vmin=-2, vmax=1)
                        fig.colorbar(f,ax=ax1,label=r'$\log_{10}(\sigma)$',fraction=0.046, pad=0.04)
                        ax1.streamplot(zoom_x[:,:,128].transpose(), zoom_y[:,:,128].transpose(), zoom_bx[:,:,128].transpose(), zoom_by[:,:,128].transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)                        
                        ax1.fill(xh,yh,'k')
                        ax1.set_xlabel(r'$x$ ($r_G$)')
                        ax1.set_ylabel(r'$y$ ($r_G$)')
                        g = ax2.pcolormesh(zoom_x[:,:,128], zoom_y[:,:,128], np.log10(zoom_tracer[:,:,128]), cmap='bwr', vmin=-1, vmax=0)
                        fig.colorbar(g,ax=ax2,label=r'$\log_{10}(\frac{B_z^2}{B^2})$',fraction=0.046, pad=0.04)
                        ax2.fill(xh,yh,'k')
                        ax2.set_xlabel(r'$x$ ($r_G$)')
                        ax2.set_ylabel(r'$y$ ($r_G$)')
                        fig.suptitle('t=%iM'%asc.t[i_dump])
                        ax3.plot(asc.t[1972:2120],asc.Phibh[1972:2120, ir],color='black')
                        ax3.axvline(asc.t[i_dump], color='black',ls='--')
                        ax3.scatter(asc.t[i_dump], asc.Phibh[i_dump,ir], color='red',alpha=0.5)
                        ax3.set_xlabel('Time t (in M)')
                        ax3.set_ylabel(r'$\Phi_{BH}$')
                        os.system("mkdir -p frames_mayank")
                        plt.savefig(fname)
                        

def mk_grframe_triple(is_magnetic=True):
        set_dump_range_gr()
        print("Processing dumps ",i_start," to ",i_end)
        for i_dump in range(i_start,i_end):
                print("Framing dump %d" %i_dump)
                fname = "frames_mayank/xz_sheet_frame_%04d.png" %i_dump
                if os.path.isfile(fname):
                        dummy = i_dump
                else:
                        plt.figure(1)
                        plt.clf()
                        a = 0.9375
                        rplus = 1.+ np.sqrt(1-a**2)
                        th = np.linspace(0.,2*np.pi,1000)
                        xh = rplus*np.sin(th)
                        yh = rplus*np.cos(th)
                        asc.rd_1d_avg()
                        ir = asc.r_to_ir(2)
                        # asc.yt_extract_box(i_dump=i_dump+1, box_radius=7.5, mhd=True, gr=True, a=0.9375)
                        # far_post = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
                        asc.yt_extract_box(i_dump=i_dump, box_radius=7.5, mhd=True, gr=True, a=0.9375)
                        # far = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
                        # cur, cur_mag = J(asc.g,asc.gi,far,far_post,asc.x,asc.y,asc.z)
                        gamma=5/3
                        sigma = asc.bsq / (asc.rho + gamma/(gamma-1) * asc.press)
                        # temp = asc.press/asc.rho
                        beta = 2*asc.press/asc.bsq
                        # R = np.sqrt(asc.x**2+asc.y**2+asc.z**2)
                        # r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*(a*asc.z)**2.0 ) )/np.sqrt(2.0)
                        uu_ks = asc.cks_vec_to_ks(asc.uu,asc.x,asc.y,asc.z,0,0,a)
                        bu_ks = asc.cks_vec_to_ks(asc.bu,asc.x,asc.y,asc.z,0,0,a)
                        Bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
                        zoom_bux = zoom(asc.bu[1,:,:,:], zoom=2, order=1)
                        zoom_bdx = zoom(asc.bd[1,:,:,:], zoom=2, order=1)
                        zoom_buy = zoom(asc.bu[2,:,:,:], zoom=2, order=1)
                        zoom_bdy = zoom(asc.bd[2,:,:,:], zoom=2, order=1)
                        zoom_buz = zoom(asc.bu[3,:,:,:], zoom=2, order=1)
                        zoom_bdz = zoom(asc.bd[3,:,:,:], zoom=2, order=1)
                        zoom_bphi = zoom(Bphi, zoom=2, order=1)
                        zoom_bsq = zoom(asc.bsq, zoom=2, order=1)
                        zoom_x = zoom(asc.x, zoom=2, order=1)
                        zoom_y = zoom(asc.y, zoom=2, order=1)
                        zoom_z = zoom(asc.z, zoom=2, order=1)
                        zoom_sigma = zoom(sigma, zoom=2, order=1)
                        # zoom_temp = zoom(temp, zoom=2, order=1)
                        zoom_beta = zoom(beta, zoom=2, order=1)
                        # zoom_cur = zoom(cur_mag, zoom=2, order=1)
                        rr = np.sqrt(zoom_x[:,128,:]**2+zoom_z[:,128,:]**2)
                        maskbh = rr<rplus
                        sigma_slice = zoom_sigma[:,128,:]
                        sigma_slice[maskbh] = sigma_slice.max()
                        dbp_x, dbp_y = np.gradient(np.sign(zoom_bphi[:,128,:]))
                        bp_gradmag = np.sqrt(dbp_x**2 + dbp_y**2)
                        bb = (zoom_bux[:,128,:]*zoom_bdx[:,128,:] + zoom_buz[:,128,:]*zoom_bdz[:,128,:])/zoom_bsq[:,128,:]
                        mask = cleanup(sigma_slice, bp_gradmag, bb, rr, rplus)
                        sk_r, sk_z, l = sheet(mask, zoom_x[:,128,:], zoom_z[:,128,:])
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
                        ax0.set_xlim(zoom_x.min(), zoom_x.max())
                        ax0.set_ylim(zoom_y.min(), zoom_y.max())
                        ax1.set_xlim(zoom_x.min(), zoom_x.max())
                        ax1.set_ylim(zoom_z.min(), zoom_z.max())
                        ax2.set_xlim(zoom_x.min(), zoom_x.max())
                        ax2.set_ylim(zoom_z.min(), zoom_z.max())
                        h = ax0.pcolormesh(zoom_x[:,:,128], zoom_y[:,:,128], np.log10(zoom_beta[:,:,128]), cmap='plasma', vmin=-1, vmax=1)
                        fig.colorbar(h,ax=ax0,label=r'$\log_{10}(\beta)$',fraction=0.046, pad=0.04)
                        ax0.streamplot(zoom_x[:,:,128].transpose(), zoom_y[:,:,128].transpose(), zoom_bx[:,:,128].transpose(), zoom_by[:,:,128].transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
                        ax0.fill(xh,yh,'k')
                        ax0.set_xlabel(r'$x$ ($r_G$)')
                        ax0.set_ylabel(r'$y$ ($r_G$)')
                        f = ax1.pcolormesh(zoom_x[:,128,:], zoom_z[:,128,:], zoom_bphi[:,128,:], cmap='bwr', vmin=-0.2, vmax=0.2)
                        fig.colorbar(f,ax=ax1,label=r'$B_\phi$',fraction=0.046, pad=0.04)
                        ax1.streamplot(zoom_x[:,128,:].transpose(), zoom_z[:,128,:].transpose(), zoom_bx[:,128,:].transpose(), zoom_bz[:,128,:].transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
                        ax1.contour(zoom_x[:,128,:], zoom_z[:,128,:], sigma_slice<1, levels=[0.5], colors='magenta', linewidths=1.5)
                        ax1.fill(xh,yh,'k')
                        ax1.set_xlabel(r'$x$ ($r_G$)')
                        ax1.set_ylabel(r'$z$ ($r_G$)')
                        g = ax2.pcolormesh(zoom_x[:,128,:], zoom_z[:,128,:], np.log10(zoom_sigma[:,128,:]), cmap='plasma', vmin=-2, vmax=1)
                        fig.colorbar(g,ax=ax2,label=r'$\log_{10}(\sigma)$',fraction=0.046, pad=0.04)
                        ax2.contour(zoom_x[:,128,:], zoom_z[:,128,:], sigma_slice<1, levels=[0.5], colors='black', linewidths=1.5)
                        ax2.scatter(sk_r, sk_z, s=1, color='black', alpha=0.5)
                        ax2.fill(xh,yh,'k')
                        ax2.set_xlabel(r'$x$ ($r_G$)')
                        ax2.set_ylabel(r'$z$ ($r_G$)')
                        fig.suptitle('t=%iM'%asc.t[i_dump])
                        ax3.plot(asc.t[2807:3191], asc.Phibh[2807:3191,ir], color='black')
                        ax3.axvline(asc.t[i_dump], color='black', ls='--')
                        ax3.scatter(asc.t[i_dump], asc.Phibh[i_dump,ir], color='red',alpha=0.5)
                        ax3.set_xlabel('Time t (in M)')
                        ax3.set_ylabel(r'$\Phi_\text{BH}$')
                        os.system("mkdir -p frames_mayank")
                        plt.savefig(fname)

def mk_grframe_triple_rot(is_magnetic=True):
	set_dump_range_gr()
	print("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print("Framing dump %d" %i_dump)
		fname = "frames_mayank/rotated_sheet_frame_%04d.png" %i_dump
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			a = 0
			rplus = 1.+ np.sqrt(1-a**2)
			th = np.linspace(0.,2*np.pi,1000)
			xh = rplus*np.sin(th)
			yh = rplus*np.cos(th)
			asc.rd_1d_avg()
			ir = asc.r_to_ir(2)
			# asc.yt_extract_box(i_dump=i_dump+1, box_radius=7.5, mhd=True, gr=True, a=0.9375)
			# far_post = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
			asc.yt_extract_box_rotated(i_dump=i_dump, box_radius=7.5, mhd=True, gr=True, a=a, th_tilt=np.pi/2, phi_tilt=np.pi/2)
			# far = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
			# cur, cur_mag = J(asc.g,asc.gi,far,far_post,asc.x,asc.y,asc.z)
			gamma=5/3
			sigma = asc.bsq / (asc.rho + gamma/(gamma-1) * asc.press)
			temp = asc.press/asc.rho
			# beta = 2*asc.press/asc.bsq
			# R = np.sqrt(asc.x**2+asc.y**2+asc.z**2)
			# r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*(a*asc.z)**2.0 ) )/np.sqrt(2.0)
			uu_ks = asc.cks_vec_to_ks(asc.uu,asc.x,asc.y,asc.z,0,0,a)
			vr = uu_ks[1]/uu_ks[0]
			bu_ks = asc.cks_vec_to_ks(asc.bu,asc.x,asc.y,asc.z,0,0,a)
			bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
			rr = np.sqrt(asc.x[:,64,:]**2+asc.z[:,64,:]**2)
			maskbh = rr<rplus
			sigma_slice = sigma[:,64,:]
			sigma_slice[maskbh] = sigma_slice.max()
			ang = np.arctan2(asc.Bcc3[:,64,:], asc.Bcc1[:,64,:])
			da_x, da_y = np.gradient(ang)
			a_gradmag = np.sqrt(da_x**2+da_y**2)
			mask = cleanup(sigma_slice, a_gradmag, rr, rplus)
			sk_r, sk_z, l = sheet(mask, sigma_slice, asc.x[:,64,:], asc.z[:,64,:])
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
			ax0.set_xlim(asc.x.min(), asc.x.max())
			ax0.set_ylim(asc.y.min(), asc.y.max())
			ax1.set_xlim(asc.x.min(), asc.x.max())
			ax1.set_ylim(asc.z.min(), asc.z.max())
			ax2.set_xlim(asc.x.min(), asc.x.max())
			ax2.set_ylim(asc.z.min(), asc.z.max())
			h = ax0.pcolormesh(asc.x[:,:,64], asc.y[:,:,64], vr[:,:,64], cmap='bwr', vmin=-0.2, vmax=0.2)
			fig.colorbar(h,ax=ax0,label=r'$v^r$',fraction=0.046, pad=0.04)
			ax0.streamplot(asc.x[:,:,64].transpose(), asc.y[:,:,64].transpose(), asc.Bcc1[:,:,64].transpose(), asc.Bcc2[:,:,64].transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
			ax0.fill(xh,yh,'k', zorder=10)
			ax0.set_xlabel(r'$x$ ($r_G$)')
			ax0.set_ylabel(r'$y$ ($r_G$)')
			f = ax1.pcolormesh(asc.x[:,64,:], asc.z[:,64,:], np.log10(sigma[:,64,:]), cmap='plasma', vmin=-1, vmax=2)
			fig.colorbar(f,ax=ax1,label=r'$\log_{10}(\sigma)$',fraction=0.046, pad=0.04)
			ax1.streamplot(asc.x[:,64,:].transpose(), asc.z[:,64,:].transpose(), asc.Bcc1[:,64,:].transpose(), asc.Bcc3[:,64,:].transpose(), color='gray', linewidth=0.5, density=2, arrowsize=0.5)
			ax1.contour(asc.x[:,64,:], asc.z[:,64,:], sigma[:,64,:]<1, levels=[0.5], colors='cyan', linewidths=1)
			ax1.contour(asc.x[:,64,:], asc.z[:,64,:], sigma[:,64,:]<10, levels=[0.5], colors='green', linewidths=1)
			ax1.scatter(sk_r, sk_z, s=1, color='black', alpha=0.5)
			ax1.fill(xh,yh,'k', zorder=10)
			ax1.set_xlabel(r'$x$ ($r_G$)')
			ax1.set_ylabel(r'$z$ ($r_G$)')
			g = ax2.pcolormesh(asc.x[:,64,:], asc.z[:,64,:], np.log10(temp[:,64,:]), cmap='viridis', vmin=-1, vmax=0)
			fig.colorbar(g,ax=ax2,label=r'$\log_{10}(T)$',fraction=0.046, pad=0.04)
			ax2.streamplot(asc.x[:,64,:].transpose(), asc.z[:,64,:].transpose(), asc.Bcc1[:,64,:].transpose(), asc.Bcc3[:,64,:].transpose(), color='gray', linewidth=0.5, density=2, arrowsize=0.5)
			ax2.fill(xh,yh,'k', zorder=10)
			ax2.set_xlabel(r'$x$ ($r_G$)')
			ax2.set_ylabel(r'$z$ ($r_G$)')
			fig.suptitle('t=%iM'%asc.t[i_dump])
			ax3.plot(asc.t, asc.Phibh[:,ir], color='black')
			ax3.axvline(asc.t[i_dump], color='black', ls='--')
			ax3.scatter(asc.t[i_dump], asc.Phibh[i_dump,ir], color='red',alpha=0.5)
			ax3.set_xlabel('Time t (in M)')
			ax3.set_ylabel(r'$\Phi_\text{BH}$')
			os.system("mkdir -p frames_mayank")
			plt.savefig(fname)

def mk_grframe_sheet(is_magnetic=True):
        set_dump_range_gr()
        print("Processing dumps ",i_start," to ",i_end)
        for i_dump in range(i_start,i_end):
                print("Framing dump %d" %i_dump)
                fname = "frames_mayank/%s_sheet_frame_%04d.png" % (sheet_type, i_dump)
                if os.path.isfile(fname):
                        dummy = i_dump
                else:
                        plt.figure(1)
                        plt.clf()
                        a = 0.9375
                        rplus = 1.+ np.sqrt(1-a**2)
                        th = np.linspace(0.,2*np.pi,1000)
                        xh = rplus*np.sin(th)
                        yh = rplus*np.cos(th)
                        asc.rd_1d_avg()
                        ir = asc.r_to_ir(2)
                        # asc.yt_extract_box(i_dump=i_dump+1, box_radius=7.5, mhd=True, gr=True, a=0.9375)
                        # far_post = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
                        asc.yt_extract_box(i_dump=i_dump, box_radius=7.5, mhd=True, gr=True, a=0.9375)
                        # far = F(asc.g, asc.Lower(asc.uu,asc.g), asc.bd)
                        # cur, cur_mag = J(asc.g,asc.gi,far,far_post,asc.x,asc.y,asc.z)
                        gamma=5/3
                        sigma = asc.bsq / (asc.rho + gamma/(gamma-1) * asc.press)
                        temp = asc.press/asc.rho
                        beta = 2*asc.press/asc.bsq
                        R = np.sqrt(asc.x**2+asc.y**2+asc.z**2)
                        r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*(a*asc.z)**2.0 ) )/np.sqrt(2.0)
                        uu_ks = asc.cks_vec_to_ks(asc.uu,asc.x,asc.y,asc.z,0,0,a)
                        bu_ks = asc.cks_vec_to_ks(asc.bu,asc.x,asc.y,asc.z,0,0,a)
                        Bphi = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])
                        zoom_bx = zoom(asc.Bcc1, zoom=2, order=1)
                        zoom_by = zoom(asc.Bcc2, zoom=2, order=1)
                        zoom_bz = zoom(asc.Bcc3, zoom=2, order=1)
                        zoom_bphi = zoom(Bphi, zoom=2, order=1)
                        zoom_sigma = zoom(sigma, zoom=2, order=1)
                        zoom_beta = zoom(beta, zoom=2, order=1)
                        zoom_temp = zoom(temp, zoom=2, order=1)
                        # zoom_cur = zoom(cur_mag, zoom=2, order=1)
                        zoom_x = zoom(asc.x, zoom=2, order=1)
                        zoom_y = zoom(asc.y, zoom=2, order=1)
                        # vals = np.zeros((4,360))
                        temp_vals = []
                        for j in range(360):
                            temp_slice, coords, r_slice, z_slice = slice(zoom_temp, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                            sigma_slice, coords, r_slice, z_slice = slice(zoom_sigma, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                            # cur_slice, coords, r_slice, z_slice = slice(zoom_cur, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                            bphi_slice, coords, r_slice, z_slice = slice(zoom_bphi, np.cos(np.pi*j/180), np.sin(np.pi*j/180), 7.5)
                            rr = np.sqrt(r_slice**2+z_slice**2)
                            maskbh = rr < rplus
                            sigma_slice[maskbh] = sigma_slice.max()
                            dbp_x, dbp_y = np.gradient(np.sign(bphi_slice))
                            bp_gradmag = np.sqrt(dbp_x**2 + dbp_y**2)
                            mask = cleanup(sigma_slice, bp_gradmag, rr, rplus)
                            if np.any(mask):
                                t = temp_slice[mask]
                                # j_m = cur_slice[mask]
                                # r_m = r_slice[mask]
                                # z_m = z_slice[mask]
                                # r_centroid = np.sum(j_m * r_m) / np.sum(j_m)
                                # z_centroid = np.sum(j_m * z_m) / np.sum(j_m)
                                # vals[0,j] = np.sqrt(r_centroid**2 + z_centroid**2)
                                # vals[1,j] = np.mean(j_m)
                                # vals[2,j] = np.max(r_m**2+z_m**2)
                                temp_vals.append(np.max(t))
                            else:
                                # vals[0,j]=0
                                # vals[1,j]=0
                                # vals[2,j]=0
                                temp_vals.append(0)
                        st = -1
                        if sheet_type=='centroid':
                            st=0
                        elif sheet_type=='current':
                            st=1
                        elif sheet_type=='radius':
                            st=2
                        if st>=0:
                            [k1] = np.where(vals[st,:]==vals[st,:].max())
                            if isinstance(k1, int):
                                k = k1
                            else:
                                st=(st+1)%3
                            [k2] = np.where(vals[st,:][k1]==vals[st,:][k1].max())
                            if isinstance(k2, int):
                                k = k1[k2]
                            else:
                                st=(st+1)%3
                                k3 = np.argmax(vals[st,:][k1][k2])
                                k = k1[k2][k3]
                        else:
                            [[k]] = np.where(np.array(temp_vals)==np.array(temp_vals).max())
                        # temp_slice, coords, r_slice, z_slice = slice(zoom_temp, np.cos(np.pi*k/180), np.sin(np.pi*k/180), 7.5)
                        sigma_slice, coords, r_slice, z_slice = slice(zoom_sigma, np.cos(np.pi*k/180), np.sin(np.pi*k/180), 7.5)
                        bphi_slice, coords, r_slice, z_slice = slice(zoom_bphi, np.cos(np.pi*k/180), np.sin(np.pi*k/180), 7.5)
                        b1_slice, b2_slice = project(zoom_bx, zoom_by, zoom_bz, coords, np.cos(np.pi*k/180), np.sin(np.pi*k/180))
                        rr = np.sqrt(r_slice**2+z_slice**2)
                        maskbh = rr < rplus
                        clean_sigma_slice = sigma_slice.copy()
                        clean_sigma_slice[maskbh] = clean_sigma_slice.max()
                        # clean_temp_slice = temp_slice.copy()
                        # clean_temp_slice[maskbh] = clean_temp_slice.min()
                        dbp_x, dbp_y = np.gradient(np.sign(bphi_slice))
                        bp_gradmag = np.sqrt(dbp_x**2 + dbp_y**2)
                        mask = cleanup(sigma_slice, bp_gradmag, rr, rplus)
                        sk_r, sk_z, l = sheet(mask, r_slice, z_slice)
                        fig = plt.figure(figsize=(13.65,6), tight_layout=True)
                        gs1 = grd.GridSpec(2,3,height_ratios=[2,1])
                        gs1.update(wspace=0.4, right=0.92, hspace=0.5, left=0.05)
                        ax0 = fig.add_subplot(gs1[0,0])
                        ax1 = fig.add_subplot(gs1[0,1])
                        ax2 = fig.add_subplot(gs1[0,2])
                        gs2 = grd.GridSpec(2,2,height_ratios=[2,1])
                        ax3 = fig.add_subplot(gs2[1,0])
                        ax4 = fig.add_subplot(gs2[1,1])
                        ax0.set_aspect('equal', adjustable='box')
                        ax1.set_aspect('equal', adjustable='box')
                        ax2.set_aspect('equal', adjustable='box')
                        ax0.set_xlim(zoom_x.min(), zoom_x.max())
                        ax0.set_ylim(zoom_y.min(), zoom_y.max())
                        ax1.set_xlim(r_slice.min(), r_slice.max())
                        ax1.set_ylim(z_slice.min(), z_slice.max())
                        ax2.set_xlim(r_slice.min(), r_slice.max())
                        ax2.set_ylim(z_slice.min(), z_slice.max())
                        h = ax0.pcolormesh(zoom_x[:,:,128], zoom_y[:,:,128], np.log10(zoom_beta[:,:,128]), cmap='plasma', vmin=-1, vmax=1)
                        fig.colorbar(h,ax=ax0,label=r'$\log_{10}(\beta)$',fraction=0.046, pad=0.04)
                        ax0.streamplot(zoom_x[:,:,128].transpose(), zoom_y[:,:,128].transpose(), zoom_bx[:,:,128].transpose(), zoom_by[:,:,128].transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
                        ax0.plot([0,15*np.cos(np.pi*k/180)], [0,15*np.sin(np.pi*k/180)], linewidth=2.5, ls='--', color='cyan')
                        ax0.fill(xh,yh,'k')
                        ax0.set_xlabel(r'$x$ ($r_G$)')
                        ax0.set_ylabel(r'$y$ ($r_G$)')
                        f = ax1.pcolormesh(r_slice, z_slice, bphi_slice, cmap='bwr', vmin=-0.2, vmax=0.2)
                        fig.colorbar(f,ax=ax1,label=r'$B_\phi$',fraction=0.046, pad=0.04)
                        ax1.streamplot(r_slice.transpose(), z_slice.transpose(), b1_slice.transpose(), b2_slice.transpose(), color='black', linewidth=0.5, density=2, arrowsize=0.5)
                        ax1.contour(r_slice, z_slice, clean_sigma_slice<1 , levels=[0.5], colors='magenta', linewidths=2.5)
                        ax1.fill(xh,yh,'k')
                        ax1.set_xlabel(r'$x$ ($r_G$)')
                        ax1.set_ylabel(r'$z$ ($r_G$)')
                        g = ax2.pcolormesh(r_slice, z_slice, np.log10(sigma_slice), cmap='plasma',vmax=1,vmin=-2)
                        fig.colorbar(g,ax=ax2,label=r'$\log_{10}(\sigma)$',fraction=0.046, pad=0.04)
                        ax2.contour(r_slice, z_slice, clean_sigma_slice<1 , levels=[0.5], colors='black', linewidths=2.5)
                        ax2.scatter(sk_r, sk_z, color='black', s=0.25)
                        ax2.fill(xh,yh,'k')
                        ax2.set_xlabel(r'$x$ ($r_G$)')
                        ax2.set_ylabel(r'$z$ ($r_G$)')
                        fig.suptitle(r't=%iM, $\phi=%i^\text{o}$'%(asc.t[i_dump],k))
                        ax3.set_ylim(-90,90)
                        ax3.plot(l,np.arctan(sk_z/sk_r)*180/np.pi, color='black')
                        ax3.set_xlabel(r'Length along Current Sheet ($r_G$)')
                        ax3.set_ylabel(r'$\theta_\text{midplane}$ (in deg)')
                        ax4.plot(asc.t[2807:3191], asc.Phibh[2807:3191,ir], color='black')
                        ax4.axvline(asc.t[i_dump], color='black', ls='--')
                        ax4.scatter(asc.t[i_dump], asc.Phibh[i_dump,ir], color='red',alpha=0.5)
                        ax4.set_xlabel('Time t (in M)')
                        ax4.set_ylabel(r'$\Phi_\text{BH}$')
                        os.system("mkdir -p frames_mayank")
                        plt.savefig(fname)

def mk_grframe_total(is_magnetic=True):
        import matplotlib.gridspec as grd
        set_dump_range_gr()
        print("Processing dumps ",i_start," to ",i_end)
        for i_dump in range(i_start,i_end):
                print("framing dump %d" %i_dump)
                fname = "frames_mayank/frame_%04d.png" % (i_dump)
                if os.path.isfile(fname):
                        dummy = i_dump
                else:
                        plt.figure(1)
                        plt.clf()
                        a = 0.9375
                        rplus = 1.+ np.sqrt(1.-a**2)
                        th = np.linspace(0.,2*np.pi,1000)
                        xh = rplus*np.sin(th)
                        yh = rplus*np.cos(th)
                        rerg = 1.+np.sqrt(1.-a**2*np.cos(th)**2)
                        xe = rerg*np.sin(th)
                        ye = rerg*np.cos(th)
                        asc.rd_1d_avg()
                        ir = asc.r_to_ir(2)
                        x = np.zeros((4,128,128,128))
                        y = np.zeros((4,128,128,128))
                        z = np.zeros((4,128,128,128))
                        rho = np.zeros((4,128,128,128))
                        sigma = np.zeros((4,128,128,128))
                        beta = np.zeros((4,128,128,128))
                        T = np.zeros((4,128,128,128))
                        Bz = np.zeros((4,128,128,128))
                        p = np.zeros((4,128,128,128))
                        j = np.array([-15,15,0,0])
                        for i in range(4):
                                asc.yt_extract_box(i_dump=i_dump, box_radius=15, mhd=True, gr=True, a=0.9375, center_x=j[i], center_y=j[3-i])
                                x[i,:,:,:] = asc.x
                                y[i,:,:,:] = asc.y
                                z[i,:,:,:] = asc.z
                                rho[i,:,:,:] = asc.rho
                                sigma[i,:,:,:] = asc.bsq / (asc.rho + asc.gamma/(asc.gamma-1) * asc.press)
                                beta[i,:,:,:] = 2 * asc.press / (asc.bsq)
                                Bz[i,:,:,:] = np.abs(asc.Bcc3)
                                p[i,:,:,:] = asc.press
                                T[i,:,:,:] = asc.press / asc.rho
                        fig = plt.figure(figsize=(14,16), tight_layout=True)
                        gs1 = grd.GridSpec(7,6,height_ratios=[0.04,1,1,1,1,1,1])
                        gs2 = grd.GridSpec(7,6,height_ratios=[0.04,1,1,1,1,1,1])
                        gs2.update(bottom=0.01,top=0.955)
                        
                        ax10 = fig.add_subplot(gs1[1,0])
                        ax10.set_aspect('equal', adjustable='box')
                        f = ax10.pcolormesh(x[0,:,:,64], y[0,:,:,64], np.log10(rho[0,:,:,64]), cmap='plasma', vmin=-1.5, vmax=0.5)
                        ax10.pcolormesh(x[1,:,:,64], y[1,:,:,64], np.log10(rho[1,:,:,64]), cmap='plasma', vmin=-1.5, vmax=0.5)
                        ax10.fill(xh,yh,'k')
                        cb = fig.colorbar(f,cax=plt.subplot(gs2[0, 0]), orientation='horizontal')
                        cb.set_label(r'$\log_{10}(\rho)$', labelpad=-35)
                        ax10.set_xlabel(r'$x$ ($r_G$)')
                        ax10.set_ylabel(r'$y$ ($r_G$)')
                        ax10.set_xlim(-15,15)
                        ax10.set_ylim(-15, 15)
                        ax20 = fig.add_subplot(gs1[2,0])
                        ax20.set_aspect('equal', adjustable='box')
                        ax20.pcolormesh(x[0,:,64,:], z[0,:,64,:], np.log10(rho[0,:,64,:]), cmap='plasma', vmin=-1.5, vmax=0.5)
                        ax20.fill(xh,yh,'k')
                        ax20.set_xlabel(r'$x$ ($r_G$)')
                        ax20.set_ylabel(r'$z$ ($r_G$)')
                        ax20.set_xlim(-30,0)
                        ax20.set_ylim(-15, 15)
                        ax30 = fig.add_subplot(gs1[3,0])
                        ax30.set_aspect('equal', adjustable='box')
                        ax30.pcolormesh(x[1,:,64,:], z[1,:,64,:], np.log10(rho[1,:,64,:]), cmap='plasma', vmin=-1.5, vmax=0.5)
                        ax30.fill(xh,yh,'k')
                        ax30.set_xlabel(r'$x$ ($r_G$)')
                        ax30.set_ylabel(r'$z$ ($r_G$)')
                        ax30.set_xlim(0,30)
                        ax30.set_ylim(-15, 15)
                        ax40 = fig.add_subplot(gs1[4,0])
                        ax40.set_aspect('equal', adjustable='box')
                        ax40.pcolormesh(y[3,64,:,:], z[3,64,:,:], np.log10(rho[3,64,:,:]), cmap='plasma', vmin=-1.5, vmax=0.5)
                        ax40.fill(xh,yh,'k')
                        ax40.set_xlabel(r'$y$ ($r_G$)')
                        ax40.set_ylabel(r'$z$ ($r_G$)')
                        ax40.set_xlim(-30,0)
                        ax40.set_ylim(-15, 15)
                        ax50 = fig.add_subplot(gs1[5,0])
                        ax50.set_aspect('equal', adjustable='box')
                        ax50.pcolormesh(y[2,64,:,:], z[2,64,:,:], np.log10(rho[2,64,:,:]), cmap='plasma', vmin=-1.5, vmax=0.5)
                        ax50.fill(xh,yh,'k')
                        ax50.set_xlabel(r'$y$ ($r_G$)')
                        ax50.set_ylabel(r'$z$ ($r_G$)')
                        ax50.set_xlim(0,30)
                        ax50.set_ylim(-15, 15)

                        ax11 = fig.add_subplot(gs1[1,1], sharey=ax10)
                        ax11.set_aspect('equal', adjustable='box')
                        plt.setp(ax11.get_yticklabels(), visible=False)
                        g = ax11.pcolormesh(x[0,:,:,64], y[0,:,:,64], np.log10(sigma[0,:,:,64]), cmap='seismic', vmin=-3, vmax=0)
                        ax11.pcolormesh(x[1,:,:,64], y[1,:,:,64], np.log10(sigma[1,:,:,64]), cmap='seismic', vmin=-3, vmax=0)
                        ax11.fill(xh,yh,'k')
                        cb = fig.colorbar(g,cax=plt.subplot(gs2[0, 1]), orientation='horizontal')
                        cb.set_label(r'$\log_{10}(\sigma)$', labelpad=-35)
                        ax11.set_xlabel(r'$x$ ($r_G$)')
                        ax11.set_xlim(-15,15)
                        ax11.set_ylim(-15, 15)
                        ax21 = fig.add_subplot(gs1[2,1],sharey=ax20)
                        ax21.set_aspect('equal', adjustable='box')
                        plt.setp(ax21.get_yticklabels(), visible=False)
                        ax21.pcolormesh(x[0,:,64,:], z[0,:,64,:], np.log10(sigma[0,:,64,:]), cmap='seismic', vmin=-3, vmax=0)
                        ax21.fill(xh,yh,'k')
                        ax21.set_xlabel(r'$x$ ($r_G$)')
                        ax21.set_xlim(-30,0)
                        ax21.set_ylim(-15, 15)
                        ax31 = fig.add_subplot(gs1[3,1],sharey=ax30)
                        ax31.set_aspect('equal', adjustable='box')
                        plt.setp(ax31.get_yticklabels(), visible=False)
                        ax31.pcolormesh(x[1,:,64,:], z[1,:,64,:], np.log10(sigma[1,:,64,:]), cmap='seismic', vmin=-3, vmax=0)
                        ax31.fill(xh,yh,'k')
                        ax31.set_xlabel(r'$x$ ($r_G$)')
                        ax31.set_xlim(0,30)
                        ax31.set_ylim(-15, 15)
                        ax41 = fig.add_subplot(gs1[4,1],sharey=ax40)
                        ax41.set_aspect('equal', adjustable='box')
                        plt.setp(ax41.get_yticklabels(), visible=False)
                        ax41.pcolormesh(y[3,64,:,:], z[3,64,:,:], np.log10(sigma[3,64,:,:]), cmap='seismic', vmin=-3, vmax=0)
                        ax41.fill(xh,yh,'k')
                        ax41.set_xlabel(r'$y$ ($r_G$)')
                        ax41.set_ylabel(r'$z$ ($r_G$)')
                        ax41.set_xlim(-30,0)
                        ax41.set_ylim(-15, 15)
                        ax51 = fig.add_subplot(gs1[5,1],sharey=ax50)
                        ax51.set_aspect('equal', adjustable='box')
                        plt.setp(ax51.get_yticklabels(), visible=False)
                        ax51.pcolormesh(y[2,64,:,:], z[2,64,:,:], np.log10(sigma[2,64,:,:]), cmap='seismic', vmin=-3, vmax=0)
                        ax51.fill(xh,yh,'k')
                        ax51.set_xlabel(r'$y$ ($r_G$)')
                        ax51.set_xlim(0,30)
                        ax51.set_ylim(-15, 15)

                        ax12 = fig.add_subplot(gs1[1,2],sharey=ax10)
                        ax12.set_aspect('equal', adjustable='box')
                        plt.setp(ax12.get_yticklabels(), visible=False)
                        h = ax12.pcolormesh(x[0,:,:,64], y[0,:,:,64], np.log10(beta[0,:,:,64]), cmap='winter', vmin=-0.5, vmax=1.5)
                        ax12.pcolormesh(x[1,:,:,64], y[1,:,:,64], np.log10(beta[1,:,:,64]), cmap='winter', vmin=-0.5, vmax=1.5)
                        ax12.fill(xh,yh,'k')
                        cb = fig.colorbar(h,cax=plt.subplot(gs2[0, 2]), orientation='horizontal')
                        cb.set_label(r'$\log_{10}(\beta)$', labelpad=-35)
                        ax12.set_xlabel(r'$x$ ($r_G$)')
                        ax12.set_xlim(-15,15)
                        ax12.set_ylim(-15, 15)
                        ax22 = fig.add_subplot(gs1[2,2],sharey=ax20)
                        ax22.set_aspect('equal', adjustable='box')
                        plt.setp(ax22.get_yticklabels(), visible=False)
                        ax22.pcolormesh(x[0,:,64,:], z[0,:,64,:], np.log10(beta[0,:,64,:]), cmap='winter', vmin=-0.5, vmax=1.5)
                        ax22.fill(xh,yh,'k')
                        ax22.set_xlabel(r'$x$ ($r_G$)')
                        ax22.set_xlim(-30,0)
                        ax22.set_ylim(-15, 15)
                        ax32 = fig.add_subplot(gs1[3,2],sharey=ax30)
                        ax32.set_aspect('equal', adjustable='box')
                        plt.setp(ax32.get_yticklabels(), visible=False)
                        ax32.pcolormesh(x[1,:,64,:], z[1,:,64,:], np.log10(beta[1,:,64,:]), cmap='winter', vmin=-0.5, vmax=1.5)
                        ax32.fill(xh,yh,'k')
                        ax32.set_xlabel(r'$x$ ($r_G$)')
                        ax32.set_xlim(0,30)
                        ax32.set_ylim(-15, 15)
                        ax42 = fig.add_subplot(gs1[4,2],sharey=ax40)
                        ax42.set_aspect('equal', adjustable='box')
                        plt.setp(ax42.get_yticklabels(), visible=False)
                        ax42.pcolormesh(y[3,64,:,:], z[3,64,:,:], np.log10(beta[3,64,:,:]), cmap='winter', vmin=-0.5, vmax=1.5)
                        ax42.fill(xh,yh,'k')
                        ax42.set_xlabel(r'$y$ ($r_G$)')
                        ax42.set_xlim(-30,0)
                        ax42.set_ylim(-15, 15)
                        ax52 = fig.add_subplot(gs1[5,2],sharey=ax50)
                        ax52.set_aspect('equal', adjustable='box')
                        plt.setp(ax52.get_yticklabels(), visible=False)
                        ax52.pcolormesh(y[2,64,:,:], z[2,64,:,:], np.log10(beta[2,64,:,:]), cmap='winter', vmin=-0.5, vmax=1.5)
                        ax52.fill(xh,yh,'k')
                        ax52.set_xlabel(r'$y$ ($r_G$)')
                        ax52.set_xlim(0,30)
                        ax52.set_ylim(-15, 15)

                        ax13 = fig.add_subplot(gs1[1,3],sharey=ax10)
                        ax13.set_aspect('equal', adjustable='box')
                        plt.setp(ax13.get_yticklabels(), visible=False)
                        i = ax13.pcolormesh(x[0,:,:,64], y[0,:,:,64], np.log10(T[0,:,:,64]), cmap='afmhot', vmin=-1.5, vmax=0.5)
                        ax13.pcolormesh(x[1,:,:,64], y[1,:,:,64], np.log10(T[1,:,:,64]), cmap='afmhot', vmin=-1.5, vmax=0.5)
                        ax13.fill(xh,yh,'k')
                        cb = fig.colorbar(i,cax=plt.subplot(gs2[0, 3]), orientation='horizontal')
                        cb.set_label(r'$\log_{10}(T)$', labelpad=-35)
                        ax13.set_xlabel(r'$x$ ($r_G$)')
                        ax13.set_xlim(-15,15)
                        ax13.set_ylim(-15, 15)
                        ax23 = fig.add_subplot(gs1[2,3],sharey=ax20)
                        ax23.set_aspect('equal', adjustable='box')
                        plt.setp(ax23.get_yticklabels(), visible=False)
                        ax23.pcolormesh(x[0,:,64,:], z[0,:,64,:], np.log10(T[0,:,64,:]), cmap='afmhot', vmin=-1.5, vmax=0.5)
                        ax23.fill(xh,yh,'k')
                        ax23.set_xlabel(r'$x$ ($r_G$)')
                        ax23.set_xlim(-30,0)
                        ax23.set_ylim(-15, 15)
                        ax33 = fig.add_subplot(gs1[3,3],sharey=ax30)
                        ax33.set_aspect('equal', adjustable='box')
                        plt.setp(ax33.get_yticklabels(), visible=False)
                        ax33.pcolormesh(x[1,:,64,:], z[1,:,64,:], np.log10(T[1,:,64,:]), cmap='afmhot', vmin=-1.5, vmax=0.5)
                        ax33.fill(xh,yh,'k')
                        ax33.set_xlabel(r'$x$ ($r_G$)')
                        ax33.set_xlim(0,30)
                        ax33.set_ylim(-15, 15)
                        ax43 = fig.add_subplot(gs1[4,3],sharey=ax40)
                        ax43.set_aspect('equal', adjustable='box')
                        plt.setp(ax43.get_yticklabels(), visible=False)
                        ax43.pcolormesh(y[3,64,:,:], z[3,64,:,:], np.log10(T[3,64,:,:]), cmap='afmhot', vmin=-1.5, vmax=0.5)
                        ax43.fill(xh,yh,'k')
                        ax43.set_xlabel(r'$y$ ($r_G$)')
                        ax43.set_xlim(-30,0)
                        ax43.set_ylim(-15, 15)
                        ax53 = fig.add_subplot(gs1[5,3],sharey=ax50)
                        ax53.set_aspect('equal', adjustable='box')
                        plt.setp(ax53.get_yticklabels(), visible=False)
                        ax53.pcolormesh(y[2,64,:,:], z[2,64,:,:], np.log10(T[2,64,:,:]), cmap='afmhot', vmin=-1.5, vmax=0.5)
                        ax53.fill(xh,yh,'k')
                        ax53.set_xlabel(r'$y$ ($r_G$)')
                        ax53.set_xlim(0,30)
                        ax53.set_ylim(-15, 15)

                        ax14 = fig.add_subplot(gs1[1,4],sharey=ax10)
                        ax14.set_aspect('equal', adjustable='box')
                        plt.setp(ax14.get_yticklabels(), visible=False)
                        j = ax14.pcolormesh(x[0,:,:,64], y[0,:,:,64], np.log10(Bz[0,:,:,64]), cmap='viridis', vmin=-2, vmax=-1)
                        ax14.pcolormesh(x[1,:,:,64], y[1,:,:,64], np.log10(Bz[1,:,:,64]), cmap='viridis', vmin=-2, vmax=-1)
                        ax14.fill(xh,yh,'k')
                        cb = fig.colorbar(j,cax=plt.subplot(gs2[0, 4]), orientation='horizontal')
                        cb.set_label(r'$\log_{10}(B_z^2)$', labelpad=-35)
                        ax14.set_xlabel(r'$x$ ($r_G$)')
                        ax14.set_xlim(-15,15)
                        ax14.set_ylim(-15, 15)
                        ax24 = fig.add_subplot(gs1[2,4],sharey=ax20)
                        ax24.set_aspect('equal', adjustable='box')
                        plt.setp(ax24.get_yticklabels(), visible=False)
                        ax24.pcolormesh(x[0,:,64,:], z[0,:,64,:], np.log10(Bz[0,:,64,:]), cmap='viridis', vmin=-2, vmax=-1)
                        ax24.fill(xh,yh,'k')
                        ax24.set_xlabel(r'$x$ ($r_G$)')
                        ax24.set_xlim(-30,0)
                        ax24.set_ylim(-15, 15)
                        ax34 = fig.add_subplot(gs1[3,4],sharey=ax30)
                        ax34.set_aspect('equal', adjustable='box')
                        plt.setp(ax34.get_yticklabels(), visible=False)
                        ax34.pcolormesh(x[1,:,64,:], z[1,:,64,:], np.log10(Bz[1,:,64,:]), cmap='viridis', vmin=-2, vmax=-1)
                        ax34.fill(xh,yh,'k')
                        ax34.set_xlabel(r'$x$ ($r_G$)')
                        ax34.set_xlim(0,30)
                        ax34.set_ylim(-15, 15)
                        ax44 = fig.add_subplot(gs1[4,4],sharey=ax40)
                        ax44.set_aspect('equal', adjustable='box')
                        plt.setp(ax44.get_yticklabels(), visible=False)
                        ax44.pcolormesh(y[3,64,:,:], z[3,64,:,:], np.log10(Bz[3,64,:,:]), cmap='viridis', vmin=-2, vmax=-1)
                        ax44.fill(xh,yh,'k')
                        ax44.set_xlabel(r'$y$ ($r_G$)')
                        ax44.set_xlim(-30,0)
                        ax44.set_ylim(-15, 15)
                        ax54 = fig.add_subplot(gs1[5,4],sharey=ax50)
                        ax54.set_aspect('equal', adjustable='box')
                        plt.setp(ax54.get_yticklabels(), visible=False)
                        ax54.pcolormesh(y[2,64,:,:], z[2,64,:,:], np.log10(Bz[2,64,:,:]), cmap='viridis', vmin=-2, vmax=-1)
                        ax54.fill(xh,yh,'k')
                        ax54.set_xlabel(r'$y$ ($r_G$)')
                        ax54.set_xlim(0,30)
                        ax54.set_ylim(-15, 15)

                        ax15 = fig.add_subplot(gs1[1,5],sharey=ax10)
                        ax15.set_aspect('equal', adjustable='box')
                        plt.setp(ax15.get_yticklabels(), visible=False)
                        k = ax15.pcolormesh(x[0,:,:,64], y[0,:,:,64], np.log10(p[0,:,:,64]), cmap='cividis', vmin=-2, vmax=-1)
                        ax15.pcolormesh(x[1,:,:,64], y[1,:,:,64], np.log10(p[1,:,:,64]), cmap='cividis', vmin=-2, vmax=-1)
                        ax15.fill(xh,yh,'k')
                        cb = fig.colorbar(k,cax=plt.subplot(gs2[0, 5]), orientation='horizontal')
                        cb.set_label(r'$\log_{10}(P)$', labelpad=-35)
                        ax15.set_xlabel(r'$x$ ($r_G$)')
                        ax15.set_xlim(-15,15)
                        ax15.set_ylim(-15, 15)
                        ax25 = fig.add_subplot(gs1[2,5],sharey=ax20)
                        ax25.set_aspect('equal', adjustable='box')
                        plt.setp(ax25.get_yticklabels(), visible=False)
                        ax25.pcolormesh(x[0,:,64,:], z[0,:,64,:], np.log10(p[0,:,64,:]), cmap='cividis', vmin=-2, vmax=-1)
                        ax25.fill(xh,yh,'k')
                        ax25.set_xlabel(r'$x$ ($r_G$)')
                        ax25.set_xlim(-30,0)
                        ax25.set_ylim(-15, 15)
                        ax35 = fig.add_subplot(gs1[3,5],sharey=ax30)
                        ax35.set_aspect('equal', adjustable='box')
                        plt.setp(ax35.get_yticklabels(), visible=False)
                        ax35.pcolormesh(x[1,:,64,:], z[1,:,64,:], np.log10(p[1,:,64,:]), cmap='cividis', vmin=-2, vmax=-1)
                        ax35.fill(xh,yh,'k')
                        ax35.set_xlabel(r'$x$ ($r_G$)')
                        ax35.set_xlim(0,30)
                        ax35.set_ylim(-15, 15)
                        ax45 = fig.add_subplot(gs1[4,5],sharey=ax40)
                        ax45.set_aspect('equal', adjustable='box')
                        plt.setp(ax45.get_yticklabels(), visible=False)
                        ax45.pcolormesh(y[3,64,:,:], z[3,64,:,:], np.log10(p[3,64,:,:]), cmap='cividis', vmin=-2, vmax=-1)
                        ax45.fill(xh,yh,'k')
                        ax45.set_xlabel(r'$y$ ($r_G$)')
                        ax45.set_xlim(-30,0)
                        ax45.set_ylim(-15, 15)
                        ax55 = fig.add_subplot(gs1[5,5],sharey=ax50)
                        ax55.set_aspect('equal', adjustable='box')
                        plt.setp(ax55.get_yticklabels(), visible=False)
                        ax55.pcolormesh(y[2,64,:,:], z[2,64,:,:], np.log10(p[2,64,:,:]), cmap='cividis', vmin=-2, vmax=-1)
                        ax55.fill(xh,yh,'k')
                        ax55.set_xlabel(r'$y$ ($r_G$)')
                        ax55.set_xlim(0,30)
                        ax55.set_ylim(-15, 15)
                        
                        ax3 = fig.add_subplot(gs1[6,:])
                        ax3.plot(asc.t[np.max(np.where(asc.t<=49350)):],asc.Phibh[np.max(np.where(asc.t<=49350)):, ir],color='black')
                        ax3.axvline(i_dump*10,color='black',ls='--')
                        ax3.scatter(i_dump*10,asc.Phibh[np.max(np.where(asc.t<=i_dump*10)),ir],color='red',alpha=0.5)
                        ax3.set_xlabel('Time t (in M)')
                        ax3.set_ylabel(r'$\Phi_{BH}$')

                        fig.suptitle(r'$t = %iM$'%(i_dump*10), fontsize=16)
                        os.system("mkdir -p frames_mayank")
                        plt.savefig(fname)
                                
                        

def mk_frame_grmhd(is_magnetic=True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
			c1 = plt.pcolormesh((asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-2,vmax=2,cmap="ocean")
			c2 = plt.pcolormesh(-(asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,asc.nz//2],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-100,100)
			plt.xlim(-100,100)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %g M$' %(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)

		fname = "frames/frame_midplane_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			continue
		else: 
			plt.figure(2)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
			c1 = plt.pcolormesh((asc.x)[:,asc.ny//2,:], (asc.y)[:,asc.ny//2,:],np.log10(asc.rho)[:,asc.ny//2,:],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			plt.ylim(-100,100)
			plt.xlim(-100,100)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %g M$' %(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)
        

def mk_frame_gr_magnetically_frustrated(is_magnetic=True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=False,gr=True,a=a)
			c1 = plt.pcolormesh((asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-2,vmax=2,cmap="ocean")
			c2 = plt.pcolormesh(-(asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,asc.nz//2],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,colors='white')
			# # if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,colors='white')
			# # if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,linestyles='--',colors='white')

			plt.ylim(-100,100)
			plt.xlim(-100,100)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)

			plt.xlim(-30,30)
			plt.ylim(-30,30)

			plt.savefig("frames/frame_zoom_in_%04d.png" % (i_dump))

			plt.xlim(-100,100)
			plt.ylim(-100,100)

			asc.x = -asc.x 
			asc.plot_fieldlines_gr(100)

			#if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,linestyles='--',colors='white')
			#if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,linestyles='--',colors='white')
			plt.savefig("frames/frame_fieldlines_%04d.png" % (i_dump))

		# fname = "frames/frame_midplane_%04d.png" % (i_dump)
		# if os.path.isfile(fname):
		# 	continue
		# else: 
		# 	plt.figure(2)
		# 	plt.clf()
		# 	asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
		# 	c1 = plt.pcolormesh((asc.x)[:,asc.ny//2,:], (asc.y)[:,asc.ny//2,:],np.log10(asc.rho)[:,asc.ny//2,:],vmin=-2,vmax=2,cmap="ocean")
		# 	cb1 = plt.colorbar(c1) 
		# 	plt.ylim(-100,100)
		# 	plt.xlim(-100,100)
		# 	plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
		# 	plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
		# 	plt.title(r'$t = %g M$' %(asc.t),fontsize = 20)

		# 	cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

		# 	for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
		# 	    label.set_fontsize(10)

		# 	os.system("mkdir -p frames")
		# 	plt.savefig(fname)
def mk_frame_grmhd_cartesian(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.yt_extract_box(i_dump,50,mhd=is_magnetic,gr=True,a=a)
			ny = asc.rho.shape[1]
			nx = asc.rho.shape[0]
			c1 = plt.pcolormesh((asc.y)[0,:,:], (asc.z)[0,:,:],np.log10(asc.rho)[nx//2,:,:],vmin=-4,vmax=0.5,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=a)
			plt.axis('off')

			##plt.axes().set_aspect('equal')

			##plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

			plt.clf()
			ny = asc.rho.shape[1]
			nx = asc.rho.shape[0]
			nz = asc.rho.shape[2]
			c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.y)[:,:,0],np.log10(asc.rho)[:,:,nz//2],vmin=-2,vmax=1,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-30,30)
			plt.xlim(-30,30)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=a)
			plt.axis('off')

			##plt.axes().set_aspect('equal')

			##plt.tight_layout()

			os.system("mkdir -p frames")
			fname = "frames/frame_midplane_%04d.png" % (i_dump)
			plt.savefig(fname)

def mk_frame_grmhd_binary(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:

			frame_suffix_array = ["","temp","gamma"]
			for i_type in arange(0,3):
				frame_suffix = frame_suffix_array[i_type]
				fname = "frames/frame_" + frame_suffix + "_%04d.png" % (i_dump)
				plt.figure(1)
				plt.clf()
				asc.yt_extract_box(i_dump,30,mhd=is_magnetic,gr=True,a=a)
				ny = asc.rho.shape[1]
				nx = asc.rho.shape[0]
				if (i_type==0): c1 = plt.pcolormesh((asc.y)[0,:,:], (asc.z)[0,:,:],np.log10(asc.rho)[nx//2,:,:],vmin=-3,vmax=0,cmap="ocean")
				elif (i_type==1): c1 = plt.pcolormesh((asc.y)[0,:,:], (asc.z)[0,:,:],np.log10(asc.press/asc.rho*1836)[nx//2,:,:],vmin=0,vmax=2,cmap="inferno")
				elif (i_type==2):
					asc.cks_binary_metric(np.array(ds.current_time),asc.x,asc.y,asc.z,aprime_=aprime,q=q,r_bh2=rbh2,a=a,t0=t0,inclination=inclination)
					tmp = asc.g[1,1]*vel1*vel1 + 2.0*asc.g[1,2]*vel1*vel2 + 2.0*asc.g[1,3]*vel1*vel3+ asc.g[2,2]*vel2*vel2 + 2.0*asc.g[2,3]*vel2*vel3+ asc,g[3,3]*vel3*vel3;
					gamma = np.sqrt(1.0 + tmp);
					c1 = plt.pcolormesh((asc.y)[0,:,:], (asc.z)[0,:,:],gamma[nx//2,:,:],vmin=1,vmax=2)

				cb1 = plt.colorbar(c1) 
				# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
				# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
				# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
				# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

				plt.ylim(-30,30)
				plt.xlim(-30,30)
				plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
				plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
				plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

				if (i_type ==0 ): cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)
				elif (i_type ==1 ): cb1.set_label(r"$\log_{10}(T)$",fontsize=17)
				elif (i_type ==2 ): cb1.set_label(r"$\Gamma$",fontsize=17)

				for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
				    label.set_fontsize(10)

				asc.bhole(az=a)
				plt.axis('off')

				plt.axes().set_aspect('equal')

				plt.tight_layout()

				os.system("mkdir -p frames")
				plt.savefig(fname)

				plt.clf()
				ny = asc.rho.shape[1]
				nx = asc.rho.shape[0]
				nz = asc.rho.shape[2]
				if (i_type==0): c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.y)[:,:,0],np.log10(asc.rho)[:,:,nz//2],vmin=-2,vmax=1,cmap="ocean")
				if (i_type==1): c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.y)[:,:,0],np.log10(asc.press/asc.rho*1836)[:,:,nz//2],vmin=0.5,vmax=2.5,cmap="inferno")
				elif (i_type==2):
					asc.cks_binary_metric(np.array(ds.current_time),asc.x,asc.y,asc.z,aprime_=aprime,q=q,r_bh2=rbh2,a=a,t0=t0,inclination=inclination)
					tmp = asc.g[1,1]*vel1*vel1 + 2.0*asc.g[1,2]*vel1*vel2 + 2.0*asc.g[1,3]*vel1*vel3+ asc.g[2,2]*vel2*vel2 + 2.0*asc.g[2,3]*vel2*vel3+ asc,g[3,3]*vel3*vel3;
					gamma = np.sqrt(1.0 + tmp);
					c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.y)[:,:,0],gamma[:,:,nz//2],vmin=1,vmax=2)
				cb1 = plt.colorbar(c1) 
				# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
				# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
				# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
				# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

				plt.ylim(-30,30)
				plt.xlim(-30,30)
				plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
				plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
				plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

				if (i_type ==0 ): cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)
				elif (i_type ==1 ): cb1.set_label(r"$\log_{10}(T)$",fontsize=17)
				elif (i_type ==2 ): cb1.set_label(r"$\Gamma$",fontsize=17)

				for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
				    label.set_fontsize(10)

				asc.bhole(az=a)
				plt.axis('off')

				##plt.axes().set_aspect('equal')

				plt.tight_layout()

				os.system("mkdir -p frames")
				fname = "frames/frame_midplane_" + frame_suffix + "_%04d.png" % (i_dump)
				plt.savefig(fname)


def mk_frame_grmhd_constant_velocity_boosted(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.yt_extract_box(i_dump,5,mhd=is_magnetic,gr=True,a=a)
			
			xbh,ybh,zbh = asc.bh2_pos_constant_velocity(np.array(asc.ds.current_time),vbh=0.1,z0=-80)
			asc.yt_extract_box(i_dump,20,mhd=is_magnetic,gr=True,a=a,center_x=xbh,center_y=ybh,center_z=zbh)
			ny = asc.rho.shape[1]
			nx = asc.rho.shape[0]
			c1 = plt.pcolormesh((asc.y)[0,:,:], (asc.z)[0,:,:],np.log10(asc.rho)[nx//2,:,:],vmin=-2,vmax=1,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-20+zbh,20+zbh)
			plt.xlim(-20+ybh,20+ybh)
			plt.xlabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			##asc.bhole()
			#plt.axis('off')

			asc.bhole2(ybh,zbh,q=1.0,aprime=0.0,facecolor='white')

			##plt.axes().set_aspect('equal')

			##plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

def mk_frame_grmhd_constant_velocity(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.yt_extract_box(i_dump,20,mhd=is_magnetic,gr=True,a=a)
			
			ny = asc.rho.shape[1]
			nx = asc.rho.shape[0]
			c1 = plt.pcolormesh((asc.y)[0,:,:], (asc.z)[0,:,:],np.log10(asc.rho)[nx//2,:,:],vmin=-2,vmax=1,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-20,20)
			plt.xlim(-20,20)
			plt.xlabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=0.0)
			plt.axis('off')

			##asc.bhole2(ybh,zbh,q=1.0,aprime=0.0,facecolor='white')

			##plt.axes().set_aspect('equal')

			##plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)
def mk_frame_grmhd_cartesian_second_bh_zoom(is_magnetic=False,spin_orbit=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	if (spin_orbit): orbit_file = glob.glob("*orbit*.dat")[0]
	else: orbit_file = None 
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_binary_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.yt_load(i_dump,gr=True,a=a)
			if (orbit_file==None): bhpos = asc.bh2_pos(np.array(asc.ds.current_time),rbh2,t0,inclination=inclination)
			else:
				asc.rd_binary_orbits(orbit_file)
				asc.get_binary_quantities(np.array(asc.t),t0)

				bhpos = [asc.x2,asc.y2,asc.z2]

				# a2x_ = asc.q * asc.a2x
				# a2y_ = asc.q * asc.a2y 
				# a2z_ = asc.q * asc.a2z

			asc.yt_extract_box(i_dump,q*30,gr=True,a=a,center_x=bhpos[0],center_y=bhpos[1],center_z = bhpos[2])
			ny = asc.rho.shape[1]
			nx = asc.rho.shape[0]
			nz = asc.rho.shape[2]
			if (midplane==True): c1 = plt.pcolormesh((asc.x)[:,:,nz//2], (asc.y)[:,:,nz//2],np.log10(asc.rho)[:,:,nz//2],vmin=-1,vmax=2,cmap="ocean")
			else: c1 = plt.pcolormesh((asc.y)[nx//2,:,:], (asc.z)[nx//2,:,:],np.log10(asc.rho)[:,:,nz//2],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			# plt.ylim(-30,30)
			# plt.xlim(-30,30)
			if (midplane==False):
				plt.xlabel(r'$y$ ($r_G$)',fontsize = 20)
				plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			else:
				plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
				plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.ds.current_time)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			if (midplane==True): asc.bhole2(bhpos[0],bhpos[1],q=q,aprime=aprime,facecolor='white')
			else: asc.bhole2(bhpos[1],bhpos[2],aprime=aprime,facecolor='white')
			#plt.axis('off')

			##plt.axes().set_aspect('equal')

			##plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

def mk_frame_grmhd_restart_cartesian(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,az=a,th=th_tilt,ph=phi_tilt)
			nz = asc.rho.shape[-1]
			ny = asc.rho.shape[1]
			# c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-3,vmax=0,cmap="ocean")
			# plt.pcolormesh(-(asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,nz//2],vmin=-3,vmax=0,cmap="ocean")
			c1 = asc.pcolormesh_corner(asc.r,asc.th,np.log10(asc.rho)[:,:,0],vmin=-5,vmax=0,cmap="ocean")
			asc.pcolormesh_corner(asc.r,asc.th,np.log10(asc.rho)[:,:,nz//2],vmin=-5,vmax=0,cmap="ocean",flip_x=True)

			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=a)
			plt.axis('off')

			##plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

			asc.cks_coord_to_ks(asc.x,asc.y,asc.z,a=a)
			asc.plot_fieldlines_gr(50,a=a,npz=True)
			fname = "frames/frame_fieldlines_%04d.png" % (i_dump)

			plt.clf()

			##c1 = plt.pcolormesh((asc.x)[:,ny//2,:], (asc.y)[:,ny//2,:],np.log10(asc.rho)[:,ny//2,:],vmin=-2,vmax=0.5,cmap="ocean")

			c1 = asc.pcolormesh_corner(asc.r,asc.th,np.log10(asc.rho)[:,ny//2,:],coords='xy',vmin=-2,vmax=0.5,cmap="ocean")
			##def pcolormesh_corner(r,th,myvar,coords = 'xz',flip_x = False,**kwargs)

			cb1 = plt.colorbar(c1) 

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=a)
			plt.axis('off')

			##plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			fname = "frames/frame_midplane_%04d.png" % (i_dump)
			plt.savefig(fname)

def mk_frame_gr_magnetically_frustrated_cartesian(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,az=a,th=th_tilt,ph=phi_tilt)
			ny = asc.rho.shape[1]
			c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-2,vmax=2,cmap="ocean")
			plt.pcolormesh(-(asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,ny//2],vmin=-2,vmax=2,cmap="ocean")

			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=a)
			plt.axis('off')

			plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

			plt.xlim(-500,500)
			plt.ylim(-500,500)
			fname = "frames/frame_jet_%04d.png" % (i_dump)
			plt.savefig(fname)

			plt.ylim(-50,50)
			plt.xlim(-50,50)


			asc.cks_coord_to_ks(asc.x,asc.y,asc.z,a=a)
			asc.plot_fieldlines_gr(50,a=a,npz=True)
			fname = "frames/frame_fieldlines_%04d.png" % (i_dump)

			plt.savefig(fname)



			plt.clf()

			c1 = plt.pcolormesh((asc.x)[:,ny//2,:], (asc.y)[:,ny//2,:],np.log10(asc.rho)[:,ny//2,:],vmin=-2,vmax=0.5,cmap="ocean")

			cb1 = plt.colorbar(c1) 

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole(az=a)
			plt.axis('off')

			plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			fname = "frames/frame_midplane_%04d.png" % (i_dump)
			plt.savefig(fname)

def mk_1d_quantities(is_magnetic =True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
		mdot =  asc.angle_average(asc.rho*asc.uu[1]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2), gr=True)
		Phibh = asc.phibh()*np.sqrt(4*np.pi)
		ud = asc.Lower(asc.uu,asc.g)
		bd = asc.Lower(asc.bu,asc.g)
		asc.Tud_calc(asc.uu,ud,asc.bu,bd,is_magnetic= is_magnetic)
		Jdot = asc.angle_average(asc.Tud[1][3]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True)
		Edot = - (asc.angle_average(asc.Tud[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True) +mdot )
		EdotEM = -asc.angle_average(asc.TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True)


		dic = {}
		dic['t'] = asc.t
		dic['r'] = asc.r[::4,0,0]
		dic['mdot'] = mdot[::4]
		dic['Jdot'] = Jdot[::4]
		dic['Edot'] = Edot[::4]
		dic['Phibh'] = Phibh[::4]
		dic['EdotEM'] = EdotEM[::4]

		np.savez("1d_dump_%05d.npz" %i_dump,**dic)
def mk_1d_quantities_binary(is_magnetic =True,spin_orbit=False):
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	if (spin_orbit): orbit_file = glob.glob("*orbit*.dat")[0]
	else: orbit_file = None 
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		dump_npz = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(i_dump,th_tilt,phi_tilt)
		if glob.glob("*out2*athdf") != []:
			dump_file_prefix = glob.glob("*out2*.athdf")[0][:-11]
			dump_hdf5 = dump_file_prefix + "%05d.athdf" %i_dump
		else: dump_hdf5 = ""
		if (os.path.isfile(dump_npz) or os.path.isfile(dump_hdf5) ): 
			print("inclination", inclination)
			asc.rd_yt_convert_to_spherical_2nd_black_hole(i_dump,MHD=is_magnetic,a=a,rmax=None,q=q,rbh2=rbh2,aprime=aprime,t0=t0,inclination=inclination,orbit_file=orbit_file)
		else: 
			print("Skipping dump: ", i_dump)
			continue 

		def SQR(arg):
			return arg**2.0
		if (orbit_file==None):

			xbh,ybh,zbh = bh2_pos(np.array(asc.t),rbh2,t0=t0,inclination=inclination)
			dxbh_dt,dybh_dt,dzbh_dt =  bh2_vel(np.array(asc.t),rbh2,t0=t0,inclination=inclination)

			a1x_ = 0.0
			a1y_ = 0.0
			a1z_ = a*1.0
			a2x_ = 0.0;
			a2y_ = 0.0;
			a2z_ = q* aprime 

	      # aprime_ = q * aprime
		else:
			asc.rd_binary_orbits(orbit_file)
			asc.get_binary_quantities(np.array(asc.t),t0)

			a2x_ = asc.q * asc.a2x
			a2y_ = asc.q * asc.a2y 
			a2z_ = asc.q * asc.a2z

		a2 = np.sqrt(a2x_**2.0 + a2y_**2.0+a2z_**2.0)



		a_dot_x = a2x_ * asc.x + a2y_ * asc.y + a2z_ * asc.z

		a_cross_x = [0,0,0]
		a_cross_x[0] = a2y_ * asc.z - a2z_ * asc.y;
		a_cross_x[1] = a2z_ * asc.x - a2x_ * asc.z;
		a_cross_x[2] = a2x_ * asc.y - a2y_ * asc.x;


		rsq_p_asq = SQR(asc.r) + SQR(a2);

		lx = (asc.r * asc.x - a_cross_x[0] + a_dot_x * a2x_/asc.r)/(rsq_p_asq);
		ly = (asc.r * asc.y - a_cross_x[1] + a_dot_x * a2y_/asc.r)/(rsq_p_asq);
		lz = (asc.r * asc.z - a_cross_x[2] + a_dot_x * a2z_/asc.r)/(rsq_p_asq);

		lz[lz>1.0] = 1.0 + 0.0* lz[lz>1.0]
		lz[lz<-1.0] = -1.0 + 0.0* lz[lz<-1.0]

		asc.th = np.arccos(lz); 
		asc.ph = np.arctan2(ly,lx); 

			##this should be correct
		gdet = np.sqrt( np.sin(asc.th)**2.0 * ( asc.r**2.0 + a_dot_x**2.0/asc.r**2.0)**2.0 )

		asc.get_mdot(mhd=True,gr=True,ax=a2x_,ay=a2y_,az=a2z_)
		# asc.th = np.arccos(asc.z/asc.r)
		# asc.ph = np.arctan2((asc.r*asc.y-aprime*asc.x), (aprime*asc.y+asc.r*asc.x) )
		mdot = asc.angle_average_npz(nan_to_num(asc.mdot), gr=True,gdet=gdet)
		Br = (asc.bu_ks[1] * asc.uu_ks[0] - asc.bu_ks[0]* asc.uu_ks[1])

		Phibh = asc.angle_average_npz(0.5*np.fabs(Br)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,gdet=gdet) 



		asc.cks_metric(asc.x,asc.y,asc.z,a2x_,a2y_,a2z_)
		ud = nan_to_num(asc.Lower(asc.uu,asc.g))
		bd = nan_to_num(asc.Lower(asc.bu,asc.g))
		asc.Tud_calc(asc.uu,ud,asc.bu,bd,is_magnetic= is_magnetic,gam=gam)

		Tud_xyz = asc.Tud * 1.0

		# uu_phi_z =  uu[1] * (-asc.y/(asc.x**2+asc.y**2+SMALL) + a2z_*asc.r*asc.x/((asc.r**2+a2**2)*sqrt_term)) + \
    #            A[2] * (x/(x**2+y**2+SMALL) + a*r*y/((r**2+a**2)*sqrt_term)) + \
    #            A[3] * (a* z/r/sqrt_term) 

		# step = -1.0*(asc.th>np.pi/2.0) +1.0*(asc.th<=np.pi/2.0)
		# Phibh_net = asc.angle_average_npz(0.5*(Br*step)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 
		# Phibh_x = asc.angle_average_npz(0.5*(asc.Bcc1)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 
		# Phibh_y = asc.angle_average_npz(0.5*(asc.Bcc2)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 
		# Phibh_z = asc.angle_average_npz(0.5*(asc.Bcc3)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 


		# asc.ks_binary_metric(np.array(asc.t),asc.x,asc.y,asc.z,aprime_=aprime,q=q,r_bh2=rbh2,a=a,t0=t0,midplane=midplane)
		g = asc.ks_metric_general(asc.x,asc.y,asc.z,a2x_,a2y_,a2z_)
		##asc.ks_metric(asc.r,asc.th)
		gdet = asc.Determinant_4b4(g)

		dth = np.diff(asc.th[0,:,0])[0]
		dph = np.diff(asc.ph[0,0,:])[0]
		angle_norm = (gdet * dth * dph).sum(-1).sum(-1)
		angle_norm = angle_norm[:,None,None]*(asc.r/asc.r)

		uu_ks = nan_to_num(asc.uu_ks)
		bu_ks = nan_to_num(asc.bu_ks)
		ud_ks = nan_to_num(asc.Lower(asc.uu_ks,g))
		bd_ks = nan_to_num(asc.Lower(asc.bu_ks,g))


		asc.Tud_calc(asc.uu_ks,ud_ks,asc.bu_ks,bd_ks,is_magnetic= is_magnetic,gam=gam)
		Jdot = asc.angle_average_npz(asc.Tud[1][3]*angle_norm,gdet=gdet,gr=True)
		Edot = - (asc.angle_average_npz(asc.Tud[1][0]*angle_norm,gdet=gdet,gr=True) +mdot )
		EdotEM = -asc.angle_average_npz(asc.TudEM[1][0]*angle_norm,gdet=gdet,gr=True)


		Tud_rot,gdet_rot = asc.get_Tud_gdet_ks_rotated(asc.x,asc.y,asc.z,a2x_,a2y_,a2z_,0,is_magnetic=is_magnetic,gam=gam)
		Jdotx = asc.angle_average_npz(Tud_rot[1][3]*angle_norm,gdet=gdet_rot,gr=True)
		Tud_rot,gdet_rot = asc.get_Tud_gdet_ks_rotated(asc.x,asc.y,asc.z,a2x_,a2y_,a2z_,1,is_magnetic=is_magnetic,gam=gam)
		Jdoty = asc.angle_average_npz(Tud_rot[1][3]*angle_norm,gdet=gdet_rot,gr=True)

		Pdotx = asc.angle_average_npz(Tud_xyz[1][1]*angle_norm,gr=True,gdet=gdet)
		Pdoty = asc.angle_average_npz(Tud_xyz[1][2]*angle_norm,gr=True,gdet=gdet)
		Pdotz = asc.angle_average_npz(Tud_xyz[1][3]*angle_norm,gr=True,gdet=gdet)


		EdotMA = Edot-EdotEM

		Lx = asc.angle_average_npz(asc.rho* (asc.y*asc.uu[3] - asc.z*asc.uu[2]),gdet=gdet,gr=True)
		Ly = asc.angle_average_npz(asc.rho* (asc.z*asc.uu[1] - asc.x*asc.uu[3]),gdet=gdet,gr=True)
		Lz = asc.angle_average_npz(asc.rho* (asc.x*asc.uu[2] - asc.y*asc.uu[1]),gdet=gdet,gr=True)

		Bx = asc.angle_average_npz(asc.Bcc1,gdet=gdet,gr=True)
		By = asc.angle_average_npz(asc.Bcc2,gdet=gdet,gr=True)
		Bz = asc.angle_average_npz(asc.Bcc3,gdet=gdet,gr=True)


		# #asc.cks_metric(asc.x,asc.y,asc.z,a)
		# asc.cks_inverse_metric(asc.x,asc.y,asc.z,a)
		# alpha = np.sqrt(-1.0/asc.gi[0,0])
		# gamma = asc.uu[0] * alpha


		dic = {}
		dic['t'] = np.array(asc.t)
		dic['r'] = asc.r[::4,0,0]
		dic['mdot'] = mdot[::4]
		dic['Jdot'] = Jdot[::4]
		dic['Jdotx'] = Jdotx[::4]
		dic['Jdoty'] = Jdoty[::4]
		dic['Edot'] = Edot[::4]
		dic['Phibh'] = Phibh[::4]

		dic['EdotEM'] = EdotEM[::4]

		dic['Lx'] = Lx[::4]
		dic['Ly'] = Ly[::4]
		dic['Lz'] = Lz[::4]
		dic['Bx'] = Bx[::4]
		dic['By'] = By[::4]
		dic['Bz'] = Bz[::4]

		dic['Pdotx'] = Pdotx[::4]
		dic['Pdoty'] = Pdoty[::4]
		dic['Pdotz'] = Pdotz[::4]


		np.savez("1d_binary_dump_%05d.npz" %i_dump,**dic)
def mk_1d_quantities_Be(is_magnetic =False):
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2='mks_pole_fix',h=0.2,gr=False,a=a)
		Omega = (np.sin(asc.th)[0,:,0]*np.diff(asc.x2f)).sum()*2.0*pi
		mdot =  asc.angle_average(asc.rho*asc.vel1 * asc.r**2) * Omega
		mdot_out = asc.angle_average(asc.rho*asc.vel1 * asc.r**2*(asc.vel1>0)) * Omega
		mdot_mid = (asc.rho*asc.vel1* asc.r**2)[:,asc.ny//2,0] * Omega

		Jdot = asc.angle_average((asc.vel1*asc.vel3*asc.r*np.sin(asc.th)*asc.r**2*asc.rho))*Omega
		Jdot_mid = ((asc.vel1*asc.vel3*asc.r*np.sin(asc.th)*asc.r**2*asc.rho))[:,asc.ny//2,0]*Omega
		rho_fft = np.fft.fft(asc.rho[::2,asc.ny//2,:],axis=-1)
		mdot_fft = np.fft.fft((asc.rho*asc.vel1*4.0*np.pi*asc.r**2.0)[::2,asc.ny//2,:],axis=-1)

		dr = np.diff(asc.x1f)
		#mdot_out_bound = -(2.0 * np.pi * asc.rho * asc.vel2 * asc.r*asc.sin(asc.th))[:,0].mean(-1)*dr + (2.0 * np.pi * asc.rho * asc.vel2 * asc.r*asc.sin(asc.th))[:,-1].mean(-1)*dr

		# flux_r,flux_th = asc.compute_boundary_fluxes_Be()
		# mdot_th_bound = (flux_th.mean(-1)*asc.r[:,0,0]*np.sin(asc.x2f)[0]*dr) * 2.0*np.pi
		# mdot_r_bound= (flux_r * Omega*np.sin(asc.th)[0]).mean(-1)
		# mdot_r_bound_tot = asc.angle_average(flux_r[None,:,:]*(asc.r/asc.r) )[0] * Omega
		# mdot_r_bound_out_tot = asc.angle_average( (flux_r*(flux_r>0))[None,:,:]*(asc.r/asc.r) )[0] * Omega

		Mnet_r_inner_bound =np.sum(asc.fluxr[0])
		Mnet_r_outer_bound = np.sum(asc.fluxr[-1])
		Mnet_r_inner_bound_pos = np.sum(asc.fluxr[0]*(asc.fluxr[0]>0))
		Mnet_r = np.sum(np.sum(asc.fluxr,axis=-1),axis=-1)
		Mnet_r_inner_bound_v_th = np.sum(asc.fluxr[0],axis=-1)

		dMfloor_r = np.sum(np.sum(asc.dM_floor,axis=-1),axis=-1)
		# Mnet_th_bound = np.sum(asc.fluxth[:,0,:])- np.sum(asc.fluxth[:,-1,:])

		dic = {}
		dic['t'] = asc.t
		dic['r'] = asc.r[::2,0,0]
		dic['th'] = asc.th[0,:,0]

		dic['mdot'] = mdot[::2]
		dic['mdot_out'] = mdot_out[::2]

		# dic['mdot_th_bound'] = mdot_th_bound[::2]
		# dic['mdot_th_bound_cum'] = mdot_th_bound.cumsum()[::2]
		# dic['mdot_th_bound_tot'] = np.sum(mdot_th_bound)
		# dic['mdot_r_bound'] = mdot_r_bound
		# dic['mdot_r_bound_tot'] = mdot_r_bound_tot
		# dic['mdot_r_bound_out_tot'] = mdot_r_bound_out_tot

		dic['Mnet_r_inner_bound'] = Mnet_r_inner_bound
		dic['Mnet_r_inner_bound_pos'] = Mnet_r_inner_bound_pos

		dic['Mnet_r_outer_bound'] = Mnet_r_outer_bound
		dic['Mnet_r'] = Mnet_r[::2]

		dic['dMfloor_r'] = dMfloor_r[::2]
		dic['dMfloor'] = asc.dM_floor.sum()

		dic['Mnet_r_inner_bound_v_th'] = Mnet_r_inner_bound_v_th
		
		# dic['Mnet_th_bound'] = Mnet_th_bound


		dic['M'] = (asc.angle_average(asc.rho*asc.r**2.0)*Omega * dr).sum()
		dic['M_cum'] = (asc.angle_average(asc.rho*asc.r**2.0)*Omega * dr).cumsum()
		dic['mdot_mid'] = mdot_mid[::2]

		dic['Jdot'] = Jdot[::2]
		dic['Jdot_mid'] = Jdot_mid[::2]

		dic['rho'] = asc.rho[::2,asc.ny//2,0]
		dic['rho_fourier'] = [rho_fft[:,0],rho_fft[:,1],rho_fft[:,2],rho_fft[:,3],rho_fft[:,4]]
		dic['mdot_fourier'] = [mdot_fft[:,0],mdot_fft[:,1],mdot_fft[:,2],mdot_fft[:,3],mdot_fft[:,4]]
		dic['phi_fourier'] = np.fft.fftfreq(asc.nz,np.diff(asc.ph[0,0,:])[0])
		ir = asc.r_to_ir_npz(5.0,asc.r)
		if (is_magnetic==True): dic['Bphi_r_5'] = asc.Bcc3[ir,:].mean(-1)
		if (is_magnetic==True): dic['BphiBr_mid'] = (asc.Bcc3*asc.Bcc1)[:,asc.ny//2,0]

		np.savez("1d_dump_%04d.npz" %i_dump,**dic)


def mk_1d_quantities_Be_smr(is_magnetic =False):
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.rd_spherical_smr(i_dump,is_magnetic)
		asc.rd_spherical_smr(i_dump,is_magnetic)
		Omega = 4.0*np.pi
		mdot =  asc.angle_average_npz(asc.rho*asc.vel1 * asc.r**2) * Omega
		mdot_out = asc.angle_average_npz(asc.rho*asc.vel1 * asc.r**2*(asc.vel1>0)) * Omega
		mdot_mid = (asc.rho*asc.vel1* asc.r**2)[:,asc.ny//2,0] * Omega

		Jdot = asc.angle_average_npz((asc.vel1*asc.vel3*asc.r*np.sin(asc.th)*asc.r**2*asc.rho))*Omega
		Jdot_mid = ((asc.vel1*asc.vel3*asc.r*np.sin(asc.th)*asc.r**2*asc.rho))[:,asc.ny//2,0]*Omega
		rho_fft = np.fft.fft(asc.rho[::2,asc.ny//2,:],axis=-1)
		mdot_fft = np.fft.fft((asc.rho*asc.vel1*4.0*np.pi*asc.r**2.0)[::2,asc.ny//2,:],axis=-1)

		Mnet_r_inner_bound =asc.dM_r_inner
		Mnet_r_outer_bound = asc.dM_r_outer
		Mnet_r_inner_bound_pos = asc.dM_r_inner_pos
		dth_new = np.gradient(asc.th,axis=1)
		dph_new = np.gradient(asc.ph,axis=-1)
		Mnet_r = np.sum(np.sum(asc.fluxr/asc.dth/asc.dph*dth_new*dph_new,axis=-1),axis=-1)


		dic = {}
		dic['t'] = asc.t
		dic['r'] = asc.r[::2,0,0]
		dic['th'] = asc.th[0,:,0]

		dic['mdot'] = mdot[::2]
		dic['mdot_out'] = mdot_out[::2]


		dic['Mnet_r_inner_bound'] = Mnet_r_inner_bound
		dic['Mnet_r_inner_bound_pos'] = Mnet_r_inner_bound_pos

		dic['Mnet_r_outer_bound'] = Mnet_r_outer_bound
		dic['Mnet_r'] = Mnet_r[::2]

		dic['dMfloor'] = asc.dM_floor_.sum()

	

		dr_new = np.gradient(asc.r[:,0,0])
		dic['M'] = (asc.angle_average_npz(asc.rho*asc.r**2.0)*Omega * dr_new).sum()
		dic['M50'] = (asc.angle_average_npz(asc.rho*asc.r**2.0)*Omega * dr_new * (asc.r[:,0,0]<50.0)).sum()
		dic['M_cum'] = (asc.angle_average_npz(asc.rho*asc.r**2.0)*Omega * dr_new).cumsum()
		dic['mdot_mid'] = mdot_mid[::2]

		dic['Jdot'] = Jdot[::2]
		dic['Jdot_mid'] = Jdot_mid[::2]

		dic['rho'] = asc.rho[::2,asc.ny//2,0]
		dic['rho_fourier'] = [rho_fft[:,0],rho_fft[:,1],rho_fft[:,2],rho_fft[:,3],rho_fft[:,4]]
		dic['mdot_fourier'] = [mdot_fft[:,0],mdot_fft[:,1],mdot_fft[:,2],mdot_fft[:,3],mdot_fft[:,4]]
		dic['phi_fourier'] = np.fft.fftfreq(int(asc.nz),np.diff(asc.ph[0,0,:])[0])
		ir = asc.r_to_ir_npz(5.0,asc.r)
		if (is_magnetic==True): dic['Bphi_r_5'] = asc.Bcc3[ir,:].mean(-1)
		if (is_magnetic==True): dic['BphiBr_mid'] = (asc.Bcc3*asc.Bcc1)[:,asc.ny//2,0]

		np.savez("1d_dump_%04d.npz" %i_dump,**dic)
def mk_1d_quantities_cartesian(is_magnetic =True,spin_orbit=False):
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	if (spin_orbit): orbit_file = glob.glob("*orbit*.dat")[0]
	else: orbit_file = None 
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		dump_npz = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(i_dump,th_tilt,phi_tilt)
		if glob.glob("*out2*athdf") != []:
			dump_file_prefix = glob.glob("*out2*.athdf")[0][:-11]
			dump_hdf5 = dump_file_prefix + "%05d.athdf" %i_dump
		else: dump_hdf5 = ""
		if (os.path.isfile(dump_npz) or os.path.isfile(dump_hdf5) ): 
			asc.yt_load(i_dump,gr=True,a=a)

			if (orbit_file==None): 
				ax=0
				ay=0
				az=a*1.0
				atot = np.sqrt(ax**2+ay**2+az**2)
			else:
				asc.rd_binary_orbits(orbit_file)
				asc.get_binary_quantities(np.array(asc.ds.current_time),t0)

				ax = asc.a1x
				ay = asc.a1y
				az = asc.a1z
				atot = np.sqrt(ax**2+ay**2+az**2)
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,ax=ax,ay=ay,az=az,th=th_tilt,ph=phi_tilt,rmin=0.5,rmax=1e3)
		else: 
			print("Skipping dump: ", i_dump)
			continue 

		def SQR(var):
			return var**2.0

		a_dot_x = ax * asc.x + ay * asc.y + az * asc.z

		a_cross_x = [0,0,0]
		a_cross_x[0] = ay * asc.z - az * asc.y;
		a_cross_x[1] = az * asc.x - ax * asc.z;
		a_cross_x[2] = ax * asc.y - ay * asc.x;


		rsq_p_asq = SQR(asc.r) + SQR(atot);

		lx = (asc.r * asc.x - a_cross_x[0] + a_dot_x * ax/asc.r)/(rsq_p_asq);
		ly = (asc.r * asc.y - a_cross_x[1] + a_dot_x * ay/asc.r)/(rsq_p_asq);
		lz = (asc.r * asc.z - a_cross_x[2] + a_dot_x * az/asc.r)/(rsq_p_asq);

		lz[lz>1.0] = 1.0 + 0.0* lz[lz>1.0]
		lz[lz<-1.0] = -1.0 + 0.0* lz[lz<-1.0]

		asc.th = np.arccos(lz); 
		asc.ph = np.arctan2(ly,lx); 

		##this should be correct
		gdet = np.sqrt( np.sin(asc.th)**2.0 * ( asc.r**2.0 + a_dot_x**2.0/asc.r**2.0)**2.0 )

		asc.get_mdot(mhd=True,gr=True,ax=ax,ay=ay,az=az)
		# asc.th = np.arccos(asc.z/asc.r)
		# asc.ph = np.arctan2(asc.y,asc.x)
		mdot = asc.angle_average_npz(nan_to_num(asc.mdot), gr=True,gdet=gdet)
		mdot_out = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0)), gr=True,gdet=gdet)
		Br = (asc.bu_ks[1] * asc.uu_ks[0] - asc.bu_ks[0]* asc.uu_ks[1])

		Phibh = asc.angle_average_npz(0.5*np.fabs(Br)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2),gr=True,gdet=gdet) 


		# step = -1.0*(asc.th>np.pi/2.0) +1.0*(asc.th<=np.pi/2.0)
		# Phibh_net = asc.angle_average_npz(0.5*(Br*step)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 
		# Phibh_x = asc.angle_average_npz(0.5*(asc.Bcc1)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 
		# Phibh_y = asc.angle_average_npz(0.5*(asc.Bcc2)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 
		# Phibh_z = asc.angle_average_npz(0.5*(asc.Bcc3)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 

		asc.g = asc.ks_metric_general(asc.x,asc.y,asc.z,ax,ay,az)
		# asc.ks_metric(asc.r,asc.th,a)

		uu_ks = nan_to_num(asc.uu_ks)
		bu_ks = nan_to_num(asc.bu_ks)
		ud_ks = nan_to_num(asc.Lower(asc.uu_ks,asc.g))
		bd_ks = nan_to_num(asc.Lower(asc.bu_ks,asc.g))


		asc.Tud_calc(asc.uu_ks,ud_ks,asc.bu_ks,bd_ks,is_magnetic= is_magnetic,gam=gam)
		Jdot = asc.angle_average_npz(asc.Tud[1][3]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2),gr=True,gdet=gdet)
		Edot = - (asc.angle_average_npz(asc.Tud[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2),gr=True,gdet=gdet) +mdot )
		EdotEM = -asc.angle_average_npz(asc.TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2),gr=True,gdet=gdet)

		r2 = asc.r**2.
		a2 = atot**2.
		sin2 = np.sin(asc.th)**2.
		sigma = asc.r**2 + atot**2.*np.cos(asc.th)**2;
		delta = r2 - 2.0*asc.r + a2

		gittBL = -1.0/delta * (r2+a2 + 2*asc.r*a2/sigma*sin2)

		# eta_lower_BL = np.array([-1.0/sqrt(-gittBL),-2.0*r/delta* 1.0/sqrt(-gittBL),0*1.0/sqrt(-gittBL),0*1.0/sqrt(-gittBL)])
		# eta_upper_BL = 
		ud_zamo =  np.array([-1.0/sqrt(-gittBL),2.0*asc.r/delta* 1.0/sqrt(-gittBL),0*1.0/sqrt(-gittBL),0*1.0/sqrt(-gittBL)])

		EdotKE_grav =  -(angle_average_npz( (asc.rho*asc.uu_ks[1]*ud_ks[0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2)),gr=True,gdet=gdet) + mdot )
		#EdotKE_grav =  -(asc.rho*asc.uu_ks[1]*ud_ks[0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2) + nan_to_num(asc.mdot) )
		Edotgrav = -angle_average_npz( (asc.rho*asc.uu_ks[1]*(ud_zamo[0]+1.0)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2) ),gr=True,gdet=gdet)
		EdotMA = Edot-EdotEM
		EdotUint = EdotMA-EdotKE_grav

		Lx = asc.angle_average_npz(asc.rho* (asc.y*asc.uu[3] - asc.z*asc.uu[2]),gr=True,gdet=gdet)
		Ly = asc.angle_average_npz(asc.rho* (asc.z*asc.uu[1] - asc.x*asc.uu[3]),gr=True,gdet=gdet)
		Lz = asc.angle_average_npz(asc.rho* (asc.x*asc.uu[2] - asc.y*asc.uu[1]),gr=True,gdet=gdet)

		Bx = asc.angle_average_npz(asc.Bcc1,gr=True,gdet=gdet)
		By = asc.angle_average_npz(asc.Bcc2,gr=True,gdet=gdet)
		Bz = asc.angle_average_npz(asc.Bcc3,gr=True,gdet=gdet)

		A_jet_p =  asc.angle_average_npz( 4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2) * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0) ,gr=True,gdet=gdet)
		A_jet_m =  asc.angle_average_npz( 4.0 * np.pi/3.0 * (3.0*asc.r**2 + atot**2) * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th>=np.pi/2.0) ,gr=True,gdet=gdet)

		if (np.amax(A_jet_p>0)*1.0>0): rjet_max_p= (np.amax(asc.r[A_jet_p>0]))
		else: rjet_max_p = 0.0 
		if (np.amax(A_jet_m>0)*1.0>0): rjet_max_m= (np.amax(asc.r[A_jet_m>0]))
		else: rjet_max_m = 0
		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0)

		x_jet_p = angle_average_npz( asc.x,weight=wgt ,gr=True,gdet=gdet)
		y_jet_p = angle_average_npz( asc.y,weight=wgt ,gr=True,gdet=gdet)
		z_jet_p = angle_average_npz( asc.z,weight=wgt ,gr=True,gdet=gdet)


		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0)


		x_cavity_p = asc.angle_average_npz(asc.x,weight=1.0/asc.rho * (asc.th<np.pi/2.0),gr=True,gdet=gdet)
		y_cavity_p = asc.angle_average_npz(asc.y,weight=1.0/asc.rho * (asc.th<np.pi/2.0),gr=True,gdet=gdet)
		z_cavity_p = asc.angle_average_npz(asc.z,weight=1.0/asc.rho * (asc.th<np.pi/2.0),gr=True,gdet=gdet)


		# rho_min_array_r = np.amin(np.amin(asc.rho[asc.th<pi/2].reshape(asc.nx,-1,asc.nz),axis=-1),axis=-1)

		# x_cavity_p = []
		# y_cavity_p = []
		# z_cavity_p = []
		# for i_ in np.arange(len(asc.r[:,0,0])):
		# 	index_ = np.where(asc.rho[i_,:,:]==rho_min_array_r[i_])
		# 	x_cavity_p.append(asc.x[index_])
		# 	y_cavity_p.append(asc.y[index_])
		# 	z_cavity_p.append(asc.z[index_])
		# x_cavity_p = np.array(x_cavity_p)
		# z_cavity_p = np.array(y_cavity_p)
		# z_cavity_p = np.array(z_cavity_p)


		asc.cks_metric(asc.x,asc.y,asc.z,ax,ay,az)

		ud = asc.Lower(asc.uu,asc.g)

		Be = -(1.0 + gam/(gam-1.0)*asc.press/asc.rho  )*ud[0] - 1.0
		asc.cks_inverse_metric(asc.x,asc.y,asc.z,ax,ay,az)
		alpha = np.sqrt(-1.0/asc.gi[0,0])
		gamma = asc.uu[0] * alpha

		#black hole spin vector = (0,ax,zy,az)
		#direction vector (0,x,y,z)
		# spin_axis_angle = np.arccos(a_dot_x/np.sqrt(ax**2+ay**2+az**2)/np.sqrt(asc.x**2+asc.y**2+asc.z**2))

		gamma_thresh = 1.005 ##(v>0.1c)

		mdot_out_up_1c =   asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th<np.pi/2.0) * (gamma>gamma_thresh) ), gr=True,gdet=gdet)

		mdot_out_down_1c = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th>2.0*np.pi/4.0) * (gamma>gamma_thresh)), gr=True,gdet=gdet)


		gamma_thresh = 1.02 ##(v>0.2c)

		mdot_out_up_2c   = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th<np.pi/2.0) * (gamma>gamma_thresh) ), gr=True,gdet=gdet)

		mdot_out_down_2c = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th>2.0*np.pi/4.0) * (gamma>gamma_thresh)), gr=True,gdet=gdet)



		gamma_thresh = 1.05 ##(v>0.3c)

		mdot_out_up_3c =  asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th<np.pi/2.0) * (gamma>gamma_thresh) ), gr=True,gdet=gdet)

		mdot_out_down_3c = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th>2.0*np.pi/4.0) * (gamma>gamma_thresh)), gr=True,gdet=gdet)


		gamma_thresh = 1.091 ##(v>0.4c)

		mdot_out_up_4c =  asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th<np.pi/2.0) * (gamma>gamma_thresh) ), gr=True,gdet=gdet)

		mdot_out_down_4c = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th>2.0*np.pi/4.0) * (gamma>gamma_thresh)), gr=True,gdet=gdet)


		gamma_thresh = 1.155 ##(v>0.5c)

		mdot_out_up_5c =  asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th<np.pi/2.0) * (gamma>gamma_thresh) ), gr=True,gdet=gdet)

		mdot_out_down_5c = asc.angle_average_npz(nan_to_num(asc.mdot * (asc.mdot>0) * (asc.th>2.0*np.pi/4.0) * (gamma>gamma_thresh)), gr=True,gdet=gdet)


		mdot_out_unbound = asc.angle_average_npz(nan_to_num(asc.mdot * (Be>0) ), gr=True,gdet=gdet)

		gamma_jet_p = angle_average_npz(gamma,weight= ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0),gr=True,gdet=gdet)
		gamma_jet_m = angle_average_npz(gamma,weight= ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th>np.pi/2.0),gr=True,gdet=gdet)


		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th>=np.pi/2.0)

		x_jet_m = asc.angle_average_npz( asc.x,weight=wgt ,gr=True,gdet=gdet)
		y_jet_m = asc.angle_average_npz( asc.y,weight=wgt ,gr=True,gdet=gdet)
		z_jet_m = asc.angle_average_npz( asc.z,weight=wgt ,gr=True,gdet=gdet)

		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) 
		vr_jet = asc.angle_average_npz(asc.uu_ks[1]/asc.uu_ks[0],weight=wgt,gr=True,gdet=gdet)


		try:
			asc.ke_ent
		except:
			print("no electrons")
		else:
			ue = asc.kappa_to_ue(asc.ke_ent,asc.rho,gr=True,mue=2.0)
			def frel(the):
				return (log(the)*(the-1)/(2*the**3) + 1/the**2) * (the>1) + 1.0 * (the<1)



			
			dr = np.gradient(asc.r[:,0,0])
			rho_Bcc3_dr_los_1 = (asc.rho*asc.Bcc3*frel(asc.theta_e))[:,-1,0] * dr
			
			ue = asc.kappa_to_ue(asc.ke_ent2,asc.rho,gr=True,mue=2.0)
			rho_Bcc3_dr_los_2 = (asc.rho*asc.Bcc3*frel(asc.theta_e))[:,-1,0] * dr
			ue = asc.kappa_to_ue(asc.ke_ent3,asc.rho,gr=True,mue=2.0)
			rho_Bcc3_dr_los_3 = (asc.rho*asc.Bcc3*frel(asc.theta_e))[:,-1,0] * dr

			rh = ( 1.0 + np.sqrt(1.0-a**2) )
			rho_Bcc3_dr_los_sum_1 = ( rho_Bcc3_dr_los_1*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)*(asc.bsq/asc.rho<1)[:,-1,0]).sum()
			rho_Bcc3_dr_los_sum_2 = ( rho_Bcc3_dr_los_2*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)*(asc.bsq/asc.rho<1)[:,-1,0]).sum()
			rho_Bcc3_dr_los_sum_3 = ( rho_Bcc3_dr_los_3*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)*(asc.bsq/asc.rho<1)[:,-1,0]).sum()

			rho_Bcc3_dr_los_cumsum_1 = ( rho_Bcc3_dr_los_1*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)*(asc.bsq/asc.rho<1)[:,-1,0]).cumsum()
			rho_Bcc3_dr_los_cumsum_2 = ( rho_Bcc3_dr_los_2*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)*(asc.bsq/asc.rho<1)[:,-1,0]).cumsum()
			rho_Bcc3_dr_los_cumsum_3 = ( rho_Bcc3_dr_los_3*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)*(asc.bsq/asc.rho<1)[:,-1,0]).cumsum()


		dic = {}
		dic['t'] = np.array(asc.t)
		dic['r'] = asc.r[::4,0,0]
		dic['mdot'] = mdot[::4]
		dic['Jdot'] = Jdot[::4]
		dic['Edot'] = Edot[::4]
		dic['Phibh'] = Phibh[::4]

		dic['mdot_out'] = mdot_out[::4]


		dic['mdot_out_up_1c'] = mdot_out_up_1c[::4]
		dic['mdot_out_up_2c'] = mdot_out_up_2c[::4]
		dic['mdot_out_up_3c'] = mdot_out_up_3c[::4]
		dic['mdot_out_up_4c'] = mdot_out_up_4c[::4]
		dic['mdot_out_up_5c'] = mdot_out_up_5c[::4]
		dic['mdot_out_unbound'] = mdot_out_unbound[::4]


		dic['mdot_out_down_1c'] = mdot_out_down_1c[::4]
		dic['mdot_out_down_2c'] = mdot_out_down_2c[::4]
		dic['mdot_out_down_3c'] = mdot_out_down_3c[::4]
		dic['mdot_out_down_4c'] = mdot_out_down_4c[::4]
		dic['mdot_out_down_5c'] = mdot_out_down_5c[::4]

		# dic['Phibh_net'] = Phibh_net[::4]
		# dic['Phibh_x'] = Phibh_x[::4]
		# dic['Phibh_y'] = Phibh_y[::4]
		# dic['Phibh_z'] = Phibh_z[::4]
		dic['EdotEM'] = EdotEM[::4]
		dic['EdotKE_grav'] = EdotKE_grav[::4]
		dic['Edotgrav'] = Edotgrav[::4]
		dic['EdotUint'] = EdotUint[::4]

		dic['Lx'] = Lx[::4]
		dic['Ly'] = Ly[::4]
		dic['Lz'] = Lz[::4]
		dic['Bx'] = Bx[::4]
		dic['By'] = By[::4]
		dic['Bz'] = Bz[::4]
		dic['A_jet_p'] = A_jet_p[::4]
		dic['A_jet_m'] = A_jet_m[::4]
		dic['x_jet_p'] = x_jet_p[::4]
		dic['y_jet_p'] = y_jet_p[::4]
		dic['z_jet_p'] = z_jet_p[::4]
		dic['x_jet_m'] = x_jet_m[::4]
		dic['y_jet_m'] = y_jet_m[::4]
		dic['z_jet_m'] = z_jet_m[::4]
		dic['x_cavity_p'] = x_cavity_p[::4]
		dic['y_cavity_p'] = y_cavity_p[::4]
		dic['z_cavity_p'] = z_cavity_p[::4]
		dic['rjet_max_p'] = rjet_max_p
		dic['rjet_max_m'] = rjet_max_m
		dic['gamma_jet_m'] = gamma_jet_m[::4]
		dic['gamma_jet_p'] = gamma_jet_p[::4]
		dic['vr_jet'] = vr_jet[::4]


		try:
			asc.ke_ent
		except: 
			print("no electrons")
		else:
			dic['rho_Bcc3_dr_los_1'] = rho_Bcc3_dr_los_1[::4]
			dic['rho_Bcc3_dr_los_sum_1'] = rho_Bcc3_dr_los_sum_1
			dic['rho_Bcc3_dr_los_cumsum_1'] = rho_Bcc3_dr_los_cumsum_1[::4]

			dic['rho_Bcc3_dr_los_2'] = rho_Bcc3_dr_los_2[::4]
			dic['rho_Bcc3_dr_los_sum_2'] = rho_Bcc3_dr_los_sum_2
			dic['rho_Bcc3_dr_los_cumsum_2'] = rho_Bcc3_dr_los_cumsum_2[::4]

			dic['rho_Bcc3_dr_los_3'] = rho_Bcc3_dr_los_3[::4]
			dic['rho_Bcc3_dr_los_sum_3'] = rho_Bcc3_dr_los_sum_3
			dic['rho_Bcc3_dr_los_cumsum_3'] = rho_Bcc3_dr_los_cumsum_3[::4]


		np.savez("1d_dump_%04d.npz" %i_dump,**dic)
		dic_torus = {}


		gamma_beta = np.sqrt( (asc.Tud[1][0]/(asc.rho*asc.uu_ks[1]))**2.0 - 1.0 )
		wgt = (gamma_beta>1)


		dic_torus['PTOT_jet'] = -asc.angle_integral_npz(asc.Tud[1][0],weight=wgt,gr=True,gdet=gdet)
		dic_torus['PEM_jet'] = -asc.angle_integral_npz(asc.TudEM[1][0],weight=wgt,gr=True,gdet=gdet)
		dic_torus['PPAKE_jet'] = -(asc.angle_integral_npz( (asc.rho*asc.uu_ks[1]*(ud_ks[0]+1.0)),weight=wgt,gr=True,gdet=gdet) )
		#PMA = PTOT-PEM
		#PEN = PMA-PPAKE
		dic_torus['PEN_jet'] = -(asc.angle_integral_npz( (ud_ks[0]*asc.uu_ks[1]*(gam/(gam-1.0)*asc.press)),weight=wgt,gr=True,gdet=gdet) )
		dic_torus['Mdot_jet']= asc.angle_integral_npz(nan_to_num(asc.rho*asc.uu_ks[1]),weight=wgt, gr=True,gdet=gdet)
		# thmin = pi/3.0
		# thmax = 2.0*pi/3.0

		thmin = 0
		thmax = np.pi

		asc.get_mdot(mhd=True,gr=True,ax=ax,ay=ay,az=az)
		uu_ks = nan_to_num(asc.uu_ks)

		dic_torus['t'] = np.array(asc.t)
		dic_torus['r'] = asc.r[:,0,0]
		dic_torus['rho'] = asc.angle_average_npz(asc.rho,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,gdet=gdet)
		dic_torus['press'] = asc.angle_average_npz(asc.press,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,gdet=gdet)
		dic_torus['beta_inv'] =asc.angle_average_npz(asc.bsq/asc.press/2.0,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,gdet=gdet)
		dic_torus['pmag'] = asc.angle_average_npz(asc.bsq/2.0,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,gdet=gdet)

		dic_torus['uu_phi'] = asc.angle_average_npz(asc.uu_ks[3],weight = (asc.th>thmin)*(asc.th<thmax),gr=True,gdet=gdet)

		dic_torus['H_r'] = asc.angle_average_npz(np.abs(np.pi/2.0-asc.th),weight = asc.rho,gr=True,gdet=gdet)



		np.savez("1d_torus_dump_%05d.npz" %i_dump,**dic_torus)

def mk_1d_quantities_cartesian_newtonian(is_magnetic =True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		dump_npz = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(i_dump,th_tilt,phi_tilt)
		if glob.glob("*out2*athdf") != []:
			dump_file_prefix = glob.glob("*out2*.athdf")[0][:-11]
			dump_hdf5 = dump_file_prefix + "%05d.athdf" %i_dump
		else: dump_hdf5 = ""
		if (os.path.isfile(dump_npz) or os.path.isfile(dump_hdf5) ): asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=False,th=th_tilt,ph=phi_tilt)
		else: 
			print("Skipping dump: ", i_dump)
			continue 
		asc.get_mdot(mhd=True,gr=False)
		asc.th = np.arccos(asc.z/asc.r)
		asc.ph = np.arctan2(asc.y,asc.x)
		mdot = asc.angle_average_npz(nan_to_num(asc.mdot), gr=False)
		Br = asc.Bcc1 * (asc.x/asc.r) + asc.Bcc2 * (asc.y/asc.r) + asc.Bcc3 * (asc.z/asc.r)

		Phibh = asc.angle_average_npz(0.5*np.fabs(Br)*np.sqrt(4.0*pi)*4.0 * np.pi*asc.r**2 ,gr=False) 



		Lx = asc.angle_average_npz(asc.rho* (asc.y*asc.vel3 - asc.z*asc.vel2),gr=False)
		Ly = asc.angle_average_npz(asc.rho* (asc.z*asc.vel1 - asc.x*asc.vel3),gr=False)
		Lz = asc.angle_average_npz(asc.rho* (asc.x*asc.vel2 - asc.y*asc.vel1),gr=False)

		Bx = asc.angle_average_npz(asc.Bcc1,gr=False)
		By = asc.angle_average_npz(asc.Bcc2,gr=False)
		Bz = asc.angle_average_npz(asc.Bcc3,gr=False)


		# try:
		# 	asc.ke_ent
		# except:
		# 	print("no electrons")
		# else:
		# 	ue = asc.kappa_to_ue(asc.ke_ent,asc.rho,gr=False,mue=2.0)
		# 	def frel(the):
		# 		return (log(the)*(the-1)/(2*the**3) + 1/the**2) * (the>1) + 1.0 * (the<1)



			
		# 	dr = np.gradient(asc.r[:,0,0])
		# 	rho_Bcc3_dr_los_1 = (asc.rho*asc.Bcc3*frel(asc.theta_e))[:,-1,0] * dr
			
		# 	ue = asc.kappa_to_ue(asc.ke_ent2,asc.rho,gr=False,mue=2.0)
		# 	rho_Bcc3_dr_los_2 = (asc.rho*asc.Bcc3*frel(asc.theta_e))[:,-1,0] * dr
		# 	ue = asc.kappa_to_ue(asc.ke_ent3,asc.rho,gr=False,mue=2.0)
		# 	rho_Bcc3_dr_los_3 = (asc.rho*asc.Bcc3*frel(asc.theta_e))[:,-1,0] * dr

		# 	rh = 2.0
		# 	rho_Bcc3_dr_los_sum_1 = ( rho_Bcc3_dr_los_1*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)[:,-1,0]).sum()
		# 	rho_Bcc3_dr_los_sum_2 = ( rho_Bcc3_dr_los_2*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)[:,-1,0]).sum()
		# 	rho_Bcc3_dr_los_sum_3 = ( rho_Bcc3_dr_los_3*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)[:,-1,0]).sum()

		# 	rho_Bcc3_dr_los_cumsum_1 = ( rho_Bcc3_dr_los_1*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)[:,-1,0]).cumsum()
		# 	rho_Bcc3_dr_los_cumsum_2 = ( rho_Bcc3_dr_los_2*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)[:,-1,0]).cumsum()
		# 	rho_Bcc3_dr_los_cumsum_3 = ( rho_Bcc3_dr_los_3*(asc.r[:,0,0]>rh)*(asc.r[:,0,0]<1600)[:,-1,0]).cumsum()


		dic = {}
		dic['t'] = np.array(asc.t)
		dic['r'] = asc.r[::4,0,0]
		dic['mdot'] = mdot[::4]
		dic['Phibh'] = Phibh[::4]


		dic['Lx'] = Lx[::4]
		dic['Ly'] = Ly[::4]
		dic['Lz'] = Lz[::4]
		dic['Bx'] = Bx[::4]
		dic['By'] = By[::4]
		dic['Bz'] = Bz[::4]


		# try:
		# 	asc.ke_ent
		# except: 
		# 	print("no electrons")
		# else:
		# 	dic['rho_Bcc3_dr_los_1'] = rho_Bcc3_dr_los_1[::4]
		# 	dic['rho_Bcc3_dr_los_sum_1'] = rho_Bcc3_dr_los_sum_1
		# 	dic['rho_Bcc3_dr_los_cumsum_1'] = rho_Bcc3_dr_los_cumsum_1[::4]

		# 	dic['rho_Bcc3_dr_los_2'] = rho_Bcc3_dr_los_2[::4]
		# 	dic['rho_Bcc3_dr_los_sum_2'] = rho_Bcc3_dr_los_sum_2
		# 	dic['rho_Bcc3_dr_los_cumsum_2'] = rho_Bcc3_dr_los_cumsum_2[::4]

		# 	dic['rho_Bcc3_dr_los_3'] = rho_Bcc3_dr_los_3[::4]
		# 	dic['rho_Bcc3_dr_los_sum_3'] = rho_Bcc3_dr_los_sum_3
		# 	dic['rho_Bcc3_dr_los_cumsum_3'] = rho_Bcc3_dr_los_cumsum_3[::4]


		np.savez("1d_dump_%04d.npz" %i_dump,**dic)


def mk_RM(moving=False):
	set_dump_range()
	os.system("cp /global/scratch/smressle/star_cluster/mhd_runs/star_inputs/PSR.dat ./")
	asc.PSR_pos()
	print ("Processing dumps ",i_start," to ",i_end)
	angle_array = np.linspace(0,2.0*np.pi,10)
	radius = np.sqrt(asc.dalpha[0]**2 + asc.ddelta[0]**2)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.yt_load(i_dump)
		sgra_RM,sgra_DM = asc.get_RM()

		# RM_rand = []
		# DM_rand = []
		# # for angle in angle_array:
		# # 	x_ = radius * np.cos(angle)
		# # 	y_ = radius * np.sin(angle)
		# # 	RM,DM = asc.get_RM(x_,y_,cum = True)[-1]
		# # 	RM_rand.append(RM)
		# # 	DM_rand.append(DM)
		# RM_rand = np.array(RM_rand)
		# DM_rand = np.array(DM_rand)

		simulation_start_time = -1.1
		if (moving == True):
			xp  = asc.dalpha[0] + (np.array(asc.ds.current_time) - (asc.t_yr[0]-2017)/1e3+simulation_start_time)*asc.valpha
			yp  = asc.ddelta[0] + (np.array(asc.ds.current_time) - (asc.t_yr[0]-2017)/1e3+simulation_start_time)*asc.vdelta
		else:
			xp = asc.dalpha[0]
			yp = asc.ddelta[0]
		pulsar_RM,pulsar_DM = asc.get_RM(xp,yp,cum=True)

		dic =  {"t":asc.ds.current_time,"pulsar_RM":pulsar_RM,"pulsar_DM": pulsar_DM,"sgra_RM":sgra_RM,"sgra_DM": sgra_DM,"z_los":asc.z_los} #,"pulsar_RM_rand":RM_rand,"pulsar_DM_rand":DM_rand}
		
		if (moving==True): np.savez("RM_dump_moving_%04d.npz" %i_dump,**dic)
		else: np.savez("RM_dump_%04d.npz" %i_dump,**dic)

def mk_RM_relativistic():
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	def frel(the):
		return (log(the)*(the-1)/(2*the**3) + 1/the**2) * (the>1) + 1.0 * (the<1)
	asc.set_constants()
	RM_0 = 1e4 * asc.e_charge**3/(2*np.pi * asc.me**2 * asc.cl**4)

	asc.set_constants()
	RM_0 = 1e4 * asc.e_charge**3/(2*pi * asc.me**2 * asc.cl**4)
	rg = asc.gm_/(asc.cl/asc.pc*asc.kyr)**2

	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)

		asc.rd_yt_convert_to_spherical(i_dump,MHD=True)		
		asc.rd_yt_convert_to_spherical(i_dump,MHD=True)

		r_in = 0.03125*2.0/128.0/2.0**11.0 * 2.0
		rmin = max(r_in*2.0,rmin_rm*rg)
		#irmin = asc.r_to_ir_npz(r_in *2.0,asc.r)
		irmin = asc.r_to_ir_npz(rmin,asc.r)
		irmax = -1

		r_cm = asc.r*asc.pc
		dr_cm = np.gradient(r_cm[:,0,0])
		ue = asc.kappa_to_ue(asc.ke_ent,asc.rho,gr=False,mue=asc.mue)
		Rm_mhd1_ = (RM_0* asc.Bcc3 * asc.Bunit * asc.rho*asc.rho_to_n_cgs/asc.mue)[:,-1,0] * dr_cm * frel(asc.theta_e[:,-1,0])
		Rm_mhd1 = nan_to_num(Rm_mhd1_)[irmin:irmax].sum()		
		
		ue = asc.kappa_to_ue(asc.ke_ent2,asc.rho,gr=False,mue=asc.mue)
		Rm_mhd2_ = (RM_0* asc.Bcc3 * asc.Bunit * asc.rho*asc.rho_to_n_cgs/asc.mue)[:,-1,0] * dr_cm * frel(asc.theta_e[:,-1,0])
		Rm_mhd2 = nan_to_num(Rm_mhd2_)[irmin:irmax].sum()
		
		ue = asc.kappa_to_ue(asc.ke_ent3,asc.rho,gr=False,mue=asc.mue)
		Rm_mhd3_ = (RM_0* asc.Bcc3 * asc.Bunit * asc.rho*asc.rho_to_n_cgs/asc.mue)[:,-1,0] * dr_cm * frel(asc.theta_e[:,-1,0])
		Rm_mhd3 = nan_to_num(Rm_mhd3_)[irmin:irmax].sum()





		dic =  {"t":asc.t,"RM1": Rm_mhd1,"RM2": Rm_mhd2,"RM3": Rm_mhd3,"RM1_": Rm_mhd1_,"RM2_": Rm_mhd2_,"RM3_": Rm_mhd3_,"r":asc.r[:,0,0] } 
		
		np.savez("RM_dump_%04d.npz" %i_dump,**dic)

def RM_movie():
	from matplotlib.offsetbox import AnchoredText

	set_dump_range()

	e_charge = 4.803e-10
	me = 9.109e-28
	cl = 2.997924e10
	mp = 1.6726e-24
	pc = 3.086e18
	kyr = 3.154e10
	msun = 1.989e33
	R_sun = 6.955e10
	km_per_s = 1e5

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

	#fig = plt.figure(figsize=(10,10))
	#fig.patch.set_facecolor('black')

	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.yt_load(i_dump)
		asc.ds.add_field(("gas","RM_integrand"),function = _RM_integrand,units="cm**-3",particle_type = False,sampling_type="cell",force_override=True)


		box_radius = 0.2
		region = asc.ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,  #512
		    (-box_radius,'pc'):(box_radius,'pc'):256j,   #512
		    (-1,'pc'):(1,'pc'):1028j ]
		RM_map = np.array(region['RM_integrand'].mean(-1).in_cgs()) * 2 * pc

		for sat in [True,False]:
			plt.clf()
			plt.style.use('dark_background')
			max_RM = 1.5
			min_RM = -1.5
			#c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = "RdBu",vmin=np.log10(6.6)-1.5,vmax=np.log10(6.6)+1.5)
			if sat == True : c = matplotlib.pyplot.contourf(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = "autumn",levels = np.linspace(np.log10(6.6)-.01,max_RM,200),extend = "max")
			else: c = matplotlib.pyplot.contourf(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = "cubehelix",levels = np.linspace(min_RM,max_RM,200),extend = "both")
			plt.xlim(box_radius,-box_radius)
			plt.ylim(-box_radius,box_radius)

			plt.axis('off')
			#plt.axes().set_aspect('equal')


			# text_box = AnchoredText(r'$t$ = %g yr' %(int(np.round(asc.t*1000-1100))), frameon=True, loc=4, pad=0.5)
			# text_box = AnchoredText(r'$t$ = %g yr' %(int(np.round(t*1000-1100))), frameon=True, loc=4, pad=0.5)
			# plt.setp(text_box.patch, facecolor='black', alpha=.9)
			# plt.gca().add_artist(text_box)

			plt.tight_layout()

			os.system("mkdir -p frames")
			if (sat == True):
			    plt.savefig("frames/frame_RM_sat_%d.png" % (i_dump))
			else: 
			    plt.savefig("frames/frame_RM_%d.png" % (i_dump))


def mk_Xray_frame():
	from matplotlib.colors import LinearSegmentedColormap

	set_dump_range()

	for i_dump in range(i_start,i_end):
		asc.yt_load(i_dump)
		plt.clf()
		asc.get_Xray_Lum('Lam_spex_Z_solar_2_8_kev',1.0,make_image=True)

		colors = ["black","#004851","#8D9093","#FFFFFF" ] #"#000000","#54585A" ,"#8D9093","#FFFFFF", "#004851"]  # R -> G -> B

		#colors = ["#000000", "#4cbb17","#FFFFFF" ] #"#000000","#54585A" ,"#8D9093","#FFFFFF", "#004851"]  # R -> G -> B
		#colors = ["#4cbb17", "#C0C0C0"]
		cmap_name = 'my_list'
		cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

		cm = 'magma'


		x_im = np.array(x_im.tolist())
		y_im = np.array(y_im.tolist())
		image = np.array(image.tolist())
		fac = int(image.shape[0]/50.)+1
		new_x = np.linspace(-0.492*50,0.492*50,fac*100)  #np.arange(-50*fac,50*fac)*0.492
		new_y = np.linspace(-0.492*50,0.492*50,fac*100)
		new_x,new_y = meshgrid(new_x,new_y,indexing='ij')
		from scipy.interpolate import griddata
		new_image = griddata((x_im.flatten()/arc_secs,y_im.flatten()/arc_secs), image.flatten(), (new_x.flatten(), new_y.flatten()), method='nearest')
		new_image = new_image.reshape(new_x.shape[0],new_y.shape[1])

		coarseness = fac
		temp = new_image.reshape((new_image.shape[0] // coarseness, coarseness,
		            new_image.shape[1] // coarseness, coarseness))
		coarse_new_image = np.mean(temp, axis=(1,3))
		temp = new_x.reshape((new_x.shape[0] // coarseness, coarseness,
		            new_x.shape[1] // coarseness, coarseness))
		coarse_new_x = np.mean(temp, axis=(1,3))
		temp = new_y.reshape((new_y.shape[0] // coarseness, coarseness,
		            new_y.shape[1] // coarseness, coarseness))
		coarse_new_y = np.mean(temp, axis=(1,3))

		#c1 = plt.pcolormesh(x_im/arc_secs,y_im/arc_secs,log10(image),levels = np.linspace(-5,0,200),cmap = 'magma') #extend= 'both',cmap = cm) #"magma")
		c1 = plt.pcolormesh(coarse_new_x,coarse_new_y,log10(coarse_new_image*3.575e-93), cmap = cm,vmin = -3.5,vmax=-0.5) #vmin = -3,vmax = 0) #levels = np.linspace(-3.5,0,200),cmap = 'magma') #extend= 'both',cmap = cm) #"magma")

		#c1 = plt.pcolormesh(coarse_new_x,coarse_new_y,log10(coarse_new_image*3.575e-93), cmap = cm,vmin = -2.5,vmax=1) #vmin = -3,vmax = 0) #levels = np.linspace(-3.5,0,200),cmap = 'magma') #extend= 'both',cmap = cm) #"magma")

		#c1 = plt.pcolormesh(new_x,new_y,log10(new_image*3.575e-93), cmap = cm,vmin = -3.5,vmax=-0.5)
		plt.gca().invert_xaxis()
		#c1 = plt.pcolormesh(x_im[::3,::3]/arc_secs,y_im[::3,::3]/arc_secs,log10(image[::3,::3]), cmap = 'magma',vmin = -2,vmax = 0)
		#
		cb1 = plt.colorbar(c1)
		cb1.set_ticks(np.arange(-10,10,.5))
		#    cb1.set_label(r"$\rho/\langle \rho \rangle - 1 $",fontsize=17)
		#    

		for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels() +cb1.ax.get_yticklabels():
			label.set_fontsize(10)


		plt.xlabel(r'$\Delta$RA Offset from Sgr A* (arcsec)',fontsize=17)
		plt.ylabel(r'$\Delta$Dec Offset from Sgr A* (arcsec)',fontsize=17)
		cb1.set_label(r"X-ray Sfc. Brt. (erg/cm$^2$/s)",fontsize=17)


		plt.axes().set_aspect('equal')
		xlim(10,-10)
		ylim(-10,10)
		plt.savefig("frame_X_ray_%03d.png")
    #circ = matplotlib.patches.Circle((dalpha[0]/arc_secs,ddelta[0]/arc_secs),radius = .5,fill=False,ls='--',lw=3,color='yellow')
    #matplotlib.pyplot.gca().add_artist(circ)
    # xlim(5,-5)
    # ylim(-5,5)


def mk_frame_disk():
	set_dump_range()

	os.system("mkdir -p frames")
	for idump in range(i_start,i_end):
		asc.rdnpz("dump_spher_disk_frame_%04d.npz" %idump)
		asc.get_mdot()
		nx = asc.x.shape[0]
		ny = asc.x.shape[1]
		nz = asc.x.shape[2]
		x = asc.x*1e3 #x_tavg*1e3
		y = asc.y*1e3
		z = asc.z*1e3 #z_tavg*1e3
		plt.figure(1)
		plt.clf()
		r_tmp = np.sqrt(x**2. + y**2. + z**2.)
		c = plt.contourf(x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,0]),levels = np.linspace(-2,2,200),extend='both',cmap = 'bds_highcontrast')
		plt.contourf(-x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,nz//2]),levels = np.linspace(-2,2,200),extend='both',cmap = 'bds_highcontrast')

		plt.xlabel(r'$x$ (mpc)',fontsize = 20)
		plt.ylabel(r'$z$ (mpc)',fontsize = 20)

		cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
		cb.set_label(r"$\langle \rho r \rangle$ $M_\odot/$pc$^2$",fontsize=17)


		for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			label.set_fontsize(10)
		plt.tight_layout()

		fac = 1
		plt.xlim(-0.003*3.*1e3*fac,0.003*3.*1e3*fac)
		plt.ylim(-0.003*3.*1e3*fac,0.003*3.*1e3*fac)
		plt.savefig('frames/frame_rho_disk_%04d.png' %idump)


		plt.figure(2)
		plt.clf()

		c = plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10(asc.rho[:,ny//2,:]),cmap = 'bds_highcontrast',vmin=1,vmax=3.5)
		#plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10(Bth*Br)[:,ny//2,:],cmap = 'cubehelix',vmin=-1,vmax=4.25)

		cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
		cb.set_label(r"$\langle \rho \rangle$ $M_\odot/$pc$^3$",fontsize=17)


		fac = 1 #30 #1 ##30
		plt.xlim(-0.003*3*1e3*fac,.003*3*1e3*fac)
		plt.ylim(-0.003*3*1e3*fac,.003*3*1e3*fac)


		for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			label.set_fontsize(10)
		plt.tight_layout()
		plt.savefig('frames/frame_rho_disk_midplane_%04d.png' %idump)


def mk_frame_L_aligned(fieldlines=False):
	set_dump_range()

	if len(sys.argv)>4:
		th_l = np.float(sys.argv[4])
		phi_l = np.float(sys.argv[5])
	else:
		th_l = 1.3
		phi_l = -1.8

	os.system("mkdir -p frames")
	for idump in range(i_start,i_end):
		asc.rdnpz("dump_spher_%d_th_%g_phi_%g.npz" %(idump,th_l,phi_l))
		nx = asc.x.shape[0]
		ny = asc.x.shape[1]
		nz = asc.x.shape[2]
		x = asc.x*1e3 #x_tavg*1e3
		y = asc.y*1e3
		z = asc.z*1e3 #z_tavg*1e3

		asc.x = x
		asc.y = y
		asc.z = z

		if (fieldlines==True): asc.get_mdot(True)

		for fac in [0.33,1,10]:
			plt.figure(1)
			plt.clf()
			r_tmp = np.sqrt(x**2. + y**2. + z**2.)
			c = plt.contourf(x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,0]),levels = np.linspace(-2,2,200),extend='both',cmap = 'viridis',vmin=-1.75,vmax=-.25)
			plt.contourf(-x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,nz//2]),levels = np.linspace(-2,2,200),extend='both',cmap = 'viridis',vmin=-1.75,vmax=-.25)

			if (fieldlines==True): asc.plot_fieldlines_slice(9*fac)

			plt.xlabel(r'$x$ (mpc)',fontsize = 20)
			plt.ylabel(r'$z$ (mpc)',fontsize = 20)

			# cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
			# cb.set_label(r"$\langle \rho r \rangle$ $M_\odot/$pc$^2$",fontsize=17)


			# for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			# 	label.set_fontsize(10)
			plt.tight_layout()

			plt.xlim(-9*fac,9*fac)
			plt.ylim(-9*fac,9*fac)

			plt.axis('off')
			plt.axes().set_aspect('equal')
			if (fieldlines==True):plt.savefig('frames/frame_rho_phi_slice_fieldlines_fac_%g_%04d.png' %(fac,idump))
			else: plt.savefig('frames/frame_rho_phi_slice_fac_%g_%04d.png' %(fac,idump))


			plt.figure(2)
			plt.clf()

			if (fac ==1 or fac ==0.33): c = plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10((asc.rho*r_tmp/1e3)[:,ny//2,:]),cmap = 'viridis',vmin=-1.5,vmax=-0.5)
			else: c = plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10((asc.rho*r_tmp/1e3)[:,ny//2,:]),cmap = 'viridis',vmin=-1,vmax=1)
			
			if (fieldlines==True): asc.plot_fieldlines_midplane(9*fac)
			#plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10(Bth*Br)[:,ny//2,:],cmap = 'cubehelix',vmin=-1,vmax=4.25)

			# cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
			# cb.set_label(r"$ \rho r $",fontsize=17)


			plt.xlim(-9*fac,9*fac)
			plt.ylim(-9*fac,9*fac)


			# for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			# 	label.set_fontsize(10)

			plt.axis('off')
			plt.axes().set_aspect('equal')
			plt.tight_layout()
			if (fieldlines==True): plt.savefig('frames/frame_rho_midplane_fieldlines_fac_%g_%04d.png' %(fac,idump))
			else: plt.savefig('frames/frame_rho_midplane_fac_%g_%04d.png' %(fac,idump))
def mk_frame_Be_star(fieldlines=True):
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2='mks_pole_fix',h=0.2)
			c1 = plt.pcolormesh((asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho*asc.r**3.0)[:,:,0],vmin=-4,vmax=-2,cmap="cubehelix")
			c2 = plt.pcolormesh(-(asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho*asc.r**3.0)[:,:,asc.nz//2],vmin=-4,vmax=-2,cmap="cubehelix")
			cb1 = plt.colorbar(c1) 

			plt.ylim(-10,10)
			plt.xlim(-10,10)
			plt.xlabel(r'$x$ ($R_\star$)',fontsize = 20)
			plt.ylabel(r'$z$ ($R_\star$)',fontsize = 20)
			plt.title(r'$t = %d t_\star$' %int(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho r^3)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)

			plt.xlim(-50,50)
			plt.ylim(-50,50)

			plt.savefig("frames/frame_medium_%04d.png" % (i_dump))

			plt.xlim(-100,100)
			plt.ylim(-100,100)

			plt.savefig("frames/frame_large_%04d.png" % (i_dump))
def mk_frame_Be_star_smr(fieldlines=True):
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",x3min=0.0,x3max=0.05,block_level=max_level_)
			c1 = plt.pcolormesh((asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho*asc.r**3.0)[:,:,0],vmin=-4,vmax=-1,cmap="cubehelix")
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",x3min=np.pi-0.05,x3max=np.pi+0.05,block_level=max_level_)
			c2 = plt.pcolormesh(-(asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho*asc.r**3.0)[:,:,0],vmin=-4,vmax=-1,cmap="cubehelix")
			cb1 = plt.colorbar(c1) 

			plt.ylim(-10,10)
			plt.xlim(-10,10)
			plt.xlabel(r'$x$ ($R_\star$)',fontsize = 20)
			plt.ylabel(r'$z$ ($R_\star$)',fontsize = 20)
			plt.title(r'$t = %d t_\star$' %int(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho r^3)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)


			plt.xlim(-50,50)
			plt.ylim(-50,50)

			plt.savefig("frames/frame_medium_%04d.png" % (i_dump))

			plt.xlim(-100,100)
			plt.ylim(-100,100)

			plt.savefig("frames/frame_large_%04d.png" % (i_dump))

			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",x2min=np.pi/2-0.05,x2max=np.pi/2+0.05,block_level=max_level_)
			c1 = plt.pcolormesh((asc.r*np.cos(asc.ph))[:,asc.ny//2,:], (asc.r*np.sin(asc.ph))[:,asc.ny//2,:],np.log10(asc.rho*asc.r**3.0)[:,asc.ny//2,:],vmin=-2,vmax=0,cmap="cubehelix")
			
			plt.ylim(-10,10)
			plt.xlim(-10,10)
			plt.xlabel(r'$x$ ($R_\star$)',fontsize = 20)
			plt.ylabel(r'$y$ ($R_\star$)',fontsize = 20)
			plt.title(r'$t = %d t_\star$' %int(asc.t),fontsize = 20)
			cb1 = plt.colorbar(c1) 
			cb1.set_label(r"$\log_{10}(\rho r^3)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			plt.savefig("frames/frame_midplane_%04d.png" % (i_dump))

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.savefig("frames/frame_midplane_medium_%04d.png" % (i_dump))
			plt.clf()
def mk_ipole_input():
	set_dump_range()
	for idump in range(i_start,i_end):
		dump_file = "dump_spher_%d_th_0_phi_0.npz" %idump
		os.system("python ressler_translate.py athinput.gr_inits %s --overwrite" % (dump_file))
# def run_ipole():
# 	set_dump_range()
# 	lambda_sq_vals = np.linspace(0.01e-5,0.25e-5,10,endpoint=True)  # in m
# 	lambda_sq_vals *= 100**2.0
# 	cl = 2.99792458e10
# 	freq_vals = cl/np.sqrt(lambda_sq_vals)
# 	os.system("mkdir -p images")
# 	for idump in range(i_start,i_end):
# 		dump_file = "dump_spher_%d_th_0_phi_0.h5" %idump
# 		n = 0 
# 		for freq in freq_vals:
# 			outfile = "images/image_dump_%d_lambda_%d.h5" %(idump,n)
# 			command_ = "./ipole --freqcgs=%ge11 --MBH=4.3e6 --M_unit=1.0 --thetacam=1.0 --dsource=8127.0" \
# 				" --fov=200 --rmax_geo=500 --dump=%s --outfile=%s" %  \
# 				(freq/1e11,dump_file,outfile)
# 			os.system(command_)
# 			n = n+1

def run_ipole():
	set_dump_range()
	# lambda_sq_vals = np.linspace(0.01e-5,0.25e-5,10,endpoint=True)  # in m
	# lambda_sq_vals *= 100**2.0
	# cl = 2.99792458e10
	# freq_vals = cl/np.sqrt(lambda_sq_vals)
	freq_vals = np.array([230e9,230e9*1.01])
	os.system("mkdir -p images")
	for idump in range(i_start,i_end):
		dump_file = "dump_spher_%d_th_0_phi_0.h5" %idump
		n = 0 
		for freq in freq_vals:
			outfile = "images/image_dump_%d_freq_%d.h5" %(idump,n)
			if os.path.isfile(outfile): continue
			command_ = "./ipole --freqcgs=%ge11 --MBH=4.3e6 --M_unit=1.0 --thetacam=1.0 --dsource=8127.0" \
				" --fov=200 --rmax_geo=500 --dump=%s --outfile=%s" %  \
				(freq/1e11,dump_file,outfile)
			os.system(command_)
			n = n+1

def run_ipole_frequency():
	set_freq_range()
	lambda_sq_vals = np.linspace(0.01e-5,0.25e-5,n_freqs,endpoint=True)  # in m
	#lambda_sq_vals = np.linspace(0.01e-5,0.25e-5,10,endpoint=True)  # in m
	lambda_sq_vals *= 100**2.0
	cl = 2.99792458e10
	freq_vals = cl/np.sqrt(lambda_sq_vals)
	idump = 4000
	dump_file = "dump_spher_%d_th_0_phi_0.h5" %idump
	for ifreq in range(i_start,i_end):
		freq = freq_vals[ifreq]
		outfile = "image_dump_%d_freq_%d.h5" %(idump,ifreq)
		command_ = "./ipole --freqcgs=%ge11 --MBH=4.3e6 --M_unit=1.0 --thetacam=1.0 --dsource=8127.0" \
		          " --fov=200 --rmax_geo=500 --dump=%s --outfile=%s" %  \
		          (freq/1e11,dump_file,outfile)
		os.system(command_)

	# freq_vals = np.array([230e9,230e9*1.01])
	# os.system("mkdir -p images")
	# for idump in range(i_start,i_end):
	# 	dump_file = "dump_spher_%d_th_0_phi_0.h5" %idump
	# 	n = 0 
	# 	for freq in freq_vals:
	# 		outfile = "images/image_dump_%d_freq_%d.h5" %(idump,n)
	# 		command_ = "./ipole --freqcgs=%ge11 --MBH=4.3e6 --M_unit=1.0 --thetacam=1.0 --dsource=8127.0" \
	# 			" --fov=200 --rmax_geo=500 --dump=%s --outfile=%s" %  \
	# 			(freq/1e11,dump_file,outfile)
	# 		os.system(command_)
	# 		n = n+1
def make_ipole_image_frames():
	set_dump_range(ipole_files=True)
	for idump in range(i_start,i_end):
		fname = "image_dump_%d_freq_0.h5" %(idump)
		if os.path.isfile(fname):
			plot_ipole_image(fname)
			plt.savefig("image_dump_%d_freq_0.png" %(idump))

if __name__ == "__main__":
	if m_type == "mk_frame_inner":
		mk_frame_inner()
	elif m_type =="mk_frame_outer":
		mk_frame_outer()
	elif m_type == "mk_frame_outer_slice":
		mk_frame_outer(isslice=True)
	elif m_type =="mk_frame_outer_slice_mhd":
		mk_frame_outer(isslice=True,mhd=mhd_switch)
	elif m_type == "mk_frame_inner_slice_mhd":
		mk_frame_inner(isslice=True,mhd=mhd_switch)
	elif m_type == "mk_frame_inner_slice":
		mk_frame_inner(isslice=True)
	elif m_type == "convert_dumps":
		convert_dumps_to_spher()
	elif m_type == "convert_dumps_mhd":
		convert_dumps_to_spher(MHD=mhd_switch)
	elif m_type == "convert_dumps_disk_frame":
		convert_dumps_disk_frame()
	elif m_type == "convert_dumps_disk_frame_mhd":
		convert_dumps_disk_frame(mhd=mhd_switch)
	elif m_type == "Lx_calc":
		calculate_L_X()
	elif m_type == "mk_3Dframe":
		mk_frame_3D_uniform_grid()
	elif m_type == "mk_3D_jet_frame":
		mk_frame_3D_jet()
	elif m_type == "mk_3D_jet_and_disk_frame":
		mk_frame_3D_jet_and_disk()
	elif m_type == "mk_grmhdframe":
		mk_frame_grmhd()
	elif m_type == "mk_grframe":
		mk_frame_grmhd(is_magnetic = mhd_switch)
	elif m_type == "mk_angle":
		mk_sheet_angle(is_magnetic = mhd_switch)
	elif m_type == "mk_frame_single":
		mk_grframe_single(is_magnetic = mhd_switch)
	elif m_type == "mk_frame_double":
		mk_grframe_double(is_magnetic = mhd_switch)
	elif m_type == "mk_frame_triple":
		mk_grframe_triple(is_magnetic = mhd_switch)
	elif m_type == "mk_frame_triple_rot":
		mk_grframe_triple_rot(is_magnetic = mhd_switch)
	elif m_type == "mk_frame_sheet":
		mk_grframe_sheet(is_magnetic = mhd_switch)
	elif m_type == "mk_frame_total":
		mk_grframe_total(is_magnetic = mhd_switch)
	elif m_type == "mk_grframe_cartesian":
		mk_frame_grmhd_cartesian(is_magnetic=mhd_switch)
	elif m_type == "mk_frame_grmhd_restart_cartesian":
		mk_frame_grmhd_restart_cartesian(is_magnetic=mhd_switch)
	elif m_type == "mk_grframe_magnetically_frustrated":
		mk_frame_gr_magnetically_frustrated(is_magnetic=mhd_switch)
	elif m_type == "mk_grframe_magnetically_frustrated_cartesian":
		mk_frame_gr_magnetically_frustrated_cartesian(is_magnetic=mhd_switch)
	elif m_type == "mk_1d":
		mk_1d_quantities(is_magnetic=mhd_switch)
	elif m_type == "mk_1d_cartesian":
		mk_1d_quantities_cartesian(is_magnetic=mhd_switch)
	elif m_type == "mk_1d_cartesian_spin_orbit":
		mk_1d_quantities_cartesian(is_magnetic=mhd_switch,spin_orbit=True)
	elif m_type == "mk_1d_cartesian_newtonian":
		mk_1d_quantities_cartesian_newtonian(is_magnetic=mhd_switch)
	elif m_type == "mk_1d_Be":
		mk_1d_quantities_Be(is_magnetic=mhd_switch)
	elif m_type == "mk_1d_Be_smr":
		mk_1d_quantities_Be_smr(is_magnetic=mhd_switch)
	elif m_type == 'mk_RM':
		mk_RM()
	elif m_type == 'mk_RM_rel':
		mk_RM_relativistic()
	elif m_type == 'mk_RM_moving':
		mk_RM(moving=mhd_switch)
	elif m_type == "RM_movie":
		RM_movie()
	elif m_type == "mk_frame_outer_cold":
		mk_frame_outer(iscold=True)
	elif m_type == "mk_frame_inner_cold":
		mk_frame_inner(iscold=True)
	elif m_type == "mk_frame_disk":
		mk_frame_disk()
	elif m_type == "mk_frame_L_aligned":
		mk_frame_L_aligned()
	elif m_type == "mk_frame_L_aligned_fieldlines":
		mk_frame_L_aligned(fieldlines=True)
	elif m_type == "mk_frame_Be_star":
		mk_frame_Be_star(fieldlines=False)
	elif m_type == "mk_frame_Be_star_smr":
		mk_frame_Be_star_smr(fieldlines=False)
	elif m_type == "mk_ipole_input":
		mk_ipole_input()
	elif m_type == "run_ipole":
		run_ipole()
	elif m_type == "run_ipole_frequency":
		run_ipole_frequency()
	elif m_type ==  "mk_images":
		make_ipole_image_frames()
	elif m_type == "mk_frame_binary":
		mk_frame_grmhd_cartesian_second_bh_zoom(is_magnetic=mhd_switch)
	elif m_type == "mk_frame_grmhd_binary":
		mk_frame_grmhd_binary(is_magnetic=mhd_switch)
	elif m_type == "mk_1d_quantities_binary":
		mk_1d_quantities_binary(is_magnetic=mhd_switch,spin_orbit=False)
	elif m_type == "mk_1d_quantities_binary_spin_orbit":
		mk_1d_quantities_binary(is_magnetic=mhd_switch,spin_orbit=True)
	elif m_type == "mk_frame_boosted":
		mk_frame_grmhd_constant_velocity_boosted(is_magnetic=mhd_switch)
	elif m_type == "mk_frame_bhl":
		mk_frame_grmhd_constant_velocity(is_magnetic=mhd_switch)
