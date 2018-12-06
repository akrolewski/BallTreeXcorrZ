import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix
import healpy as hp
import argparse
from astropy.cosmology import Planck15 as LCDM
from astropy.io import fits
from astropy import units as u
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Arguments go here

cli = argparse.ArgumentParser("Cross correlate with QSO data of a selected redshift range")

cli.add_argument("--zmin",default=0.0,type=float,help="minimum redshift")
cli.add_argument("--zmax",default=4.0,type=float,help="maximum redshift")
cli.add_argument("--dz",default=0.1,type=float,help="delta z")
cli.add_argument("--nside",default=64,type=int,help="Healpix chunk size for bootstrapping")
cli.add_argument("--nboot",default=500,type=int,help="Number of bootstraps")
cli.add_argument("phot_name", help="internal catalogue of fits type.")
cli.add_argument("phot_name_randoms", help="internal catalogue of fits type.")
cli.add_argument("spec_name", help="internal catalogue of fits type.")
cli.add_argument("outdir", help="Directory of text files to store the correlation function." )
cli.add_argument("plotdir", help="Directory of plots.")

ns = cli.parse_args()

def truncate(name):
	'''Truncates a filename so I can use it to name things'''
	return name.split('/')[-1].split('.fits')[0]
	
def downgrade(pixel,nside1,nside2):
	'''downgrades a HEALPix pixel from nside1 to nside2'''
	theta,phi = hp.pix2ang(nside1,pixel)
	return hp.ang2pix(nside2,theta,phi)
	

# Binning parameters (min/max in Mpc/h)
Smin = 0.05
Smax = 50
nbins = 15

# Original data is binned into deltaz = 0.01 bins and nside=256
# These should be much higher resolution than any practical application
orig_deltaz = 0.01
nside_base = 256

zlen = int(ns.dz/orig_deltaz)
zbin = int(round((ns.zmax-ns.zmin)/ns.dz))

# open files
data1file = fits.open(ns.phot_name)[1].data
data2file = fits.open(ns.spec_name)[1].data
rand1file = fits.open(ns.phot_name_randoms)[1].data

data1RA = data1file['RA'][:]
data1DEC = data1file['DEC'][:]

rand1RA = rand1file['RA'][:]
rand1DEC = rand1file['DEC'][:]

data2RA = data2file['RA'][:]
data2DEC = data2file['DEC'][:]

Nr1 = len(rand1RA)
Nd1 = len(data1RA)

# get a list of all possible healpixels from the randoms
all_healpixels = hp.ang2pix(ns.nside,rand1RA,rand1DEC,lonlat=True)
unq_all_healpixels = np.unique(all_healpixels)

# set up a function that takes each healpixel to an index in the list of all possible healpixels
unq_all_healpixels_inds = np.arange(len(unq_all_healpixels))
unq_all_healpixels_fn = np.zeros(np.max(all_healpixels)+1,dtype=int)
unq_all_healpixels_fn[unq_all_healpixels ] = unq_all_healpixels_inds

# Get the length of all possible healpixels
len_unq_hp = len(unq_all_healpixels)
			
# A function that we need later to set up the sparse matrix
def make_sparse_mat(list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,nside,m,n):	
	pl = np.array(map(lambda x: downgrade(int(x),nside_base,nside),list[m][n][1]))
	pls = pl.argsort()
	
	indl = np.searchsorted(pl[pls],np.unique(pl),side='left')
	indu = np.searchsorted(pl[pls],np.unique(pl),side='right')
	
	unq_pl = np.unique(pl)
	counts = np.array(map(lambda k: np.sum(np.array(list[m][n][0])[pls][indl[k]:indu[k]]),range(len(indl)))) 
	#print counts
	#print unq_all_healpixels_fn[pix[m]]
	return csr_matrix((counts, (np.tile(unq_all_healpixels_fn[pix[m]], len(counts)), unq_all_healpixels_fn[unq_pl])),shape=(len_unq_hp,len_unq_hp))

for i in range(zbin):
	z1 = (i+int(round(ns.zmin/ns.dz)))*ns.dz
	z2 = (i+int(round(ns.zmin/ns.dz))+1)*ns.dz
	
	data2mask  = data2file['Z'][:] >= z1
	data2mask &= data2file['Z'][:] <  z2

	flag = 0
	for j in range(zlen):
		name_ind = int((i*ns.dz + j*orig_deltaz)/orig_deltaz)+120
		try:
			pixel_lists = pickle.load(open('%s-%s/%i_pix_list.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind),'rb'))
			inds = pixel_lists[0]
			dd_pix_list = pixel_lists[1]
			dr_pix_list = pixel_lists[2]
						
			if flag == 0:
				flag += 1
				
				all_dd_pix_list = dd_pix_list
				all_dr_pix_list = dr_pix_list
				allinds = inds
			else:
				
				all_dd_pix_list = np.concatenate((all_dd_pix_list,dd_pix_list))
				all_dr_pix_list = np.concatenate((all_dr_pix_list,dr_pix_list))
				allinds = np.concatenate((allinds,inds))
		except IOError:
			continue
	if flag != 0:
		pix = hp.ang2pix(ns.nside,data2RA[allinds],data2DEC[allinds],lonlat=True)

		# Set up the sparse matrices
		pair_mats_dd = []
		pair_mats_dr = []
		
		for n in range(np.shape(all_dd_pix_list)[1]):
			dd_flag = 0
			dr_flag = 0
			for m in range(np.shape(all_dd_pix_list)[0]):
				if all_dd_pix_list[m][n]:												
					if dd_flag == 0:
						pair_mat_dd = make_sparse_mat(all_dd_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n)
					else:
						pair_mat_dd += make_sparse_mat(all_dd_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n)
					dd_flag += 1
					
				if all_dr_pix_list[m][n]:
					if dr_flag == 0:
						pair_mat_dr = make_sparse_mat(all_dr_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n)
					else:
						pair_mat_dr += make_sparse_mat(all_dr_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n)
					dr_flag += 1
			
			pair_mats_dd.append(pair_mat_dd)
			pair_mats_dr.append(pair_mat_dr)
		
	
		# Choose the bootstrap pixels
		boot_pix = np.random.choice(range(len_unq_hp),size=(len_unq_hp,ns.nboot),replace=True)
		
		# Make arrays for the 3 bootstraps: literal, sqrt, and marked
		literal_bs_dd = np.zeros((nbins,ns.nboot))
		sqrt_bs_dd = np.zeros((nbins,ns.nboot))
		marked_bs_dd = np.zeros((nbins,ns.nboot))
		
		literal_bs_dr = np.zeros((nbins,ns.nboot))
		sqrt_bs_dr = np.zeros((nbins,ns.nboot))
		marked_bs_dr = np.zeros((nbins,ns.nboot))
		
		boot_counts = np.zeros(ns.nboot)
		
		# Function for bootstrapping
		def bootstrap(pair_mats,wts):
			wted_by_qso = map(lambda x: (x.transpose().multiply(wts)).transpose(), pair_mats)		
			wted_by_qso_plus_galaxy = map(lambda x: x.dot(wts), wted_by_qso)			
			return np.sum(wted_by_qso_plus_galaxy,axis=1)

		for k in range(ns.nboot):
			wts = np.bincount(boot_pix[:,k],minlength=len_unq_hp)
				
			literal_bs_dd[:,k] = bootstrap(pair_mats_dd,wts)
			literal_bs_dr[:,k] = bootstrap(pair_mats_dr,wts)
			
			sqrt_bs_dd[:,k] = bootstrap(pair_mats_dd,np.sqrt(wts))
			sqrt_bs_dr[:,k] = bootstrap(pair_mats_dr,np.sqrt(wts))
			
			# Marked bootstrap is a bit simpler
			summed_qso = map(lambda x: np.sum(x,axis=1), pair_mats_dd)
			marked_bs_dd[:,k] = np.squeeze(np.dot(wts,summed_qso))
			
			summed_qso = map(lambda x: np.sum(x,axis=1), pair_mats_dr)
			marked_bs_dr[:,k] = np.squeeze(np.dot(wts,summed_qso))
			
			boot_counts[k] = np.sum(wts[unq_all_healpixels_fn[pix]])

		dd_data = np.array(map(lambda x: np.sum(x), pair_mats_dd)).astype('float')
		dr_data = np.array(map(lambda x: np.sum(x), pair_mats_dr)).astype('float')
		
		wmeas = dd_data/dr_data * float(Nr1)/float(Nd1) - 1.
		
		wpoisson = np.sqrt(dd_data)/dr_data * float(Nr1)/float(Nd1)
		
		wliteral = literal_bs_dd/literal_bs_dr * float(Nr1)/float(Nd1) - 1.
		wsqrt = sqrt_bs_dd/sqrt_bs_dr * float(Nr1)/float(Nd1) - 1.
		wmarked = marked_bs_dd/marked_bs_dr * float(Nr1)/float(Nd1) - 1.
		
		data2RA_sel = data2file['RA'][:][data2mask]
		data2DEC_sel = data2file['DEC'][:][data2mask]
		data2Z_sel = data2file['Z'][:][data2mask]

		h0 = LCDM.H0 / (100 * u.km / u.s / u.Mpc)
		zmean = data2Z_sel.mean()
		R = (LCDM.comoving_distance(zmean) / (u.Mpc / h0 ))

		thmin = (Smin/R)*180./np.pi
		thmax = (Smax/R)*180./np.pi
		#print(thmin)
		#print(np.logspace(-3,0,16,endpoint=True))
		b = np.logspace(np.log10(thmin),np.log10(thmax),nbins+1,endpoint=True)
		
		theta = 0.5*(b[1:]+b[:-1])

		s = np.radians(theta.value)*R

		header='\n'.join([
				"SPEC=%s" % (ns.spec_name),
				"z1=%g z2=%g zmean=%g" % (z1, z2, zmean),
				"PHOTO=%s" % (ns.phot_name),
				"PHOTO_RANDOM=%s" % (ns.phot_name_randoms),
				"N_SPEC=%d" % (data2mask.sum()),
				"N_PHOT=%d" % (np.shape(data1RA)[0]),
				"N_PHOT_RANDOM=%d" % (np.shape(rand1RA)[0]),
				"<N_SPEC_BOOT>=%d" % (np.mean(boot_counts)),
				"STD(N_SPEC_BOOT)=%d" % (np.std(boot_counts)),
				"NBOOT=%d" % (ns.nboot),
				"ESTIMATOR=Davis&Peebles",
				"NSIDE=%d" % (ns.nside),
				"COSMO=Planck15",
				"theta [deg] thmin [arcsec] thmax [arcsec] s [Mpc/h] smin [Mpc/h] smax [Mpc/h] " +
				"w [measured] err [Poisson] w std wsamples [literal bootstrap] " +
				"w std wsamples [sqrt bootstrap] w std wsamples [marked bootstrap] ",
				])

		theta_low = 3600.*np.logspace(np.log10(thmin), np.log10(thmax), nbins+1, endpoint=True)[:-1]
		theta_high  = 3600.*np.logspace(np.log10(thmin), np.log10(thmax), nbins+1, endpoint=True)[1:]
		slow = np.radians(theta_low.value/3600.)*R
		shigh = np.radians(theta_high.value/3600.)*R

		myout = np.concatenate(([theta, theta_low, theta_high, s, slow, shigh, 
		wmeas, wpoisson, 
		np.nanmean(wliteral,axis=1), np.nanstd(wliteral,axis=1,ddof=1)], wliteral.transpose(),
		[np.nanmean(wsqrt,axis=1), np.nanstd(wsqrt,axis=1,ddof=1)], wsqrt.transpose(), 
		[np.nanmean(wmarked,axis=1), np.nanstd(wmarked,axis=1,ddof=1)],wmarked.transpose()),axis=0).T
		np.savetxt(ns.outdir + 'z%.2f_%.2f.txt' % (z1,z2) , myout, header=header)
		#print i, wmeas, mean

		plt.figure()
		plt.errorbar(s,wmeas,yerr=np.nanstd(wliteral,axis=1))
		plt.xscale('log')
		plt.xlabel(r'R ($h^{-1}$ Mpc)$',size=20)
		plt.ylabel(r'$w(\theta)$',size=20)
		plt.plot(np.linspace(0.1,100,1000),np.zeros(1000),color='k',linestyle='--')
		plt.savefig(ns.plotdir + 'z%.2f_%.2f.pdf' % (z1,z2))

	