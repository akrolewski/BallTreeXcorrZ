import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix
import healpy as hp
import argparse
from astropy.cosmology import Planck15 as LCDM
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Arguments go here

cli = argparse.ArgumentParser("Cross correlate with QSO data of a selected redshift range")

cli.add_argument('--jackknife', dest='jackknife', action='store_true',help="flag to use jackknife errors")
cli.set_defaults(jackknife=False)
cli.add_argument('--bootstrap', dest='bootstrap', action='store_true',help="flag to use bootstrap errors")
cli.set_defaults(bootstrap=False)
cli.add_argument('--equalize', dest='equalize', action='store_true',help="whether to aggregate healpixels to make them the same size")
cli.set_defaults(equalize=False)
cli.add_argument("--zmin",default=0.0,type=float,help="minimum redshift")
cli.add_argument("--zmax",default=4.0,type=float,help="maximum redshift")
cli.add_argument("--dz",default=0.1,type=float,help="delta z")
cli.add_argument("--nside",default=64,type=int,help="Healpix chunk size for bootstrapping")
cli.add_argument("--nboot",default=500,type=int,help="Number of bootstraps")
cli.add_argument("--Smin",default=0.05,type=float,help="Minimum bin radius in h^-1 Mpc")
cli.add_argument("--Smax",default=50,type=float,help="Maximum bin radius in h^-1 Mpc")
cli.add_argument("--nbins",default=15,type=int,help="Number of bins")
cli.add_argument("--orig_deltaz",default=0.01,type=float,help="Original deltaz")
cli.add_argument("--nside_base",default=256,type=int,help="Original nside")
cli.add_argument("--small_nside",default=8,type=int,help="nside for local wise density estimation")
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
	
def equalize_pixels(unq_all_healpixels,hp_cnts):
	'''Quick and dirty function for load balancing among pixels.
	Work your way down the pixels in order of number of randoms in
	each pixel, and assign neighbors to a central pixel until the total
	exceeds a certain threshold. Explicitly search through several
	values of ratio, where threshold = ratio * max(hp_cnts), to find
	the optimal value of ratio that minimizes the difference between
	the maximum and minimum counts in a pixel'''
	rats = [0.8,0.85,0.9,0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30]
	allnewsizes = []
	allnewhps = []
	stats =[]

	for rat in rats:

		s = np.argsort(hp_cnts)[::-1]

		hps = unq_all_healpixels[s]
		sizes = hp_cnts[s]

		target = rat*np.max(hp_cnts)

		newsizes = []
		newhps = []
						

		i = 0
		cnt = 0
		while cnt < len(hp_cnts):
			pix = hps[0]
			size = sizes[0]
			if size < target:
				neighb = hp.get_all_neighbours(ns.nside,pix)
				allc = []
				alln = []
				for j,n in enumerate(neighb):
					if n in hps:
						allc.append(sizes[hps==n][0])
						alln.append(n)
				allc = np.array(allc)
				alln = np.array(alln)

				sallc = np.argsort(allc)
				cum_allc = np.cumsum(allc[sallc])

				ss = np.searchsorted(cum_allc+size,target)

				if ss != 0:
					#print sizes[i] + cum_allc[ss-1], ss
					newsizes.append(size + cum_allc[ss-1])
					cnt += 1 + ss
					print(size, alln[sallc][:ss], cum_allc[ss-1], type(alln[sallc][:ss]))

				else:
					#print sizes[i], ss
					newsizes.append(size)
					cnt += 1
					print(size)


				excl = np.concatenate(([pix],alln[sallc][:ss]))
				excl = excl.astype('int')
	
				#if ss != 0:
				print(type(excl[0]), type(neighb[0]), type(pix))
				newhps.append(excl)
				#else:
				#	newhps.append([hps[i]])

				mod_inds = list(filter(lambda j: hps[j] not in excl ,range(len(hps))))

				hps = hps[mod_inds]
				sizes = sizes[mod_inds]
			else:
				newhps.append([pix])
				newsizes.append(size)
		
				excl = np.array([pix])
		
				mod_inds = list(filter(lambda j: hps[j] not in excl ,range(len(hps))))

				hps = hps[mod_inds]
				sizes = sizes[mod_inds]

				cnt += 1
			i += 1		
		
		allnewsizes.append(newsizes)
		allnewhps.append(newhps)
		stats.append((np.max(newsizes)-np.min(newsizes))/np.mean(newsizes))
		
		print(np.sum(newsizes))

	allnewsizes = np.array(allnewsizes)
	allnewhps = np.array(allnewhps)
	stats = np.array(stats)
	
	return allnewsizes[np.argmin(stats)], allnewhps[np.argmin(stats)]	

def weighted_cov_jack_loo(w,wmean,cnts):
	a = np.sqrt((np.sum(cnts)-cnts)/np.sum(cnts)) *(w-wmean[:,np.newaxis])
	return np.sum(a * a[:,np.newaxis], axis=2)
	
def weighted_cov_jack_l2o(w,wmean,cnts):
	'''Does not work'''
	return np.sum((np.sum(cnts)-cnts-cnts[:,np.newaxis])/(np.sum(cnts)*(np.sum(cnts)-np.mean(cnts))) *(w-wmean[:,np.newaxis]) * (w-wmean[:,np.newaxis])[:,np.newaxis],axis=2)
	
t0 = time.time()	

# Binning parameters (min/max in Mpc/h)
Smin = ns.Smin
Smax = ns.Smax
nbins = ns.nbins

# Original data is binned into deltaz = 0.01 bins and nside=256
# These should be much higher resolution than any practical application
orig_deltaz = ns.orig_deltaz
nside_base = ns.nside_base

downgrade_vec = np.zeros(12*nside_base**2)
nside_base_vec = np.arange(12*nside_base**2)
downgrade_vec = downgrade(nside_base_vec,nside_base,ns.nside)

zlen = int(ns.dz/orig_deltaz)
#zlen = 1
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

#data2file_wted = fits.open('boss_dr12/lowz_masked-r1-v2-flag.fits')[1].data
#data2wted_RA = data2file_wted['RA'][:]
#data2wted_DEC = data2file_wted['DEC'][:]

# This is for correcting the number counts of WISE galaxies to match
# the local number counts
small_nside = ns.small_nside
data1_smallpix = hp.ang2pix(small_nside,data1RA,data1DEC,lonlat=True)
data1_smallmap = np.bincount(data1_smallpix,minlength=12*small_nside**2)

rand1_smallpix = hp.ang2pix(small_nside,rand1RA,rand1DEC,lonlat=True)
rand1_smallmap = np.bincount(rand1_smallpix,minlength=12*small_nside**2)

ratio_smallmap = rand1_smallmap.astype('float')/data1_smallmap.astype('float')


data2_smallpix = hp.ang2pix(small_nside,data2RA,data2DEC,lonlat=True)

downgrade_vec_small = downgrade(nside_base_vec,nside_base,small_nside)

small_nside_vec = np.arange(12*small_nside**2)
downgrade_vec_main_small = downgrade(small_nside_vec,small_nside,ns.nside)


# get a list of all possible healpixels from the randoms
all_healpixels = hp.ang2pix(ns.nside,rand1RA,rand1DEC,lonlat=True)
unq_all_healpixels,hp_cnts = np.unique(all_healpixels,return_counts=True)

d1pix = hp.ang2pix(ns.nside, data1RA, data1DEC, nest=False, lonlat=True)
unq_all_healpixels_d1,hp_d1_cnts = np.unique(d1pix,return_counts=True)

d2pix = hp.ang2pix(ns.nside, data2RA, data2DEC, nest=False, lonlat=True)

if not os.path.exists(ns.outdir):
        os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:3]))
        os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:4]))
        os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:5]))
        os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:6]))
        os.system('mkdir %s' % ns.outdir)
if not os.path.exists(ns.plotdir):
        os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:3]))
        os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:4]))
        os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:5]))
        os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:6]))
        os.system('mkdir %s' % ns.plotdir)



if len(hp_d1_cnts) != len(hp_cnts):
	new_hp_d1_cnts = np.zeros(len(hp_cnts))
	for i,hpp in enumerate(unq_all_healpixels):
		try:
			new_hp_d1_cnts[i] = hp_d1_cnts[unq_all_healpixels_d1 == hpp]
		except ValueError:
			continue
	hp_d1_cnts = np.copy(new_hp_d1_cnts)

# load balance among pixels
if ns.equalize:
	cnts,hps =equalize_pixels(unq_all_healpixels,hp_cnts)

	# set up a function that takes each healpixel to an index in the list of all possible healpixels
	#unq_all_healpixels_fn = np.zeros(np.max(all_healpixels)+1,dtype=int)
	#for i in range(len(hps)):
	#	unq_all_healpixels_fn[hps[i]] = i

	## Get the length of all possible healpixels
	#len_unq_hp = len(hps)
#else:
# set up a function that takes each healpixel to an index in the list of all possible healpixels
unq_all_healpixels_inds = np.arange(len(unq_all_healpixels))
unq_all_healpixels_fn = np.zeros(np.max(all_healpixels)+1,dtype=int)
unq_all_healpixels_fn[unq_all_healpixels ] = unq_all_healpixels_inds

# Get the length of all possible healpixels
len_unq_hp = len(unq_all_healpixels)

# Set up healpixels for the small nside as well...
all_healpixels_small = hp.ang2pix(small_nside,rand1RA,rand1DEC,lonlat=True)
unq_all_healpixels_small,hp_cnts_small = np.unique(all_healpixels_small,return_counts=True)
unq_all_healpixels_inds_small = np.arange(len(unq_all_healpixels_small))
unq_all_healpixels_fn_small = np.zeros(np.max(all_healpixels_small)+1,dtype=int)
unq_all_healpixels_fn_small[unq_all_healpixels_small] = unq_all_healpixels_inds_small

# Get the length of all possible healpixels
len_unq_hp_small = len(unq_all_healpixels_small)
			
# A function that we need later to set up the sparse matrix
def make_sparse_mat(llist,unq_all_healpixels_fn,pix,len_unq_hp,n,downgrade_vec):	
	#print 'began sparse mat',time.time()-t0
	#pl = np.array(map(lambda x: downgrade(int(x),nside_base,nside),list[m][n][1]))

	pl = []
	to_iterate_over = np.array(tuple(llist[:,n]))[:,1]
	for i in range(len(np.array(tuple(llist[:,n]))[:,1])):
	
		x = to_iterate_over[i]
		print(time.time()-t0)
		if type(x) == np.int64 or type(x) == np.float64 or type(x) == int or type(x) == float:
			x = [x]
		print(time.time()-t0)
		out = []
		for j in range(len(x)):
			y = downgrade_vec[int(x[j])]
			out.append(y)
		pl.append(out)
		print(i, time.time()-t0)
	#pl = map(lambda x: map(lambda y: downgrade_vec[int(y)],x),np.array(tuple(list[:,n]))[:,1])
	#print pl
	#print np.shape(pl)
	cts = np.array(tuple(llist[:,n]))[:,0]
	unqinv = list(map(lambda x: np.unique(x,return_inverse=True)[1], pl))
	unqpl = list(map(lambda x: np.unique(x), pl))
	lenpl = list(map(len, unqpl))

	counts = list(map(lambda x: np.dot(np.heaviside(unqinv[x]-np.arange(lenpl[x])[:,np.newaxis],1)*np.heaviside(np.arange(lenpl[x])[:,np.newaxis]-unqinv[x],1),cts[x]), range(len(lenpl))))
	allh = list(map(lambda x: np.tile(unq_all_healpixels_fn[pix[x]], lenpl[x]), range(len(lenpl))))
	
	recounts = []
	for i in range(len(counts)):
		if len(np.shape(counts[i])) > 1:
			recounts.append(counts[i][0])
		else:
			recounts.append(counts[i])
	counts = recounts

	c = csr_matrix((np.concatenate(counts), (np.concatenate(allh), unq_all_healpixels_fn[np.concatenate(unqpl)])),shape=(len_unq_hp,len_unq_hp))
	return c

for i in range(zbin):

	#z1 = (i+int(round(ns.zmin/ns.dz)))*ns.dz
	#z2 = (i+int(round(ns.zmin/ns.dz))+1)*ns.dz
	z1 = ns.zmin + ns.dz * i
	z2 = ns.zmin + ns.dz * (i+1)
	
	data2mask  = data2file['Z'][:] >= z1
	data2mask &= data2file['Z'][:] <  z2
	
	Nd1 = len(data1RA) #* np.mean(data1_smallmap[data2_smallpix[data2mask]])/np.mean(data1_smallmap[data2_smallpix])

	Nr1 = len(rand1RA) #* np.mean(rand1_smallmap[data2_smallpix[data2mask]])/np.mean(rand1_smallmap[data2_smallpix])


	flag = 0
	flag2 = -1
	for j in range(zlen):
		name_ind = int(round((i*ns.dz + j*orig_deltaz)/orig_deltaz)) + int(round(ns.zmin/orig_deltaz))
		print(i, j, name_ind)
		try:
			print('%s-%s/%i_pix_list.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind))
			#pixel_lists = pickle.load(open('red_16.6_lowz_masked-lowz_masked-r1-v2/%i_pix_list.p' % name_ind,'rb'))
			pixel_lists = pickle.load(open('%s-%s/%i_pix_list.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind),'rb'))
			#print j
			inds = pixel_lists[0]
			dd_pix_list = pixel_lists[1]
			dr_pix_list = pixel_lists[2]
			if np.any(inds):		
				if flag == 1:
					flag2 = 1
				else:
					flag2 = 0
						
				if flag == 0:
					flag += 1
				
					all_dd_pix_list = dd_pix_list
					all_dr_pix_list = dr_pix_list
					#print j
					allinds = inds
					print('flag=0', j, np.shape(allinds), np.shape(inds), '%s-%s/%i_pix_list.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind))
				else:
					
					all_dd_pix_list = np.concatenate((all_dd_pix_list,dd_pix_list))
					all_dr_pix_list = np.concatenate((all_dr_pix_list,dr_pix_list))
				
					#print j
					allinds = np.concatenate((allinds,inds))
					print('flag!=0', j, np.shape(allinds), np.shape(inds), '%s-%s/%i_pix_list.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind))
		except (IOError, ValueError):
			continue
		#if np.shape(all_dd_pix_list)[0] == 1:
		if flag2 == 0:
			all_dd_pix_list = np.array(all_dd_pix_list)
			#if np.shape(all_dr_pix_list)[0] == 1:
			all_dr_pix_list = np.array(all_dr_pix_list)
	if flag != 0:
		print('loaded files', time.time()-t0)
		allinds = np.array(allinds).astype('int')
		pix = hp.ang2pix(ns.nside,data2RA[allinds],data2DEC[allinds],lonlat=True)
		
		pix_small = hp.ang2pix(small_nside,data2RA[allinds],data2DEC[allinds],lonlat=True)

		# Set up the sparse matrices
		pair_mats_dd = []
		pair_mats_dr = []
		dat_random_ratio_realpixes = []
		counts_realpixes = []
		
		for n in range(np.shape(all_dd_pix_list)[1]):
			dd_flag = 0
			dr_flag = 0
			print('sparse matrix',time.time()-t0)
			pair_mat_dd = make_sparse_mat(all_dd_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,n,downgrade_vec)
			pair_mat_dr = make_sparse_mat(all_dr_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,n,downgrade_vec)
			# Average the data/random ratio over the number of random pairs (as a proxy for the unmasked size of the annulus)
			pair_mat_dr2 = make_sparse_mat(all_dr_pix_list,unq_all_healpixels_fn_small,pix_small,len_unq_hp_small,n,downgrade_vec_small)
			ratio_smallmap_pixels = 1./ratio_smallmap[unq_all_healpixels_small].astype('float')
			ratio_smallmap_pixels[np.isnan(ratio_smallmap_pixels) | np.isinf(ratio_smallmap_pixels)] = 0
			pd2 = pair_mat_dr2.toarray()
			#pd2 = np.diag(np.diag(pd2))
			ones = np.ones_like(ratio_smallmap_pixels)
			#ones[2264] = 0
			counts = np.dot(pd2,ones)
			#np.random.seed(100)
			#np.random.shuffle(ratio_smallmap_pixels)
			dat_random_ratio = np.dot(pd2,ratio_smallmap_pixels)/counts
			big_pixels = downgrade_vec_main_small[unq_all_healpixels_small]
			dat_random_ratio_realpix = np.zeros(len_unq_hp)
			counts_realpix = np.zeros(len_unq_hp)
			for m,bp in enumerate(np.unique(big_pixels)):
				inds = np.where(big_pixels == bp)
				dat_random_ratio_realpix[m] = np.nansum(dat_random_ratio[inds]*counts[inds])/np.nansum(counts[inds])
				counts_realpix[m] = np.nansum(counts[inds])
			#print 5/0
			#print np.nansum(np.dot(pd2,ratio_smallmap_pixels))/np.nansum(counts)
			
			#for m in range(np.shape(all_dd_pix_list)[0]):
			#	if all_dd_pix_list[m][n]:												
			#		#if dd_flag == 0:
			#		#	pair_mat_dd = make_sparse_mat(all_dd_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n,t0)
			#		#else:
			#		#	pair_mat_dd += make_sparse_mat(all_dd_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n,t0)
			#		#dd_flag += 1
			#		#
			#	if all_dr_pix_list[m][n]:
			#		if dr_flag == 0:
			#			pair_mat_dr = make_sparse_mat(all_dr_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n,t0)
			#		else:
			#			pair_mat_dr += make_sparse_mat(all_dr_pix_list,unq_all_healpixels_fn,pix,len_unq_hp,nside_base,ns.nside,m,n,t0)
			#		dr_flag += 1
			
			pair_mats_dd.append(pair_mat_dd)
			pair_mats_dr.append(pair_mat_dr)
			
			dat_random_ratio_realpixes.append(dat_random_ratio_realpix)
			counts_realpixes.append(counts_realpix)
		
		print('made sparse matrix',time.time()-t0)
		dd_data = np.array(list(map(lambda x: np.sum(x), pair_mats_dd))).astype('float')
		dr_data = np.array(list(map(lambda x: np.sum(x), pair_mats_dr))).astype('float')
		
		rat = np.zeros(len(pair_mats_dd))
		for n in range(len(pair_mats_dd)):
			#rat[n] 
			#data1cnts = np.nansum(counts_realpixes[n]*dat_random_ratio_realpixes[n])/np.nansum(counts_realpixes[n])
			#rat[n] = float(len(rand1RA))/(float(len(data1RA)) * (data1cnts/np.mean(data1_smallmap[data2_smallpix])))
			#rat[n] = float(len(rand1RA))/(float(len(data1RA)) * (data1cnts/(np.sum(ratio_smallmap_pixels.astype('float') * hp_cnts_small.astype('float'))/np.sum(hp_cnts_small.astype('float')))))
			#print data1cnts
			rat[n] = 1./(np.nansum(counts_realpixes[n]*dat_random_ratio_realpixes[n])/np.nansum(counts_realpixes[n]))
			print(rat[n])
		
		np.savetxt(ns.outdir + 'z%.2f_%.2f_rat.txt' % (z1,z2),rat)
		wmeas = dd_data/dr_data * rat - 1.
		wmeas_no_adj = dd_data/dr_data * float(Nr1)/float(Nd1) - 1.
		
                #print 5/0
		wpoisson = np.sqrt(dd_data)/dr_data * float(Nr1)/float(Nd1)
		
		# "cross correlation" version of formula in Mo Jing and Boerner 1992
		# the geometric mean of the two is purely empirical...
		ng =  np.sqrt(np.shape(data2RA)[0]*np.shape(data1RA)[0])
		w_mjb = np.sqrt(dd_data + dd_data**2. * 4./ng)/dr_data * float(Nr1)/float(Nd1)
		
		data2RA_sel = data2file['RA'][:][allinds]
		data2DEC_sel = data2file['DEC'][:][allinds]
		data2Z_sel = data2file['Z'][:][allinds]


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
		
		theta_low = 3600.*np.logspace(np.log10(thmin), np.log10(thmax), nbins+1, endpoint=True)[:-1]
		theta_high  = 3600.*np.logspace(np.log10(thmin), np.log10(thmax), nbins+1, endpoint=True)[1:]
		slow = np.radians(theta_low.value/3600.)*R
		shigh = np.radians(theta_high.value/3600.)*R
		
		print('done with data',time.time()-t0)
		
		# Function for bootstrapping
		def resample(pair_mats,wts):
			wted_by_qso = list(map(lambda x: (x.transpose().multiply(wts)).transpose(), pair_mats))		
			wted_by_qso_plus_galaxy = list(map(lambda x: x.dot(wts), wted_by_qso))			
			return np.sum(wted_by_qso_plus_galaxy,axis=1)
		
		if ns.bootstrap:
			# Choose the bootstrap pixels
			np.random.seed(272)
			boot_pix = np.random.choice(range(len_unq_hp),size=(len_unq_hp,ns.nboot),replace=True)
			
			# Use Yu Feng method
			# Virtually identical to my method (at nside=32) and this implementation of it is very slow...
			# For smaller nsides this method becomes more preferred!
			'''N = np.bincount(all_healpixels)
			active_straps = N.nonzero()[0]
			N = N[active_straps]
			size = N.sum()
			sizes = N
			rng = np.random
			
			boot_pix = []
			for k in range(ns.nboot):
				boot_pix_ind = []
				Nremain = size
				while Nremain > 0:
					ch = rng.choice(len(active_straps),size=1,replace=True)
					accept = rng.uniform() <= Nremain / (1.0 * sizes[ch])
					if accept:
						Nremain -= sizes[ch]
						#print ch
						boot_pix_ind.append(ch)
					else:
						break
				boot_pix_ind =np.squeeze(np.array(boot_pix_ind))
				boot_pix.append(boot_pix_ind)'''
			print('made boot pixels', time.time()-t0)
		
			# Make arrays for the 3 bootstraps: literal, sqrt, and marked
			literal_bs_dd = np.zeros((nbins,ns.nboot))
			sqrt_bs_dd = np.zeros((nbins,ns.nboot))
			marked_bs_dd = np.zeros((nbins,ns.nboot))
		
			literal_bs_dr = np.zeros((nbins,ns.nboot))
			sqrt_bs_dr = np.zeros((nbins,ns.nboot))
			marked_bs_dr = np.zeros((nbins,ns.nboot))

			boot_counts = np.zeros(ns.nboot)
		
			for k in range(ns.nboot):
				wts = np.bincount(boot_pix[:,k],minlength=len_unq_hp)
				
				literal_bs_dd[:,k] = resample(pair_mats_dd,wts)
				literal_bs_dr[:,k] = resample(pair_mats_dr,wts)
			
				sqrt_bs_dd[:,k] = resample(pair_mats_dd,np.sqrt(wts))
				sqrt_bs_dr[:,k] = resample(pair_mats_dr,np.sqrt(wts))
			
				# Marked bootstrap is a bit simpler
				summed_qso = list(map(lambda x: np.sum(x,axis=1), pair_mats_dd))
				marked_bs_dd[:,k] = np.squeeze(np.dot(wts,summed_qso))
			
				summed_qso = list(map(lambda x: np.sum(x,axis=1), pair_mats_dr))
				marked_bs_dr[:,k] = np.squeeze(np.dot(wts,summed_qso))
			
				boot_counts[k] = np.sum(wts[unq_all_healpixels_fn[pix]])
		
			wliteral = literal_bs_dd/literal_bs_dr * float(Nr1)/float(Nd1) - 1.
			wsqrt = sqrt_bs_dd/sqrt_bs_dr * float(Nr1)/float(Nd1) - 1.
			wmarked = marked_bs_dd/marked_bs_dr * float(Nr1)/float(Nd1) - 1.
			print('resampled',time.time()-t0)
			
			def header(method):
				header_out='\n'.join([
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
						"ERROR=%s" % (method),
						"ESTIMATOR=Davis&Peebles",
						"NSIDE=%d" % (ns.nside),
						"COSMO=Planck15",
						"theta [deg] thmin [arcsec] thmax [arcsec] s [Mpc/h] smin [Mpc/h] smax [Mpc/h] " +
						"w [measured] err [Poisson] w std cov wsamples [%s] " % (method),
						])
				return header_out
				
			base_out = np.array([theta, theta_low, theta_high, s, slow, shigh, wmeas, wmeas_no_adj, wpoisson, w_mjb])
			
			def make_output(base_out, arr):
				myout = np.concatenate((base_out, [np.nanmean(arr,axis=1), np.nanstd(arr,axis=1,ddof=1)],
					np.cov(arr,bias=False), arr.transpose()),axis=0).T
				return myout
			
			np.savetxt(ns.outdir + 'z%.2f_%.2f_bs_literal.txt' % (z1,z2) , make_output(base_out,wliteral), header=header('bootstrap-literal'))
			np.savetxt(ns.outdir + 'z%.2f_%.2f_bs_sqrt.txt' % (z1,z2) , make_output(base_out,wsqrt), header=header('bootstrap-sqrt'))
			np.savetxt(ns.outdir + 'z%.2f_%.2f_bs_marked.txt' % (z1,z2) , make_output(base_out,wmarked), header=header('bootstrap-marked'))

			plt.figure()
			plt.errorbar(s,wmeas,yerr=np.nanstd(wliteral,axis=1))
			plt.xscale('log')
			plt.xlabel(r'R ($h^{-1}$ Mpc)$',size=20)
			plt.ylabel(r'$w(\theta)$',size=20)
			plt.plot(np.linspace(0.1,100,1000),np.zeros(1000),color='k',linestyle='--')
			if not os.path.exists(ns.plotdir):
				os.system('mkdir %s' % ns.plotdir)
			plt.savefig(ns.plotdir + 'z%.2f_%.2f_bs_literal.pdf' % (z1,z2))
			print('wrote files', time.time()-t0)
		
		if ns.jackknife:			
			if ns.equalize:		
				#print 5/0
				loo_jackknife_dd = np.zeros((nbins,len(hps)))
				loo_jackknife_dr = np.zeros((nbins,len(hps)))
				wjack_loo = np.zeros((nbins,len(hps)))	
				wjack_loo_no_adjust = np.zeros((nbins,len(hps)))
				ratios = []
				for k in range(len(hps)):
					wts = np.ones(len_unq_hp)
					#print len_unq_hp
					wts[unq_all_healpixels_fn[hps[k]]] = 0
					#wts[k] = 0
					#Nd1 = np.shape(data1RA)[0] * np.mean(data1_smallmap[data2_smallpix[data2mask]])/np.mean(data1_smallmap[data2_smallpix])
					Nd1 = np.shape(data1RA)[0]
					print('modulated by ', np.mean(data1_smallmap[data2_smallpix[data2mask]])/np.mean(data1_smallmap[data2_smallpix]))
					for l,hpk in enumerate(hps[k]):
						Nd1 -=  hp_d1_cnts[np.where(unq_all_healpixels == hpk)]
						#cond = data2mask & (d1pix != hpk)
						if l == 0:
							cond = (d2pix != hpk)
						else:
							cond &= (d2pix != hpk)
						#Nd1 -= 
						#Nd1 = float(np.sum(hp_d1_cnts[np.where(unq_all_healpixels == hps[k])]))
					#data1_smallpix = hp.ang2pix(small_nside,data1RA[cond],data1DEC[cond],lonlat=True)
					#data1_smallmap = np.bincount(data1_smallpix,minlength=12*small_nside**2)
					print(Nd1)
					#Nd1 = Nd1 * np.mean(data1_smallmap[data2_smallpix[data2mask & cond]])/np.mean(data1_smallmap[data2_smallpix[cond]])
					print(Nd1)
					Nr1 = np.shape(rand1RA)[0]-cnts[k]
					#Nr1 = Nr1 * np.mean(rand1_smallmap[data2_smallpix[data2mask & cond]])/np.mean(rand1_smallmap[data2_smallpix[cond]])
					print(hps[k])
					print(Nd1,Nr1)
					loo_jackknife_dd[:,k] = resample(pair_mats_dd,wts)
					loo_jackknife_dr[:,k] = resample(pair_mats_dr,wts)
					
					counts_realpixes_jk = list(map(lambda x: x * wts, counts_realpixes))
					dat_random_ratio_realpixes_jk = list(map(lambda x: x * wts, dat_random_ratio_realpixes))
					rat = np.zeros(len(pair_mats_dd))
					for n in range(len(pair_mats_dd)):
						#rat[n] = np.nansum(counts_realpixes_jk[n]*dat_random_ratio_realpixes_jk[n])/np.nansum(counts_realpixes_jk[n])
						rat[n] = 1./(np.nansum(counts_realpixes_jk[n]*dat_random_ratio_realpixes_jk[n])/np.nansum(counts_realpixes_jk[n]))
		
					#print loo_jackknife_dd[:,k],loo_jackknife_dr[:,k], wjack_loo[:,k]
					wjack_loo[:,k] =  (loo_jackknife_dd[:,k]/loo_jackknife_dr[:,k]) * rat - 1.
					wjack_loo_no_adjust[:,k] = (loo_jackknife_dd[:,k]/loo_jackknife_dr[:,k]) * float(Nr1)/float(Nd1) - 1.

					#wjack_loo[:,k] =  (loo_jackknife_dd[:,k]/loo_jackknife_dr[:,k]) * float(Nr1)/float(Nd1)- 1.
					print(loo_jackknife_dd[:,k],loo_jackknife_dr[:,k], wjack_loo[:,k])
					ratios.append(rat)
				#print 5/0
			else:
				loo_jackknife_dd = np.zeros((nbins,len_unq_hp))
				loo_jackknife_dr = np.zeros((nbins,len_unq_hp))		
				for k in range(len_unq_hp):
					wts = np.ones(len_unq_hp)
					print(len_unq_hp)
					wts[k] = 0
					loo_jackknife_dd[:,k] = resample(pair_mats_dd,wts)
					loo_jackknife_dr[:,k] = resample(pair_mats_dr,wts)	
					# This is slightly fucked up because Nr1 and Nd1 should be corrected as for th equalized case
					# Haven't done that yet
					wjack_loo = loo_jackknife_dd/loo_jackknife_dr * float(Nr1)/float(Nd1) - 1.		
			
			#print(len(hps)*(len(hps)-1)/2,nbins)
			l2o_jackknife_dd = np.zeros((nbins,int(len(hps)*(len(hps)-1)/2)))
			l2o_jackknife_dr = np.zeros((nbins,int(len(hps)*(len(hps)-1)/2)))
			wjack_l2o = np.zeros((nbins,int(len(hps)*(len(hps)-1)/2)))
			wjack_l2o_cnts = np.zeros(int(len(hps)*(len(hps)-1)/2))
			
			cnter = 0
			for k in range(len(hps)):
				for l in range(len(hps)):
					if k < l:
						wts = np.ones(len_unq_hp)
						wts[unq_all_healpixels_fn[hps[k]]] = 0
						wts[unq_all_healpixels_fn[hps[l]]] = 0
						l2o_jackknife_dd[:,cnter] = resample(pair_mats_dd,wts)
						l2o_jackknife_dr[:,cnter] = resample(pair_mats_dr,wts)
						Nd1 = np.shape(data1RA)[0]
						for hpk in hps[k]:
							Nd1 -=  hp_d1_cnts[np.where(unq_all_healpixels == hpk)]
						for hpl in hps[l]:
							Nd1 -=  hp_d1_cnts[np.where(unq_all_healpixels == hpl)]
						#Nd1 = float(np.sum(hp_d1_cnts[np.where(unq_all_healpixels == hps[k])]))
						Nr1 = np.shape(rand1RA)[0]-cnts[k]-cnts[l]
						print(hps[k])
						print(Nd1,Nr1)
						wjack_l2o[:,cnter] = l2o_jackknife_dd[:,cnter]/l2o_jackknife_dr[:,cnter] * float(Nr1)/float(Nd1) - 1.
						wjack_l2o_cnts[cnter] = Nr1
						cnter += 1

		
			#wjack_loo = loo_jackknife_dd/loo_jackknife_dr * float(Nr1)/float(Nd1) - 1.
			
			
			
			def header(method):
				header_out='\n'.join([
						"SPEC=%s" % (ns.spec_name),
						"z1=%g z2=%g zmean=%g" % (z1, z2, zmean),
						"PHOTO=%s" % (ns.phot_name),
						"PHOTO_RANDOM=%s" % (ns.phot_name_randoms),
						"N_SPEC=%d" % (data2mask.sum()),
						"N_PHOT=%d" % (np.shape(data1RA)[0]),
						"N_PHOT_RANDOM=%d" % (np.shape(rand1RA)[0]),
						"ERROR=%s" % (method),
						"ESTIMATOR=Davis&Peebles",
						"NSIDE=%d" % (ns.nside),
						"COSMO=Planck15",
						"theta [deg] thmin [arcsec] thmax [arcsec] s [Mpc/h] smin [Mpc/h] smax [Mpc/h] " +
						"w [measured] err [Poisson] w std cov wsamples [%s] " % (method),
						])
				return header_out
				
			base_out = np.array([theta, theta_low, theta_high, s, slow, shigh, wmeas, wmeas_no_adj, wpoisson, w_mjb])
			
			if ns.equalize:
				cnts = np.array(cnts).astype('float')
				cov = weighted_cov_jack_loo(wjack_loo,wmeas,cnts)
				cov_no_adjust = weighted_cov_jack_loo(wjack_loo_no_adjust, wmeas, cnts)
			else:
				cov = (len_unq_hp-1.)*np.cov(wjack_loo,bias=True)
			
			# See https://www.stat.berkeley.edu/~hhuang/STAT152/Jackknife-Bootstrap.pdf for delete-2 jackknife
			# Note that they always have N in the denominator, not N-1, so I need to specify bias=True when
			# computing the covariance			
			#def make_output(base_out, arr):
			if ns.equalize:
				cnts_arr = np.zeros_like(wjack_loo)
				cnts_arr[0,:] = cnts
				myout = np.concatenate((base_out, [np.nanmean(wjack_loo,axis=1), np.sqrt(np.diag(cov)), np.nanmean(wjack_loo_no_adjust,axis=1)],
					cov, wjack_loo.transpose(),wjack_loo_no_adjust.transpose(),cnts_arr.transpose()),axis=0).T
			else:
				myout = np.concatenate((base_out, [np.nanmean(wjack_loo,axis=1), np.sqrt(np.diag(cov))],
					cov, wjack_loo.transpose()),axis=0).T
					
			cov_loo = np.copy(cov)
					
			if not os.path.exists(ns.outdir):
				os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:3]))
				os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:4]))
				os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:5]))
				os.system('mkdir %s' % '/'.join(ns.outdir.split('/')[:6]))
				os.system('mkdir %s' % ns.outdir)
			if not os.path.exists(ns.plotdir):
				os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:3]))
				os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:4]))
				os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:5]))
				os.system('mkdir %s' % '/'.join(ns.plotdir.split('/')[:6]))
				os.system('mkdir %s' % ns.plotdir)			

			if ns.equalize:
				np.savetxt(ns.outdir + 'z%.2f_%.2f_jk_loo_equalized.txt' % (z1,z2) , myout, header=header('jackknife-leave one out'))
			else:
				np.savetxt(ns.outdir + 'z%.2f_%.2f_jk_loo.txt' % (z1,z2) , myout, header=header('jackknife-leave one out'))
			
			#if ns.equalize:
			#	cov = weighted_cov_jack_l2o(wjack_l2o,wmeas,cnts)
			#else:
			cov = (len_unq_hp-2.)/2.*np.cov(wjack_l2o,bias=True)
			
			cnts_arr = np.zeros_like(wjack_l2o)
			cnts_arr[0,:] = wjack_l2o_cnts
			myout = np.concatenate((base_out, [np.nanmean(wjack_l2o,axis=1), np.sqrt(np.diag(cov))],
				cov, wjack_l2o.transpose(),cnts_arr.transpose()),axis=0).T

			#if ns.equalize:
			#	np.savetxt(ns.outdir + 'z%.2f_%.2f_jk_l2o_equalized.txt' % (z1,z2) , myout, header=header('jackknife-leave two out'))
			#else:
			np.savetxt(ns.outdir + 'z%.2f_%.2f_jk_l2o.txt' % (z1,z2) , myout, header=header('jackknife-leave two out'))
			
			plt.figure()
			plt.errorbar(s,wmeas,yerr=np.sqrt(np.diag(cov_loo)))
			plt.xscale('log')
			plt.xlabel(r'R ($h^{-1}$ Mpc)$',size=20)
			plt.ylabel(r'$w(\theta)$',size=20)
			plt.plot(np.linspace(0.1,100,1000),np.zeros(1000),color='k',linestyle='--')
			if not os.path.exists(ns.plotdir):
				os.system('mkdir %s' % ns.plotdir)
			plt.savefig(ns.plotdir + 'z%.2f_%.2f_jk_loo.pdf' % (z1,z2))
			print('wrote files', time.time()-t0)
