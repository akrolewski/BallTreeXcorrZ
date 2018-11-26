import numpy as np
import healpy as hp
import argparse
from astropy.cosmology import Planck15 as LCDM
from astropy.io import fits
from astropy import units as u
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

# Binning parameters (min/max in Mpc/h)
Smin = 0.05
Smax = 50
nbins = 15

orig_deltaz = 0.01

zlen = int(ns.dz/orig_deltaz)
zbin = int(round((ns.zmax-ns.zmin)/ns.dz))

ncols = 2 * nbins + 1

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

for i in range(zbin):
	flag = 0
	for j in range(zlen):
		name_ind = int((i*ns.dz + j*orig_deltaz)/orig_deltaz)
		try:
			arrin = np.fromfile('%s-%s/%i.bin' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind),dtype=int)
			arrin = arrin.reshape((len(arrin)/ncols,ncols))
			if flag == 0:
				allin = arrin
				flag += 1
			else:
				allin = np.concatenate((allin,arrin),axis=0)
		except IOError:
			continue
	if flag != 0:
		pix = hp.ang2pix(ns.nside,data2RA[allin[:,0]],data2DEC[allin[:,0]],lonlat=True)

		argsort_pix = np.argsort(pix)
		argsort_allin = allin[argsort_pix,:]

		unq_pix, counts = np.unique(pix,return_counts=True)

		inds_lower = np.searchsorted(pix[argsort_pix],unq_pix,side='left')
		inds_upper = np.searchsorted(pix[argsort_pix],unq_pix,side='right')

		summed_allin = np.array(map(lambda k: np.sum(argsort_allin[inds_lower[k]:inds_upper[k],1:],axis=0),range(len(inds_lower))))
		dd = np.sum(summed_allin[:,:nbins],axis=0).astype('float')
		dr = np.sum(summed_allin[:,nbins:],axis=0).astype('float')
		wmeas = dd/dr * float(Nr1)/float(Nd1) - 1.
		
		wpoisson = np.sqrt(dd)/dr * float(Nr1)/float(Nd1)

		boot_pix = np.random.choice(range(len(unq_pix)),size=(len(unq_pix),ns.nboot),replace=True)

		summed_allin_boot = summed_allin[boot_pix,:]

		boot_counts = np.sum(counts[boot_pix],axis=0) # No of quasars in each bootstrap

		dd_boot = np.sum(summed_allin_boot[:,:,:nbins],axis=0).astype('float')
		dr_boot = np.sum(summed_allin_boot[:,:,nbins:],axis=0).astype('float')
		wsamples = dd_boot/dr_boot * float(Nr1)/float(Nd1) - 1.
	
		mean = np.nanmean(wsamples, axis=0)
		std = np.nanstd(wsamples, axis=0)
		sem = std / (~np.isnan(wsamples)).sum(axis=0) ** 0.5

		z1 = i*ns.dz
		z2 = (i+1)*ns.dz
		data2mask  = data2file['Z'][:] >= z1
		data2mask &= data2file['Z'][:] <  z2

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
				"theta [deg] thmin [arcsec] thmax [arcsec] s [Mpc/h] smin [Mpc/h] smax [Mpc/h] w [mean bootstrap] std SEM w [measured] err [Poisson]",
				])

		theta_low = 3600.*np.logspace(np.log10(thmin), np.log10(thmax), nbins+1, endpoint=True)[:-1]
		theta_high  = 3600.*np.logspace(np.log10(thmin), np.log10(thmax), nbins+1, endpoint=True)[1:]
		slow = np.radians(theta_low.value/3600.)*R
		shigh = np.radians(theta_high.value/3600.)*R

		myout = np.concatenate(([theta, theta_low, theta_high, s, slow, shigh, mean, std, sem, wmeas, wpoisson], wsamples),axis=0).T
		np.savetxt(ns.outdir + 'z%.2f_%.2f.txt' % (z1,z2), myout, header=header)
		print i, wmeas, mean

		plt.figure()
		plt.errorbar(s,wmeas,yerr=std)
		plt.xscale('log')
		plt.xlabel(r'R ($h^{-1}$ Mpc)$',size=20)
		plt.ylabel(r'$w(\theta)$',size=20)
		plt.plot(np.linspace(0.1,100,1000),np.zeros(1000),color='k',linestyle='--')
		plt.savefig(ns.plotdir + 'z%.2f_%.2f.pdf' % (z1,z2))

	