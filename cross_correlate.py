from astropy.io import fits
import numpy as np
import argparse
import os
from sklearn import neighbors
import healpy as hp
import pickle
from astropy.cosmology import Planck15 as LCDM
from astropy import units as u
import time

# AK version of Ellie K photo-spectro code
# Changes files from hdf5 to fits and removes reference to completeness, tycho masks

cli = argparse.ArgumentParser("Cross correlate with QSO data of a selected redshift range")

cli.add_argument('--loadtree', dest='loadtree', action='store_true',help="Load the existing kdtree or make a new one")
cli.add_argument('--no-loadtree', dest='loadtree', action='store_false')
cli.set_defaults(loadtree=True)
cli.add_argument("--zmin",default=0.0,type=float,help="minimum redshift")
cli.add_argument("--zmax",default=4.0,type=float,help="maximum redshift")
cli.add_argument("--dz",default=0.01,type=float,help="delta z")
cli.add_argument("phot_name", help="internal catalogue of fits type.")
cli.add_argument("phot_name_randoms", help="internal catalogue of fits type.")
cli.add_argument("spec_name", help="internal catalogue of fits type.")

ns = cli.parse_args()

deltaz = ns.dz # deltaz for this guy
zmin = ns.zmin # For test purposes
zmax = ns.zmax

# Binning (min and max in Mpc/h)
Smin = 0.05
Smax = 50
nbins = 15

def truncate(name):
	'''Truncates a filename so I can use it to name things'''
	return name.split('/')[-1].split('.fits')[0]

#def main():
t0 = time.time()
data1file = fits.open(ns.phot_name)[1].data
data2file = fits.open(ns.spec_namee)[1].data
rand1file = fits.open(ns.phot_name_randoms)[1].data

data1RA = data1file['RA'][:]
data1DEC = data1file['DEC'][:]

rand1RA = rand1file['RA'][:]
rand1DEC = rand1file['DEC'][:]

d1rad = np.array([data1DEC*np.pi/180.,data1RA*np.pi/180.]).transpose()
r1rad = np.array([rand1DEC*np.pi/180.,rand1RA*np.pi/180.]).transpose()
print("Loaded data")

if not ns.loadtree:
	t0 = time.time()
	d1tree = neighbors.BallTree(d1rad,metric='haversine')
	pickle.dump(d1tree,open('%s-d1tree.p' % (truncate(ns.phot_name)),'w'))
	print time.time()-t0 # 5x as long as flatsky case

	r1tree = neighbors.BallTree(r1rad,metric='haversine')
	pickle.dump(r1tree,open('%s-r1tree.p' % (truncate(ns.phot_name_randoms)),'w'))
else:
	d1tree = pickle.load(open('%s-d1tree.p' % (truncate(ns.phot_name)),'r'))
	print("Loaded data tree")
	r1tree = pickle.load(open('%s-r1tree.p' % (truncate(ns.phot_name_randoms)),'r'))
	print("Loaded random tree")


zs = np.arange(zmin,zmax+deltaz,deltaz)

os.system('mkdir %s-%s/' % (truncate(ns.phot_name),truncate(ns.spec_name)))

for i in range(len(zs)-1):
	z1 = zs[i]
	z2 = zs[i+1]
	data2mask  = data2file['Z'][:] >= z1
	data2mask &= data2file['Z'][:] <  z2
	
	if data2mask.sum() == 0:
		pass
	else:
		print("Selected %d QSOs from %.2f to %.2f" % (data2mask.sum(),z1,z2))
	
	

		data2RA = data2file['RA'][:][data2mask]
		data2DEC = data2file['DEC'][:][data2mask]
		data2Z = data2file['Z'][:][data2mask]

		h0 = LCDM.H0 / (100 * u.km / u.s / u.Mpc)
		zmean = data2Z.mean()
		R = (LCDM.comoving_distance(zmean) / (u.Mpc / h0 ))


		thmin = (Smin/R)*180./np.pi
		thmax = (Smax/R)*180./np.pi
		#print(thmin)
		#print(np.logspace(-3,0,16,endpoint=True))
		b = np.logspace(np.log10(thmin),np.log10(thmax),nbins+1,endpoint=True)

		d2rad = np.array([data2DEC*np.pi/180.,data2RA*np.pi/180.]).transpose()

		t0 = time.time()
		dd_tree_out = d1tree.query_radius(d2rad, np.max(b)*np.pi/180., return_distance=True, count_only=False)
		print time.time()-t0, " queried data"

		t0 = time.time()
		dr_tree_out = r1tree.query_radius(d2rad, np.max(b)*np.pi/180., return_distance=True, count_only=False)
		print time.time()-t0, " queried random"

		dd = map(lambda x: np.histogram(x,bins=b*np.pi/180.)[0],dd_tree_out[1])

		dr = map(lambda x: np.histogram(x,bins=b*np.pi/180.)[0],dr_tree_out[1])

		inds = np.where(data2mask==True)[0]
		arr_out = np.concatenate((inds[:,np.newaxis],dd,dr),axis=1)
		if zmin == 0:
			name_ind = i
		else:
			name_ind = int(round(zmin/deltaz)) + i
			print z1, z2, name_ind
		arr_out.tofile('%s-%s/%i.bin' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind))