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
import pickle

# AK version of Ellie K photo-spectro code
# Changes files from hdf5 to fits and removes reference to completeness, tycho masks

cli = argparse.ArgumentParser("Cross correlate with QSO data of a selected redshift range")

cli.add_argument('--loadtree', dest='loadtree', action='store_true',help="Load the existing kdtree or make a new one")
cli.add_argument('--no-loadtree', dest='loadtree', action='store_false')
cli.set_defaults(loadtree=True)
cli.add_argument("--zmin",default=0.0,type=float,help="minimum redshift")
cli.add_argument("--zmax",default=4.0,type=float,help="maximum redshift")
cli.add_argument("--dz",default=0.01,type=float,help="delta z")
cli.add_argument("--Smin",default=0.05,type=float,help="minimum bin radius")
cli.add_argument("--Smax",default=50,type=float,help="maximum bin radius")
cli.add_argument("--Nbins",default=15,type=int,help="nbins")
cli.add_argument("--rand_ind",default=0,type=int,help='index of the spectroscopic randoms')
cli.add_argument("phot_name", help="internal catalogue of fits type.")
cli.add_argument("phot_name_randoms", help="internal catalogue of fits type.")
cli.add_argument("spec_name", help="internal catalogue of fits type.")
cli.add_argument("spec_name_randoms", help="internal catalogue of fits type.")

ns = cli.parse_args()

deltaz = ns.dz # deltaz for this guy
zmin = ns.zmin # For test purposes
zmax = ns.zmax

# Binning (min and max in Mpc/h)
Smin = ns.Smin
#Smax = 12.5594322
#nbins = 12
Smax = ns.Smax
nbins = ns.Nbins

def truncate(name):
	'''Truncates a filename so I can use it to name things'''
	return name.split('/')[-1].split('.fits')[0]
	
def sparse_histogram(dataset):
	'''Defines a sparse histogram'''
	if len(dataset) == 0:
		#return []
		return (0,0)
	else:
		maxx = np.max(dataset)
		minn = np.min(dataset)
		if minn == maxx:
			cnts = len(dataset)
			lowbin = np.min(dataset)
		else:
			h = np.histogram(dataset,range=(minn,maxx+1),bins=maxx-minn+1)
			cnts = h[0][h[0] != 0]
			lowbin = h[1][:-1][h[0] != 0]
		return cnts, lowbin

#def main():
t0 = time.time()
data1file = fits.open(ns.phot_name)[1].data
data2file = fits.open(ns.spec_name)[1].data
rand1file = fits.open(ns.phot_name_randoms)[1].data
rand2file = fits.open(ns.spec_name_randoms)[1].data

data1RA = data1file['RA'][:]
data1DEC = data1file['DEC'][:]

rand1RA = rand1file['RA'][:]
rand1DEC = rand1file['DEC'][:]

d1rad = np.array([data1DEC*np.pi/180.,data1RA*np.pi/180.]).transpose()
r1rad = np.array([rand1DEC*np.pi/180.,rand1RA*np.pi/180.]).transpose()
print("Loaded data",t0-time.time())

nside_base = 256 # Let's see if I can get the stupid thing to work for nside=256, i.e. a whole bunch of histograms
# Maybe represent them as sparse matrices or something to speed stuff up?
# Actually, maybe it's easier to just write a whole bunch of lists of things.

d1pix = hp.ang2pix(nside_base, data1RA, data1DEC, nest=False, lonlat=True)
r1pix = hp.ang2pix(nside_base, rand1RA, rand1DEC, nest=False, lonlat=True)
print("Computed healpixels",t0-time.time())

if not ns.loadtree:
	t0 = time.time()
	d1tree = neighbors.BallTree(d1rad,metric='haversine')
	pickle.dump(d1tree,open('%s-d1tree.p' % (truncate(ns.phot_name)),'w'))
	print time.time()-t0 # 5x as long as flatsky case

	r1tree = neighbors.BallTree(r1rad,metric='haversine')
	pickle.dump(r1tree,open('%s-r1tree.p' % (truncate(ns.phot_name_randoms)),'w'))
else:
	d1tree = pickle.load(open('%s-d1tree.p' % (truncate(ns.phot_name)),'r'))
	print("Loaded data tree",t0-time.time())
	r1tree = pickle.load(open('%s-r1tree.p' % (truncate(ns.phot_name_randoms)),'r'))
	print("Loaded random tree",t0-time.time())


#zs = np.arange(zmin,zmax+deltaz,deltaz)
zs = np.linspace(zmin,zmax+deltaz,1+int(round((zmax+deltaz-zmin)/deltaz)))
print deltaz
print zs
os.system('mkdir %s-%s/' % (truncate(ns.phot_name),truncate(ns.spec_name)))

for i in range(len(zs)-1):
        z1 = zs[i]                                                                                                                                        
        z2 = zs[i+1] 
        if zmin == 0:                                                                                                                                
                name_ind = i                                                                                                                         
        else:                                                                                                                                        
                name_ind = int(round(zmin/deltaz)) + i                                                                                               
                print z1, z2, name_ind 
	#if os.path.exists('%s-%s/%i.bin' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind)):
	#	continue
	#else:
        #        print i
	data2mask  = data2file['Z'][:] >= z1
	data2mask &= data2file['Z'][:] <  z2
	
	
	rand2mask  = rand2file['Z'][:] >= z1
	rand2mask &= rand2file['Z'][:] <  z2
	
	if data2mask.sum() == 0:
		pass
	else:
		print("Selected %d QSOs from %.6f to %.6f, i = %i" % (data2mask.sum(),z1,z2, i))
	
	

		data2RA = data2file['RA'][:][data2mask]
		data2DEC = data2file['DEC'][:][data2mask]
		data2Z = data2file['Z'][:][data2mask]
		
		rand2RA = rand2file['RA'][:][rand2mask]
		rand2DEC = rand2file['DEC'][:][rand2mask]
		rand2Z = rand2file['Z'][:][rand2mask]

		h0 = LCDM.H0 / (100 * u.km / u.s / u.Mpc)
		zmean = data2Z.mean()
		R = (LCDM.comoving_distance(zmean) / (u.Mpc / h0 ))


		thmin = (Smin/R)*180./np.pi
		thmax = (Smax/R)*180./np.pi
		#print(thmin)
		#print(np.logspace(-3,0,16,endpoint=True))
		b = np.logspace(np.log10(thmin),np.log10(thmax),nbins+1,endpoint=True)

		d2rad = np.array([data2DEC*np.pi/180.,data2RA*np.pi/180.]).transpose()
		r2rad = np.array([rand2DEC*np.pi/180.,rand2RA*np.pi/180.]).transpose()

		print time.time()-t0, "preliminaries"
		#t0 = time.time()
		dd_tree_out = d1tree.query_radius(d2rad, np.max(b)*np.pi/180., return_distance=True, count_only=False)
		print time.time()-t0, " queried data-data"
		
		t0 = time.time()
		drspec_tree_out = d1tree.query_radius(r2rad, np.max(b)*np.pi/180., return_distance=True, count_only=False)
		print time.time()-t0, " queried data-spec random"

		#t0 = time.time()
		dr_tree_out = r1tree.query_radius(d2rad, np.max(b)*np.pi/180., return_distance=True, count_only=False)
		print time.time()-t0, " queried random-phot data"

		t0 = time.time()
		rr_tree_out = r1tree.query_radius(r2rad, np.max(b)*np.pi/180., return_distance=True, count_only=False)
		print time.time()-t0, " queried random-random"

		dd = map(lambda x: np.histogram(x,bins=b*np.pi/180.)[0],dd_tree_out[1])		
		dd_pix = map(lambda x: d1pix[x], dd_tree_out[0])	
		
		dd_pix_list = []
		
		drspec = map(lambda x: np.histogram(x,bins=b*np.pi/180.)[0],drspec_tree_out[1])		
		drspec_pix = map(lambda x: d1pix[x], drspec_tree_out[0])	
		
		drspec_pix_list = []
		
		dr = map(lambda x: np.histogram(x,bins=b*np.pi/180.)[0],dr_tree_out[1])
		dr_pix = map(lambda x: r1pix[x], dr_tree_out[0])
			
		dr_pix_list = []
		
		rr = map(lambda x: np.histogram(x,bins=b*np.pi/180.)[0],rr_tree_out[1])
		rr_pix = map(lambda x: r1pix[x], rr_tree_out[0])
			
		rr_pix_list = []
		
		print time.time()-t0," made histograms"
		
		for j in range(len(dd)):	
			dd_hist_inds_orig = np.digitize(dd_tree_out[1][j],bins=b*np.pi/180.)-1
			dd_hist_inds = dd_hist_inds_orig[dd_hist_inds_orig >= 0]
			
			dd_pixj = dd_pix[j]
			dd_pixj = dd_pixj[dd_hist_inds_orig >= 0]
			
			dd_hist_inds_s = np.argsort(dd_hist_inds)
		
			cs = np.concatenate((np.array([0]),np.cumsum(dd[j])))
		
			dd_pix_list.append(map(lambda k: sparse_histogram(dd_pixj[dd_hist_inds_s][cs[k]:cs[k+1]]), range(len(dd[j]))))
	
			dr_hist_inds_orig = np.digitize(dr_tree_out[1][j],bins=b*np.pi/180.)-1
			dr_hist_inds = dr_hist_inds_orig[dr_hist_inds_orig >= 0]
			
			dr_pixj = dr_pix[j]
			dr_pixj = dr_pixj[dr_hist_inds_orig >= 0]
			
			dr_hist_inds_s = np.argsort(dr_hist_inds)
			
			cs = np.concatenate((np.array([0]),np.cumsum(dr[j])))

			dr_pix_list.append(map(lambda k: sparse_histogram(dr_pixj[dr_hist_inds_s][cs[k]:cs[k+1]]), range(len(dd[j]))))


		for j in range(len(drspec)):							
			drspec_hist_inds_orig = np.digitize(drspec_tree_out[1][j],bins=b*np.pi/180.)-1
			drspec_hist_inds = drspec_hist_inds_orig[drspec_hist_inds_orig >= 0]
			
			drspec_pixj = drspec_pix[j]
			drspec_pixj = drspec_pixj[drspec_hist_inds_orig >= 0]
			
			drspec_hist_inds_s = np.argsort(drspec_hist_inds)
			
			cs = np.concatenate((np.array([0]),np.cumsum(drspec[j])))

			drspec_pix_list.append(map(lambda k: sparse_histogram(drspec_pixj[drspec_hist_inds_s][cs[k]:cs[k+1]]), range(len(drspec[j]))))

	
			rr_hist_inds_orig = np.digitize(rr_tree_out[1][j],bins=b*np.pi/180.)-1
			rr_hist_inds = rr_hist_inds_orig[rr_hist_inds_orig >= 0]
			
			rr_pixj = rr_pix[j]
			rr_pixj = rr_pixj[rr_hist_inds_orig >= 0]
			
			rr_hist_inds_s = np.argsort(rr_hist_inds)
			
			cs = np.concatenate((np.array([0]),np.cumsum(rr[j])))
			
			rr_pix_list.append(map(lambda k: sparse_histogram(rr_pixj[rr_hist_inds_s][cs[k]:cs[k+1]]), range(len(drspec[j]))))


		print time.time()-t0," made pixel lists"
		
		inds = np.where(data2mask==True)[0]
		#arr_out = np.concatenate((inds[:,np.newaxis],dd,dr),axis=1)
		#arr_out.tofile('%s-%s-landy_szalay/%i_data_%i.bin' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind,ns.rand_ind))
		inds_rand = np.where(rand2mask==True)[0]
		#arr_out = np.concatenate((inds[:,np.newaxis],drspec,rr),axis=1)
		#arr_out.tofile('%s-%s-landy_szalay/%i_data_rands_%i.bin' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind,ns.rand_ind))
		print time.time()-t0," wrote histograms"
		#pickle.dump([inds,inds_rand,dd_pix_list,dr_pix_list,drspec_pix_list,rr_pix_list],open('%s-%s-landy_szalay/%i_pix_list_%i-ALL.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind,ns.rand_ind),'wb'))
		pickle.dump([inds,inds_rand,dd_pix_list,dr_pix_list,drspec_pix_list,rr_pix_list],open('%s-%s-landy_szalay/%i_pix_list_%i.p' % (truncate(ns.phot_name),truncate(ns.spec_name),name_ind,ns.rand_ind),'wb'))

		print time.time()-t0," wrote pixel lists"
		
		del data2RA
		del data2DEC
		del data2Z
		
		del rand2RA
		del rand2DEC
		del rand2Z
		del d2rad
		del r2rad
		del drspec_tree_out
		del rr_tree_out
		del drspec
		del rr
		del drspec_pix
		del rr_pix
		del drspec_pix_list
		del rr_pix_list
		#del arr_out