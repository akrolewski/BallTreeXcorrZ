import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
import argparse
import os

cli = argparse.ArgumentParser("Cross correlate with QSO data of a selected redshift range")

cli.add_argument("--xcorrdir_name",help="name of xcorr output directory")
cli.add_argument("--gal_name",help="name of galaxy directory in xcorr output directory")
cli.add_argument("--bdndz_name",help="name of bdndz output directory")
cli.add_argument("--error",help="boot or poisson-diag or jackknife. Specifies method for computing correlation function error bars.")
cli.add_argument("--zmin",default=0.0,type=float,help="minimum redshift")
cli.add_argument("--zmax",default=4.0,type=float,help="maximum redshift")
cli.add_argument("--dz",default=0.05,type=float,help="delta z")
cli.add_argument("--boot_name",help="type of bootstrap error (literal, sqrt, marked)")
cli.add_argument("--jack_name",help="type of jackknife error (loo, l2o)")
cli.add_argument("--njack",type=int,help="number of jackknife resamples")
cli.add_argument("--lower_ind",type=int,help="index of the minimum r_p bin to use. In the default binning scheme, 6: r_p,min=1.59 h^-1 Mpc, 7: r_p,min=2.52 h^-1 Mpc, 8: r_p,min=4 h^-1 Mpc")
cli.add_argument("--nbins",type=int,help="number of bins in the outpu")


ns = cli.parse_args()

nbins = ns.nbins
inds = ns.lower_ind


t0 = time.time()
plt.ion()
plt.figure()


zs = np.linspace(ns.zmin,ns.zmax,int(round((ns.zmax-ns.zmin)/ns.dz))+1)

dndzall = np.zeros(len(zs)-1)
dndzall_err = np.zeros(len(zs)-1)

nboot = int(ns.gal_name.split('_')[-1])

for i in range(len(zs)-1):
	try:
		if ns.error == 'bootstrap':
			ind = 'bs'
			dat = np.loadtxt('%s/%s/z%.2f_%.2f_%s_%s.txt' % (ns.xcorrdir_name,ns.gal_name,zs[i],zs[i+1],ind,ns.boot_name))
		elif ns.error == 'jackknife':
			ind = 'jk'
			dat = np.loadtxt('%s/%s/z%.2f_%.2f_%s_%s.txt' % (ns.xcorrdir_name,ns.gal_name,zs[i],zs[i+1],ind,ns.jack_name))
			f = open('%s/%s/z%.2f_%.2f_%s_%s.txt' % (ns.xcorrdir_name,ns.gal_name,zs[i],zs[i+1],ind,ns.jack_name),'r')
			for line in f:
				if line[:9] == '# N_SPEC=':
					nspec = float(line[9:])
				
			f.close()
		s = dat[:,3]
		w = dat[:,6]
		ds = dat[:,5]-dat[:,4] # will replace with ds from actual bins...
		dndz = np.sum(ds[ind:]/s[ind:]*w[ind:])
		err_poisson = dat[:,7]
		#cov = dat[:,11:11+nbins]
		cov = dat[:,13:13+nbins]
		wjack_loo = dat[:,13+nbins:13+nbins+ns.njack]
		cnts = dat[0,13+nbins+2*ns.njack:13+nbins+3*ns.njack]
		
		print cnts
		print np.sqrt(np.diag(cov[4:,4:]))
		
		print nspec, ns.njack, i
	
		if (np.any(cov[ind:,ind:] == 0)) or (nspec < ns.njack):
			dndzall[i] = np.nan
			dndzall_err[i] = np.nan
		else:
			wbar = np.sum((ds[ind:]/s[ind:])[:,np.newaxis]*wjack_loo[ind:,:],axis=0)
			dndz_err = np.sqrt(np.sum((np.sum(cnts)-cnts)/np.sum(cnts) *(wbar-np.mean(wbar))**2.))	
			print i, dndz_err, np.sqrt(np.sum(np.dot(cov[ind:,ind:],ds[ind:]**2/s[ind:]**2)))		
			dndzall[i] = dndz
			dndzall_err[i] = dndz_err
			print time.time()-t0
	except IOError:
		dndzall[i] = np.nan
		dndzall_err[i] = np.nan
print time.time()-t0

if ns.gal_name.find('/') != -1:
	gal_name = ns.gal_name.replace('/','_')
	
#np.savetxt('bdndz/unwise_DR14_QSO/ak/%s/dz0.05/%s/nostar_%s.txt' % (hemi,error,name), np.array([dndzall,dndzall_err]).transpose())
os.system('mkdir %s' % ('/'.join(ns.bdndz_name.split('/')[:-3])))
os.system('mkdir %s' % ('/'.join(ns.bdndz_name.split('/')[:-2])))
os.system('mkdir %s' % ('/'.join(ns.bdndz_name.split('/')[:-1])))
os.system('mkdir %s' % (ns.bdndz_name))
print 'mkdir %s/dz%.2f/' % (ns.bdndz_name, ns.dz)
os.system('mkdir %s/dz%.2f/' % (ns.bdndz_name, ns.dz))
os.system('mkdir %s/dz%.2f/%s' % (ns.bdndz_name,ns.dz,ns.error))
if ns.error == 'bootstrap':
	np.savetxt('%s/dz%.2f/%s/%s_%s.txt' % (ns.bdndz_name,ns.dz,ns.error,gal_name,ns.boot_name), np.array([dndzall,dndzall_err]).transpose())
elif ns.error == 'jackknife':
	np.savetxt('%s/dz%.2f/%s/%s_%s.txt' % (ns.bdndz_name,ns.dz,ns.error,gal_name,ns.jack_name), np.array([dndzall,dndzall_err]).transpose())
elif ns.error == 'poisson-diag':
	np.savetxt('%s/dz%.2f/%s/%s.txt' % (ns.bdndz_name,ns.dz,ns.error,gal_name), np.array([dndzall,dndzall_err]).transpose())


plt.errorbar(0.5*(zs[:-1]+zs[1:]),dndzall,yerr=dndzall_err)
plt.ylim(-0.1,0.4)

plt.xlabel('z',size=30)
plt.ylabel(r'$b \frac{dN}{dz}$',size=30)
plt.title(gal_name, size=20)

plt.plot(zs,np.zeros_like(zs),color='r',linestyle='--')

plt.tight_layout()

if ns.error == 'bootstrap':
	plt.savefig('%s/dz%.2f/%s/%s_%s.pdf' % (ns.bdndz_name,ns.dz,ns.error,gal_name,ns.boot_name))
elif ns.error == 'jackknife':
	plt.savefig('%s/dz%.2f/%s/%s_%s.pdf' % (ns.bdndz_name,ns.dz,ns.error,gal_name,ns.jack_name))
elif ns.error == 'poisson-diag':
	plt.savefig('%s/dz%.2f/%s/%s.pdf' % (ns.bdndz_name,ns.dz,ns.error,gal_name))
print time.time()-t0