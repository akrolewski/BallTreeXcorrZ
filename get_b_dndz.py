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

ns = cli.parse_args()


t0 = time.time()
plt.ion()
plt.figure()

#error = sys.argv[1]
#name = sys.argv[2]
#hemi = sys.argv[3]

zs = np.linspace(ns.zmin,ns.zmax,int(round((ns.zmax-ns.zmin)/ns.dz))+1)

dndzall = np.zeros_like(zs)
dndzall_err = np.zeros_like(zs)

nboot = int(ns.gal_name.split('_')[-1])

for i in range(len(zs)-1):
	#i = 3
	#i = 5
	try:
		#dat = np.loadtxt('xcorr_out/unwise_DR14_QSO/ak/%s/nostar_%s/z%.2f_%.2f.txt' % (hemi,name,zs[i],zs[i+1]))
		if ns.error == 'bootstrap':
			ind = 'bs'
			dat = np.loadtxt('%s/%s/z%.2f_%.2f_%s_%s.txt' % (ns.xcorrdir_name,ns.gal_name,zs[i],zs[i+1],ind,ns.boot_name))
		elif ns.error == 'jackknife':
			ind = 'jk'
			dat = np.loadtxt('%s/%s/z%.2f_%.2f_%s_%s.txt' % (ns.xcorrdir_name,ns.gal_name,zs[i],zs[i+1],ind,ns.jack_name))
			
		s = dat[:,3]
		#wsamples = dat[:,11:]
		#cov = np.cov(wsamples)
		#w = np.mean(wsamples,axis=1)
		w = dat[:,6]
		ds = dat[:,5]-dat[:,4] # will replace with ds from actual bins...
		ind = np.where(s > 0.4)[0][0]
		dndz = np.sum(ds[ind:]/s[ind:]*w[ind:])
		err_poisson = dat[:,7]
		cov = dat[:,10:25]

		dndz_err = np.sqrt(np.sum(np.dot(cov[ind:,ind:],ds[ind:]**2/s[ind:]**2)))			
		dndzall[i] = dndz
		dndzall_err[i] = dndz_err
	except IOError:
		continue
print time.time()-t0

if ns.gal_name.find('/') != -1:
	gal_name = ns.gal_name.replace('/','_')
	
#np.savetxt('bdndz/unwise_DR14_QSO/ak/%s/dz0.05/%s/nostar_%s.txt' % (hemi,error,name), np.array([dndzall,dndzall_err]).transpose())
os.system('mkdir %s/dz%.2f/' % (ns.bdndz_name, ns.dz))
os.system('mkdir %s/dz%.2f/%s' % (ns.bdndz_name,ns.dz,ns.error))
if ns.error == 'bootstrap':
	np.savetxt('%s/dz%.2f/%s/%s_%s.txt' % (ns.bdndz_name,ns.dz,ns.error,gal_name,ns.boot_name), np.array([dndzall,dndzall_err]).transpose())
elif ns.error == 'jackknife':
	np.savetxt('%s/dz%.2f/%s/%s_%s.txt' % (ns.bdndz_name,ns.dz,ns.error,gal_name,ns.jack_name), np.array([dndzall,dndzall_err]).transpose())
elif ns.error == 'poisson-diag':
	np.savetxt('%s/dz%.2f/%s/%s.txt' % (ns.bdndz_name,ns.dz,ns.error,gal_name), np.array([dndzall,dndzall_err]).transpose())


plt.errorbar(zs,dndzall,yerr=dndzall_err)
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