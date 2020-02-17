import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits

data = np.loadtxt('z0.30_0.35_jk_loo_equalized.txt')

# Measured correlation function
corr = data[:,7]

# Minimum bin radius in arcsec
bin_min = data[:,1]
# Maximum bin radius
bin_max = data[:,2]

# Load data and randoms
cmass = fits.open('test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits')[1].data
cmass_random = fits.open('test/cmass_random0_masked-r1-v2-flag_30_35_cmass+lowz_mask.fits')[1].data

cmass_coords = SkyCoord(ra=cmass['ra']*u.deg,dec=cmass['dec']*u.deg)
random_coords = SkyCoord(ra=cmass_random['ra']*u.deg,dec=cmass_random['dec']*u.deg)

Nd = float(len(cmass_coords))
Nr = float(len(random_coords))

astropy_corr = np.zeros(len(bin_min))

# Compute correlation function
for i in range(len(bin_min)):
	idx1_max,_,_, _ =cmass_coords.search_around_sky(cmass_coords,bin_max[i]*u.arcsec)
	idx1_min,_,_, _ =cmass_coords.search_around_sky(cmass_coords,bin_min[i]*u.arcsec)
	
	counts_data = len(idx1_max)-len(idx1_min)
	
	idx1_max,_,_, _ =random_coords.search_around_sky(cmass_coords,bin_max[i]*u.arcsec)
	idx1_min,_,_, _ =random_coords.search_around_sky(cmass_coords,bin_min[i]*u.arcsec)
	
	counts_random = len(idx1_max)-len(idx1_min)
	
	astropy_corr[i] = (counts_data/counts_random) * Nr/Nd - 1.
	
print('BallTreeXcorrZ correlation function', corr)
print('Astropy correlation function', astropy_corr)
