# BallTreeXcorrZ
cross-correlation redshifts using sklearn's BallTree

We implement the clustering redshift method of Menard++13 (see also Newman+08,McQuinn&White+2013, among others) using sklearn's BallTree class. This code can quickly compute angular clustering, b*dN/dz, and their errors, for ~1M photometric and ~100k spectroscopic objects.

Requires numpy, healpy, astropy, scikit-learn.

The code separates pair-counting and error estimation & redshift binning, allowing the user to efficiently test different redshift bin widths and error-estimation schemes.

Pair counting is done in cross_correlate.py, which starts by generating a BallTree of the photometric catalog or photometric randoms. Optionally one can load a pre-existing tree, which allows you to run different deltaz slices at once. See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html for details on BallTree. I use the default leaf_size as I find tree-generation is <20 minutes for my largest catalogs, but one could imagine optimizing by playing with leaf_size. I use the 'haversine' metric to compute curved-sky distances between points.

After generating (or loading) the tree, I then loop over redshift slices of deltaz=0.01, chosen to be as small or smaller than you would want to use on data. For each redshift slice I return the pair counts around each spectroscopic object and the index of that spectroscopic object in the spectroscopic catalog.

Generating and querying the tree are the most computationally expensive steps of the process. I have not set up cross_correlate.py with MPI, but instead leave it up to the user to choose a zmin and zmax and potentially run several chunks in parallel.

error_estimator.py computes the cross-correlation and is capable of several methods of error estimation.  It supports both jackknife 
and bootstrap error estimation, and always computes Poisson error bars by default.  If the user wants bootstrap error bars, error_estimator.py computes the bootstrap three different ways: the "literal" bootstrap in which certain regions are fully masked and others are counted twice (thus a pair with both objects in a double counted region will be quadruple counted); a "marked" bootstrap (Loh &Stein 2004, Loh 2008) where only the positions of the spectroscopic objects are resampled; and a "sqrt" bootstrap in which I take the square root of the weights from the literal bootstrap.  For the jackknife error bars I compute both leave-one-out and leave-two-out jackknifes.  I also include a method to equalize the areas of the jackknife regions and weight the jackknife covariance estimate by the  (slightly non-equal) areas of the resulting regions (e.g. Myers et al. 2004).

Finally, the b*dN/dz distribution is computed in get_b_dndz.py.

For example, to run the test:

python cross_correlate.py --no-loadtree --dz 0.05 --Smin 0.1006799 --Smax 10 --Nbins 10 test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits test/cmass_random0_masked-r1-v2-flag_30_35_cmass+lowz_mask.fits test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits

python error_estimator.py --jackknife --equalize --Smin 0.1006799 --Smax 10 --nbins 10 --orig_deltaz 0.05 --nside 4 --nboot 1000 --dz 0.05 --zmin 0.30 --zmax 0.35 --small_nside 8 test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits test/cmass_random0_masked-r1-v2-flag_30_35_cmass+lowz_mask.fits test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits ./ ./'

This creates 10 logspaced bins between smin = 0.1 and smax = 10 h^-1 Mpc. Angle is converted to distance
in bins given by the --dz argument in cross_correlate.py--these bins can be made finer than the bin
in which you compute the correlation function. I typically use --dz = 0.01 or 0.001. dz=0.05
is only used to construct the test here.
The --loadtree argument will reload the tree, saving time for larger datasets where creating the tree is expensive.

In error_estimator, the --jackknife arguments tells it to compute the jackknife errors, and --equalize
tells it to correct for the slightly different sizes of the jackknife regions.  --orig_deltaz
corresponds to the --dz argument of cross_correlate.  --nside is the nside used to construct
jackknife regions--the code bins all points into healpixels of this nside, and then combines
them together to make regions of roughly the same size.
zmin and zmax give the minimum and maximum of the z range, and --dz gives its spacing.
"small_nside" allows the user to adjust the size of the pixels used to estimate
the local galaxy density--small_nside = 8 means that the code uses nside=8 pixels
to estimate the local density.

To average over the relevant scales and create a b*dndz file:
python get_b_dndz.py --xcorrdir_name path/to/directory --gal_name path/to/output
--error jackknife --dz 0.05 --njack njack, where njack is the number of jackknife
regions used.

run_test.sh runs a test set, which is compared to the correlation function
computed from astropy's SkyCoord module in test_correlation_function.py.
This verifies that we recover the correct correlation function on the curved sky.