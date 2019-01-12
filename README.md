# BallTreeXcorrZ
cross-correlation redshifts using sklearn's BallTree

We implement the clustering redshift method of Menard++13 (see also Newman+08,McQuinn&White+2013, among others) using sklearn's BallTree class. This code can quickly compute angular clustering, b*dN/dz, and their errors, for ~1M photometric and ~100k spectroscopic objects.

Requires numpy, healpy, astropy, scikit-learn.

The code separates pair-counting and error estimation & redshift binning, allowing the user to efficiently test different redshift bin widths and error-estimation schemes.

Pair counting is done in cross_correlate.py, which starts by generating a BallTree of the photometric catalog or photometric randoms. Optionally one can load a pre-existing tree, which allows you to run different deltaz slices at once. See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html for details on BallTree. I use the default leaf_size as I find tree-generation is <20 minutes for my largest catalogs, but one could imagine optimizing by playing with leaf_size. I use the 'haversine' metric to compute curved-sky distances between points.

After generating (or loading) the tree, I then loop over redshift slices of deltaz=0.01, chosen to be as small or smaller than you would want to use on data. For each redshift slice I return the pair counts around each quasar and the index of that quasar in the spectroscopic catalog.

Generating and querying the tree are the most computationally expensive steps of the process. I have not set up cross_correlate.py with MPI, but instead leave it up to the user to choose a zmin and zmax and potentially run several chunks in parallel.

error_estimator.py computes the cross-correlation and is capable of several methods of error estimation.  It supports both jackknife 
and bootstrap error estimation, and always computes Poisson error bars by default.  If the user wants bootstrap error bars, error_estimator.py computes the bootstrap three different ways: the "literal" bootstrap in which certain regions are fully masked and others are counted twice (thus a pair with both objects in a double counted region will be quadruple counted); a "marked" bootstrap (Loh &Stein 2004, Loh 2008) where only the positions of the spectroscopic objects are resampled; and a "sqrt" bootstrap in which I take the square root of the weights from the literal bootstrap.  For the jackknife error bars I compute both leave-one-out and leave-two-out jackknifes.  I also include a method to equalize the areas of the jackknife regions and weight the jackknife covariance estimate by the  (slightly non-equal) areas of the resulting regions (e.g. Myers et al. 2004).

In bootstrap.py I bootstrap resample and combine redshift bins. I begin by tying the index of each quasar back to the spectroscopic catalog, and then summing paircounts within each healpix in each narrow deltaz=0.01 slice.  I bootstrap by healpixels rather than individual objects to ameliorate the issues raised in Fisher,Davis,Strauss et al. 1994.  Good reference for nside of bootstrap? Default=64.  The bootstrap code then combines healpixels within a user-selected redshift slice, resamples, and outputs the resulting error bars as well as the Poisson error bars. Implement analytic formulae in Mo, Jing, Borner+1992?

Finally, the b*dN/dz distribution is computed in get_b_dndz.py.

For example, to run on the unwise "blue" sample and Northern cap quasars, begin by running:
python cross_correlate.py --no-loadtree --zmin 0.00 --zmax 0.50 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits

Then run cross_correlate_blue.sh using slurm (note the really dumb way of parallelizing the cross-correlation...I haven't done any sort of optimization of the number of z-bins in each srun or how many bins are appropriate in that first run...)

Then:
python error_estimator.py --jackknife --equalize --nside 4 --dz 0.05 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits xcorr_out/unwise_DR14_QSO/ak/N/nostar_blue/ plots/unwise_DR14_QSO/ak/N/nostar_blue/

Then:
python get_b_dndz.py xcorr_out/unwise_DR14_QSO/ak/N nostar_blue bdndz/unwise_DR14_QSO/ak/N boot 0.0 4.0 0.05
for bootstrap errors, bins of deltz=0.05 or substitute 'poisson-diag' for 'boot' to get poisson errors
