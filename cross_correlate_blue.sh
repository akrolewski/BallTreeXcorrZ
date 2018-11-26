#!/bin/bash -l                                                                                                                                                                                     
#SBATCH -p debug                                                                                                                                                                                   
#SBATCH --account=m3058                                                                                                                                                                            
#SBATCH -N 20                                                                                                                                                                                     
#SBATCH -t 00:30:00                                                                                                                                                                                
#SBATCH --mail-type=BEGIN,END,FAIL                                                                                                                                                                 
#SBATCH --mail-user=alex@krolewski.com                                                                                                                                                             
#SBATCH -C haswell                                                                                                                                                                                 

export PATH=/global/homes/a/akrolew/miniconda2/bin:$PATH

srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 0.50 --zmax 0.60 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 0.60 --zmax 0.70 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 0.70 --zmax 0.80 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 0.80 --zmax 0.90 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 0.90 --zmax 1.00 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.00 --zmax 1.10 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.10 --zmax 1.20 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.20 --zmax 1.30 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.30 --zmax 1.40 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.40 --zmax 1.50 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.50 --zmax 1.60 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.60 --zmax 1.70 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.70 --zmax 1.80 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.80 --zmax 1.90 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 1.90 --zmax 2.00 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 2.00 --zmax 2.10 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 2.10 --zmax 2.20 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 2.20 --zmax 2.30 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 2.30 --zmax 2.40 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
srun -N 1 -n 1 -c 48 python cross_correlate.py --loadtree --zmin 2.40 --zmax 4.00 --dz 0.01 unwise/sdsssampall_nostar_blue_masked.fits unwise/sdsssampall_nostar_blue_randoms_masked.fits eboss/data_DR14_QSO_N_masked.fits &
wait