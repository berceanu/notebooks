"""This module reimplements the ``EnergyHistogram`` function from ``Auswertung``."""

import os, h5py
import scipy.constants as const
import numpy as np


#import pdb
#pdb.set_trace()


def energy_histogram(timestep=50000, root='.', prefix='h5_all', slice_size=2**20,
                     species='e', n_bins=1024, bin_min=20, bin_max=150, mask='all',
                     logfile=None):

    if logfile: #turn on logging to file
        import logging
        logging.basicConfig(filename=logfile, level=logging.DEBUG)

        #logging.debug('This message should go to the log file')
        #logging.warning('And this, too')



    # build full path to file
    filename = os.path.join(root, 'h5/{}_{}.h5'.format(prefix, timestep))

    logging.info('opening file {} for reading..'.format(filename))
    f = h5py.File(filename, 'r')
    logging.info('done')

    handler  = f['/data/{}/particles/{}/'.format(timestep, species)]
    w = f['/data/{}/particles/{}/weighting'.format(timestep, species)]
    N_all = w.size


    if mask == 'random':
        # seed the random number generator for reproducibility
        np.random.seed(42)
        mask = np.random.choice([True, False], N_all)
    elif mask == 'all':
        mask = np.full(N_all, True)
    else:
        raise ValueError('mask must be either \'all\' or \'random\'.')



    x = f['/data/{}/particles/{}/position/x'.format(timestep, species)][...][mask]
    N = x.size


    E = np.zeros(N)
    weights = np.zeros(N)


    convert = (handler['momentum/x'].attrs['unitSI']) / const.speed_of_light / const.electron_mass

    # compute startsUnfiltered, startsFiltered
    N_Chunks=np.int(np.round(N_all/slice_size))
    startsUnfiltered=np.arange(0,N_Chunks+1)*slice_size
    startsFiltered=np.zeros(N_Chunks+1,dtype=np.int)
    for i in range(N_Chunks+1):
        startsUnfiltered_i=startsUnfiltered[i]
        startsFiltered[i]=np.int(np.sum(mask[0:startsUnfiltered_i]))


    # major looping going on
    for i in range(len(startsFiltered)-1):
        weights[startsFiltered[i]:startsFiltered[i+1]] = w[startsUnfiltered[i]:startsUnfiltered[i+1]][
            mask[startsUnfiltered[i]:startsUnfiltered[i + 1]]]
        pxsq = (handler['momentum/x'][startsUnfiltered[i]:startsUnfiltered[i+1]][
                    mask[startsUnfiltered[i]:startsUnfiltered[i + 1]]] / weights[startsFiltered[i]:startsFiltered[i + 1]]) ** 2
        pysq = (handler['momentum/y'][startsUnfiltered[i]:startsUnfiltered[i+1]][
                    mask[startsUnfiltered[i]:startsUnfiltered[i + 1]]] / weights[startsFiltered[i]:startsFiltered[i + 1]]) ** 2
        pzsq = (handler['momentum/z'][startsUnfiltered[i]:startsUnfiltered[i+1]][
                    mask[startsUnfiltered[i]:startsUnfiltered[i + 1]]] / weights[startsFiltered[i]:startsFiltered[i + 1]]) ** 2
        usq = (pxsq+pysq+pzsq)*(convert**2)
        gamma = np.sqrt(1+usq)
        E[startsFiltered[i]:startsFiltered[i+1]]=(gamma-1)*0.511

    f.close()
    return (np.histogram(E, bins=n_bins, range=(bin_min, bin_max), weights=weights))[0]
