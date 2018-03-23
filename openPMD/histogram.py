"""This module reimplements the ``EnergyHistogram`` function from ``Auswertung``."""

import os, h5py
import scipy.constants as const
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

#import dask.array as da
#x = da.from_array(dset, chunks=(1000, 1000))

#import pdb
#pdb.set_trace()

#TODO: move to variouz.py
#def chunk_range(x1, x2, numb):
#    size = ((x2 - x1 + 1) // numb) + 1
#    while x1 <= x2:
#        next = x1 + size
#        yield x1, min(next - 1, x2)
#        x1 = next

#def chunk_range(x1, x2, size):
#    while x1 <= x2:
#        next = x1 + size
#        yield x1, min(next - 1, x2)
#        x1 = next

def chunk_range(x1, x2, size):
    while x1 <= x2:
        next = x1 + size
        yield x1, min(next, x2)
        x1 = next

def energy_histogram(timestep=50000, root='.', prefix='h5_all', slice_size=2**20,
                     species='e', n_bins=1024, bin_min=20, bin_max=150, mask='all',
                     logfile=None):

    if logfile: #turn on logging to file
        logging.basicConfig(filename=logfile)


    # build full path to file
    filename = os.path.join(root, 'h5/{}_{}.h5'.format(prefix, timestep))

    logging.info('opening {} for reading'.format(filename))
    f = h5py.File(filename, 'r')
    logging.info('done')


    h5_key = '/data/{}/particles/{}'.format(timestep, species)
    particle_group = f[h5_key]
    weights_dset = particle_group['weighting']
    px_dset = particle_group['momentum/x']
    py_dset = particle_group['momentum/y']
    pz_dset = particle_group['momentum/z']
    N_all = weights_dset.size


    if mask == 'random':
        # seed the random number generator for reproducibility
        np.random.seed(42)
        mask = np.random.choice([True, False], N_all)
    elif mask == 'all':
        mask = np.full(N_all, True)
    else:
        raise ValueError('mask must be either \'all\' or \'random\'.')


    n_trues = np.sum(mask) # nr. of True values
    E = np.zeros(n_trues, dtype=np.float32)
    weights = np.zeros(n_trues, dtype=np.float32)


    convert = px_dset.attrs['unitSI'] / const.speed_of_light / const.electron_mass

    a, b = 0, 0

    for st, end in chunk_range(0, N_all, slice_size):
        logfile.info('processing chuck from {} to {}'.format(st, end))

        px, py, pz = px_dset[st:end], py_dset[st:end], pz_dset[st:end]
        w = weights_dset[st:end]
        m = mask[st:end]

        a, b = b, b + np.sum(m)

        logfile.info('a = {}, b = {}'.format(a, b))

        weights[a:b] = w[m]
        norm = weights[a:b]

        pxsq = px[m]**2
        pysq = py[m]**2
        pzsq = pz[m]**2

        usq = (pxsq + pysq + pzsq) * convert**2 / norm**2
        gamma = np.sqrt(1 + usq)
        E[a:b] = (gamma - 1) * 0.511

    f.close()

    hist, _ = np.histogram(E, bins=n_bins, range=(bin_min, bin_max), weights=weights)
    return hist



if __name__ == '__main__':
    energy_histogram(mask='random', timestep=5000)
