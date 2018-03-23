"""This module reimplements the ``EnergyHistogram`` function from ``Auswertung``."""

import os
import h5py
import scipy.constants as const
import numpy as np
import logging


def chunk_range(x1, x2, size):
    while x1 <= x2:
        nxt = x1 + size
        yield x1, min(nxt, x2)
        x1 = nxt


def energy_histogram(timestep=50000, root='.', prefix='h5_all', slice_size=2**20,
                     species='e', n_bins=1024, bin_min=20, bin_max=150, mask='all'):

    # build full path to file
    filename = os.path.join(root, 'h5/{}_{}.h5'.format(prefix, timestep))

    logging.info('opening {} for reading'.format(filename))
    f = h5py.File(filename, 'r')
    logging.info('done')

    h5_key = '/data/{}/particles/{}'.format(timestep, species)
    particle_group = f[h5_key]
    weights_dset = particle_group['weighting']
    mom_dset = particle_group['momentum']
    n_datapoints = weights_dset.size

    if mask == 'random':
        # seed the random number generator for reproducibility
        np.random.seed(42)
        mask = np.random.choice([True, False], n_datapoints)
    elif mask == 'all':
        mask = np.full(n_datapoints, True)
    else:
        raise ValueError('mask must be either \'all\' or \'random\'.')

    n_trues = np.sum(mask)  # nr. of True values
    energy = np.zeros(n_trues, dtype=np.float32)
    weights = np.zeros(n_trues, dtype=np.float32)

    ratio = mom_dset['x'].attrs['unitSI'] / const.speed_of_light / const.electron_mass

    a, b = 0, 0

    for st, end in chunk_range(0, n_datapoints, slice_size):
        logging.info('processing chunk from {} to {} of {}'.format(st, end, n_datapoints))

        logging.info('slicing datasets')
        px, py, pz = mom_dset['x'][st:end], mom_dset['y'][st:end], mom_dset['z'][st:end]
        w = weights_dset[st:end]
        m = mask[st:end]
        logging.info('done')

        a, b = b, b + np.sum(m)
        logging.info('a = {}, b = {}'.format(a, b))

        usq = (px[m]**2 + py[m]**2 + pz[m]**2) / w[m]**2
        gamma = np.sqrt(1 + usq * ratio**2)

        energy[a:b] = (gamma - 1) * 0.511
        weights[a:b] = w[m]

    f.close()

    hist, _ = np.histogram(energy, bins=n_bins, range=(bin_min, bin_max), weights=weights)
    return hist


if __name__ == '__main__':
    energy_histogram(mask='random', timestep=5000)
