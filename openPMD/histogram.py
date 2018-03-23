"""This module reimplements the ``EnergyHistogram`` function from ``Auswertung``."""

import os
import h5py
import scipy.constants as const
import numpy as np
import logging
import numexpr as ne


def chunk_range(x1, x2, size):
    while x1 <= x2:
        nxt = x1 + size
        yield x1, min(nxt, x2)
        x1 = nxt


#@profile
def energy_histogram(timestep=50000, root='.', prefix='h5_all', slice_size=2**20,
                     part_species='e', n_bins=1024, bin_min=20, bin_max=150, mask='all'):

    # build full path to file
    h5_fname = os.path.join(root, 'h5/{}_{}.h5'.format(prefix, timestep))

    logging.info('opening {} for reading'.format(h5_fname))
    h5_file = h5py.File(h5_fname, 'r')
    logging.info('done')

    spc_path= '/data/{}/particles/{}'.format(timestep, part_species)
    species = h5_file[spc_path]
    weighting = species['weighting']
    momentum = species['momentum']
    n_datapoints = weighting.size

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

    ratio = momentum['x'].attrs['unitSI'] / const.speed_of_light / const.electron_mass

    a, b = 0, 0

    for st, end in chunk_range(0, n_datapoints, slice_size):
        logging.info('processing chunk from {} to {} of {}'.format(st, end, n_datapoints))

        logging.info('slicing datasets')
        px, py, pz = momentum['x'][st:end], momentum['y'][st:end], momentum['z'][st:end]
        w = weighting[st:end]
        m = mask[st:end]
        logging.info('done')

        a, b = b, b + np.sum(m)
        logging.info('a = {}, b = {}'.format(a, b))

        usq = ne.evaluate("(px**2 + py**2 + pz**2) / w**2")
        gamma = np.sqrt(1 + usq * ratio**2)

        energy[a:b] = (gamma[m] - 1) * 0.511
        weights[a:b] = w[m]

    h5_file.close()

    hist, _ = np.histogram(energy, bins=n_bins, range=(bin_min, bin_max), weights=weights)
    return hist


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='debug_histogram.log')
    energy_histogram(mask='random', timestep=50000)
