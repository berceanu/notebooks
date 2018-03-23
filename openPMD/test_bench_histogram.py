import PlotFkt
import Auswertung
import numpy as np
import pytest
import mock
import histogram as hst


filename, species, slice_size = './', 'e', 2**20
n_bins, bin_min, bin_max = 1024, 20, 150
timestep = mock.Mock(value=50000)


def old_histogram():
    darstellungen = PlotFkt.PlotFkt(filename, species, slice_size, [])
    m = darstellungen.filtern({}, timestep.value)
    b = Auswertung.Auswertung(filename, n_bins, bin_min, bin_max, species, slice_size, m, timestep.value)
    return b.EnergyHistogram(timestep.value)


def test_old_histogram(benchmark):
    eh = benchmark(old_histogram)
    assert np.sum(eh) == pytest.approx(41479085.59, 0.1)


def test_histogram_all(benchmark):
    eh = benchmark(hst.energy_histogram, timestep=timestep.value, root=filename, species=species,
                   slice_size=slice_size, n_bins=n_bins, bin_min=bin_min, bin_max=bin_max,
                   mask='all')
    assert np.sum(eh) == pytest.approx(41479085.59, 0.1)


def test_histogram_random(benchmark):
    eh = benchmark(hst.energy_histogram, timestep=timestep.value, root=filename, species=species,
                   slice_size=slice_size, n_bins=n_bins, bin_min=bin_min, bin_max=bin_max,
                   mask='random')
    assert np.sum(eh) == pytest.approx(20947234.74, 0.1)
