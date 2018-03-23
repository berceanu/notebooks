import cProfile
import logging
import histogram as h


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='debug_histogram.log')
    prof = cProfile.Profile()
    prof.runcall(h.energy_histogram, timestep=5000)
    prof.dump_stats('stats.prof')
