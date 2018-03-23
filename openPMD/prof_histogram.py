import histogram as h
import cProfile


if __name__ == '__main__':
    prof = cProfile.Profile()
    prof.runcall(h.energy_histogram, timestep=5000, logfile='log_histogram.log')
    prof.dump_stats('stats.prof')
