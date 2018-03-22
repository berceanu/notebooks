import histogram as h
import cProfile


if __name__ == '__main__':
    prof = cProfile.Profile()
    prof.runcall(h.energy_histogram, logfile='log_histogram.log')
    prof.dump_stats('stats.prof')
