#!/bin/sh

kernprof -l histogram.py
python -m line_profiler histogram.py.lprof
