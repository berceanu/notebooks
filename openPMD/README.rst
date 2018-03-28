To profile the code, after 
sudo apt install kcachegrind
pip install pyprof2calltree

python prof_histogram.py
pyprof2calltree -i stats.prof -k




conda install pytest-benchmark mock 

To test_ and bechmark_, run ``pytest`` in this folder.

.. _test: https://docs.pytest.org/en/latest/
.. _benchmark: https://pypi.python.org/pypi/pytest-benchmark
