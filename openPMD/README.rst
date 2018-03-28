To profile the code, after::

    sudo apt install kcachegrind
    pip install pyprof2calltree

do::

    python prof_histogram.py
    pyprof2calltree -i stats.prof -k



To `test <https://docs.pytest.org/en/latest>`_ and `bechmark <https://pypi.python.org/pypi/pytest-benchmark>`_ run ``pytest`` in this folder, after::

    conda install pytest-benchmark mock 
