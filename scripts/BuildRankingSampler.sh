#!/bin/bash

TMPFILE=`mktemp --tmpdir=. "$(basename $0).XXXXXXXXXX"`

cat << 'END' > ${TMPFILE}
import numpy as np

from distutils.core import setup
from Cython.Build import cythonize

# python setup.py build_ext --inplace
setup(ext_modules=cythonize("RankingSampler.pyx"), include_dirs=np.get_include())
END

python ${TMPFILE} build_ext --inplace

rm -rf build/ RankingSampler.c ${TMPFILE}