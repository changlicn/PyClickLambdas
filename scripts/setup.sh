#!/bin/bash

TMPFILE=`mktemp "$(pwd)/$(basename $0).XXXXXXXXXX"`

cat << 'END' > ${TMPFILE}
import numpy as np

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["./samplers/samplers_inner.pyx", "./users/users_inner.pyx", "./rankbs/bandits.pyx"]), include_dirs=np.get_include())
END

python ${TMPFILE} build_ext --inplace

rm -rf build/ samplers/samplers_inner.c users/users_inner.c rankbs/bandits.c ${TMPFILE}
