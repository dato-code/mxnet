#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
WORKSPACE=${SCRIPT_DIR}/..

cd ${WORKSPACE}
source ${WORKSPACE}/scripts/python_env.sh

if [ -d "/usr/local/cuda-7.5" ]; then
	./configure --cleanup_if_invalid --yes --cuda_path=/usr/local/cuda-7.5
	HAS_GPU=1
else
	./configure --cleanup_if_invalid --yes
fi

make clean_all
make -j4

if [[ $OSTYPE != msys ]]; then
  nosecmd="${PYTHON_EXECUTABLE} ${NOSETEST_EXECUTABLE}"
else
  nosecmd="${NOSETEST_EXECUTABLE}"
fi

if [ ! -z "$HAS_GPU" ]; then
  ${nosecmd} -v --with-id ${WORKSPACE}/tests/python/unittest ${WORKSPACE}/tests/python/train ${WORKSPACE}/tests/python/gpu --with-xunit --xunit-file=alltests.nosetests.xml
else
  ${nosecmd} -v --with-id ${WORKSPACE}/tests/python/unittest ${WORKSPACE}/tests/python/train --with-xunit --xunit-file=alltests.nosetests.xml
fi
