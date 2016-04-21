#!/bin/bash

##########
# Environment variables:
# CUDA_PATH: set to build with cuda at particular location
# TEST_GPU: set to run gpu unittest
# PLATFORM: platform name of the final artifact, optional
# BUILD_NUMBER: the build number of the final artifact
##########

if [[ -z "$BUILD_NUMBER" ]]; then
  BUILD_NUMBER=0.1
fi

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
WORKSPACE=${SCRIPT_DIR}/..

cd ${WORKSPACE}
source ${WORKSPACE}/scripts/python_env.sh

## Build 
function build {
echo "============= Build =============="
./configure --cleanup_if_invalid --yes
make clean_all
make -j4
echo "==================================="
}

## Build with cuda
function build_with_cuda {
echo "============= Build with CUDA =============="
echo "CUDA path: ${CUDA_PATH}"
./configure --cleanup_if_invalid --yes --cuda_path=${CUDA_PATH}
make clean_all
make -j4
echo "==================================="
}

## Test
function unittest {
echo "============= UnitTest =============="
if [[ $OSTYPE != msys ]]; then
  nosecmd="${PYTHON_EXECUTABLE} ${NOSETEST_EXECUTABLE}"
else
  nosecmd="${NOSETEST_EXECUTABLE}"
fi
${nosecmd} -v --with-id ${WORKSPACE}/tests/python/unittest ${WORKSPACE}/tests/python/train --with-xunit --xunit-file=alltests.nosetests.xml
echo "==================================="
}

## Test with cuda
function unittest_with_cuda {
echo "============= UnitTest =============="
if [[ $OSTYPE != msys ]]; then
  nosecmd="${PYTHON_EXECUTABLE} ${NOSETEST_EXECUTABLE}"
else
  nosecmd="${NOSETEST_EXECUTABLE}"
fi
${nosecmd} -v --with-id ${WORKSPACE}/tests/python/unittest ${WORKSPACE}/tests/python/gpu --with-xunit --xunit-file=alltests.nosetests.xml
echo "==================================="
}

## Copy artifacts
function copy_artifact {
echo "============= Copy artifacts =============="
TARGET_DIR=${WORKSPACE}/target/build
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p ${TARGET_DIR}
  mkdir -p ${TARGET_DIR}/python
fi

if [[ $OSTYPE == linux* ]]; then
  dll_ext='so'
elif [[ $OSTYPE == darwin* ]]; then
  dll_ext='so'
elif [[ $OSTYPE == msys ]]; then
  dll_ext='dll'
fi

if [[ -z "${LIB_NAME}" ]]; then
  LIB_NAME='libmxnet'
fi

set -x
cp -r python/mxnet ${TARGET_DIR}/python/
cp -r lib/libmxnet.${dll_ext} ${TARGET_DIR}/python/${LIB_NAME}.${dll_ext}
if [[ $OSTYPE == linux* ]]; then
  strip -s ${TARGET_DIR}/python/${LIB_NAME}.${dll_ext}
fi
set +x
echo "====================================="
}

# Package
function package {
echo "============= Package =============="
echo "Build number: $BUILD_NUMBER"

if [[ -z ${PLATFORM} ]]; then
  if [[ $OSTYPE == linux* ]]; then
    PLATFORM='linux'
  elif [[ $OSTYPE == darwin* ]]; then
    PLATFORM='mac'
  elif [[ $OSTYPE == msys ]]; then
    PLATFORM='windows'
  fi
fi

TARGET_DIR=${WORKSPACE}/target
archive_file_ext="tar.gz"
cd ${TARGET_DIR}/build/python
FNAME=${TARGET_DIR}/mxnet_${PLATFORM}_${BUILD_NUMBER}.${archive_file_ext}
tar -czvf ${FNAME} mxnet/*.py mxnet/builtin_symbols mxnet/module libmxnet*
echo "====================================="
}

function clean_target_dir {
TARGET_DIR=${WORKSPACE}/target
rm -rf ${TARGET_DIR}/build
rm -rf ${TARGET_DIR}/*.tar.gz
}

# Cleanup previous build ##
clean_target_dir

## Standard build ##
build
unittest
LIB_NAME='libmxnet'
copy_artifact

## GPU build ##
if [[ ! -z "$CUDA_PATH" ]]; then
  build_with_cuda
  if [[ ! -z "$TEST_GPU" ]]; then
    unittest_with_cuda
  fi
  LIB_NAME='libmxnet.cuda'
  copy_artifact
fi

package
