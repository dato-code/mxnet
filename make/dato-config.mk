#-------------------------------------------------------------------------------
#  Template configuration for compiling mxnet
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of mxnet. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

ROOTDIR = ${PROJECT_HOME}

#---------------------
# choice of compiler
#--------------------

export CC = ${CC}
export CXX = ${CXX}
export NVCC = ${NVCC}

# whether compile with debug
DEBUG = 0
WIN32 = ${WIN32}

# the additional link flags you want to add
ADD_LDFLAGS = ${ADD_LDFLAGS}
ADD_LDFLAGS += -L${DEPS}/lib -L${DEPS}/lib64 ${SHARED_LINKER_FLAGS}

# the additional compile flags you want to add
ADD_CFLAGS = ${ADD_CFLAGS}
ADD_CFLAGS += -Iplugin/SFrameSubtree/oss_src
ADD_CFLAGS += -I${DEPS}/include

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# whether use CUDA during compile
USE_CUDA = ${USE_CUDA}

# add the path to CUDA libary to link and compile flag
# if you have already add them to enviroment variable, leave it as NONE
# USE_CUDA_PATH = /usr/local/cuda
USE_CUDA_PATH = ${USE_CUDA_PATH}

# whether use CUDNN R3 library
USE_CUDNN = 0

# whether use cuda runtime compiling for writing kernels in native language (i.e. Python)
USE_NVRTC = 0

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 0

# use openmp for parallelization
USE_OPENMP = 0
ifneq ($(USE_OPENMP), 1)
	export NO_OPENMP = 1
	ADD_CFLAGS += -DDISABLE_OPENMP
endif


# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
USE_BLAS = ${USE_BLAS}

# add path to intel libary, you may need it for MKL, if you did not add the path
# to enviroment variable
USE_INTEL_PATH = NONE

#----------------------------
# distributed computing
#----------------------------

# whether or not to enable mullti-machine supporting
USE_DIST_KVSTORE = 0

# whether or not allow to read and write HDFS directly. If yes, then hadoop is
# required
USE_HDFS = 0

# path to libjvm.so. required if USE_HDFS=1
LIBJVM=${JAVA_HOME}/jre/lib/amd64/server

# whether or not allow to read and write AWS S3 directly. If yes, then
# libcurl4-openssl-dev is required, it can be installed on Ubuntu by
# sudo apt-get install -y libcurl4-openssl-dev
USE_S3 = 0

#----------------------------
# additional operators
#----------------------------

# path to folders containing projects specific operators that you don't want to put in src/operators
EXTRA_OPERATORS =

#----------------------------
# plugins
#----------------------------
