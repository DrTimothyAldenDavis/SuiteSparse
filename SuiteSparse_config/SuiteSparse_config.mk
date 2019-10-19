#===============================================================================
# SuiteSparse_config.mk:  common configuration file for the SuiteSparse
#===============================================================================

# This file contains all configuration settings for all packages authored or
# co-authored by Tim Davis:
#
# Package Version       Description
# ------- -------       -----------
# AMD     1.2 or later  approximate minimum degree ordering
# COLAMD  2.4 or later  column approximate minimum degree ordering
# CCOLAMD 1.0 or later  constrained column approximate minimum degree ordering
# CAMD    any           constrained approximate minimum degree ordering
# UMFPACK 4.5 or later  sparse LU factorization, with the BLAS
# CHOLMOD any           sparse Cholesky factorization, update/downdate
# KLU     0.8 or later  sparse LU factorization, BLAS-free
# BTF     0.8 or later  permutation to block triangular form
# LDL     1.2 or later  concise sparse LDL'
# CXSparse any          extended version of CSparse (int/long, real/complex)
# SuiteSparseQR any     sparse QR factorization
# RBio    2.0 or later  read/write sparse matrices in Rutherford-Boeing format
#
# By design, this file is NOT included in the CSparse makefile.  That package
# is fully stand-alone.  CSparse is primarily for teaching; production code
# should use CXSparse.
#
# To enable an option of the form "## OPTION = ...", edit this file and
# delete the "##" in the first column of the option you wish to use.
# The double "##" notation in this file is used to tag lines with non-default
# options that you may with to uncomment and edit.
#
# The use of METIS 4.0.1 is optional.  To exclude METIS, you must compile with
# CHOLMOD_CONFIG set to -DNPARTITION.  See below for details.  However, if you
# do not have a metis-4.0 directory inside the SuiteSparse directory, the
# */Makefile's that optionally rely on METIS will automatically detect this and
# compile without METIS.

#-------------------------------------------------------------------------------
# Generic configuration
#-------------------------------------------------------------------------------

# Using standard definitions from the make environment, typically:
#
#   CC              cc      C compiler
#   CXX             g++     C++ compiler
#   CFLAGS          [ ]     flags for C and C++ compiler
#   CPPFLAGS        [ ]     flags for C and C++ compiler
#   TARGET_ARCH     [ ]     target architecture
#   FFLAGS          [ ]     flags for Fortran compiler
#   RM              rm -f   delete a file
#   AR              ar      create a static *.a library archive
#   ARFLAGS         rv      flags for ar
#   MAKE            make    make itself (sometimes called gmake)
#
# You can redefine them by editting the following lines, but by default they
# are used from the default make environment.

##  CC              =
##  CXX             =
##  CFLAGS          =
##  CPPFLAGS        =
##  TARGET_ARCH     =
##  FFLAGS          =
##  RM              = rm -f
##  AR              = ar
##  ARFLAGS         = rv
##  MAKE            = make

#-------------------------------------------------------------------------------
# determine what system we are on
#-------------------------------------------------------------------------------

UNAME =
ifeq ($(OS),Windows_NT)
    # Windows, untested
    UNAME = Windows
else
    # Linux, SunOS, and Darwin (Mac) have been tested.  See also AIX below.
    UNAME := $(shell uname)
endif

#-------------------------------------------------------------------------------
# optimization level
#-------------------------------------------------------------------------------

# Edit this to select your level of optimization
OPTIMIZATION = -O3
# with debugging:
## OPTIMIZATION = -g

#===============================================================================
# Defaults
#===============================================================================

    # These options are overridden by the system-dependent settings below
    # (Linux, Darwin, SunOS, etc).

    #---------------------------------------------------------------------------
    # statement coverage for */Tcov
    #---------------------------------------------------------------------------

    ifneq ($(TCOV),)
        # Tcov tests require Linux and gcc, and use the vanilla BLAS
        MKLROOT =
        CC = gcc
        CXX = g++
    endif

    #---------------------------------------------------------------------------
    # CFLAGS for the C/C++ compiler
    #---------------------------------------------------------------------------

    # The CF macro is used by SuiteSparse Makefiles as a combination of
    # CFLAGS, CPPFLAGS, TARGET_ARCH, and system-dependent settings.
    # You normally should not edit the CF0 string below:
    CF0 = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) $(OPTIMIZATION) \
          -fexceptions -fPIC

    # extra flags for using the Intel MKL compiler, detected automatically
    ifeq ($(MKLROOT),)
        CF1 = $(CF0)
    else
        # use the Intel MKL for BLAS and LAPACK.
        CF1 = $(CF0) -qopenmp -I$(MKLROOT)/include -D_GNU_SOURCE
        # If OpenMP is used, it is recommended to define
        # CHOLMOD_OMP_NUM_THREADS as the number of cores per socket on the
        # machine being used to maximize memory performance.
    endif

    #---------------------------------------------------------------------------
    # code formatting (for Tcov only)
    #---------------------------------------------------------------------------

    PRETTY = grep -v "^\#" | indent -bl -nce -bli0 -i4 -sob -l120

    #---------------------------------------------------------------------------
    # required libraries
    #---------------------------------------------------------------------------

    # SuiteSparse requires the BLAS, LAPACK, and -lm (Math) libraries.
    # Linux might also requires the -lrt library.
    LIB = -lm

    # See http://www.openblas.net for a recent and freely available optimzed
    # BLAS.  LAPACK is at http://www.netlib.org/lapack/ .  You can use the
    # standard Fortran LAPACK along with OpenBLAS to obtain very good
    # performance.  This script can also detect if the Intel MKL BLAS is
    # installed.

    ifeq ($(MKLROOT),)
        # use the OpenBLAS at http://www.openblas.net
        BLAS = -lopenblas
	# use the vanilla reference BLAS.  This will be slow
        ## BLAS = -lrefblas -lgfortran
        LAPACK = -llapack
    else
        # use the Intel MKL for BLAS and LAPACK
        BLAS = -Wl,--start-group \
            $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
            $(MKLROOT)/lib/intel64/libmkl_core.a \
            $(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
            -Wl,--end-group -lpthread -lm
        LAPACK = 
    endif

    # For ACML, use this instead:
    ## BLAS = -lacml -lgfortran
    ## LAPACK =

    #---------------------------------------------------------------------------
    # shell commands
    #---------------------------------------------------------------------------

    # ranlib, and ar, for generating libraries.  If you don't need ranlib,
    # just change it to RANLAB = echo
    RANLIB = ranlib
    ARCHIVE = $(AR) $(ARFLAGS)
    CP = cp -f
    MV = mv -f

    #---------------------------------------------------------------------------
    # Fortran compiler (not required for 'make' or 'make library')
    #---------------------------------------------------------------------------

    # A Fortran compiler is optional.  Only required for the optional Fortran
    # interfaces to AMD and UMFPACK.  Not needed by 'make' or 'make install'
    F77 = gfortran
    F77FLAGS = $(FFLAGS) $(OPTIMIZATION)
    F77LIB =

    #---------------------------------------------------------------------------
    # installation location
    #---------------------------------------------------------------------------

    # For "make install" and "make uninstall"
    INSTALL_LIB = /usr/local/lib
    INSTALL_INCLUDE = /usr/local/include

    #---------------------------------------------------------------------------
    # xerbla
    #---------------------------------------------------------------------------

    # The BLAS might not contain xerbla, an error-handling routine for LAPACK
    # and the BLAS.  Also, the standard xerbla requires the Fortran I/O
    # library, and stops the application program if an error occurs.  A C
    # version of xerbla distributed with this software includes
    # Fortran-callable xerbla routine in SuiteSparse_config/xerbla that prints
    # nothing and does not stop the application program.  This is optional.

    # Assuming you can use the XERBLA in LAPACK and/or the BLAS:
    XERBLA = 

    # If you need the C version SuiteSparse_config/xerbla
    # XERBLA = ../../SuiteSparse_config/xerbla/libcerbla.a 

    # If you need the Fortran version SuiteSparse_config/xerbla
    # XERBLA = ../../SuiteSparse_config/xerbla/libxerbla.a 

    #---------------------------------------------------------------------------
    # for removing files not in the distribution
    #---------------------------------------------------------------------------

    # remove object files and profile output, but keep compiled libraries
    CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d \
        *.gcda *.gcno *.aux *.bbl *.blg *.log *.toc *.dvi *.lof *.lot

    # also remove compiled libraries
    PURGE = *.so* *.a *.dll *.dylib *.dSYM

    #---------------------------------------------------------------------------
    # NVIDIA CUDA configuration for CHOLMOD and SPQR
    #---------------------------------------------------------------------------

    CUDA_ROOT = /usr/local/cuda
    ifeq ($(wildcard $(CUDA_ROOT)),)
        # CUDA is not present
        CUDA_ROOT     =
        GPU_BLAS_PATH =
        GPU_CONFIG    =
        CUDA_PATH     =
        CUDART_LIB    =
        CUBLAS_LIB    =
        CUDA_INC_PATH =
        CUDA_INC      =
        NVCC          = echo
        NVCCFLAGS     =
    else
        # with CUDA for CHOLMOD and SPQR
        GPU_BLAS_PATH = $(CUDA_ROOT)
	# GPU_CONFIG must include -DGPU_BLAS to compile SuiteSparse for the
	# GPU.  You can add additional GPU-related flags to it as well.
        # with 4 cores (default):
        GPU_CONFIG    = -DGPU_BLAS
        # For example, to compile CHOLMOD for 10 CPU cores when using the GPU:
        ## GPU_CONFIG  = -DGPU_BLAS -DCHOLMOD_OMP_NUM_THREADS=10
        CUDA_PATH     = $(CUDA_ROOT)
        CUDART_LIB    = $(CUDA_ROOT)/lib64/libcudart.so
        CUBLAS_LIB    = $(CUDA_ROOT)/lib64/libcublas.so
        CUDA_INC_PATH = $(CUDA_ROOT)/include/
        CUDA_INC      = -I$(CUDA_INC_PATH)
        NVCC          = $(CUDA_ROOT)/bin/nvcc
        NVCCFLAGS     = -Xcompiler -fPIC -O3 \
			    -gencode=arch=compute_20,code=sm_20 \
                            -gencode=arch=compute_30,code=sm_30 \
                            -gencode=arch=compute_35,code=sm_35
    endif

    #---------------------------------------------------------------------------
    # METIS, optionally used by CHOLMOD, SPQR, and UMFPACK
    #---------------------------------------------------------------------------

    # Automatically detect if METIS is in the right place.  If so, use it.
    # Otherwise, compile SuiteSparse without it, and with -DNPARTITION.
    # The path is relative to where it is used, in CHOLMOD/Lib, CHOLMOD/MATLAB,
    # etc.  You may wish to use an absolute path.  METIS is optional.

    METIS_PATH = ../../metis-4.0
    ifeq ($(wildcard $(METIS_PATH)),)
	# METIS is not present
        METIS_PATH = 
        METIS =
        CF2 = $(CF1) -DNPARTITION
    else
	# METIS is present
        METIS = $(METIS_PATH)/libmetis.a
        CF2 = $(CF1)
    endif

    #---------------------------------------------------------------------------
    # UMFPACK configuration:
    #---------------------------------------------------------------------------

    # Configuration for UMFPACK.  See UMFPACK/Source/umf_config.h for details.
    #
    # -DNBLAS       do not use the BLAS.  UMFPACK will be very slow.
    # -D'LONGBLAS=long' or -DLONGBLAS='long long' defines the integers used by
    #               LAPACK and the BLAS (defaults to 'int')
    # -DNSUNPERF    do not use the Sun Perf. Library on Solaris
    # -DNRECIPROCAL do not multiply by the reciprocal
    # -DNO_DIVIDE_BY_ZERO   do not divide by zero
    # -DNCHOLMOD    do not use CHOLMOD as a ordering method.  If -DNCHOLMOD is
    #               included in UMFPACK_CONFIG, then UMFPACK  does not rely on
    #               CHOLMOD, CAMD, CCOLAMD, COLAMD, and METIS.

    UMFPACK_CONFIG =

    # For example, uncomment this line to compile UMFPACK without CHOLMOD:
    ## UMFPACK_CONFIG = -DNCHOLMOD

    #---------------------------------------------------------------------------
    # CHOLMOD configuration
    #---------------------------------------------------------------------------

    # CHOLMOD Library Modules, which appear in libcholmod.a:
    # Core       requires: none
    # Check      requires: Core
    # Cholesky   requires: Core, AMD, COLAMD. optional: Partition, Supernodal
    # MatrixOps  requires: Core
    # Modify     requires: Core
    # Partition  requires: Core, CCOLAMD, METIS.  optional: Cholesky
    # Supernodal requires: Core, BLAS, LAPACK
    #
    # CHOLMOD test/demo Modules (these do not appear in libcholmod.a or .so):
    # Tcov       requires: Core, Check, Cholesky, MatrixOps, Modify, Supernodal
    #            optional: Partition
    # Valgrind   same as Tcov
    # Demo       requires: Core, Check, Cholesky, MatrixOps, Supernodal
    #            optional: Partition
    #
    # Configuration flags:
    # -DNCHECK      do not include the Check module.       License GNU LGPL
    # -DNCHOLESKY   do not include the Cholesky module.    License GNU LGPL
    # -DNPARTITION  do not include the Partition module.   License GNU LGPL
    #               also do not include METIS.
    # -DNCAMD       do not use CAMD, etc from Partition module.    GNU LGPL
    # -DNGPL        do not include any GNU GPL Modules in the CHOLMOD library:
    # -DNMATRIXOPS  do not include the MatrixOps module.   License GNU GPL
    # -DNMODIFY     do not include the Modify module.      License GNU GPL
    # -DNSUPERNODAL do not include the Supernodal module.  License GNU GPL
    #
    # -DNPRINT      do not print anything.
    # -D'LONGBLAS=long' or -DLONGBLAS='long long' defines the integers used by
    #               LAPACK and the BLAS (defaults to 'int')
    # -DNSUNPERF    for Solaris only.  If defined, do not use the Sun
    #               Performance Library

    # append options to this line (leave in GPU_CONFIG; if you do not have
    # a GPU then GPU_CONFIG is already empty):
    CHOLMOD_CONFIG = $(GPU_CONFIG)

    # For example, to compile CHOLMOD without METIS:
    ## CHOLMOD_CONFIG = $(GPU_CONFIG) -DNPARTITION

    #---------------------------------------------------------------------------
    # SuiteSparseQR configuration:
    #---------------------------------------------------------------------------

    # The SuiteSparseQR library can be compiled with the following options:
    #
    # -DNPARTITION      do not include the CHOLMOD partition module
    # -DNEXPERT         do not include the functions in SuiteSparseQR_expert.cpp
    # -DHAVE_TBB        enable the use of Intel's Threading Building Blocks

    # append options to this line (leave in GPU_CONFIG; if you do not have
    # a GPU then GPU_CONFIG is already empty):
    SPQR_CONFIG = $(GPU_CONFIG)

    # For example, to use TBB:
    ## SPQR_CONFIG = $(GPU_CONFIG) -DHAVE_TBB

    # without TBB:
    TBB =
    # with TBB, you must uncomment this line:
    ## TBB = -ltbb

    # TODO: figure out how to auto-detect the presence of Intel's TBB

    #---------------------------------------------------------------------------
    # default C flags, for all systems
    #---------------------------------------------------------------------------

    CF = $(CF2)

#===============================================================================
# System-dependent configurations
#===============================================================================

    #---------------------------------------------------------------------------
    # Linux
    #---------------------------------------------------------------------------

    ifeq ($(UNAME), Linux)
        # use both the Math (-lm) and /l
        LIB = -lm -lrt
        # use the Intel compilers, unless compiling in */Tcov
        ifeq ($(TCOV),)
            ifneq ($(shell which icc 2>/dev/null),)
                # use the Intel icc compiler for C codes
                CC = icc
            endif
            ifneq ($(shell which icpc 2>/dev/null),)
                # use the Intel icc compiler for C++ codes
                CXX = icc
            endif
            ifneq ($(shell which ifort 2>/dev/null),)
                # use the Intel ifort compiler for Fortran codes
                F77 = ifort
            endif
        endif
    endif

    #---------------------------------------------------------------------------
    # Mac
    #---------------------------------------------------------------------------

    ifeq ($(UNAME), Darwin)
        # To compile on the Mac, you must install Xcode.  Then do this at the
        # command line in the Termal, before doing 'make':
        #       xcode-select --install
        # As recommended by macports, http://suitesparse.darwinports.com/
        # Also compile with no timers (-DNTIMER)
        ## CF = $(CF2) -fno-common -DNTIMER
        CF = $(CF2) -fno-common
        BLAS = -framework Accelerate
        LAPACK = -framework Accelerate
    endif

    #---------------------------------------------------------------------------
    # Solaris
    #---------------------------------------------------------------------------

    ifeq ($(UNAME), SunOS)
        # Using the Sun compiler and the Sun Performance Library
	# This hasn't been tested recently.
        CF = $(CF2) -fast -KPIC -xc99=%none -xlibmieee -xlibmil -m64 -Xc
        F77FLAGS = -O -fast -KPIC -dalign -xlibmil -m64
        BLAS = -xlic_lib=sunperf
        LAPACK =
        # Using the GCC compiler and the reference BLAS
        ## CC = gcc
        ## CXX = g++
        ## MAKE = gmake
        ## CF = $(CF2)
        ## BLAS = -lrefblas -lgfortran
        ## LAPACK = -llapack
    endif

    #---------------------------------------------------------------------------
    # IBM AIX
    #---------------------------------------------------------------------------

    ifeq ($(UNAME), AIX)
        # hasn't been tested for a very long time...
        CF = $(CF2) -O4 -qipa -qmaxmem=16384 -q64 -qproto -DBLAS_NO_UNDERSCORE
        F77FLAGS =  -O4 -qipa -qmaxmem=16384 -q64
        BLAS = -lessl
        LAPACK =
    endif

#===============================================================================
# Building the shared and static libraries
#===============================================================================

# How to build/install shared and static libraries for Mac and Linux/Unix.
# This assumes that LIBRARY and VERSION have already been defined by the
# Makefile that includes this file.

ifeq ($(UNAME),Windows)
    # Windows (untested)
    AR_TARGET = $(LIBRARY).dll
    SO_TARGET =
else
    # Mac or Linux/Unix
    AR_TARGET = $(LIBRARY).a
    ifeq ($(UNAME),Darwin)
        # Mac
        SO_PLAIN  = $(LIBRARY).dylib
        SO_TARGET = $(LIBRARY).$(VERSION).dylib
        SO_OPTS   = -dynamiclib  -compatibility_version $(VERSION) \
                    -current_version $(VERSION) \
                    -shared -undefined dynamic_lookup
    else
        # Linux and other variants of Unix
        SO_PLAIN  = $(LIBRARY).so
        SO_TARGET = $(LIBRARY).so.$(VERSION)
        SO_OPTS   = -shared -Wl,-soname -Wl,$(SO_TARGET)
    endif
endif

