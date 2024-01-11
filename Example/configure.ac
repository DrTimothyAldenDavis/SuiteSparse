# simple configure file for example project in SuiteSparse

# initialize
AC_INIT([example], [1.6.0])

AM_INIT_AUTOMAKE([foreign subdir-objects])

LT_INIT

# check for working compilers
AC_PROG_CC
AC_PROG_CXX

# find pkg-config executable
PKG_PROG_PKG_CONFIG()

if test "$enable_static" = yes; then
  PKG_CONFIG="$PKG_CONFIG --static"
fi

# find installed SuiteSparse libraries
PKG_CHECK_MODULES([SUITESPARSE], [SuiteSparse_config AMD BTF CAMD CCOLAMD CHOLMOD COLAMD CXSparse KLU KLU_CHOLMOD LDL SuiteSparse_Mongoose ParU RBio SPEX SPQR UMFPACK])

PKG_CHECK_MODULES([GRAPHBLAS], [GraphBLAS], [use_graphblas=yes], [use_graphblas=no])
AM_CONDITIONAL([AMCOND_HAVE_GRAPHBLAS], [test $use_graphblas = yes])

PKG_CHECK_MODULES([LAGRAPH], [LAGraph], [use_lagraph=yes], [use_lagraph=no])
AM_CONDITIONAL([AMCOND_HAVE_LAGRAPH], [test $use_lagraph = yes])

# Check if linker flags are needed for function fmax.
# Adds -lm to LIBS if needed.
AC_CHECK_FUNC(fmax, [], AC_CHECK_LIB(m, fmax))

# Check for C++ STL.
# If found, this will add -lstdc++ to LIBS.
# FIXME: This is a workaround to allow linking CUDA libraries to a C binary.
AC_CHECK_LIB(stdc++, __gxx_personality_v0)

# create Makefile.in from Makefile.am
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
