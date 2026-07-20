# simple configure file for example project in SuiteSparse

# initialize
AC_INIT([example], [1.6.0])

AM_INIT_AUTOMAKE([foreign subdir-objects])

LT_INIT

# check for working compilers
AC_PROG_CXX

# find pkg-config executable
PKG_PROG_PKG_CONFIG()

if test "$enable_static" = yes; then
  PKG_CONFIG="$PKG_CONFIG --static"
fi

# find installed SuiteSparse library
PKG_CHECK_MODULES([SUITESPARSE], [BTF])

# create Makefile.in from Makefile.am
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
