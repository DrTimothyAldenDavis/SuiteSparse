#-------------------------------------------------------------------------------
# SuiteSparse/Example/Makefile
#-------------------------------------------------------------------------------

# Example: Copyright (c) 2023, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#-------------------------------------------------------------------------------

# A simple Makefile for the MY library in SuiteSparse/Example, which relies on
# libtool.

# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To clean up the files:
#
#       make clean

noinst_LTLIBRARIES = \
  %reldir%/libmy.la \
  %reldir%/libmy_cxx.la

%canon_reldir%_libmy_la_SOURCES = %reldir%/Source/my.c
%canon_reldir%_libmy_la_CPPFLAGS = -I../Include @SUITESPARSE_CFLAGS@
%canon_reldir%_libmy_la_LIBADD = @SUITESPARSE_LIBS@

%canon_reldir%_libmy_cxx_la_SOURCES = %reldir%/Source/my_cxx.cc
%canon_reldir%_libmy_cxx_la_CPPFLAGS = -I../Include @SUITESPARSE_CFLAGS@
%canon_reldir%_libmy_cxx_la_LIBADD = @SUITESPARSE_LIBS@

if AMCOND_HAVE_GRAPHBLAS
%canon_reldir%_libmy_la_CPPFLAGS += @GRAPHBLAS_CFLAGS@
%canon_reldir%_libmy_la_LIBADD += @GRAPHBLAS_LIBS@

%canon_reldir%_libmy_cxx_la_CPPFLAGS += @GRAPHBLAS_CFLAGS@
%canon_reldir%_libmy_cxx_la_LIBADD += @GRAPHBLAS_LIBS@
else
%canon_reldir%_libmy_la_CPPFLAGS += -DNO_GRAPHBLAS

%canon_reldir%_libmy_cxx_la_CPPFLAGS += -DNO_GRAPHBLAS
endif

if AMCOND_HAVE_LAGRAPH
%canon_reldir%_libmy_la_CPPFLAGS += @LAGRAPH_CFLAGS@
%canon_reldir%_libmy_la_LIBADD += @LAGRAPH_LIBS@

%canon_reldir%_libmy_cxx_la_CPPFLAGS += @LAGRAPH_CFLAGS@
%canon_reldir%_libmy_cxx_la_LIBADD += @LAGRAPH_LIBS@
else
%canon_reldir%_libmy_la_CPPFLAGS += -DNO_LAGRAPH

%canon_reldir%_libmy_cxx_la_CPPFLAGS += -DNO_LAGRAPH
endif

demo_PROGRAMS = \
  %reldir%/my_demo \
  %reldir%/my_cxx_demo

demodir = %canon_reldir%

%canon_reldir%_my_demo_SOURCES = %reldir%/Demo/my_demo.c
%canon_reldir%_my_demo_CPPFLAGS = -I../Include
%canon_reldir%_my_demo_LDADD = %reldir%/libmy.la

%canon_reldir%_my_cxx_demo_SOURCES = %reldir%/Demo/my_demo.cc
%canon_reldir%_my_cxx_demo_CPPFLAGS = -I../Include
%canon_reldir%_my_cxx_demo_LDADD = %reldir%/libmy_cxx.la
