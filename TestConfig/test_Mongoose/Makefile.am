#-------------------------------------------------------------------------------
# SuiteSparse/Example/.../Makefile
#-------------------------------------------------------------------------------

# Example: Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#-------------------------------------------------------------------------------

demo_PROGRAMS = \
  %reldir%/suitesparse_demo

demodir = %canon_reldir%

%canon_reldir%_suitesparse_demo_SOURCES = %reldir%/demo.cc
%canon_reldir%_suitesparse_demo_CPPFLAGS = @SUITESPARSE_CFLAGS@
%canon_reldir%_suitesparse_demo_LDADD = @SUITESPARSE_LIBS@
