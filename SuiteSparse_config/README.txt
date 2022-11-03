SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause

--------------------------------------------------------------------------------

SuiteSparse_config contains configuration settings for all many of the software
packages that I develop or co-author.  Note that older versions of some of
these packages do not require SuiteSparse_config.

Files in SuiteSparse_config:

    CMakeLists.txt              for compiling SuiteSparse_config
    Makefile                    simple Makefile to control cmake (optional)
    README.txt                  this file
    SuiteSparse_config.c        SuiteSparse-wide utilities
    SuiteSparse_config.h        SuiteSparse-wide include file
                                (created from Config/SuiteSparse_config.h)

    build/      where SuiteSparse_config is compiled
    Config/SuiteSparse_config.h.in      source for SuiteSparse_config.h

For packages that use cmake and require SuiteSparse_config, see:

    ../SuiteSparse_config/cmake_modules/FindSuiteSparse_config.cmake

To compile/install SuiteSparse_config on Linux/MacOS, in this directory do:

    make
    sudo make install

To compile all of SuiteSparse for installation only in ../lib
and ../include instead:

    make local

Within each package, to install only in ../lib and ../include,
for example for UMFPACK:

    cd UMFPACK
    make local
    make install

To clean up:

    make clean

See the SuiteSparse/SuiteSparse_config/Makefile for more options.

--------------------------------------------------------------------------------
SuiteSparse packages:
--------------------------------------------------------------------------------

  Package  Description
  -------  -----------
  AMD      approximate minimum degree ordering
  CAMD     constrained AMD
  COLAMD   column approximate minimum degree ordering
  CCOLAMD  constrained approximate minimum degree ordering
  UMFPACK  sparse LU factorization, with the BLAS
  CXSparse int/long/real/complex version of CSparse
  CHOLMOD  sparse Cholesky factorization, update/downdate
  KLU      sparse LU factorization, BLAS-free
  BTF      permutation to block triangular form
  LDL      concise sparse LDL'
  LPDASA   LP Dual Active Set Algorithm
  RBio     read/write files in Rutherford/Boeing format
  SPQR     sparse QR factorization (full name: SuiteSparseQR)
  SPEX     sparse left-looking integer-preserving LU factorization

SuiteSparse_config is not required by these packages:

  CSparse       a Concise Sparse matrix package
  MATLAB_Tools  toolboxes for use in MATLAB
  GraphBLAS     graph algorithms in the language of linear algebra

If you edit this directory then you should do "make purge ; make" in the parent
directory to recompile all of SuiteSparse.  Otherwise, the changes will not
necessarily be applied.

