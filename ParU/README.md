# SuiteSparse:ParU

ParU, Copyright (c) 2022-2023, Mohsen Aznaveh and Timothy A. Davis,
All Rights Reserved.
SPDX-License-Identifier: GNU GPL 3.0

--------------------------------------------------------------------------------

## NOTE: This is a pre-release of ParU, not yet version 1.0.

There are several FIXME's and TODO's in the code we need to resolv for the
final version, and the API may change in the final stable v1.0 version.

Stay tuned.

## Introduction

ParU: is a set of routings for solving sparse linear system via parallel
multifrontal LU factorization algorithms.  Requires OpenMP 4.0+, BLAS, CHOLMOD,
UMFPACK, AMD, COLAMD, CAMD, CCOLAMD, and METIS (in particular, the
CHOLMOD/SuiteSparse_metis variant; see the CHOLMOD documentation for details).

##  How to install

You should first install all dependencies for ParU which is UMFPACK and all its 
dependencies (AMD, CHOLMOD, ...). By default ParU also needs metis. The 
configuration of ParU is mostly done via SuiteSparse config file (BLAS library,
OpenMP settings and ...) which is in SuiteSparse/SuiteSparse_config.
All SuiteSparse dependencies should be in the same directory as in ParU.
After that simply call make.

##  How to use

You should include ParU.hpp in your C++ project. Then for solving Ax=b in which
A is a sparse matrix in matrix market format with double entries and b is a
dense vector of double (or a dense matrix B for multiple rhs):

     // you can have different Controls for each
     info = ParU_Analyze(A, &Sym, &Control);
     // you can have multiple different factorization with a single ParU_Analyze
     info = ParU_Factorize(A, Sym, &Num, &Control);
     info = ParU_Solve(Sym, Num, b, x, &Control);
     ParU_Freenum(Sym, &Num, &Control);
     ParU_Freesym(&Sym, &Control);

See Demo for more examples.

--------------------------------------------------------------------------------
## License
Copyright (C) 2022-2023 Mohsen Aznaveh and Timothy A. Davis

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------
