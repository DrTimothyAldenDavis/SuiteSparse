# SuiteSparse:ParU

ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
All Rights Reserved.
SPDX-License-Identifier: GNU GPL 3.0

--------------------------------------------------------------------------------

## Introduction

ParU: is a set of routings for solving sparse linear system via parallel
multifrontal LU factorization algorithms.  Requires OpenMP 4.0+, BLAS, CHOLMOD,
UMFPACK, AMD, COLAMD, CAMD, CCOLAMD, and METIS.

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
Copyright (C) 2022 Mohsen Aznaveh and Timothy A. Davis

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
