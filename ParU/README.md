# SuiteSparse:ParU

ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
All Rights Reserved.
SPDX-License-Identifier: GPL-3.0-or-later

--------------------------------------------------------------------------------

## Introduction

ParU is a set of routines for solving sparse linear system via parallel
multifrontal LU factorization algorithms.  Requires OpenMP 4.5+, BLAS, CHOLMOD,
UMFPACK, AMD, COLAMD, CAMD, CCOLAMD, and METIS (in particular, the
`CHOLMOD/SuiteSparse_metis` variant; see the CHOLMOD documentation for
details).

##  How to install

See the SuiteSparse/README.md for instructions on building all of SuiteSparse
via the SuiteSparse/CMakeLists.txt file.  You may also build each individual
package that ParU depends on (`SuiteSparse_config`, AMD, COLAMD, CCAMD,
CCOLAMD, CHOLMOD, and UMFPACK).  Then simply do:

```
    cd ParU/build
    cmake ..
    cmake --build . --config Release
    sudo cmake --install .
```

Alternatively, on Linux, Mac, or MINGW, simply type `make` in the ParU
folder, then `sudo make install`.

##  How to use

You should include ParU.h in your C++ or C project. Then for solving Ax=b in
which A is a CHOLMOD real sparse double-precision matrix and b is a dense
double vector:

     ParU_Analyze (A, &Sym, &Control) ;
     ParU_Factorize (A, Sym, &Num, &Control) ;
     ParU_Solve (Sym, Num, b, x, &Control) ;
     ParU_FreeNumeric (Sym, &Num, &Control) ;
     ParU_FreeSymbolic (&Sym, &Control) ;

ParU_Analyze only considers the sparsity pattern of A, not its values, so the
Sym object can be reused for matrices with the same pattern but different
values.

See Demo for more examples.  See the ParU/ParU folder for a MATLAB interface.

--------------------------------------------------------------------------------
## License
Copyright (C) 2022-2024 Mohsen Aznaveh and Timothy A. Davis

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
