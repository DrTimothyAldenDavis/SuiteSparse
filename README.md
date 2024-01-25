-----------------------------------------------------------------------------
SuiteSparse:  A Suite of Sparse matrix packages at http://suitesparse.com
-----------------------------------------------------------------------------

Jan 20, 2024, SuiteSparse VERSION 7.6.0

SuiteSparse is a set of sparse-matrix-related packages written or co-authored
by Tim Davis, available at https://github.com/DrTimothyAldenDavis/SuiteSparse .

Primary author of SuiteSparse (codes and algorithms, excl. METIS): Tim Davis

Code co-authors, in alphabetical order (not including METIS or LAGraph):
    Patrick Amestoy, Mohsen Aznaveh, David Bateman, Jinhao Chen, Yanqing Chen,
    Iain Duff, Joe Eaton, Les Foster, William Hager, Raye Kimmerer, Scott
    Kolodziej, Chris Lourenco, Stefan Larimore, Lorena Mejia Domenzain, Erick
    Moreno-Centeno, Markus Mützel, Corey Nolel, Ekanathan Palamadai,
    Sivasankaran Rajamanickam, Sanjay Ranka, Wissam Sid-Lakhdar, and
    Nuri Yeralan.

LAGraph has been developed by the highest number of developers of any of
the packages in SuiteSparse and deserves its own list.  The list also
appears in LAGraph/Contibutors.txt:

    Janos B. Antal,    Budapest University of Technology and Economics, Hungary
    Mohsen Aznaveh,    Texas A&M University
    David A. Bader     New Jersey Institute of Technology
    Aydin Buluc,       Lawrence Berkeley National Lab
    Jinhao Chen,       Texas A&M University
    Tim Davis,         Texas A&M University
    Florentin Dorre,   Technische Univeritat Dresden, Neo4j
    Marton Elekes,     Budapest University of Technology and Economics, Hungary
    Balint Hegyi,      Budapest University of Technology and Economics, Hungary
    Tanner Hoke,       Texas A&M University
    James Kitchen,     Anaconda
    Scott Kolodziej,   Texas A&M University
    Pranav Konduri,    Texas A&M University
    Roi Lipman,        Redis Labs (now FalkorDB)
    Tze Meng Low,      Carnegie Mellon University
    Tim Mattson,       Intel
    Scott McMillan,    Carnegie Mellon University
    Markus Muetzel
    Michel Pelletier,  Graphegon
    Gabor Szarnyas,    CWI Amsterdam, The Netherlands
    Erik Welch,        Anaconda, NVIDIA
    Carl Yang,         University of California at Davis, Waymo
    Yongzhe Zhang,     SOKENDAI, Japan

METIS is authored by George Karypis.

Additional algorithm designers: Esmond Ng and John Gilbert.

Refer to each package for license, copyright, and author information.

-----------------------------------------------------------------------------
Documentation
-----------------------------------------------------------------------------

Refer to each package for the documentation on each package, typically in the
Doc subfolder.

-----------------------------------------------------------------------------
SuiteSparse branches
-----------------------------------------------------------------------------

* dev: the default branch, with recent updates of features to appear in
    the next stable release.  The intent is to keep this branch in
    fully working order at all times, but the features will not be
    finalized at any given time.
* stable: the most recent stable release.
* dev2: working branch.  All submitted PRs should made to this branch.
    This branch might not always be in working order.

-----------------------------------------------------------------------------
SuiteSparse Packages
-----------------------------------------------------------------------------

Packages in SuiteSparse, and files in this directory:

* `AMD`

  approximate minimum degree ordering.  This is the built-in AMD function in
  MATLAB.

  authors: Tim Davis, Patrick Amestoy, Iain Duff

* `bin`

  where programs are placed when compiled, for `make local`

* `BTF`

  permutation to block triangular form

  authors: Tim Davis, Ekanathan Palamadai

* `build`

  folder for default build tree

* `CAMD`

  constrained approximate minimum degree ordering

  authors: Tim Davis, Patrick Amestoy, Iain Duff, Yanqing Chen

* `CCOLAMD`

  constrained column approximate minimum degree ordering

  authors: Tim Davis, Sivasankaran Rajamanickam, Stefan Larimore.

  Algorithm design collaborators: Esmond Ng, John Gilbert (for COLAMD)

* `ChangeLog`

  a summary of changes to SuiteSparse.  See `*/Doc/ChangeLog` for details for
  each package.

* `CHOLMOD`

  sparse Cholesky factorization.  Requires AMD, COLAMD, CCOLAMD, the BLAS, and
  LAPACK.  Optionally uses METIS.  This is `chol` and `x=A\b` in MATLAB.

  author for all modules: Tim Davis

  CHOLMOD/Modify module authors: Tim Davis and William W. Hager

  CHOLMOD/SuiteSparse_metis: a modified version of METIS, embedded into the
  CHOLMOD library.  See the README.txt files for details.  author: George
  Karypis.  This is a slightly modified copy included with SuiteSparse via the
  open-source license provided by George Karypis.  SuiteSparse cannot use an
  unmodified copy of METIS.

* `CITATION.bib`

  citations for SuiteSparse packages, in bibtex format.

* `CMakeLists.txt`

  optional, to compile all of SuiteSparse.  See below.

* `CODE_OF_CONDUCT.md`

  community guidelines

* `COLAMD`

  column approximate minimum degree ordering.  This is the built-in COLAMD
  function in MATLAB.

  authors (of the code): Tim Davis and Stefan Larimore

  Algorithm design collaborators: Esmond Ng, John Gilbert

* `Contents.m`

  a list of contents for 'help SuiteSparse' in MATLAB.

* `CONTRIBUTING.md`

  how to contribute to SuiteSparse

* `CONTRIBUTOR-LICENSE.txt`

  required contributor agreement

* `CSparse`

  a concise sparse matrix package, developed for my book, "Direct Methods for
  Sparse Linear Systems", published by SIAM.  Intended primarily for teaching.
  Note that the code is (c) Tim Davis, as stated in the book.

  For production, use CXSparse instead.  In particular, both CSparse and
  CXSparse have the same include filename: `cs.h`.  This package is used for
  the built-in DMPERM in MATLAB.

  author: Tim Davis

* `CXSparse`

  CSparse Extended.  Includes support for complex matrices and both int or long
  integers.  Use this instead of CSparse for production use; it creates a
  libcsparse.so (or dylib on the Mac) with the same name as CSparse.  It is a
  superset of CSparse.  Any code that links against CSparse should also be able
  to link against CXSparse instead.

  author: Tim Davis, David Bateman

* `Example`

  a simple package that relies on almost all of SuiteSparse

* `.github`

  workflows for CI testing on GitHub.

* `GraphBLAS`

  graph algorithms in the language of linear algebra.

  https://graphblas.org

  authors: Tim Davis, Joe Eaton, Corey Nolet

* `include`

  `make install` places user-visible include files for each package here, after
  `make local`.

* `KLU`

  sparse LU factorization, primarily for circuit simulation.  Requires AMD,
  COLAMD, and BTF.  Optionally uses CHOLMOD, CAMD, CCOLAMD, and METIS.

  authors: Tim Davis, Ekanathan Palamadai

* `LAGraph`

  a graph algorithms library based on GraphBLAS.  See also
  https://github.com/GraphBLAS/LAGraph

  Authors: many.

* `LDL`

  a very concise LDL' factorization package

  author: Tim Davis

* `lib`

  `make install` places shared libraries for each package here, after
  `make local`.

* `LICENSE.txt`

  collected licenses for each package.

* `Makefile`

  optional, to compile all of SuiteSparse using `make`, which is used as a
  simple wrapper for `cmake` in each subproject.

  * `make`

    compiles SuiteSparse libraries.  Subsequent `make install` will install
    in `CMAKE_INSTALL_PATH` (might default to `/usr/local/lib` on Linux or Mac).

  * `make local`

    compiles SuiteSparse.  Subsequent `make install` will install in `./lib`,
    `./include`.  Does not install in `CMAKE_INSTALL_PATH`.

  * `make global`

    compiles SuiteSparse libraries.  Subsequent `make install` will install in
    `/usr/local/lib` (or whatever the configured `CMAKE_INSTALL_PREFIX` is).
    Does not install in `./lib` and `./include`.

  * `make install`

    installs in the current directory (`./lib`, `./include`), or in
    `/usr/local/lib` and `/usr/local/include`, (the latter defined by
    `CMAKE_INSTALL_PREFIX`) depending on whether `make`, `make local`, or
    `make global` has been done.

  * `make uninstall`

    undoes `make install`.

  * `make distclean`

    removes all files not in distribution, including `./bin`, `./share`,
    `./lib`, and `./include`.

  * `make purge`

    same as `make distclean`.

  * `make clean`

    removes all files not in distribution, but keeps compiled libraries and
    demos, `./lib`, `./share`, and `./include`.

  Each individual subproject also has each of the above `make` targets.

  Things you don't need to do:

  * `make docs`

    creates user guides from LaTeX files

  * `make cov`

    runs statement coverage tests (Linux only)

* `MATLAB_Tools`

  various m-files for use in MATLAB

  author: Tim Davis (all parts)

  for `spqr_rank`: author Les Foster and Tim Davis

  * `Contents.m`

    list of contents

  * `dimacs10`

    loads matrices for DIMACS10 collection

  * `Factorize`

    object-oriented `x=A\b` for MATLAB

  * `find_components`

    finds connected components in an image

  * `GEE`

    simple Gaussian elimination

  * `getversion.m`

    determine MATLAB version

  * `gipper.m`

    create MATLAB archive

  * `hprintf.m`

    print hyperlinks in command window

  * `LINFACTOR`

    predecessor to `Factorize` package

  * `MESHND`

    nested dissection ordering of regular meshes

  * `pagerankdemo.m`

    illustrates how PageRank works

  * `SFMULT`

    `C=S*F` where `S` is sparse and `F` is full

  * `shellgui`

    display a seashell

  * `sparseinv`

    sparse inverse subset

  * `spok`

    check if a sparse matrix is valid

  * `spqr_rank`

    SPQR_RANK package.  MATLAB toolbox for rank deficient sparse matrices: null
    spaces, reliable factorizations, etc.  With Leslie Foster, San Jose State
    Univ.

  * `SSMULT`

    `C=A*B` where `A` and `B` are both sparse.
    This was the basis for the built-in `C=A*B` in MATLAB, until it was
    superseded by GraphBLAS in MATLAB R2021a.

  * `SuiteSparseCollection`

    for the SuiteSparse Matrix Collection

  * `waitmex`

    waitbar for use inside a mexFunction

* `Mongoose`

  graph partitioning.

  authors: Nuri Yeralan, Scott Kolodziej, William Hager, Tim Davis

* `ParU`

  a parallel unsymmetric pattern multifrontal method.

  Currently a pre-release.

  authors: Mohsen Aznaveh and Tim Davis

* `RBio`

  read/write sparse matrices in Rutherford/Boeing format

  author: Tim Davis

* `README.md`

  this file

* `SPEX`

  solves sparse linear systems in exact arithmetic.

  Requires the GNU GMP and MPRF libraries.

  This will be soon replaced by a more general package, SPEX v3 that includes
  this method (exact sparse LU) and others (sparse exact Cholesky, and sparse
  exact update/downdate).  The API of v3 will be changing significantly.

  authors: Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
  Lorena Lorena Mejia Domenzain, and Tim Davis.

  See https://github.com/clouren/SPEX for the latest version.

* `SPQR`

  sparse QR factorization.  This the built-in `qr` and `x=A\b` in MATLAB.  Also
  called SuiteSparseQR.

  Includes two GPU libraries: `SPQR/GPUQREngine` and
  `SPQR/SuiteSparse_GPURuntime`.

  author of the CPU code: Tim Davis

  author of GPU modules: Tim Davis, Nuri Yeralan, Wissam Sid-Lakhdar,
  Sanjay Ranka

* `ssget`

  MATLAB interface to the SuiteSparse Matrix Collection

  author: Tim Davis

* `SuiteSparse_config`

  library with common functions and configuration for all the above packages.
  `CSparse`, `GraphBLAS`, `LAGraph`, and `MATLAB_Tools` do not use
  `SuiteSparse_config`.

  author: Tim Davis

* `SuiteSparse_demo.m`

  a demo of SuiteSparse for MATLAB

* `SuiteSparse_install.m`

  install SuiteSparse for MATLAB

* `SuiteSparse_paths.m`

  set paths for SuiteSparse MATLAB mexFunctions

* `SuiteSparse_test.m`

  exhaustive test for SuiteSparse in MATLAB

* `UMFPACK`

  sparse LU factorization.  Requires `AMD` and the `BLAS`.

  This is the built-in `lu` and `x=A\b` in MATLAB.

  author: Tim Davis

  algorithm design collaboration: Iain Duff

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis (email: davis@tamu.edu),
except for METIS (by George Karypis), `GraphBLAS/cpu_features` (by Google),
GraphBLAS/lz4, zstd, and xxHash (by Yann Collet, now at Facebook), and
GraphBLAS/CUDA/jitify.hpp (by NVIDIA).  Parts of GraphBLAS/CUDA are
Copyright (c) by NVIDIA.  Please refer to each of these licenses.

-----------------------------------------------------------------------------
For distro maintainers (Linux, homebrew, spack, R, Octave, Trilinos, ...):
-----------------------------------------------------------------------------

Thanks for packaging SuiteSparse!  Here are some suggestions:

* GraphBLAS takes a long time to compile because it creates many fast
  "FactoryKernels" at compile-time.  If you want to reduce the compile time and
  library size, enable the `GRAPHBLAS_COMPACT` mode, but keep the JIT compiler
  enabled.  Then GraphBLAS will compile the kernels it needs at run-time, via
  its JIT compiler.  Performance will be the same as the FactoryKernels once
  the JIT kernels are compiled.  User compiled kernels are placed in
  `~/.SuiteSparse`, by default.  You do not need to distribute the source for
  GraphBLAS to enable the JIT compiler: just `libgraphblas.so` and
  `GraphBLAS.h` is enough.

* GraphBLAS needs OpenMP!  It's fundamentally a parallel code so please
  distribute it with OpenMP enabled.  Performance will suffer otherwise.

* CUDA acceleration:  CHOLMOD and SPQR can benefit from their CUDA kernels.  If
  you do not have CUDA or do not want to include it in your distro, this
  version of SuiteSparse skips the building of the `CHOLMOD_CUDA` and `SPQR_CUDA`
  libraries, and does not link against the `GPUQREngine` and
  `SuiteSparse_GPURuntime` libraries.

-----------------------------------------------------------------------------
How to cite the SuiteSparse meta-package and its component packages:
-----------------------------------------------------------------------------

SuiteSparse is a meta-package of many packages, each with their own published
papers.  To cite the whole collection, use the URLs:

* https://github.com/DrTimothyAldenDavis/SuiteSparse
* http://suitesparse.com (which is a forwarding URL
        to https://people.engr.tamu.edu/davis/suitesparse.html)

Please also cite the specific papers for the packages you use.  This is a long
list; if you want a shorter list, just cite the most recent "Algorithm XXX:"
papers in ACM TOMS, for each package.

* For the MATLAB x=A\b, see below for AMD, COLAMD, CHOLMOD, UMFPACK,
    and SuiteSparseQR (SPQR).

* for GraphBLAS, and C=AB in MATLAB (sparse-times-sparse):

  T. A. Davis. Algorithm 1037: SuiteSparse:GraphBLAS: Parallel Graph Algorithms
  in the Language of Sparse Linear Algebra. ACM Trans. Math.  Softw. 49, 3,
  Article 28 (September 2023), 30 pages.  https://doi.org/10.1145/3577195

  T. Davis, Algorithm 1000: SuiteSparse:GraphBLAS: graph algorithms in the
  language of sparse linear algebra, ACM Trans on Mathematical Software, vol
  45, no 4, Dec. 2019, Article No 44.  https://doi.org/10.1145/3322125.

* for LAGraph:

  G. Szárnyas et al., "LAGraph: Linear Algebra, Network Analysis Libraries, and
  the Study of Graph Algorithms," 2021 IEEE International Parallel and
  Distributed Processing Symposium Workshops (IPDPSW), Portland, OR, USA, 2021,
  pp. 243-252. https://doi.org/10.1109/IPDPSW52791.2021.00046.

* for CSparse/CXSParse:

  T. A. Davis, Direct Methods for Sparse Linear Systems, SIAM Series on the
  Fundamentals of Algorithms, SIAM, Philadelphia, PA, 2006.
  https://doi.org/10.1137/1.9780898718881

* for SuiteSparseQR (SPQR): (also cite AMD, COLAMD):

  T. A. Davis, Algorithm 915: SuiteSparseQR: Multifrontal multithreaded
  rank-revealing sparse QR factorization, ACM Trans. on Mathematical Software,
  38(1), 2011, pp. 8:1--8:22.  https://doi.org/10.1145/2049662.2049670

* for SuiteSparseQR/GPU:

  Sencer Nuri Yeralan, T. A. Davis, Wissam M. Sid-Lakhdar, and Sanjay Ranka.
  2017. Algorithm 980: Sparse QR Factorization on the GPU.  ACM Trans. Math.
  Softw. 44, 2, Article 17 (June 2018), 29 pages.
  https://doi.org/10.1145/3065870

* for CHOLMOD: (also cite AMD, COLAMD):

  Y. Chen, T. A. Davis, W. W. Hager, and S. Rajamanickam, Algorithm 887:
  CHOLMOD, supernodal sparse Cholesky factorization and update/downdate, ACM
  Trans. on Mathematical Software, 35(3), 2008, pp. 22:1--22:14.
  https://dl.acm.org/doi/abs/10.1145/1391989.1391995

  T. A. Davis and W. W. Hager, Dynamic supernodes in sparse Cholesky
  update/downdate and triangular solves, ACM Trans. on Mathematical Software,
  35(4), 2009, pp. 27:1--27:23.  https://doi.org/10.1145/1462173.1462176

* for CHOLMOD/Modify Module: (also cite AMD, COLAMD):

  T. A. Davis and William W. Hager, Row Modifications of a Sparse Cholesky
  Factorization SIAM Journal on Matrix Analysis and Applications 2005 26:3,
  621-639.  https://doi.org/10.1137/S089547980343641X

  T. A. Davis and William W. Hager, Multiple-Rank Modifications of a Sparse
  Cholesky Factorization SIAM Journal on Matrix Analysis and Applications 2001
  22:4, 997-1013.  https://doi.org/10.1137/S0895479899357346

  T. A. Davis and William W. Hager, Modifying a Sparse Cholesky Factorization,
  SIAM Journal on Matrix Analysis and Applications 1999 20:3, 606-627.
  https://doi.org/10.1137/S0895479897321076

* for CHOLMOD/GPU Modules:

  Steven C. Rennich, Darko Stosic, Timothy A. Davis, Accelerating sparse
  Cholesky factorization on GPUs, Parallel Computing, Vol 59, 2016, pp 140-150.
  https://doi.org/10.1016/j.parco.2016.06.004

* for AMD and CAMD:

  P. Amestoy, T. A. Davis, and I. S. Duff, Algorithm 837: An approximate
  minimum degree ordering algorithm, ACM Trans. on Mathematical Software,
  30(3), 2004, pp. 381--388.
  https://dl.acm.org/doi/abs/10.1145/1024074.1024081

  P. Amestoy, T. A. Davis, and I. S. Duff, An approximate minimum degree
  ordering algorithm, SIAM J. Matrix Analysis and Applications, 17(4), 1996,
  pp. 886--905.  https://doi.org/10.1137/S0895479894278952

* for COLAMD, SYMAMD, CCOLAMD, and CSYMAMD:

  T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836:  COLAMD, an
  approximate column minimum degree ordering algorithm, ACM Trans. on
  Mathematical Software, 30(3), 2004, pp. 377--380.
  https://doi.org/10.1145/1024074.1024080

  T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, A column approximate minimum
  degree ordering algorithm, ACM Trans. on Mathematical Software, 30(3), 2004,
  pp. 353--376.  https://doi.org/10.1145/1024074.1024079

* for UMFPACK: (also cite AMD and COLAMD):

  T. A. Davis, Algorithm 832:  UMFPACK - an unsymmetric-pattern multifrontal
  method with a column pre-ordering strategy, ACM Trans. on Mathematical
  Software, 30(2), 2004, pp. 196--199.
  https://dl.acm.org/doi/abs/10.1145/992200.992206

  T. A. Davis, A column pre-ordering strategy for the unsymmetric-pattern
  multifrontal method, ACM Trans. on Mathematical Software, 30(2), 2004, pp.
  165--195.  https://dl.acm.org/doi/abs/10.1145/992200.992205

  T. A. Davis and I. S. Duff, A combined unifrontal/multifrontal method for
  unsymmetric sparse matrices, ACM Trans. on Mathematical Software, 25(1),
  1999, pp. 1--19.  https://doi.org/10.1145/305658.287640

  T. A. Davis and I. S. Duff, An unsymmetric-pattern multifrontal method for
  sparse LU factorization, SIAM J. Matrix Analysis and Computations, 18(1),
  1997, pp. 140--158.  https://doi.org/10.1137/S0895479894246905

* for the FACTORIZE m-file:

  T. A. Davis, Algorithm 930: FACTORIZE, an object-oriented linear system
  solver for MATLAB, ACM Trans. on Mathematical Software, 39(4), 2013, pp.
  28:1-28:18.  https://doi.org/10.1145/2491491.2491498

* for KLU and BTF (also cite AMD and COLAMD):

  T. A. Davis and Ekanathan Palamadai Natarajan. 2010. Algorithm 907: KLU, A
  Direct Sparse Solver for Circuit Simulation Problems. ACM Trans.  Math.
  Softw. 37, 3, Article 36 (September 2010), 17 pages.
  https://dl.acm.org/doi/abs/10.1145/1824801.1824814

* for LDL:

  T. A. Davis. Algorithm 849: A concise sparse Cholesky factorization package.
  ACM Trans. Math. Softw. 31, 4 (December 2005), 587–591.
  https://doi.org/10.1145/1114268.1114277

* for ssget and the SuiteSparse Matrix Collection:

  T. A. Davis and Yifan Hu. 2011. The University of Florida sparse matrix
  collection. ACM Trans. Math. Softw. 38, 1, Article 1 (November 2011), 25
  pages.  https://doi.org/10.1145/2049662.2049663

  Kolodziej et al., (2019). The SuiteSparse Matrix Collection Website
  Interface. Journal of Open Source Software, 4(35), 1244.
  https://doi.org/10.21105/joss.01244

* for `spqr_rank`:

  Leslie V. Foster and T. A. Davis. 2013. Algorithm 933: Reliable calculation
  of numerical rank, null space bases, pseudoinverse solutions, and basic
  solutions using suitesparseQR. ACM Trans. Math.  Softw. 40, 1, Article 7
  (September 2013), 23 pages.  https://doi.org/10.1145/2513109.2513116

* for Mongoose:

  T. A. Davis, William W. Hager, Scott P. Kolodziej, and S. Nuri Yeralan.
  2020. Algorithm 1003: Mongoose, a Graph Coarsening and Partitioning Library.
  ACM Trans. Math. Softw. 46, 1, Article 7 (March 2020), 18 pages.
  https://doi.org/10.1145/3337792

* for SPEX:

  Christopher Lourenco, Jinhao Chen, Erick Moreno-Centeno, and T. A.  Davis.
  2022. Algorithm 1021: SPEX Left LU, Exactly Solving Sparse Linear Systems via
  a Sparse Left-Looking Integer-Preserving LU Factorization. ACM Trans. Math.
  Softw. June 2022.  https://doi.org/10.1145/3519024

-----------------------------------------------------------------------------
About the BLAS and LAPACK libraries
-----------------------------------------------------------------------------

NOTE: if you use OpenBLAS, be sure to use version 0.3.27 or later.

To select your BLAS/LAPACK, see the instructions in SuiteSparseBLAS.cmake in
`SuiteSparse_config/cmake_modules`.  If `SuiteSparse_config` finds a BLAS with
64-bit integers (such as the Intel MKL ilp64 BLAS), it configures
`SuiteSparse_config.h` with the `SUITESPARSE_BLAS_INT` defined as `int64_t`.
Otherwise, if a 32-bit BLAS is found, this type is defined as `int32_t`.  If
later on, UMFPACK, CHOLMOD, or SPQR are compiled and linked  with a BLAS that
has a different integer size, you must override the definition with `-DBLAS64`
(to assert the use of 64-bit integers in the BLAS) or `-DBLAS32`, (to assert
the use of 32-bit integers in the BLAS).

The size of the BLAS integer has nothing to do with `sizeof(void *)`.

When distributed in a binary form (such as a Debian, Ubuntu, Spack, or Brew
package), SuiteSparse should probably be compiled to expect a 32-bit BLAS,
since this is the most common case.  The default is to use a 32-bit BLAS, but
this can be changed by setting the cmake variable
`SUITESPARSE_USE_64BIT_BLAS` to `ON`.

By default, SuiteSparse hunts for a suitable BLAS library.  To enforce a
particular BLAS library use either:

    CMAKE_OPTIONS="-DBLA_VENDOR=OpenBLAS" make
    cd Package ; cmake -DBLA_VENDOR=OpenBLAS .. make

To use the default (hunt for a BLAS), do not set `BLA_VENDOR`, or set it to
`ANY`.  In this case, if `SUITESPARSE_USE_64BIT_BLAS` is ON, preference is
given to a 64-bit BLAS, but a 32-bit BLAS library will be used if no 64-bit
library is found.  However, if both `SUITESPARSE_USE_64BIT_BLAS` and
`SUITESPARSE_USE_STRICT` are ON, then only a 64-bit BLAS is considered.

When selecting a particular BLAS library, the `SUITESPARSE_USE_64BIT_BLAS`
setting is strictly followed.  If set to true, only a 64-bit BLAS library will
be used.  If false (the default), only a 32-bit BLAS library will be used.  If
no such BLAS is found, the build will fail.

-----------------------------------------------------------------------------
QUICK START FOR THE C/C++ LIBRARIES:
-----------------------------------------------------------------------------

Type the following in this directory (requires system priviledge to do the
`sudo make install`):
```
    mkdir -p build && cd build
    cmake ..
    cmake --build .
    sudo cmake --install .
```

All libraries will be created and installed into the default system-wide folder
(/usr/local/lib on Linux).  All include files needed by the applications that
use SuiteSparse are installed into /usr/local/include/suitesparse (on Linux).

To build only a subset of libraries, set `SUITESPARSE_ENABLE_PROJECTS` when
configuring with CMake.  E.g., to build and install CHOLMOD and CXSparse
(including their dependencies), use the following commands:
```
    mkdir -p build && cd build
    cmake -DSUITESPARSE_ENABLE_PROJECTS="cholmod;cxsparse" ..
    cmake --build .
    sudo cmake --install .
```

For Windows (MSVC), import the `CMakeLists.txt` file into MS Visual Studio.
Be sure to specify the build type as Release; for example, to build SuiteSparse
on Windows in the command window, run:
```
    mkdir -p build && cd build
    cmake ..
    cmake --build . --config Release
    cmake --install .
```

Be sure to first install all required libraries:  BLAS and LAPACK for UMFPACK,
CHOLMOD, and SPQR, and GMP and MPFR for SPEX.  Be sure to use the latest
libraries; SPEX requires MPFR 4.0.2 and GMP 6.1.2 (these version numbers
do NOT correspond to the X.Y.Z suffix of libgmp.so.X.Y.Z and libmpfr.so.X.Y.Z;
see the SPEX user guide for details).

To compile the libraries and install them only in SuiteSparse/lib (not
/usr/local/lib), do this instead in the top-level of SuiteSparse:
```
    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=.. ..
    cmake --build .
    cmake --install .
```

If you add /home/me/SuiteSparse/lib to your library search path
(`LD_LIBRARY_PATH` in Linux), you can do the following (for example):
```
    S = /home/me/SuiteSparse
    cc myprogram.c -I$(S)/include/suitesparse -lumfpack -lamd -lcholmod -lsuitesparseconfig -lm
```

To change the C and C++ compilers, and to compile in parallel use:
```
    cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER==g++ ..
```

for example, which changes the compiler to gcc and g++.

This will work on Linux/Unix and the Mac.  It should automatically detect if
you have the Intel compilers or not, and whether or not you have CUDA.

See `SuiteSparse_config/cmake_modules/SuiteSparsePolicy.cmake` to select your BLAS.

You may also need to add SuiteSparse/lib to your path.  If your copy of
SuiteSparse is in /home/me/SuiteSparse, for example, then add this to your
`~/.bashrc` file:

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/me/SuiteSparse/lib
export LD_LIBRARY_PATH
```

For the Mac, use this instead:
```
DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/home/me/SuiteSparse/lib
export DYLD_LIBRARY_PATH
```

Default install location of files is below, where PACKAGE is one of the
packages in SuiteSparse:

    * `CMAKE_INSTALL_PREFIX/include/suitesparse/`: include files
    * `CMAKE_INSTALL_PREFIX/lib/`: compiled libraries
    * `CMAKE_INSTALL_PREFIX/lib/cmake/SuiteSparse/`: `*.cmake` scripts
        for all of SuiteSparse
    * `CMAKE_INSTALL_PREFIX/lib/cmake/PACKAGE/`: `*Config.cmake` scripts for a
        specific package
    * `CMAKE_INSTALL_PREFIX/lib/pkgconfig/PACKAGE.pc`: `.pc` scripts for
        a specific package pkgconfig

-----------------------------------------------------------------------------
QUICK START FOR MATLAB USERS (Linux or Mac):
-----------------------------------------------------------------------------

Suppose you place SuiteSparse in the `/home/me/SuiteSparse` folder.

Add the `SuiteSparse/lib` folder to your run-time library path.  On Linux, add
this to your `~/.bashrc` script, assuming `/home/me/SuiteSparse` is the
location of your copy of SuiteSparse:
```
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/me/SuiteSparse/lib
    export LD_LIBRARY_PATH
```

For the Mac, use this instead, in your `~/.zshrc` script, assuming you place
SuiteSparse in `/Users/me/SuiteSparse`:
```
    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/me/SuiteSparse/lib
    export DYLD_LIBRARY_PATH
```

Compile all of SuiteSparse with `make local`.

Next, compile the GraphBLAS MATLAB library.  In the system shell while in the
SuiteSparse folder, type `make gbmatlab` if you want to install it system-wide
with `make install`, or `make gblocal` if you want to use the library in
your own SuiteSparse/lib.

Then in the MATLAB Command Window, cd to the SuiteSparse directory and type
`SuiteSparse_install`.  All packages will be compiled, and several demos will be
run.  To run a (long!) exhaustive test, do `SuiteSparse_test`.

Save your MATLAB path for future sessions with the MATLAB pathtool or savepath
commands.  If those methods fail because you don't have system-wide permission,
add the new paths to your startup.m file, normally in
Documents/MATLAB/startup.m.  You can also use the `SuiteSparse_paths` m-file to
set all your paths at the start of each MATLAB session.

-----------------------------------------------------------------------------
Compilation options
-----------------------------------------------------------------------------

You can set specific options for CMake with the command (for example):
```
    cmake -DCHOLMOD_PARTITION=OFF -DBUILD_STATIC_LIBS=OFF -DCMAKE_BUILD_TYPE=Debug ..
```

That command will compile all of SuiteSparse except for CHOLMOD/Partition
Module (because of `-DCHOLMOD_PARTITION=OFF`).  Debug mode will be used (the
build type).  The static libraries will not be built (since
`-DBUILD_STATIC_LIBS=OFF` is set).

* `SUITESPARSE_ENABLE_PROJECTS`:

  Semicolon separated list of projects to be built or `all`.
  Default: `all` in which case the following projects are built:

  `suitesparse_config;mongoose;amd;btf;camd;ccolamd;colamd;cholmod;cxsparse;ldl;klu;umfpack;paru;rbio;spqr;spex;graphblas;lagraph`

  Additionally, `csparse` can be included in that list to build CSparse.

* `CMAKE_BUILD_TYPE`:

  Default: `Release`, use `Debug` for debugging.

* `SUITESPARSE_USE_STRICT`:

  SuiteSparse has many user-definable settings of the form `SUITESPARSE_USE_*`
  or `(package)_USE_*` for some particular package.  In general, these settings
  are not strict.  For example, if `SUITESPARSE_USE_OPENMP` is `ON` then OpenMP
  is preferred, but SuiteSparse can be used without OpenMP so no error is
  generated if OpenMP is not found.  However, if `SUITESPARSE_USE_STRICT` is
  `ON` then all `*_USE_*` settings are treated strictly and an error occurs
  if any are set to `ON` but the corresponding package or setting is not
  available.  The `*_USE_SYSTEM_*` settings are always treated as strict.
  Default: `OFF`.

* `SUITESPARSE_USE_CUDA`:

  If set to `ON`, CUDA is enabled for all of SuiteSparse.  Default: `ON`,

  CUDA on Windows with MSVC appears to be working with this release, but it
  should be considered as a prototype and may not be fully functional.  I have
  limited resources for testing CUDA on Windows.  If you encounter issues,
  disable CUDA and post this as an issue on GitHub.

* `CHOLMOD_USE_CUDA`:

  Default: `ON`.  Both `SUITESPARSE_USE_CUDA` and `CHOLMOD_USE_CUDA` must be
  enabled to use CUDA in CHOLMOD.

* `SPQR_USE_CUDA`:

  Default: `ON`.  Both `SUITESPARSE_USE_CUDA` and `SPQR_USE_CUDA` must be
  enabled to use CUDA in SPQR.

* `CMAKE_INSTALL_PREFIX`:

  Defines the install location (default on Linux is `/usr/local`).  For example,
  this command while in a folder `build` in the top level SuiteSparse folder
  will set the install directory to `/stuff`, used by the subsequent
  `sudo cmake --install .`:
```
    cmake -DCMAKE_INSTALL_PREFIX=/stuff ..
    sudo cmake --install .
```

* `SUITESPARSE_PKGFILEDIR`:

  Directory where CMake Config and pkg-config files will be installed.  By
  default, CMake Config files will be installed in the subfolder `cmake` of the
  directory where the (static) libraries will be installed (e.g., `lib`).  The
  `.pc` files for pkg-config will be installed in the subfolder `pkgconfig` of
  the directory where the (static) libraries will be installed.

  This option allows to install them at a location different from the (static)
  libraries.  This allows to install multiple configurations of the SuiteSparse
  libraries at the same time (e.g., by also setting a different
  `CMAKE_RELEASE_POSTFIX` and `CMAKE_INSTALL_LIBDIR` for each of them).  To pick
  up the respective configuration in downstream projects, set, e.g.,
  `CMAKE_PREFIX_PATH` (for CMake) or `PKG_CONFIG_PATH` (for build systems using
  pkg-config) to the path containing the respective CMake Config files or
  pkg-config files.

* `SUITESPARSE_INCLUDEDIR_POSTFIX`:

  Postfix for installation target of header from SuiteSparse. Default:
  suitesparse, so the default include directory is:
  `CMAKE_INSTALL_PREFIX/include/suitesparse`

* `BUILD_SHARED_LIBS`:

  If `ON`, shared libraries are built.
  Default: `ON`.

* `BUILD_STATIC_LIBS`:

  If `ON`, static libraries are built.
  Default: `ON`, except for GraphBLAS, which takes a long time to compile so
  the default for GraphBLAS is `OFF` unless `BUILD_SHARED_LIBS` is `OFF`.

* `SUITESPARSE_CUDA_ARCHITECTURES`:

  A string, such as `"all"` or `"35;50;75;80"` that lists the CUDA
  architectures to use when compiling CUDA kernels with `nvcc`.  The `"all"`
  option requires CMake 3.23 or later.  Default: `"52;75;80"`.

* `BLA_VENDOR`:

  A string.  Leave unset, or use `"ANY"` to select any BLAS library (the
  default).  Or set to the name of a `BLA_VENDOR` defined by FindBLAS.cmake.
  See:
  https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors

* `SUITESPARSE_USE_64BIT_BLAS`:

  If `ON`, look for a 64-bit BLAS.  If `OFF`: 32-bit only.  Default: `OFF`.

* `SUITESPARSE_USE_OPENMP`:

  If `ON`, OpenMP is used by default if it is available.  Default: `ON`.

  GraphBLAS, LAGraph, and ParU will be vastly slower if OpenMP is not used.
  CHOLMOD will be somewhat slower without OpenMP (as long as it still has a
  parallel BLAS/LAPACK).  Three packages (UMFPACK, CHOLMOD, and SPQR) rely
  heavily on parallel BLAS/LAPACK libraries and those libraries may use OpenMP
  internally.  If you wish to disable OpenMP in an entire application, select a
  single-threaded BLAS/LAPACK, or a parallel BLAS/LAPACK that does not use
  OpenMP (such as the Apple Accelerate Framework).  Using a single-threaded
  BLAS/LAPACK library will cause UMFPACK, CHOLMOD, and SPQR to be vastly
  slower.

  WARNING: GraphBLAS may not be thread-safe if built without OpenMP or pthreads
  (see the GraphBLAS User Guide for details).

* `SUITESPARSE_CONFIG_USE_OPENMP`:

  If `ON`, `SuiteSparse_config` uses OpenMP if it is available.
  Default: `SUITESPARSE_USE_OPENMP`.
  It is not essential and only used to let `SuiteSparse_time` call
  `omp_get_wtime`.

* `CHOLMOD_USE_OPENMP`:

  If `ON`, OpenMP is used in CHOLMOD if it is available.
  Default: `SUITESPARSE_USE_OPENMP`.

* `GRAPHBLAS_USE_OPENMP`:

  If `ON`, OpenMP is used in GraphBLAS if it is available.
  Default: `SUITESPARSE_USE_OPENMP`.

* `LAGRAPH_USE_OPENMP`:

  If `ON`, OpenMP is used in LAGraph if it is available.
  Default: `SUITESPARSE_USE_OPENMP`.

* `PARU_USE_OPENMP`:

  If `ON`, OpenMP is used in ParU if it is available.
  Default: `SUITESPARSE_USE_OPENMP`.

* `SUITESPARSE_DEMOS`:

  If `ON`, build the demo programs for each package.  Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_BTF`:

  If `ON`, use BTF libraries installed on the build system. If `OFF`,
  automatically build BTF as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_CHOLMOD`:

  If `ON`, use CHOLMOD libraries installed on the build system. If `OFF`,
  automatically build CHOLMOD as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_AMD`:

  If `ON`, use AMD libraries installed on the build system. If `OFF`,
  automatically build AMD as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_COLAMD`:

  If `ON`, use COLAMD libraries installed on the build system. If `OFF`,
  automatically build COLAMD as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_CAMD`:

  If `ON`, use CAMD libraries installed on the build system. If `OFF`,
  automatically build CAMD as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_CCOLAMD`:

  If `ON`, use CCOLAMD libraries installed on the build system. If `OFF`,
  automatically build CCOLAMD as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_GRAPHBLAS`:

  If `ON`, use GraphBLAS libraries installed on the build system. If `OFF`,
  automatically build GraphBLAS as dependency if needed. Default: `OFF`.

* `SUITESPARSE_USE_SYSTEM_SUITESPARSE_CONFIG`:

  If `ON`, use `SuiteSparse_config` libraries installed on the build system. If
  `OFF`, automatically build `SuiteSparse_config` as dependency if needed.
  Default: `OFF`.

* `SUITESPARSE_USE_FORTRAN`

  If `ON`, use the Fortran compiler to determine how C calls Fortan, and to
  build several optional Fortran routines. If `OFF`, use
  `SUITESPARSE_C_TO_FORTRAN` to define how C calls Fortran (see
  `SuiteSparse_config/cmake_modules/SuiteSparsePolicy.cmake` for details).
  Default: `ON`.

Additional options are available for specific packages:

* `UMFPACK_USE_CHOLMOD`:

  If `ON`, UMFPACK uses CHOLMOD for additional (optional)
  ordering options.  Default: `ON`.

* `KLU_USE_CHOLMOD`:

  If `ON`, KLU uses CHOLMOD for additional (optional)
  ordering options.  Default: `ON`.

CHOLMOD is composed of a set of Modules that can be independently selected;
all options default to `ON`:

* `CHOLMOD_GPL`

  If `OFF`, do not build any GPL-licensed module (MatrixOps, Modify, Supernodal,
  and GPU modules)

* `CHOLMOD_CHECK`

  If `OFF`, do not build the Check module.

* `CHOLMOD_MATRIXOPS`

  If `OFF`, do not build the MatrixOps module.

* `CHOLMOD_CHOLESKY`
  If `OFF`, do not build the Cholesky module. This also disables the Supernodal
  and Modify modules.

* `CHOLMOD_MODIFY`

  If `OFF`, do not build the Modify module.

* `CHOLMOD_CAMD`

  If `OFF`, do not link against CAMD and CCOLAMD. This also disables the
  Partition module.

* `CHOLMOD_PARTITION`

  If `OFF`, do not build the Partition module.

* `CHOLMOD_SUPERNODAL`

  If `OFF`, do not build the Supernodal module.

-----------------------------------------------------------------------------
Possible build/install issues
-----------------------------------------------------------------------------

One common issue can affect all packages:  getting the right #include files
that match the current libraries being built.  It's possible that your Linux
distro has an older copy of SuiteSparse headers in /usr/include or
/usr/local/include, or that Homebrew has installed its suite-sparse bundle into
/opt/homebrew/include or other places.  Old libraries can appear in in
/usr/local/lib, /usr/lib, etc.  When building a new copy of SuiteSparse, the
cmake build system is normally (or always?) able to avoid these, and use the
right header for the right version of each library.

As an additional guard against this possible error, each time one SuiteSparse
package #include's a header from another one, it checks the version number in
the header file, and reports an #error to the compiler if a stale version is
detected.  In addition, the Example package checks both the header version and
the library version (by calling a function in each library).  If the versions
mismatch in any way, the Example package reports an error at run time.

For example, CHOLMOD 5.1.0 requires AMD 3.3.0 or later.  If it detects an
older one in `amd.h`, it will report an `#error`:

```
    #include "amd.h"
    #if ( ... AMD version is stale ... )
    #error "CHOLMOD 5.1.0 requires AMD 3.3.0 or later"
    #endif
```

and the compilation will fail.  The Example package makes another check,
by calling `amd_version` and comparing it with the versions from the `amd.h`
header file.

If this error or one like it occurs, check to see if you have an old copy of
SuiteSparse, and uninstall it before compiling your new copy of SuiteSparse.

There are other many possible build/install issues that are covered by the
corresponding user guides for each package, such as finding the right BLAS,
OpenMP, and other libraries, and how to compile on the Mac when using GraphBLAS
inside MATLAB, and so on.  Refer to the User Guides for more details.

-----------------------------------------------------------------------------
Interfaces to SuiteSparse
-----------------------------------------------------------------------------

MATLAB/Octave/R/Mathematica interfaces:

  Many built-in methods in MATLAB and Octave rely on SuiteSparse, including
  `C=A*B` `x=A\b`, `L=chol(A)`, `[L,U,P,Q]=lu(A)`, `R=qr(A)`, `dmperm(A)`,
  `p=amd(A)`, `p=colamd(A)`, ...
  See also Mathematica, R, and many many more.  The list is too long.

Julia interface:

  https://github.com/JuliaSparse/SparseArrays.jl

python interface to GraphBLAS by Anaconda and NVIDIA:

  https://pypi.org/project/python-graphblas

Intel's Go interface to GraphBLAS:

  https://pkg.go.dev/github.com/intel/forGraphBLASGo

See scikit-sparse and scikit-umfpack for the Python interface via SciPy:

  https://github.com/scikit-sparse/scikit-sparse
  https://github.com/scikit-umfpack/scikit-umfpack

See russell for a Rust interface:

  https://github.com/cpmech/russell

-----------------------------------------------------------------------------
Acknowledgements
-----------------------------------------------------------------------------

Markus Mützel contributed the most recent update of the SuiteSparse build
system for all SuiteSparse packages, extensively porting it and modernizing it.

I would also like to thank François Bissey, Sebastien Villemot, Erik Welch, Jim
Kitchen, and Fabian Wein for their valuable feedback on the
SuiteSparse build system and how it works with various Linux / Python distros
and other package managers.  If you are a maintainer of a SuiteSparse packaging
for a Linux distro, conda-forge, R, spack, brew, vcpkg, etc, please feel free
to contact me if there's anything I can do to make your life easier.
I would also like to thank Raye Kimmerer for adding support for 32-bit
row/column indices in SPQR v4.2.0.

See also the various Acknowledgements within each package.

