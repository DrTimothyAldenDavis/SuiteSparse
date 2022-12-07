-----------------------------------------------------------------------------
SuiteSparse:  A Suite of Sparse matrix packages at http://suitesparse.com
-----------------------------------------------------------------------------

Nov 12, 2022.  SuiteSparse VERSION 6.0.1

SuiteSparse is a set of sparse-matrix-related packages written or co-authored
by Tim Davis, available at https://github.com/DrTimothyAldenDavis/SuiteSparse .

Primary author of SuiteSparse (codes and algorithms, excl. METIS): Tim Davis

Code co-authors, in alphabetical order (not including METIS):
    Patrick Amestoy, David Bateman, Jinhao Chen, Yanqing Chen, Iain Duff,
    Les Foster, William Hager, Scott Kolodziej, Chris Lourenco, Stefan
    Larimore, Erick Moreno-Centeno, Ekanathan Palamadai, Sivasankaran
    Rajamanickam, Sanjay Ranka, Wissam Sid-Lakhdar, Nuri Yeralan.

METIS is authored by George Karypis.

Additional algorithm designers: Esmond Ng and John Gilbert.

Refer to each package for license, copyright, and author information.

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

    * for GraphBLAS, and `C=A*B` in MATLAB (sparse-times-sparse):

        T. Davis, Algorithm 10xx: SuiteSparse:GraphBLAS: parallel graph
        algorithms in the language of sparse linear algebra, ACM Trans on
        Mathematical Software, submitted, under revision, 2022.
        In GraphBLAS/Doc v7.0.1, to appear here shortly.  See:
        https://github.com/DrTimothyAldenDavis/GraphBLAS/tree/stable/Doc

        T. Davis, Algorithm 1000: SuiteSparse:GraphBLAS: graph algorithms in
        the language of sparse linear algebra, ACM Trans on Mathematical
        Software, vol 45, no 4, Dec. 2019, Article No 44.
        https://doi.org/10.1145/3322125.

    * for CSparse/CXSParse:

        T. A. Davis, Direct Methods for Sparse Linear Systems, SIAM Series on
        the Fundamentals of Algorithms, SIAM, Philadelphia, PA, 2006.
        https://doi.org/10.1137/1.9780898718881

    * for SuiteSparseQR (SPQR): (also cite AMD, COLAMD):

        T. A. Davis, Algorithm 915: SuiteSparseQR: Multifrontal multithreaded
        rank-revealing sparse QR factorization, ACM Trans. on Mathematical
        Software, 38(1), 2011, pp. 8:1--8:22.
        https://doi.org/10.1145/2049662.2049670

    * for SuiteSparseQR/GPU:

        Sencer Nuri Yeralan, T. A. Davis, Wissam M. Sid-Lakhdar, and Sanjay
        Ranka. 2017. Algorithm 980: Sparse QR Factorization on the GPU.  ACM
        Trans. Math. Softw. 44, 2, Article 17 (June 2018), 29 pages.
        https://doi.org/10.1145/3065870

    * for CHOLMOD: (also cite AMD, COLAMD):

        Y. Chen, T. A. Davis, W. W. Hager, and S. Rajamanickam, Algorithm 887:
        CHOLMOD, supernodal sparse Cholesky factorization and update/downdate,
        ACM Trans. on Mathematical Software, 35(3), 2008, pp. 22:1--22:14.
        https://dl.acm.org/doi/abs/10.1145/1391989.1391995

        T. A. Davis and W. W. Hager, Dynamic supernodes in sparse Cholesky
        update/downdate and triangular solves, ACM Trans. on Mathematical
        Software, 35(4), 2009, pp. 27:1--27:23.
        https://doi.org/10.1145/1462173.1462176

    * for CHOLMOD/Modify Module: (also cite AMD, COLAMD):

        T. A. Davis and William W. Hager, Row Modifications of a Sparse
        Cholesky Factorization SIAM Journal on Matrix Analysis and Applications
        2005 26:3, 621-639
        https://doi.org/10.1137/S089547980343641X

        T. A. Davis and William W. Hager, Multiple-Rank Modifications of a
        Sparse Cholesky Factorization SIAM Journal on Matrix Analysis and
        Applications 2001 22:4, 997-1013
        https://doi.org/10.1137/S0895479899357346

        T. A. Davis and William W. Hager, Modifying a Sparse Cholesky
        Factorization, SIAM Journal on Matrix Analysis and Applications 1999
        20:3, 606-627
        https://doi.org/10.1137/S0895479897321076

    * for CHOLMOD/GPU Modules:

        Steven C. Rennich, Darko Stosic, Timothy A. Davis, Accelerating sparse
        Cholesky factorization on GPUs, Parallel Computing, Vol 59, 2016, pp
        140-150.
        https://doi.org/10.1016/j.parco.2016.06.004

    * for AMD and CAMD:

        P. Amestoy, T. A. Davis, and I. S. Duff, Algorithm 837: An approximate
        minimum degree ordering algorithm, ACM Trans. on Mathematical Software,
        30(3), 2004, pp. 381--388.
        https://dl.acm.org/doi/abs/10.1145/1024074.1024081

        P. Amestoy, T. A. Davis, and I. S. Duff, An approximate minimum degree
        ordering algorithm, SIAM J. Matrix Analysis and Applications, 17(4),
        1996, pp. 886--905.
        https://doi.org/10.1137/S0895479894278952

    * for COLAMD, SYMAMD, CCOLAMD, and CSYMAMD:

        T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836:  COLAMD,
        an approximate column minimum degree ordering algorithm, ACM Trans. on
        Mathematical Software, 30(3), 2004, pp. 377--380.
        https://doi.org/10.1145/1024074.1024080

        T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, A column approximate
        minimum degree ordering algorithm, ACM Trans. on Mathematical Software,
        30(3), 2004, pp. 353--376.
        https://doi.org/10.1145/1024074.1024079

    * for UMFPACK: (also cite AMD and COLAMD):

        T. A. Davis, Algorithm 832:  UMFPACK - an unsymmetric-pattern
        multifrontal method with a column pre-ordering strategy, ACM Trans. on
        Mathematical Software, 30(2), 2004, pp. 196--199.
        https://dl.acm.org/doi/abs/10.1145/992200.992206

        T. A. Davis, A column pre-ordering strategy for the unsymmetric-pattern
        multifrontal method, ACM Trans. on Mathematical Software, 30(2), 2004,
        pp. 165--195.
        https://dl.acm.org/doi/abs/10.1145/992200.992205

        T. A. Davis and I. S. Duff, A combined unifrontal/multifrontal method
        for unsymmetric sparse matrices, ACM Trans. on Mathematical Software,
        25(1), 1999, pp. 1--19.
        https://doi.org/10.1145/305658.287640

        T. A. Davis and I. S. Duff, An unsymmetric-pattern multifrontal method
        for sparse LU factorization, SIAM J. Matrix Analysis and Computations,
        18(1), 1997, pp. 140--158.
        https://doi.org/10.1137/S0895479894246905

    * for the FACTORIZE m-file:

        T. A. Davis, Algorithm 930: FACTORIZE, an object-oriented linear system
        solver for MATLAB, ACM Trans. on Mathematical Software, 39(4), 2013,
        pp. 28:1-28:18.
        https://doi.org/10.1145/2491491.2491498

    * for KLU and BTF (also cite AMD and COLAMD):

        T. A. Davis and Ekanathan Palamadai Natarajan. 2010. Algorithm 907:
        KLU, A Direct Sparse Solver for Circuit Simulation Problems. ACM Trans.
        Math. Softw. 37, 3, Article 36 (September 2010), 17 pages.
        https://dl.acm.org/doi/abs/10.1145/1824801.1824814

    * for LDL:

        T. A. Davis. Algorithm 849: A concise sparse Cholesky factorization
        package. ACM Trans. Math. Softw. 31, 4 (December 2005), 587–591.
        https://doi.org/10.1145/1114268.1114277

    * for ssget and the SuiteSparse Matrix Collection:

        T. A. Davis and Yifan Hu. 2011. The University of Florida sparse
        matrix collection. ACM Trans. Math. Softw. 38, 1, Article 1 (November
        2011), 25 pages.
        https://doi.org/10.1145/2049662.2049663

        Kolodziej et al., (2019). The SuiteSparse Matrix Collection Website
        Interface. Journal of Open Source Software, 4(35), 1244,
        https://doi.org/10.21105/joss.01244

    * for `spqr_rank`:

        Leslie V. Foster and T. A. Davis. 2013. Algorithm 933: Reliable
        calculation of numerical rank, null space bases, pseudoinverse
        solutions, and basic solutions using suitesparseQR. ACM Trans. Math.
        Softw. 40, 1, Article 7 (September 2013), 23 pages.
        https://doi.org/10.1145/2513109.2513116

    * for Mongoose:

        T. A. Davis, William W. Hager, Scott P. Kolodziej, and S. Nuri Yeralan.
        2020. Algorithm 1003: Mongoose, a Graph Coarsening and Partitioning
        Library. ACM Trans. Math. Softw. 46, 1, Article 7 (March 2020), 18
        pages.
        https://doi.org/10.1145/3337792

    * for SPEX:

        Christopher Lourenco, Jinhao Chen, Erick Moreno-Centeno, and T. A.
        Davis. 2022. Algorithm 1021: SPEX Left LU, Exactly Solving Sparse
        Linear Systems via a Sparse Left-Looking Integer-Preserving LU
        Factorization. ACM Trans. Math. Softw. June 2022.
        https://doi.org/10.1145/3519024

-----------------------------------------------------------------------------
About the BLAS and LAPACK libraries
-----------------------------------------------------------------------------

NOTE: Use of the Intel MKL BLAS is strongly recommended.  In a 2019 test,
OpenBLAS caused result in severe performance degradation.  The reason for this
is being investigated, and this may be resolved in the near future.

To select your BLAS/LAPACK, see the instructions in SuiteSparseBLAS.cmake in
`SuiteSparse_config/cmake_modules`.  If `SuiteSparse_config` finds a BLAS with
64-bit integers (such as the Intel MKL ilp64 BLAS), it configures
`SuiteSparse_config.h` with the `SUITESPARSE_BLAS_INT` defined as `int64_t`.
Otherwise, if a 32-bit BLAS is found, this type is defined as `int32_t`.  If
later on, UMFPACK, CHOLMOD, or SPQR are compiled and linked  with a BLAS that
has a different integer size, you must override the definition with -DBLAS64
(to assert the use of 64-bit integers in the BLAS) or -DBLAS32, (to assert the
use of 32-bit integers in the BLAS).

When distributed in a binary form (such as a Debian, Ubuntu, Spack, or Brew
package), SuiteSparse should probably be compiled to expect a 32-bit BLAS,
since this is the most common case.  The default is to use a 32-bit BLAS, but
this can be changed in SuiteSparseBLAS.cmake or by compiling with
`-DALLOW_64BIT_BLAS=1`.

By default, SuiteSparse hunts for a suitable BLAS library.  To enforce a
particular BLAS library use either:

    CMAKE_OPTIONS="-DBLA_VENDOR=OpenBLAS" make
    cd Package ; cmake -DBLA_VENDOR=OpenBLAS .. make

To use the default (hunt for a BLAS), do not set `BLA_VENDOR`, or set it to
ANY.  In this case, if `ALLOW_64BIT_BLAS` is set, preference is given to a
64-bit BLAS, but a 32-bit BLAS library will be used if no 64-bit library is
found.

When selecting a particular BLAS library, the `ALLOW_64BIT_BLAS` setting is
strictly followed.  If set to true, only a 64-bit BLAS library will be used.
If false (the default), only a 32-bit BLAS library will be used.  If no such
BLAS is found, the build will fail.

------------------
SuiteSparse/README
------------------

Packages in SuiteSparse, and files in this directory:

    GraphBLAS   graph algorithms in the language of linear algebra.
                https://graphblas.org
                author: Tim Davis

    SPEX        solves sparse linear systems in exact arithmetic.
                Requires the GNU GMP and MPRF libraries.
                This will be soon replaced by a more general package, SPEX v3
                that includes this method (exact sparse LU) and others (sparse
                exact Cholesky, and sparse exact update/downdate).  The API
                of v3 will be changing significantly.

    AMD         approximate minimum degree ordering.  This is the built-in AMD
                function in MATLAB.
                authors: Tim Davis, Patrick Amestoy, Iain Duff

    bin         where programs are placed when compiled

    BTF         permutation to block triangular form
                authors: Tim Davis, Ekanathan Palamadai

    CAMD        constrained approximate minimum degree ordering
                authors: Tim Davis, Patrick Amestoy, Iain Duff, Yanqing Chen

    CCOLAMD     constrained column approximate minimum degree ordering
                authors: Tim Davis, Sivasankaran Rajamanickam, Stefan Larimore.
                    Algorithm design collaborators: Esmond Ng, John Gilbert
                    (for COLAMD)

    ChangeLog   a summary of changes to SuiteSparse.  See */Doc/ChangeLog
                for details for each package.

    CHOLMOD     sparse Cholesky factorization.  Requires AMD, COLAMD, CCOLAMD,
                the BLAS, and LAPACK.  Optionally uses METIS.  This is chol and
                x=A\b in MATLAB.
                author for all modules: Tim Davis
                CHOLMOD/Modify module authors: Tim Davis and William W. Hager

    COLAMD      column approximate minimum degree ordering.  This is the
                built-in COLAMD function in MATLAB.
                authors (of the code): Tim Davis and Stefan Larimore
                Algorithm design collaborators: Esmond Ng, John Gilbert

    Contents.m  a list of contents for 'help SuiteSparse' in MATLAB.

    CSparse     a concise sparse matrix package, developed for my
                book, "Direct Methods for Sparse Linear Systems",
                published by SIAM.  Intended primarily for teaching.
                Note that the code is (c) Tim Davis, as stated in the book.
                For production, use CXSparse instead.  In particular, both
                CSparse and CXSparse have the same include filename: cs.h.
                This package is used for the built-in DMPERM in MATLAB.
                author: Tim Davis

    CXSparse    CSparse Extended.  Includes support for complex matrices
                and both int or long integers.  Use this instead of CSparse
                for production use; it creates a libcsparse.so (or *dylib on
                the Mac) with the same name as CSparse.  It is a superset
                of CSparse.  Any code that links against CSparse should
                also be able to link against CXSparse instead.
                author: Tim Davis, David Bateman

    include     'make install' places user-visible include files for each
                package here, after 'make local'

    KLU         sparse LU factorization, primarily for circuit simulation.
                Requires AMD, COLAMD, and BTF.  Optionally uses CHOLMOD,
                CAMD, CCOLAMD, and METIS.
                authors: Tim Davis, Ekanathan Palamadai

    LDL         a very concise LDL' factorization package
                author: Tim Davis

    lib         'make install' places shared libraries for each package
                here, after 'make local'

    Makefile    to compile all of SuiteSparse

                make            compiles SuiteSparse libraries.
                                Subsequent "make install" will install
                                in just /usr/local/lib.
                                Normally requires "sudo make install"

                make both       compiles SuiteSparse, and then "make install"
                                will instal in both ./lib and /usr/local/lib
                                (the latter controlled by CMAKE_INSTALL_PATH).
                                Normally requires "sudo make install"

                make local      compiles SuiteSparse.
                                Subsequent "make install will install only
                                in ./lib, ./include only.  No sudo required.
                                Does not install in /usr/local/lib.

                make global     compiles SuiteSparse libraries.
                                Subsequent "make install" will install in
                                just /usr/local/lib (or whatever your
                                CMAKE_INSTALL_PREFIX is).
                                Normally requires "sudo make install"
                                Does not install in ./lib and ./include.

                make install    installs in the current directory
                                (./lib, ./include), and/or in
                                /usr/local/lib and /usr/local/include,
                                depending on whether "make", "make local",
                                "make global", or "make both",
                                etc has been done.

                make uninstall  undoes 'make install'

                make distclean  removes all files not in distribution, including
                                ./bin, ./share, ./lib, and ./include.

                make purge      same as 'make distclean'

                make clean      removes all files not in distribution, but
                                keeps compiled libraries and demoes, ./lib,
                                ./share, and ./include.

                Each individual package also has each of the above 'make'
                targets.

                Things you don't need to do:
                make docs       creates user guides from LaTeX files
                make cov        runs statement coverage tests (Linux only)

    MATLAB_Tools    various m-files for use in MATLAB
                author: Tim Davis (all parts)
                for spqr_rank: author Les Foster and Tim Davis

                Contents.m      list of contents
                dimacs10        loads matrices for DIMACS10 collection
                Factorize       object-oriented x=A\b for MATLAB
                find_components finds connected components in an image
                GEE             simple Gaussian elimination
                getversion.m    determine MATLAB version
                gipper.m        create MATLAB archive
                hprintf.m       print hyperlinks in command window
                LINFACTOR       predecessor to Factorize package
                MESHND          nested dissection ordering of regular meshes
                pagerankdemo.m  illustrates how PageRank works
                SFMULT          C=S*F where S is sparse and F is full
                shellgui        display a seashell
                sparseinv       sparse inverse subset
                spok            check if a sparse matrix is valid
                spqr_rank       SPQR_RANK package.  MATLAB toolbox for rank
                                deficient sparse matrices: null spaces,
                                reliable factorizations, etc.  With Leslie
                                Foster, San Jose State Univ.
                SSMULT          C=A*B where A and B are both sparse
                SuiteSparseCollection    for the SuiteSparse Matrix Collection
                waitmex         waitbar for use inside a mexFunction

                The SSMULT and SFMULT functions are the basis for the
                built-in C=A*B functions in MATLAB.

    Mongoose    graph partitioning.
                authors: Nuri Yeralan, Scott Kolodziej, William Hager, Tim Davis

    CHOLMOD/SuiteSparse_metis: a modified version of METIS, embedded into
                the CHOLMOD library.  See the README.txt files
                for details.  author: George Karypis.  This is a slightly
                modified copy included with SuiteSparse via the open-source
                license provided by George Karypis.  SuiteSparse cannot use
                an unmodified copy METIS.

    RBio        read/write sparse matrices in Rutherford/Boeing format
                author: Tim Davis

    README.txt  this file

    SPQR        sparse QR factorization.  This the built-in qr and x=A\b in
                MATLAB.  Also called SuiteSparseQR.
                author of the CPU code: Tim Davis
                author of GPU modules: Tim Davis, Nuri Yeralan,
                    Wissam Sid-Lakhdar, Sanjay Ranka

    GPUQREngine: GPU support package for SPQR
                (not built into MATLAB, however)
                authors: Tim Davis, Nuri Yeralan, Sanjay Ranka,
                    Wissam Sid-Lakhdar

    SuiteSparse_config    configuration file for all the above packages.
                CSparse and MATLAB_Tools do not use SuiteSparse_config.
                author: Tim Davis

    SuiteSparse_GPURuntime      GPU support package for SPQR and CHOLMOD
                (not builtin to MATLAB, however).

    SuiteSparse_install.m       install SuiteSparse for MATLAB
    SuiteSparse_paths.m         set paths for SuiteSparse MATLAB mexFunctions

    SuiteSparse_test.m          exhaustive test for SuiteSparse in MATLAB

    ssget       MATLAB interface to the SuiteSparse Matrix Collection
                author: Tim Davis

    UMFPACK     sparse LU factorization.  Requires AMD and the BLAS.
                This is the built-in lu and x=A\b in MATLAB.
                author: Tim Davis
                algorithm design collaboration: Iain Duff

Some codes optionally use METIS 5.1.0.  This package is located in SuiteSparse
in the `CHOLMOD/SuiteSparse_metis` directory.  Its use is optional.  To compile
CHOLMOD without it, use the CMAKE_OPTIONS="-DNPARTITION=1" setting.  The use of
METIS can improve ordering quality for some matrices, particularly large 3D
discretizations.  METIS has been slightly modified for use in SuiteSparse; see
the `CHOLMOD/SuiteSparse_metis/README.txt` file for details.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis (email: davis@tamu.edu),
except for METIS, which is by George Karypis.

Licenses for each package are located in the following files, all in
PACKAGENAME/Doc/License.txt, and these files are also concatenated into
the top-level LICENSE.txt file.

METIS 5.1.0 is distributed with SuiteSparse (slightly modified for use in
SuiteSparse, with all modifications marked), and is Copyright (c) by George
Karypis.  Please refer to that package for its License.

-----------------------------------------------------------------------------
QUICK START FOR MATLAB USERS (Linux or Mac):
-----------------------------------------------------------------------------

Uncompress the SuiteSparse.zip or SuiteSparse.tar.gz archive file (they contain
the same thing).  Suppose you place SuiteSparse in the /home/me/SuiteSparse
folder.

Add the SuiteSparse/lib folder to your run-time library path.  On Linux, add
this to your ~/.bashrc script, assuming /home/me/SuiteSparse is the location of
your copy of SuiteSparse:

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/me/SuiteSparse/lib
    export LD_LIBRARY_PATH

For the Mac, use this instead, in your ~/.zshrc script, assuming you place
SuiteSparse in /Users/me/SuiteSparse:

    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/me/SuiteSparse/lib
    export DYLD_LIBRARY_PATH

Compile all of SuiteSparse with "make local".

Next, compile the GraphBLAS MATLAB library.  In the system shell while in the
SuiteSparse folder, type "make gbmatlab" if you want to install it system- wide
with "sudo make install", or "make gblocal" if you want to use the library in
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
QUICK START FOR THE C/C++ LIBRARIES:
-----------------------------------------------------------------------------

For Linux and Mac: type the following in this directory (requires system
priviledge to do the `sudo make install`):

    make
    sudo make install

All libraries will be created and copied into SuiteSparse/lib and into
/usr/local/lib.  All include files need by the applications that use
SuiteSparse are copied into SuiteSparse/include and into /usr/local/include.

For Windows, import each `*/CMakeLists.txt` file into MS Visual Studio.

Be sure to first install all required libraries:  BLAS and LAPACK for UMFPACK,
CHOLMOD, and SPQR, and GMP and MPFR for SPEX.  Be sure to use the latest
libraries; SPEX requires MPFR 4.0.2 and GMP 6.1.2 (these version numbers
do NOT correspond to the X.Y.Z suffix of libgmp.so.X.Y.Z and libmpfr.so.X.Y.Z;
see the SPEX user guide for details).

To compile the libraries and install them only in SuiteSparse/lib (not
/usr/local/lib), do this instead in the top-level of SuiteSparse:

    make local

If you add /home/me/SuiteSparse/lib to your library search path
(`LD_LIBRARY_PATH` in Linux), you can do the following (for example):

    S = /home/me/SuiteSparse
    cc myprogram.c -I$(S)/include -lumfpack -lamd -lcholmod -lsuitesparseconfig -lm

To change the C and C++ compilers, and to compile in parallel use:

    CC=gcc CX=g++ JOBS=32 make

for example, which changes the compiler to gcc and g++, and runs make with
'make -j32', in parallel with 32 jobs.

This will work on Linux/Unix and the Mac.  It should automatically detect if
you have the Intel compilers or not, and whether or not you have CUDA.

NOTE: Use of the Intel MKL BLAS is strongly recommended.  The OpenBLAS can
(rarely) result in severe performance degradation, in CHOLMOD in particular.
The reason for this is still under investigation and might already be resolved
in the current version of OpenBLAS.  See
`SuiteSparse_config/cmake_modules/SuiteSparsePolicy.cmake` to select your BLAS.

You may also need to add SuiteSparse/lib to your path.  If your copy of
SuiteSparse is in /home/me/SuiteSparse, for example, then add this to your
~/.bashrc file:

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/me/SuiteSparse/lib
    export LD_LIBRARY_PATH

For the Mac, use this instead:

    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/home/me/SuiteSparse/lib
    export DYLD_LIBRARY_PATH

-----------------------------------------------------------------------------
Python interface
-----------------------------------------------------------------------------

See scikit-sparse and scikit-umfpack for the Python interface via SciPy:

https://github.com/scikit-sparse/scikit-sparse

https://github.com/scikit-umfpack/scikit-umfpack

-----------------------------------------------------------------------------
Compilation options
-----------------------------------------------------------------------------

You can set specific options for CMake with the command (for example):

    CMAKE_OPTIONS="-DNPARTITION=1 -DNSTATIC=1 -DCMAKE_BUILD_TYPE=Debug" make

That command will compile all of SuiteSparse except for CHOLMOD/Partition
Module.  Debug mode will be used.  The static libraries will not be built
(NSTATIC is true).

    CMAKE_BUILD_TYPE:   Default: "Release", use "Debug" for debugging.

    ENABLE_CUDA:        if set to true, CUDA is enabled for the project.
                        Default: true for CHOLMOD and SPQR; false otherwise

    GLOBAL_INSTALL:     if true, "make install" will install
                        into /usr/local/lib and /usr/local/include.
                        Default: true

    LOCAL_INSTALL:      if true, "make install" will install
                        into SuiteSparse/lib and SuiteSparse/include.
                        Default: false

    NSTATIC:            if true, static libraries are not built.
                        Default: false, except for GraphBLAS, which
                        takes a long time to compile so the default for
                        GraphBLAS is true.  For Mongoose, the NSTATIC setting
                        is treated as if it always false, since the mongoose
                        program is built with the static library.

    SUITESPARSE_CUDA_ARCHITECTURES:  a string, such as "all" or
                        "35;50;75;80" that lists the CUDA architectures to use
                        when compiling CUDA kernels with nvcc.  The "all"
                        option requires cmake 3.23 or later.
                        Default: "52;75;80".

    BLA_VENDOR          a string.  Leave unset, or use "ANY" to select any BLAS
                        library (the default).  Or set to the name of a
                        BLA_VENDOR defined by FindBLAS.cmake.  See:
                        https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors

    ALLOW_64BIT_BLAS    if true: look for a 64-bit BLAS.  If false: 32-bit only.
                        Default: false.

    NOPENMP             if true: OpenMP is not used.  Default: false.
                        UMFPACK, CHOLMOD, SPQR, and GraphBLAS will be slow.
                        Note that BLAS and LAPACK may still use OpenMP
                        internally; if you wish to disable OpenMP in an entire
                        application, select a single-threaded BLAS/LAPACK.
                        WARNING: GraphBLAS may not be thread-safe if built
                        without OpenMP (see the User Guide for details).

    DEMO                if true: build the demo programs for each package.
                        Default: false.

Additional options are available within specific packages:

    NCHOLMOD            if true, UMFPACK and KLU do not use CHOLMOD for
                        additional (optional) ordering options

CHOLMOD is composed of a set of Modules that can be independently selected;
all options default to false:

    NGL                 if true: do not build any GPL-licensed module
                        (MatrixOps, Modify, Supernodal, and GPU modules)
    NCHECK              if true: do not build the Check module.
    NMATRIXOPS          if true: do not build the MatrixOps module.
    NCHOLESKY           if true: do not build the Cholesky module.
                        This also disables the Supernodal and Modify modules.
    NMODIFY             if true: do not build the Modify module.
    NCAMD               if true: do not link against CAMD and CCOLAMD.
                        This also disables the Partition module.
    NPARTITION          if true: do not build the Partition module.
    NSUPERNODAL         if true: do not build the Supernodal module.

-----------------------------------------------------------------------------
Acknowledgements
-----------------------------------------------------------------------------

I would like to thank François Bissey, Sebastien Villemot, Erik Welch, and Jim
Kitchen for their valuable feedback on the SuiteSparse build system and how it
works with various Linux / Python distros and other package managers.  If you
are a maintainer of a SuiteSparse packaging for a Linux distro, conda-forge, R,
spack, brew, vcpkg, etc, please feel free to contact me if there's anything I
can do to make your life easier.

See also the various Acknowledgements within each package.

