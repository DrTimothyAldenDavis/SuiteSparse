-----------------------------------------------------------------------------
SuiteSparse:  A Suite of Sparse matrix packages at http://suitesparse.com
-----------------------------------------------------------------------------

Apr 10, 2022.  SuiteSparse VERSION 5.12.0

    Now includes GraphBLAS, SLIP_LU, and a new interface to the SuiteSparse
    Matrix Collection (ssget), via MATLAB and a Java GUI, to
    http://sparse.tamu.edu.

Primary author of SuiteSparse (codes and algorithms, excl. METIS): Tim Davis

Code co-authors, in alphabetical order (not including METIS):

    Patrick Amestoy, David Bateman, Jinhao Chen.  Yanqing Chen, Iain Duff,
    Les Foster, William Hager, Scott Kolodziej, Chris Lourenco, Stefan
    Larimore, Erick Moreno-Centeno, Ekanathan Palamadai, Sivasankaran
    Rajamanickam, Sanjay Ranka, Wissam Sid-Lakhdar, Nuri Yeralan.

Additional algorithm designers: Esmond Ng and John Gilbert.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis.

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
        and SuiteSparseQR.

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

    * for SuiteSparseQR: (also cite AMD, COLAMD):

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
        CHOLMOD, supernodal sparse Cholesky factorization and update/downdate, ACM
        Trans. on Mathematical Software, 35(3), 2008, pp. 22:1--22:14.
        https://dl.acm.org/doi/abs/10.1145/1391989.1391995

        T. A. Davis and W. W. Hager, Dynamic supernodes in sparse Cholesky
        update/downdate and triangular solves, ACM Trans. on Mathematical Software,
        35(4), 2009, pp. 27:1--27:23.
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

    * for `SLIP_LU` and SPEX:

        Christopher Lourenco, Jinhao Chen, Erick Moreno-Centeno, and T. A.
        Davis. 2022. Algorithm 1XXX: SPEX Left LU, Exactly Solving Sparse
        Linear Systems via a Sparse Left-Looking Integer-Preserving LU
        Factorization. ACM Trans. Math. Softw. Just Accepted (February 2022).
        https://doi.org/10.1145/3519024

-----------------------------------------------------------------------------
About the BLAS and LAPACK libraries
-----------------------------------------------------------------------------

*NOTE: Use of the Intel MKL BLAS is strongly recommended.  A recent OpenBLAS
can result in severe performance degradation.  The reason for this is being
investigated, and this may be resolved in the near future.  Ignore the comments
about OpenBLAS in the various user guides; those are out of date.*

------------------
SuiteSparse/README
------------------

Packages in SuiteSparse, and files in this directory:

    GraphBLAS   graph algorithms in the language of linear algebra.
                https://graphblas.org
                A stand-alone package that uses cmake to compile; see
                GraphBLAS/README.txt.  The rest of SuiteSparse still uses
                'make'.  A cmake setup for all of SuiteSparse is in progress.
                author: Tim Davis

    SLIP_LU     solves sparse linear systems in exact arithmetic.
                Requires the GNU GMP and MPRF libraries.

    AMD         approximate minimum degree ordering.  This is the built-in AMD
                function in MATLAB.
                authors: Tim Davis, Patrick Amestoy, Iain Duff

    bin         where the metis-5.1.0 programs are placed when METIS is compiled

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
                It does have a 'make install' but I recommend using
                CXSparse instead.  In particular, both CSparse and CXSparse
                have the same include filename: cs.h.

                This package is used for the built-in DMPERM in MATLAB.
                author: Tim Davis

    CSparse_to_CXSparse
                a Perl script to create CXSparse from CSparse and
                CXSparse_newfiles
                author: David Bateman, Motorola

    CXSparse    CSparse Extended.  Includes support for complex matrices
                and both int or long integers.  Use this instead of CSparse
                for production use; it creates a libcsparse.so (or *dylib on
                the Mac) with the same name as CSparse.  It is a superset
                of CSparse.  Any code that links against CSparse should
                also be able to link against CXSparse instead.
                author: Tim Davis, David Bateman

    CXSparse_newfiles
                Files unique to CXSparse
                author: Tim Davis, David Bateman

    share       'make' places documentation for each package here

    include     'make' places user-visible include fomes for each package here

    KLU         sparse LU factorization, primarily for circuit simulation.
                Requires AMD, COLAMD, and BTF.  Optionally uses CHOLMOD,
                CAMD, CCOLAMD, and METIS.
                authors: Tim Davis, Ekanathan Palamadai

    LDL         a very concise LDL' factorization package
                author: Tim Davis

    lib         'make' places shared libraries for each package here

    Makefile    to compile all of SuiteSparse
                make            compiles SuiteSparse libraries and runs demos
                make install    compiles SuiteSparse and installs in the
                                current directory (./lib, ./include).
                                Use "sudo make INSTALL=/usr/local" to install
                                in /usr/local/lib and /usr/local/include.
                make uninstall  undoes 'make install'
                make library    compiles SuiteSparse libraries (not demos)
                make distclean  removes all files not in distribution, including
                                ./bin, ./share, ./lib, and ./include.
                make purge      same as 'make distclean'
                make clean      removes all files not in distribution, but
                                keeps compiled libraries and demoes, ./lib,
                                ./share, and ./include.
                make config     displays parameter settings; does not compile

                Each individual package also has each of the above 'make'
                targets.  Doing 'make config' in each package */Lib directory
                displays the exact shared and static library names.

                Things you don't need to do:
                make cx         creates CXSparse from CSparse
                make docs       creates user guides from LaTeX files
                make cov        runs statement coverage tests (Linux only)
                make metis      compiles METIS (also done by 'make')

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

    metis-5.1.0 a modified version of METIS.  See the README.txt files for
                details.
                author: George Karypis; not an integral component of
                SuiteSparse, however.  This is just a copy included with
                SuiteSparse via the open-source license provided by
                George Karypis

    RBio        read/write sparse matrices in Rutherford/Boeing format
                author: Tim Davis

    README.txt  this file

    SPQR        sparse QR factorization.  This the built-in qr and x=A\b in
                MATLAB.
                author of the CPU code: Tim Davis
                author of GPU modules: Tim Davis, Nuri Yeralan,
                    Wissam Sid-Lakhdar, Sanjay Ranka

                SPQR/GPUQREngine: GPU support package for SPQR
                (not built into MATLAB, however)
                authors: Tim Davis, Nuri Yeralan, Sanjay Ranka,
                    Wissam Sid-Lakhdar

    SuiteSparse_config    configuration file for all the above packages.  The
                SuiteSparse_config/SuiteSparse_config.mk is included in the
                Makefile's of all packages.  CSparse and MATLAB_Tools do not
                use SuiteSparse_config.
                author: Tim Davis

    SuiteSparse_GPURuntime      GPU support package for SPQR and CHOLMOD
                (not builtin to MATLAB, however).

    SuiteSparse_install.m       install SuiteSparse for MATLAB
    SuiteSparse_paths.m         set paths for SuiteSparse MATLAB mexFunctions

    SuiteSparse_test.m          exhaustive test for SuiteSparse in MATLAB

    ssget       MATLAB interface to the SuiteSparse Matrix Collection
                (formerly called the UF Sparse Matrix Collection).
                Includes a UFget function for backward compatibility.
                author: Tim Davis

    UMFPACK     sparse LU factorization.  Requires AMD and the BLAS.
                This is the built-in lu and x=A\b in MATLAB.
                author: Tim Davis
                algorithm design collaboration: Iain Duff

Some codes optionally use METIS 5.1.0.  This package is located in SuiteSparse
in the metis-5.1.0 directory.  Its use is optional, so you can remove it before
compiling SuiteSparse, if you desire.  The use of METIS will improve the
ordering quality.  METIS has been slightly modified for use in SuiteSparse; see
the metis-5.1.0/README.txt file for details.  SuiteSparse can use the
unmodified METIS 5.1.0, however.  To use your own copy of METIS, or a
pre-installed copy of METIS use 'make MY_METIS_LIB=-lmymetis' or
'make MY_METIS_LIB=/my/stuff/metis-5.1.0/whereeveritis/libmetis.so 
      MY_METIS_INC=/my/stuff/metis-5.1.0/include'.
If you want to use METIS in MATLAB, however, you MUST use the version provided
here, in SuiteSparse/metis-5.1.0.  The MATLAB interface to METIS required some
small changes in METIS itself to get it to work.  The original METIS 5.1.0
will segfault MATLAB.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis.
email: davis@tamu.edu

Licenses for each package are located in the following files, all in
PACKAGENAME/Doc/License.txt:

    AMD/Doc/License.txt
    BTF/Doc/License.txt
    CAMD/Doc/License.txt
    CCOLAMD/Doc/License.txt
    CHOLMOD/Doc/License.txt
    COLAMD/Doc/License.txt
    CSparse/Doc/License.txt
    CXSparse/Doc/License.txt
    GPUQREngine/Doc/License.txt
    KLU/Doc/License.txt
    LDL/Doc/License.txt
    MATLAB_Tools/Doc/License.txt
    Mongoose/Doc/License.txt
    RBio/Doc/License.txt
    SPQR/Doc/License.txt
    SuiteSparse_GPURuntime/Doc/License.txt
    ssget/Doc/License.txt
    UMFPACK/Doc/License.txt
    GraphBLAS/Doc/License.txt

These files are also present, but they are simply copies of the above license
files for CXSparse and ssget:

    CXSparse_newfiles/Doc/License.txt
    CSparse/MATLAB/ssget/Doc/License.txt
    CXSparse/MATLAB/ssget/Doc/License.txt

METIS 5.0.1 is distributed with SuiteSparse, and is Copyright (c)
by George Karypis.  Please refer to that package for its License.

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

Next, compile the GraphBLAS library.  In the system shell while in the
SuiteSparse folder, type "make gbinstall" if you have MATLAB R2020b or earlier,
or type "make gbrenamed" if you have MATLAB 9.10 (R2021a) or later.

Then in the MATLAB Command Window, cd to the SuiteSparse directory and type
SuiteSparse_install.  All packages will be compiled, and several demos will be
run.  To run a (long!) exhaustive test, do SuiteSparse_test.

Save your MATLAB path for future sessions with the MATLAB pathtool or savepath
commands.  If those methods fail because you don't have system-wide permission,
add the new paths to your startup.m file, normally in
Documents/MATLAB/startup.m.  You can also use the SuiteSparse_paths method to
set all your paths at the start of each MATLAB session.

For Windows:  My apologies, but I don't support Windows so you will need to
revise the above instructions for Windows yourself.

-----------------------------------------------------------------------------
QUICK START FOR THE C/C++ LIBRARIES:
-----------------------------------------------------------------------------

Type the following in this directory:

    make ; make install

or, if want to use GraphBLAS in recent versions of MATLAB, do:

    make ; make gbrenamed ; make install

All libraries will be created and copied into SuiteSparse/lib.  All include
files need by the applications that use SuiteSparse are copied into
SuiteSparse/include.   All user documenation is copied into
SuiteSparse/share/doc.

Be sure to first install all required libraries:  BLAS and LAPACK for UMFPACK,
CHOLMOD, and SPQR, and GMP and MPFR for SLIP_LU.  Be sure to use the latest
libraries; SLIP_LU requires MPFR 4.0 for example.

When compiling the libraries, do NOT use the INSTALL=... options for
installing. Just do:

    make

or to compile just the libraries without running the demos, do:

    make library

Any program that uses SuiteSparse can thus use a simpler rule as compared to
earlier versions of SuiteSparse.  If you add /home/me/SuiteSparse/lib to
your library search patch, you can do the following (for example):

    S = /home/me/SuiteSparse
    cc myprogram.c -I$(S)/include -lumfpack -lamd -lcholmod -lsuitesparseconfig -lm

To change the C and C++ compilers, and to compile in parallel use:

    AUTOCC=no CC=gcc CX=g++ JOBS=32 make

for example, which changes the compiler to gcc and g++, and runs make with
'make -j32', in parallel with 32 jobs.

Now you can install the libraries, if you wish, in a location other than
SuiteSparse/lib, SuiteSparse/include, and SuiteSparse/share/doc, using
'make install INSTALL=...'

Do 'make install' if you want to install the libraries and include files in
SuiteSparse/lib and SuiteSparse/include, and the documentation in
SuiteSparse/doc/suitesparse-VERSION.
This will work on Linux/Unix and the Mac.  It should automatically detect if
you have the Intel compilers or not, and whether or not you have CUDA.  If this
fails, see the SuiteSparse_config/SuiteSparse_config.mk file.  There are many
options that you can either list on the 'make' command line, or you can just
edit that file.  For example, to compile with your own BLAS:

    make BLAS=-lmyblaslibraryhere

NOTE: Use of the Intel MKL BLAS is strongly recommended.  The OpenBLAS can
result in severe performance degradation, in CHOLMOD in particular.

To list all configuration options (but not compile anything), do:

    make config

Any parameter you see in the output of 'make config' with an equal sign
can be modified at the 'make' command line.

If you do "make install" by itself, then the packages are all installed in
SuiteSparse/lib (libraries), SuiteSparse/include (include .h files), and
SuiteSparse/doc/suitesparse-VERSION (documentation).  To install in
/usr/local, the default location for Linux, do:

    make library
    sudo make install INSTALL=/usr/local

If you want to install elsewhere, say in /my/path, first ensure that /my/path
is in your LD_LIBRARY_PATH.  How to do that depends on your system, but in the
bash shell, add this to your ~/.bashrc file:

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/path
    export LD_LIBRARY_PATH

You may also need to add SuiteSparse/lib to your path.  If your copy of
SuiteSparse is in /home/me/SuiteSparse, for example, then add this to your
~/.bashrc file:

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/me/SuiteSparse/lib:/my/path
    export LD_LIBRARY_PATH

For the Mac, use this instead:

    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/home/me/SuiteSparse/lib:/my/path
    export DYLD_LIBRARY_PATH

Then do the following (use "sudo make ..." if needed):

    make library
    make install INSTALL=/my/path

which puts the files in /my/path/lib, /my/path/include, and /my/path/doc.
If you want to selectively put the libraries, include files, and doc files
in different locations, do:

    make install INSTALL_LIB=/my/libs INSTALL_INCLUDE=/myotherstuff/include INSTALL_DOC=/mydocs

for example.  Any term not defined will be set to its default, so if you don't
want to install the documentation, but wish to install the libraries and
includes in /usr/local/lib and /usr/local/include, do:

    make install INSTALL_DOC=/tmp/doc

which copies the documentation to /tmp/doc where you can then remove it later.

Both the static (.a) and shared (.so) libraries are compiled.  The lib.a
libraries are left in the package Lib folder (AMD/Lib/libamd.a for example).
The main exception to this rule is the SuiteSparse_config library, which is in
SuiteSparse/libsuiteSparseconfig.a.  SuiteSparse_config is required by all
packages.  The (extremely) optional xerbla library is also an exception, but it
is highly unlikely that you need that library.

The 'make uninstall' takes the same command-line arguments.

----------------------------------
Step-by-step details:
----------------------------------

(1) Use the right BLAS and LAPACK libraries.
    Determine where your BLAS and LAPACK libraries are.  If the default
    'make' does not find them, use
    'make BLAS=-lmyblaslibraryhere LAPACK=-lmylapackgoeshere'

(2) Install Intel's Threading Building Blocks (TBB).
    This is optionally used by SuiteSparseQR.  Refer to the User Guide in 
    SuiteSparse/SPQR/Doc/spqr_user_guide.pdf for details.

(3) Determine what other command line options you need for 'make'.  All options
    can be set at the 'make' command line without the need to edit this file.
    Browse that file to see what options you can control.  If you choose
    different options and wish to recompile, be sure to do 'make distclean' in
    this directory first, to remove all files not in the original distribution.

(4) Type "make" in this directory.  All packages will be be compiled.  METIS
    5.1.0 will be compiled if you have it (note that METIS require CMake to
    build it).  Several demos will be run.  To compile just the libraries,
    without running any demos, use "make library".  The libraries will appear
    in */Lib/*.so.* (*.dylib for the Mac).  Include files, as needed by user
    programs that use CHOLMOD, AMD, CAMD, COLAMD, CCOLAMD, BTF, KLU, UMFPACK,
    LDL, etc. are in */Include/*.h.  The include files required by user
    programs are then copied into SuiteSparse/include, and the compiled
    libraries are copied into SuiteSparse/lib.  Documentation is copied into
    SuiteSparse/doc.  The GraphBLAS libraries are created by cmake and placed
    in GraphBLAS/build.  NOTE: on Linux, you may see some errors when you
    compile METIS ('make: *** No rule to make target 'w'.).  You can safely
    ignore those errors.

(6) To install, type "make install".  This will place copies of all
    libraries in SuiteSparse/lib, and all include files in SuiteSparse/include,
    and all documentation in SuiteSparse/doc/suitesparse-VERSION.  You can
    change the install location by "make install INSTALL=/my/path" which puts
    the libraries in /my/path/lib, the include files in /my/path/include, and
    documentation in /my/path/doc.  These directories are created if they do
    not already exist.

(7) To uninstall, type "make uninstall", which reverses "make install"
    by removing the SuiteSparse libraries, include files, and documentation
    from the place they were installed.  If you pass INSTALL_***= options
    to 'make install', you must pass the same to 'make uninstall'.

-----------------------------------------------------------------------------
Python interface
-----------------------------------------------------------------------------

See scikit-sparse and scikit-umfpack for the Python interface via SciPy:

https://github.com/scikit-sparse/scikit-sparse

https://github.com/scikit-umfpack/scikit-umfpack

