-----------------------------------------------------------------------------
SuiteSparse:  A Suite of Sparse matrix packages at http://www.suitesparse.com
-----------------------------------------------------------------------------

Apr 8, 2020.  SuiteSparse VERSION 5.7.2

Now includes GraphBLAS and a new interface to the SuiteSparse Matrix
Collection (ssget), via MATLAB and a Java GUI, to http://sparse.tamu.edu.

Primary author of SuiteSparse (codes and algorithms, excl. METIS): Tim Davis

Code co-authors, in alphabetical order (not including METIS):

    Patrick Amestoy, David Bateman, Yanqing Chen, Iain Duff, Les Foster,
    William Hager, Scott Kolodziej, Stefan Larimore, Ekanathan Palamadai,
    Sivasankaran Rajamanickam, Sanjay Ranka, Wissam Sid-Lakhdar, Nuri Yeralan.

Additional algorithm designers: Esmond Ng and John Gilbert.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis.

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

    Makefile    to compile all of SuiteSparse (except GraphBLAS)
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
QUICK START FOR MATLAB USERS (Linux, Mac, or Windows):
-----------------------------------------------------------------------------

Uncompress the SuiteSparse.zip or SuiteSparse.tar.gz archive file (they contain
the same thing), then in the MATLAB Command Window, cd to the SuiteSparse
directory and type SuiteSparse_install.  All packages will be compiled, and
several demos will be run.  To run a (long!) exhaustive test, do
SuiteSparse_test.

-----------------------------------------------------------------------------
QUICK START FOR THE C/C++ LIBRARIES:
-----------------------------------------------------------------------------

For just GraphBLAS, do this:

    cd GraphBLAS/build ; cmake .. ; make ; cd ../Demo ; ./demo 
    cd ../build ; sudo make install

For all other packages, type 'make' in this directory.  All libraries will be
created and copied into SuiteSparse/lib.  All include files need by the
applications that use SuiteSparse are copied into SuiteSparse/include.   All
user documenation is copied into SuiteSparse/share/doc.

When compiling the libraries, do NOT use the INSTALL=... options for
installing. Just do:

    make

or to compile just the libraries without running the demos, do:

    make library

Any program that uses SuiteSparse can thus use a simpler rule as compared to
earlier versions of SuiteSparse.  If you add /home/myself/SuiteSparse/lib to
your library search patch, you can do the following (for example):

    S = /home/myself/SuiteSparse
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
SuiteSparse/lib (libraries), SuiteSparse/include (include *.h files), and
SuiteSparse/doc/suitesparse-VERSION (documentation).  If you want to install
elsewhere, do:

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

Both the static (*.a) and shared (*.so) libraries are compiled.  The *.a
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

