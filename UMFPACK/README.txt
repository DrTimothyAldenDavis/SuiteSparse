UMFPACK, Copyright (c) 1995-2022 by Timothy A. Davis,
http://www.suitesparse.com

UMFPACK :  a set of routines solving sparse linear systems via LU
    factorization.  Requires three other packages:  the BLAS (dense matrix
    operations), AMD (sparse matrix minimum degree ordering), and
    SuiteSparse_config.

    Includes a C-callable and MATLAB interface, and a basic FORTRAN 77
    interface to a subset of the C-callable routines.  Requires AMD Version
    2.0 or later.

    Unless you compile with -DNCHOLMOD, addtional packages are required:
    CHOLMOD, CAMD, CCOLAMD, COLAMD, and CHOLMOD/SuiteSparse_metis.

The AMD, SuiteSparse_config, and UMFPACK directories must all reside in the
same parent directory.  If the -DNCHOLMOD is not used, the CHOLMOD, CAMD,
CCOLAMD, COLAMD, and also also exist in the same parent.

Quick start:

    To compile and install the library for system-wide usage:

        make
        sudo make install

    To compile/install for local usage (SuiteSparse/lib and SuiteSparse/include)

        make local
        make install

    To run the demos

        make demos

Quick start (for MATLAB users);

    To compile, test, and install the UMFPACK mexFunction, cd to the
    UMFPACK/MATLAB directory and type umfpack_install at the MATLAB prompt.
    Then save your path for future sessions.

--------------------------------------------------------------------------------

UMFPACK is available under alternate licences; contact T. Davis for details.

UMFPACK License:  See UMFPACK/Doc/License.txt for the license.

Availability:

    http://www.suitesparse.com

    UMFPACK (including versions 2.2.1 and earlier, in FORTRAN) is available at
    http://www.suitesparse.com.  MA38 is available in the Harwell
    Subroutine Library.  This version of UMFPACK includes a modified form of
    COLAMD Version 2.0, originally released on Jan. 31, 2000, also available at
    http://www.suitesparse.com.  COLAMD V2.0 is also incorporated
    as a built-in function in MATLAB version 6.1, by The MathWorks, Inc.
    (http://www.mathworks.com).  COLAMD V1.0 appears as a column-preordering
    in SuperLU (SuperLU is available at http://www.netlib.org).
    UMFPACK v4.0 is a built-in routine in MATLAB 6.5.
    UMFPACK v4.3 is a built-in routine in MATLAB 7.1.

--------------------------------------------------------------------------------

Refer to ../AMD/Doc/License.txt for the License for AMD, which is a separate
package for ordering sparse matrices that is required by UMFPACK.
UMFPACK v4.5 cannot use AMD v1.1 or earlier.  UMFPACK 5.x
requires AMD v2.0 or later.

--------------------------------------------------------------------------------

This is the UMFPACK README.txt file.  It is a terse overview of UMFPACK.
Refer to the User Guide (Doc/UserGuide.pdf) for how to install and use UMFPACK,
or to the Quick Start Guide, QuickStart.pdf.

Description:

    UMFPACK is a set of routines for solving unsymmetric sparse linear systems,
    Ax=b, using the Unsymmetric MultiFrontal method.  Written in ANSI/ISO C,
    with a MATLAB (Version 6.0 or later) interface.

    For best performance, UMFPACK requires an optimized BLAS library.  It can
    also be compiled without any BLAS at all.  UMFPACK requires AMD Version 2.0.

Authors:

    Timothy A. Davis (DrTimothyAldenDavis@gmail.com)

    Includes a modified version of COLAMD V2.0, by Stefan I. Larimore and
    Timothy A. Davis.  The COLAMD algorithm was developed
    in collaboration with John Gilbert, Xerox Palo Alto Research Center, and
    Esmond Ng, Lawrence Berkeley National Laboratory.

    Includes AMD, by Timothy A. Davis, Patrick R. Amestoy, and Iain S. Duff.

    UMFPACK Version 2.2.1 (MA38 in the Harwell Subroutine Library) is
    co-authored with Iain S. Duff, Rutherford Appleton Laboratory.

Acknowledgements:

    This work was supported by the National Science Foundation, under
    grants DMS-9504974, DMS-9803599, and CCR-0203270.

    Portions of this work were done while on sabbatical at Stanford University
    and Lawrence Berkeley National Laboratory (with funding from the SciDAC
    program).  I would like to thank Gene Golub, Esmond Ng, and Horst Simon
    for making this sabbatical possible.

    I would also like to thank the many researchers who provided sparse
    matrices from a wide range of domains and used earlier versions of UMFPACK/
    MA38 in their applications, and thus assisted in the practical development
    of the algorithm (see http://www.suitesparse.com, future
    contributions of matrices are always welcome).

    The MathWorks, Inc., provided a pre-release of MATLAB V6 which allowed me
    to release the first umfpack mexFunction (v3.0) about 6 months earlier than
    I had originally planned.  They also supported the extension of UMFPACK to
    complex, singular, and rectangular matrices (UMFPACK v4.0).

    Penny Anderson (The MathWorks, Inc.), Anshul Gupta (IBM), and Friedrich
    Grund (WAIS) assisted in porting UMFPACK to different platforms.  Penny
    Anderson also incorporated UMFPACK v4.0 into MATLAB, for lu, backslash (\),
    and forward slash (/).

    David Bateman (Motorola) wrote the initial version of the packed complex
    input option, and umfpack_get_determinant.

--------------------------------------------------------------------------------
Files and directories in the UMFPACK distribution:
--------------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    Subdirectories of the UMFPACK directory:
    ----------------------------------------------------------------------------

    Doc		documentation
    Source	primary source code
    Include	include files for use in your code that calls UMFPACK
    Demo	demo programs.  also serves as test of the UMFPACK installation.
    MATLAB	UMFPACK mexFunction for MATLAB, and supporting m-files
    build       where the compiled libraries and demos are placed
    Config      source file to construct umfpack.h

    ----------------------------------------------------------------------------
    Files in the UMFPACK directory:
    ----------------------------------------------------------------------------

    Makefile	a very simple Makefile (optional); just for simplifying cmake
    CMakeLists.txt  cmake script for building UMFPACK
    README.txt	this file

    ----------------------------------------------------------------------------
    Doc directory: documentation
    ----------------------------------------------------------------------------

    ChangeLog			change log
    License.txt			the UMFPACK License (GPL)
    gpl.txt			the GNU GPL
    Makefile			for creating the documentation
    UMFPACK_QuickStart.tex	Quick Start guide (source)
    UMFPACK_QuickStart.pdf	Quick Start guide (PDF)
    UserGuide.bib		User Guide (references)
    UMFPACK_UserGuide.tex	User Guide (LaTeX)
    UMFPACK_UserGuide.pdf	User Guide (PDF)

    ----------------------------------------------------------------------------
    Source2 directory:
    ----------------------------------------------------------------------------

    This directory contains all source files used directly in CMakeLists.txt.
    Each of them sets various #define's, and then #include's files in the
    Source/ directory.

    ----------------------------------------------------------------------------
    Source directory:
    ----------------------------------------------------------------------------

    umfpack_col_to_triplet.c	convert col form to triplet
    umfpack_defaults.c		set Control defaults
    umfpack_free_numeric.c	free Numeric object
    umfpack_free_symbolic.c	free Symbolic object
    umfpack_get_determinant.c	compute determinant from Numeric object
    umfpack_get_lunz.c		get nz's in L and U
    umfpack_get_numeric.c	get Numeric object
    umfpack_get_symbolic.c	get Symbolic object
    umfpack_load_numeric.c	load Numeric object from file
    umfpack_load_symbolic.c	load Symbolic object from file
    umfpack_numeric.c		numeric factorization
    umfpack_qsymbolic.c		symbolic factorization, user Q
    umfpack_report_control.c	print Control settings
    umfpack_report_info.c	print Info statistics
    umfpack_report_matrix.c	print col or row-form sparse matrix
    umfpack_report_numeric.c	print Numeric object
    umfpack_report_perm.c	print permutation
    umfpack_report_status.c	print return status
    umfpack_report_symbolic.c	print Symbolic object
    umfpack_report_triplet.c	print triplet matrix
    umfpack_report_vector.c	print dense vector
    umfpack_save_numeric.c	save Numeric object to file
    umfpack_save_symbolic.c	save Symbolic object to file
    umfpack_scale.c		scale a vector
    umfpack_solve.c		solve a linear system
    umfpack_symbolic.c		symbolic factorization
    umfpack_tictoc.c		timer
    umfpack_timer.c		timer
    umfpack_transpose.c		transpose a matrix
    umfpack_triplet_to_col.c	convert triplet to col form

    umf_config.h		configuration file (BLAS, memory, timer)
    umf_internal.h		definitions internal to UMFPACK
    umf_version.h		version definitions (int/SuiteSparse_long, real/complex)

    umf_analyze.[ch]		symbolic factorization of A'*A
    umf_apply_order.[ch]	apply column etree postorder
    umf_assemble.[ch]		assemble elements into current front
    umf_blas3_update.[ch]	rank-k update.  Uses level-3 BLAS
    umf_build_tuples.[ch]	construct tuples for elements
    umf_colamd.[ch]		COLAMD pre-ordering, modified for UMFPACK
    umf_cholmod.[ch]		interface to CHOLMOD
    umf_create_element.[ch]	create a new element
    umf_dump.[ch]		debugging routines, not normally active
    umf_extend_front.[ch]	extend the current frontal matrix
    umf_free.[ch]		free memory
    umf_fsize.[ch]		determine largest front in each subtree
    umf_garbage_collection.[ch]	compact Numeric->Memory
    umf_get_memory.[ch]		make Numeric->Memory bigger
    umf_grow_front.[ch]		make current frontal matrix bigger
    umf_init_front.[ch]		initialize a new frontal matrix
    umf_is_permutation.[ch]	checks the validity of a permutation vector
    umf_kernel.[ch]		the main numeric factorization kernel
    umf_kernel_init.[ch]	initializations for umf_kernel
    umf_kernel_wrapup.[ch]	wrapup for umf_kernel
    umf_local_search.[ch]	local row and column pivot search
    umf_lsolve.[ch]		solve Lx=b
    umf_ltsolve.[ch]		solve L'x=b and L.'x=b
    umf_malloc.[ch]		malloc some memory
    umf_mem_alloc_element.[ch]		allocate element in Numeric->Memory
    umf_mem_alloc_head_block.[ch]	alloc. block at head of Numeric->Memory
    umf_mem_alloc_tail_block.[ch]	alloc. block at tail of Numeric->Memory
    umf_mem_free_tail_block.[ch]	free block at tail of Numeric->Memory
    umf_mem_init_memoryspace.[ch]	initialize Numeric->Memory
    umf_realloc.[ch]		realloc memory
    umf_report_perm.[ch]	print a permutation vector
    umf_report_vector.[ch]	print a double vector
    umf_row_search.[ch]		look for a pivot row
    umf_scale.[ch]		scale the pivot column
    umf_scale_column.[ch]	move pivot row & column into place, log P and Q
    umf_set_stats.[ch]		set statistics (final or estimates)
    umf_singletons.[ch]		find all zero-cost pivots
    umf_solve.[ch]		solve a linear system
    umf_start_front.[ch]	start a new frontal matrix for one frontal chain
    umf_store_lu.[ch]		store LU factors of current front
    umf_symbolic_usage.[ch]	determine memory usage for Symbolic object
    umf_transpose.[ch]		transpose a matrix in row or col form
    umf_triplet.[ch]		convert triplet to column form
    umf_tuple_lengths.[ch]	determine the tuple list lengths
    umf_usolve.[ch]		solve Ux=b
    umf_utsolve.[ch]		solve U'x=b and U.'x=b
    umf_valid_numeric.[ch]	checks the validity of a Numeric object
    umf_valid_symbolic.[ch]	check the validity of a Symbolic object

    ----------------------------------------------------------------------------
    Include directory:
    ----------------------------------------------------------------------------

    umfpack.h			include file for user programs.  Also serves as
                                source-code level documentation.

    ----------------------------------------------------------------------------
    Demo directory:
    ----------------------------------------------------------------------------

    Makefile			to compile the demos

    umfpack_simple.c		a simple demo
    umpack_xx_demo.c		template to create the demo codes below

    umfpack_di_demo.sed		for creating umfpack_di_demo.c
    umfpack_dl_demo.sed		for creating umfpack_dl_demo.c
    umfpack_zi_demo.sed		for creating umfpack_zi_demo.c
    umfpack_zl_demo.sed		for creating umfpack_zl_demo.c

    umfpack_di_demo.c		a full demo (real/int version)
    umfpack_dl_demo.c		a full demo (real/SuiteSparse_long version)
    umfpack_zi_demo.c		a full demo (complex/int version)
    umfpack_zl_demo.c		a full demo (complex/SuiteSparse_long version)

    umfpack_di_demo.out		umfpack_di_demo output
    umfpack_dl_demo.out		umfpack_dl_demo output
    umfpack_zi_demo.out		umfpack_zi_demo output
    umfpack_zl_demo.out		umfpack_zl_demo output

    umf4.c			a demo (real/int) for Harwell/Boeing matrices
    umf4.out			output of "make hb"
    HB/			        directory of sample Harwell/Boeing matrices
    readhb.f			reads HB matrices, keeps zero entries
    readhb_nozeros.f		reads HB matrices, removes zero entries
    readhb_size.f		reads HB matrix dimension, nnz

    umf4_f77wrapper.c		a simple FORTRAN interface for UMFPACK.
				compile with "make fortran"
    umf4hb.f			a demo of the FORTRAN interface
    umf4hb.out			output of "make fortran"

    umf4_f77zwrapper.c		a simple FORTRAN interface for the complex
				UMFPACK routines.  compile with "make fortran"
    umf4zhb.f			a demo of the FORTRAN interface (complex)
    umf4zhb.out			output of umf4zhb with HB/qc324.cua

    umf4hb64.f			64-bit version of umf4hb.f

    ----------------------------------------------------------------------------
    MATLAB directory:
    ----------------------------------------------------------------------------

    Contents.m			for "help umfpack" listing of toolbox contents

    lu_normest.m		1-norm estimate of A-L*U (by Hager & Davis).
    luflop.m			for "help luflop"
    luflopmex.c			luflop mexFunction, for computing LU flop count
    umfpack.m			for "help umfpack"
    umfpack_btf.m		solve Ax=b using umfpack and dmperm
    umfpack_demo.m		a full umfpack demo
    umfpack_details.m		the details of how to use umfpack
    umfpack_make.m		compile the umfpack mexFunction within MATLAB
    umfpack_report.m		report statistics
    umfpack_simple.m		a simple umfpack demo
    umfpack_solve.m		x=A\b or b/A for arbitrary b
    umfpack_test.m		extensive test, requires ssget
    umfpackmex.c		the umfpack mexFunction
    west0067.mat		sparse matrix for umfpack_demo.m

    umfpack_demo.m.out		output of umfpack_demo.m
    umfpack_simple.m.out	output of umfpack_simple

