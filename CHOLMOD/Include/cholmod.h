//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod.h: include file for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod.h.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.

// Each Module of CHOLMOD has its own license, and a shared cholmod.h file.

// CHOLMOD/Check:      SPDX-License-Identifier: LGPL-2.1+
// CHOLMOD/Cholesky:   SPDX-License-Identifier: LGPL-2.1+
// CHOLMOD/Utility:    SPDX-License-Identifier: LGPL-2.1+
// CHOLMOD/Partition:  SPDX-License-Identifier: LGPL-2.1+

// CHOLMOD/Demo:       SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/GPU:        SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/MATLAB:     SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/MatrixOps:  SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Modify:     SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Supernodal: SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Tcov:       SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
// CHOLMOD consists of a set of Modules, each with their own license: either
// LGPL-2.1+ or GPL-2.0+.  This cholmod.h file includes defintions of the
// CHOLMOD API for all Modules, and this cholmod.h file itself is provided to
// you with a permissive license (Apache-2.0).  You are permitted to provide
// the hooks for an optional interface to CHOLMOD in a non-GPL/non-LGPL code,
// without requiring you to agree to the GPL/LGPL license of the Modules, as
// long as you don't use the *.c files in the relevant Modules.  The Modules
// themselves can only be functional if their GPL or LGPL licenses are used.
//
// The Modify Module is co-authored by William W. Hager.
//
// Acknowledgements:  this work was supported in part by the National Science
// Foundation (NFS CCR-0203270 and DMS-9803599), and a grant from Sandia
// National Laboratories (Dept. of Energy) which supported the development of
// CHOLMOD's Partition Module.
// -----------------------------------------------------------------------------

// Each routine in CHOLMOD has a consistent interface.
//
// Naming convention:
// ------------------
//
//  All routine names, data types, and CHOLMOD library files use the
//  cholmod_ prefix.  All macros and other #define's use the CHOLMOD
//  prefix.
//
// Return value:
// -------------
//
//  Most CHOLMOD routines return an int (TRUE (1) if successful, or FALSE
//      (0) otherwise.  An int32_t, int64_t, double, or float return value
//      is >= 0 if successful, or -1 otherwise.  A size_t return value
//      is > 0 if successful, or 0 otherwise.
//
//      If a routine returns a pointer, it is a pointer to a newly allocated
//      object or NULL if a failure occured, with one exception.  cholmod_free
//      always returns NULL.
//
// "Common" parameter:
// ------------------
//
//  The last parameter in all CHOLMOD routines is a pointer to the CHOLMOD
//  "Common" object.  This contains control parameters, statistics, and
//  workspace used between calls to CHOLMOD.  It is always an input/output
//  parameter.
//
// Input, Output, and Input/Output parameters:
// -------------------------------------------
//
//  Input parameters are listed first.  They are not modified by CHOLMOD.
//
//  Input/output are listed next.  They must be defined on input, and
//  are modified on output.
//
//  Output parameters are listed next.  If they are pointers, they must
//  point to allocated space on input, but their contents are not defined
//  on input.
//
//  Workspace parameters appear next.  They are used in only two routines
//  in the Supernodal module.
//
//  The cholmod_common *Common parameter always appears as the last
//  parameter.  It is always an input/output parameter.

#ifndef CHOLMOD_H
#define CHOLMOD_H

//==============================================================================
// version control
//==============================================================================

#define CHOLMOD_DATE "Oct 31, 2023"
#define CHOLMOD_MAIN_VERSION   5
#define CHOLMOD_SUB_VERSION    0
#define CHOLMOD_SUBSUB_VERSION 1

#define CHOLMOD_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define CHOLMOD_VERSION \
    CHOLMOD_VER_CODE(CHOLMOD_MAIN_VERSION,CHOLMOD_SUB_VERSION)
#define CHOLMOD_HAS_VERSION_FUNCTION

#ifdef __cplusplus
extern "C" {
#endif

int cholmod_version     // returns CHOLMOD_VERSION, defined above
(
    // if version is not NULL, then cholmod_version returns its contents as:
    // version [0] = CHOLMOD_MAIN_VERSION
    // version [1] = CHOLMOD_SUB_VERSION
    // version [2] = CHOLMOD_SUBSUB_VERSION
    int version [3]
) ;
int cholmod_l_version (int version [3]) ;

#ifdef __cplusplus
}
#endif

//==============================================================================
// Large file support
//==============================================================================

// CHOLMOD assumes large file support.  If problems occur, compile with
// -DNLARGEFILE

// Definitions required for large file I/O, which must come before any other
// #includes.  These are not used if -DNLARGEFILE is defined at compile time.
// Large file support may not be portable across all platforms and compilers;
// if you encounter an error here, compile your code with -DNLARGEFILE.  In
// particular, you must use -DNLARGEFILE for MATLAB 6.5 or earlier (which does
// not have the io64.h include file).

// skip all of this if NLARGEFILE is defined at the compiler command line
#ifndef NLARGEFILE
    #if defined(MATLAB_MEX_FILE) || defined(MATHWORKS)
        // CHOLMOD compiled as a MATLAB mexFunction, or for use in MATLAB
        #include "io64.h"
    #else
        // CHOLMOD is being compiled in a stand-alone library
        #undef  _LARGEFILE64_SOURCE
        #define _LARGEFILE64_SOURCE
        #undef  _FILE_OFFSET_BITS
        #define _FILE_OFFSET_BITS 64
    #endif
#endif

//==============================================================================
// SuiteSparse_config
//==============================================================================

#include "SuiteSparse_config.h"

//==============================================================================
// CHOLMOD configuration
//==============================================================================

// You do not have to edit any CHOLMOD files to compile and install CHOLMOD.
// However, if you do not use all of CHOLMOD's modules, you need to compile
// with the appropriate flag, or edit this file to add the appropriate #define.
//
// Compiler flags for CHOLMOD
//
// -DNCHECK         do not include the Check module.
// -DNCHOLESKY      do not include the Cholesky module.
// -DNPARTITION     do not include the Partition module.
// -DNCAMD          do not include the interfaces to CAMD,
//                  CCOLAMD, CSYMAND in Partition module.
// -DNMATRIXOPS     do not include the MatrixOps module.
// -DNMODIFY        do not include the Modify module.
// -DNSUPERNODAL    do not include the Supernodal module.
//
// -DNPRINT         do not print anything
//
// The Utility Module is always included in the CHOLMOD library.

// Use the compiler flag, or uncomment the definition(s), if you want to use
// one or more non-default installation options:

//      #define NCHECK
//      #define NCHOLESKY
//      #define NCAMD
//      #define NPARTITION
//      #define NMATRIXOPS
//      #define NMODIFY
//      #define NSUPERNODAL
//      #define NPRINT
//      #define NGPL

// The NGPL option disables the MatrixOps, Modify, and Supernodal modules.  The
//  existence of this #define here, and its use in these 3 modules, does not
//  affect the license itself; see CHOLMOD/Doc/License.txt for your actual
//  license.

#ifdef NGPL
    #undef  NMATRIXOPS
    #define NMATRIXOPS
    #undef  NMODIFY
    #define NMODIFY
    #undef  NSUPERNODAL
    #define NSUPERNODAL
#endif

//==============================================================================
// CHOLMOD:Utility Module
//==============================================================================

// the CHOLMOD:Utility Module is always required
#if 1

//------------------------------------------------------------------------------
// CUDA BLAS
//------------------------------------------------------------------------------

// Define buffering parameters for GPU processing
#ifndef SUITESPARSE_GPU_EXTERN_ON
#ifdef SUITESPARSE_CUDA
#include <cublas_v2.h>
#endif
#endif

#define CHOLMOD_DEVICE_SUPERNODE_BUFFERS 6
#define CHOLMOD_HOST_SUPERNODE_BUFFERS 8
#define CHOLMOD_DEVICE_STREAMS 2

//------------------------------------------------------------------------------
// CHOLMOD objects
//------------------------------------------------------------------------------

// CHOLMOD object enums
#define CHOLMOD_COMMON 0
#define CHOLMOD_SPARSE 1
#define CHOLMOD_FACTOR 2
#define CHOLMOD_DENSE 3
#define CHOLMOD_TRIPLET 4

// enums used by cholmod_symmetry and cholmod_write
#define CHOLMOD_MM_RECTANGULAR       1
#define CHOLMOD_MM_UNSYMMETRIC       2
#define CHOLMOD_MM_SYMMETRIC         3
#define CHOLMOD_MM_HERMITIAN         4
#define CHOLMOD_MM_SKEW_SYMMETRIC    5
#define CHOLMOD_MM_SYMMETRIC_POSDIAG 6
#define CHOLMOD_MM_HERMITIAN_POSDIAG 7

//------------------------------------------------------------------------------
// CHOLMOD Common object
//------------------------------------------------------------------------------

// itype: integer sizes
// The itype is held in the Common object and must match the method used.
#define CHOLMOD_INT  0    /* int32, for cholmod_* methods (no _l_) */
#define CHOLMOD_LONG 2    /* int64, for cholmod_l_* methods        */

// dtype: floating point sizes (double or float)
// The dtype of all parameters for all CHOLMOD routines must match.
// NOTE: CHOLMOD_SINGLE is still under development.
#define CHOLMOD_DOUBLE 0  /* matrix or factorization is double precision */
#define CHOLMOD_SINGLE 4  /* matrix or factorization is single precision */

// xtype: pattern, real, complex, or zomplex
#define CHOLMOD_PATTERN 0 /* no numerical values                            */
#define CHOLMOD_REAL    1 /* real (double or single), not complex           */
#define CHOLMOD_COMPLEX 2 /* complex (double or single), interleaved        */
#define CHOLMOD_ZOMPLEX 3 /* complex (double or single), with real and imag */
                          /* parts held in different arrays                 */

// xdtype is (xtype + dtype), which combines the two type parameters into
// a single number handling all 8 cases:
//
//  (0) CHOLMOD_DOUBLE + CHOLMOD_PATTERN    a pattern-only matrix
//  (1) CHOLMOD_DOUBLE + CHOLMOD_REAL       a double real matrix
//  (2) CHOLMOD_DOUBLE + CHOLMOD_COMPLEX    a double complex matrix
//  (3) CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX    a double zomplex matrix
//  (4) CHOLMOD_SINGLE + CHOLMOD_PATTERN    a pattern-only matrix
//  (5) CHOLMOD_SINGLE + CHOLMOD_REAL       a float real matrix
//  (6) CHOLMOD_SINGLE + CHOLMOD_COMPLEX    a float complex matrix
//  (7) CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX    a float zomplex matrix

// max # of ordering methods in Common
#define CHOLMOD_MAXMETHODS 9

// Common->status for error handling: 0 is ok, negative is a fatal error,
// and positive is a warning
#define CHOLMOD_OK            (0)
#define CHOLMOD_NOT_INSTALLED (-1) /* module not installed                  */
#define CHOLMOD_OUT_OF_MEMORY (-2) /* malloc, calloc, or realloc failed     */
#define CHOLMOD_TOO_LARGE     (-3) /* integer overflow                      */
#define CHOLMOD_INVALID       (-4) /* input invalid                         */
#define CHOLMOD_GPU_PROBLEM   (-5) /* CUDA error                            */
#define CHOLMOD_NOT_POSDEF    (1)  /* matrix not positive definite          */
#define CHOLMOD_DSMALL        (2)  /* diagonal entry very small             */

// ordering method
#define CHOLMOD_NATURAL     0 /* no preordering                             */
#define CHOLMOD_GIVEN       1 /* user-provided permutation                  */
#define CHOLMOD_AMD         2 /* AMD: approximate minimum degree            */
#define CHOLMOD_METIS       3 /* METIS: mested dissection                   */
#define CHOLMOD_NESDIS      4 /* CHOLMOD's nested dissection                */
#define CHOLMOD_COLAMD      5 /* AMD for A, COLAMD for AA' or A'A           */
#define CHOLMOD_POSTORDERED 6 /* natural then postordered                   */

// supernodal strategy
#define CHOLMOD_SIMPLICIAL 0 /* always use simplicial method                */
#define CHOLMOD_AUTO       1 /* auto select simplicial vs supernodal        */
#define CHOLMOD_SUPERNODAL 2 /* always use supernoda method                 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cholmod_common_struct
{

    //--------------------------------------------------------------------------
    // primary parameters for factorization and update/downdate
    //--------------------------------------------------------------------------

    double dbound ; // Bounds the diagonal entries of D for LDL'
        // factorization and update/downdate/rowadd.  Entries outside this
        // bound are replaced with dbound.  Default: 0.
        // dbound is used for double precision factorization only.
        // See sbound for single precision factorization.

    double grow0 ;      // default: 1.2
    double grow1 ;      // default: 1.2
    size_t grow2 ;      // default: 5
        // Initial space for simplicial factorization is max(grow0,1) times the
        // required space.  If space is exhausted, L is grown by
        // max(grow0,1.2) times the required space.  grow1 and grow2 control
        // how each column of L can grow in an update/downdate; if space runs
        // out, then grow1*(required space) + grow2 is allocated.

    size_t maxrank ;    // maximum rank for update/downdate.  Valid values are
        // 2, 4, and 8.  Default is 8.  If a larger update/downdate is done,
        // it is done in steps of maxrank.

    double supernodal_switch ;  // default: 40
    int supernodal ;            // default: CHOLMOD_AUTO.
        // Controls supernodal vs simplicial factorization.  If
        // Common->supernodal is CHOLMOD_SIMPLICIAL, a simplicial factorization
        // is always done; if CHOLMOD_SUPERNODAL, a supernodal factorization is
        // always done.  If CHOLMOD_AUTO, then a simplicial factorization is
        // down if flops/nnz(L) < Common->supernodal_switch.

    int final_asis ;    // if true, other final_* parameters are ignored,
        // except for final_pack and the factors are left as-is when done.
        // Default: true.

    int final_super ;   // if true, leave factor in supernodal form.
        // if false, convert to simplicial.  Default: true.

    int final_ll ;      // if true, simplicial factors are converted to LL',
        // otherwise left as LDL.  Default: false.

    int final_pack ;    // if true, the factorize are allocated with exactly
        // the space required.  Set this to false if you expect future
        // updates/downdates (giving a little extra space for future growth),
        // Default: true.

    int final_monotonic ;   // if true, columns are sorted when done, by
        // ascending row index.  Default: true.

    int final_resymbol ;    // if true, a supernodal factorization converted
        // to simplicial is reanalyzed, to remove zeros added for relaxed
        // amalgamation.  Default: false.

    double zrelax [3] ; size_t nrelax [3] ;
        // The zrelax and nrelax parameters control relaxed supernodal
        // amalgamation,  If ns is the # of columns in two adjacent supernodes,
        // and z is the fraction of zeros in the two supernodes if merged, then
        // the two supernodes are merged if any of the 5 following condition
        // are true:
        //
        //      no new zero entries added if the two supernodes are merged
        //      (ns <= nrelax [0])
        //      (ns <= nrelax [1] && z < zrelax [0])
        //      (ns <= nrelax [2] && z < zrelax [1])
        //      (z < zrelax [2])
        //
        // With the defaults, the rules become:
        //
        //      no new zero entries added if the two supernodes are merged
        //      (ns <=  4)
        //      (ns <= 16 && z < 0.8)
        //      (ns <= 48 && z < 0.1)
        //      (z < 0.05)

    int prefer_zomplex ;    // if true, and a complex system is solved,
        // X is returned as zomplex (with two arrays, one for the real part
        // and one for the imaginary part).  If false, then X is returned as
        // a single array with interleaved real and imaginary parts.
        // Default: false.

    int prefer_upper ;  // if true, then a preference is given for holding
        // a symmetric matrix by just its upper triangular form.  This gives
        // the best performance by the CHOLMOD analysis and factorization
        // methods.  Only used by cholmod_read.  Default: true.

    int quick_return_if_not_posdef ;    // if true, a supernodal factorization
        // returns immediately if it finds the matrix is not positive definite.
        // If false, the failed supernode is refactorized, up to but not
        // including the failed column (required by MATLAB).

    int prefer_binary ; // if true, cholmod_read_triplet converts a symmetric
        // pattern-only matrix to a real matrix with all values set to 1.
        // if false, diagonal entries A(k,k) are set to one plus the # of
        // entries in row/column k, and off-diagonals are set to -1.
        // Default: false.

    int print ;     // print level.  Default is 3.
    int precise ;   // if true, print 16 digits, otherwise 5. Default: false.

    int try_catch ; // if true, ignore errors (CHOLMOD is assumed to be inside
        // a try/catch block.  No error messages are printed and the
        // error_handler function is not called.  Default: false.

    void (*error_handler) (int status, const char *file, int line,
        const char *message) ;
        // User error handling routine; default is NULL.
        // This function is called if an error occurs, with parameters:
        // status: the Common->status result.
        // file: filename where the error occurred.
        // line: line number where the error occurred.
        // message: a string that describes the error.

    //--------------------------------------------------------------------------
    // ordering options
    //--------------------------------------------------------------------------

    // CHOLMOD can try many ordering options and then pick the best result it
    // finds.  The default is to use one or two orderings: the user's
    // permutation (if given), and AMD.

    // Common->nmethods is the number of methods to try.  If the
    // Common->method array is left unmodified, the methods are:

    // (0) given (skipped if no user permutation)
    // (1) amd
    // (2) metis
    // (3) nesdis with defaults (CHOLMOD's nested dissection, based on METIS)
    // (4) natural
    // (5) nesdis: stop at subgraphs of 20000 nodes
    // (6) nesdis: stop at subgraphs of 4 nodes, do not use CAMD
    // (7) nesdis: no pruning on of dense rows/cols
    // (8) colamd

    // To use all 9 of the above methods, set Common->nmethods to 9.  The
    // analysis will take a long time, but that might be worth it if the
    // ordering will be reused many many times.

    // Common->nmethods and Common->methods can be revised to use a different
    // set of orderings.  For example, to use just a single method
    // (AMD with a weighted postordering):
    //
    //      Common->nmethods = 1 ;
    //      Common->method [0].ordering = CHOLMOD_AMD ;
    //      Common->postorder = TRUE ;
    //
    //

    int nmethods ;  // Number of methods to try, default is 0.
        // The value of 0 is a special case, and tells CHOLMOD to use the user
        // permutation (if not NULL) and then AMD.  Next, if fl is lnz are the
        // flop counts and number of nonzeros in L as found by AMD, then the
        // this ordering is used if fl/lnz < 500 or lnz/anz < 5, where anz is
        // the number of entries in A.  If this condition fails, METIS is tried
        // as well.
        //
        // Otherwise, if Common->nmethods > 0, then the methods defined by
        // Common->method [0 ... Common->nmethods-1] are used.

    int current ;   // The current method being tried in the analysis.
    int selected ;  // The selected method: Common->method [Common->selected]

    // The Common->method parameter is an array of structs that defines up
    // to 9 methods:
    struct cholmod_method_struct
    {

        //----------------------------------------------------------------------
        // statistics from the ordering
        //----------------------------------------------------------------------

        double lnz ;    // number of nonzeros in L
        double fl ;     // Cholesky flop count for this ordering (each
            // multiply and each add counted once (doesn't count complex
            // flops).

        //----------------------------------------------------------------------
        // ordering parameters:
        //----------------------------------------------------------------------

        double prune_dense ;    // dense row/col control.  Default: 10.
            // Rows/cols with more than max (prune_dense*sqrt(n),16) are
            // removed prior to orderingm and placed last.  If negative,
            // only completely dense rows/cols are removed.  Removing these
            // rows/cols with many entries can speed up the ordering, but
            // removing too many can reduce the ordering quality.
            //
            // For AMD, SYMAMD, and CSYMAMD, this is the only dense row/col
            // parameter.  For COLAMD and CCOLAMD, this parameter controls
            // how dense columns are handled.

        double prune_dense2 ;   // dense row control for COLAMD and CCOLAMD.
            // Default -1.  When computing the Cholesky factorization of AA'
            // rows with more than max(prune_dense2*sqrt(n),16) entries
            // are removed prior to ordering.  If negative, only completely
            // dense rows are removed.

        double nd_oksep ;   // for CHOLMOD's nesdis method. Default 1.
            // A node separator with nsep nodes is discarded if
            // nsep >= nd_oksep*n.

        double other_1 [4] ;    // unused, for future expansion

        size_t nd_small ;   // for CHOLMOD's nesdis method. Default 200.
            // Subgraphs with fewer than nd_small nodes are not partitioned.

        double other_2 [4] ;    // unused, for future expansion

        int aggressive ;    // if true, AMD, COLAMD, SYMAMD, CCOLAMD, and
            // CSYMAMD perform aggresive absorption.  Default: true

        int order_for_lu ;  // Default: false.  If the CHOLMOD analysis/
            // ordering methods are used as an ordering method for an LU
            // factorization, then set this to true.  For use in a Cholesky
            // factorization by CHOLMOD itself, never set this to true.

        int nd_compress ;   // if true, then the graph and subgraphs are
            // compressed before partitioning them in CHOLMOD's nesdis
            // method.  Default: true.

        int nd_camd ;   // if 1, then CHOLMOD's nesdis is followed by
            // CAMD.  If 2: followed by CSYMAMD.  If nd_small is very small,
            // then use 0, which skips CAMD or CSYMAMD.  Default: 1.

        int nd_components ; // CHOLMOD's nesdis can partition a graph and then
            // find that the subgraphs are unconnected.  If true, each of these
            // components is partitioned separately.  If false, the whole
            // subgraph is partitioned.  Default: false.

        int ordering ;  // ordering method to use

        size_t other_3 [4] ;    // unused, for future expansion

    }
    method [CHOLMOD_MAXMETHODS + 1] ;

    int postorder ; // if true, CHOLMOD performs a weighted postordering
        // after its fill-reducing ordering, which improves supernodal
        // amalgamation.  Has no effect on flop count or nnz(L).
        // Default: true.

    int default_nesdis ;    //  If false, then the default ordering strategy
        // when Common->nmethods is zero is to try the user's permutation
        // if given, then AMD, and then METIS if the AMD ordering results in
        // a lot of fill-in.  If true, then nesdis is used instead of METIS.
        // Default: false.

    //--------------------------------------------------------------------------
    // METIS workarounds
    //--------------------------------------------------------------------------

    // These workarounds were put into place for METIS 4.0.1.  They are safe
    // to use with METIS 5.1.0, but they might not longer be necessary.

    double metis_memory ;   // default: 0.  If METIS terminates your
        // program when it runs out of memory, try 2, or higher.
    double metis_dswitch ;  // default: 0.66
    size_t metis_nswitch ;  // default: 3000
        // If a matrix has n > metis_nswitch and a density (nnz(A)/n^2) >
        // metis_dswitch, then METIS is not used.

    //--------------------------------------------------------------------------
    // workspace
    //--------------------------------------------------------------------------

    // This workspace is kept in the CHOLMOD Common object.  cholmod_start
    // sets these arrays to NULL, and cholmod_finish frees them.

    size_t nrow ;       // Flag has size nrow, Head has size nrow+1
    int64_t mark ;      // Flag is cleared if Flag [0..nrow-1] < mark.
    size_t iworksize ;  // size of Iwork, in Ints (int32 or int64).
                        // This is at most 6*nrow + ncol.
    size_t xworkbytes ; // size of Xwork, in bytes.  for update/downdate:
        // maxrank*nrow*sizeof(double or float), 2*nrow*sizeof(double or float)
        // otherwise.  NOTE: in CHOLMOD v4 and earlier, xworkwise was in terms
        // of # of doubles, not # of bytes.

    void *Flag ;    // size nrow.  If this is "cleared" then
        // Flag [i] < mark for all i = 0:nrow-1.  Flag is kept cleared between
        // calls to CHOLMOD.

    void *Head ;    // size nrow+1.  If Head [i] = EMPTY (-1) then that
        // entry is "cleared".  Head is kept cleared between calls to CHOLMOD.

    void *Xwork ;   // a double or float array.  It has size nrow for most
        // routines, or 2*nrow if complex matrices are being handled.
        // It has size 2*nrow for cholmod_rowadd/rowdel, and maxrank*nrow for
        // cholmod_updown, where maxrank is 2, 4, or 8.  Xwork is kept all
        // zero between calls to CHOLMOD.

    void *Iwork ;   // size iworksize integers (int32's or int64's).
        // Uninitialized integer workspace, of size at most 6*nrow+ncol.

    int itype ;     // cholmod_start (for int32's) sets this to CHOLMOD_INT,
        // and cholmod_l_start sets this to CHOLMOD_LONG.  It defines the
        // integer sizes for th Flag, Head, and Iwork arrays, and also
        // defines the integers for all objects created by CHOLMOD.
        // The itype of the Common object must match the function name
        // and all objects passed to it.

    int other_5 ;   // unused: for future expansion

    int no_workspace_reallocate ;   // an internal flag, usually false.
        // This is set true to disable any reallocation of the workspace
        // in the Common object.

    //--------------------------------------------------------------------------
    // statistics
    //--------------------------------------------------------------------------

    int status ;    // status code (0: ok, negative: error, pos: warning
    double fl ;     // flop count from last analysis
    double lnz ;    // nnz(L) from last analysis
    double anz ;    // in last analysis: nnz(tril(A)) or nnz(triu(A)) if A
                    // symmetric, or tril(A*A') if A is unsymmetric.
    double modfl ;  // flop count from last update/downdate/rowadd/rowdel,
                    // not included the flops to revise the solution to Lx=b,
                    // if that was performed.

    size_t malloc_count ;   // # of malloc'd objects not yet freed
    size_t memory_usage ;   // peak memory usage in bytes
    size_t memory_inuse ;   // current memory usage in bytes

    double nrealloc_col ;   // # of column reallocations
    double nrealloc_factor ;// # of factor reallocations due to col. reallocs
    double ndbounds_hit ;   // # of times diagonal modified by dbound

    double rowfacfl ;       // flop count of cholmod_rowfac
    double aatfl ;          // flop count to compute A(:,f)*A(:,f)'

    int called_nd ; // true if last analysis used nesdis or METIS.
    int blas_ok ;   // true if no integer overflow has occured when trying to
        // call the BLAS.  The typical BLAS library uses 32-bit integers for
        // its input parameters, even on a 64-bit platform.  CHOLMOD uses int64
        // in its cholmod_l_* methods, and these must be typecast to the BLAS
        // integer.  If integer overflow occurs, this is set false.

    //--------------------------------------------------------------------------
    // SuiteSparseQR control parameters and statistics
    //--------------------------------------------------------------------------

    // SPQR uses the CHOLMOD Common object for its control and statistics.
    // These parameters are not used by CHOLMOD itself.

    // control parameters:
    double SPQR_grain ;     // task size is >= max (total flops / grain)
    double SPQR_small ;     // task size is >= small
    int SPQR_shrink ;       // controls stack realloc method
    int SPQR_nthreads ;     // number of TBB threads, 0 = auto

    // statistics:
    double SPQR_flopcount ;         // flop count for SPQR
    double SPQR_analyze_time ;      // analysis time in seconds for SPQR
    double SPQR_factorize_time ;    // factorize time in seconds for SPQR
    double SPQR_solve_time ;        // backsolve time in seconds
    double SPQR_flopcount_bound ;   // upper bound on flop count
    double SPQR_tol_used ;          // tolerance used
    double SPQR_norm_E_fro ;        // Frobenius norm of dropped entries

    //--------------------------------------------------------------------------
    // Revised for CHOLMOD v5.0
    //--------------------------------------------------------------------------

    // was size 10 in CHOLMOD v4.2; reduced to 8 in CHOLMOD v5:
    int64_t SPQR_istat [8] ;        // other statistics

    //--------------------------------------------------------------------------
    // Added for CHOLMOD v5.0
    //--------------------------------------------------------------------------

    // These terms have been added to the CHOLMOD Common struct for v5.0, and
    // on most systems they will total 16 bytes.  The preceding term,
    // SPQR_istat, was reduced by 16 bytes, since those last 2 entries were
    // unused in CHOLMOD v4.2.  As a result, the Common struct in v5.0 has the
    // same size as v4.0, and all entries would normally be in the same offset,
    // as well.  This mitigates any changes between v4.0 and v5.0, and may make
    // it easier to upgrade from v4 to v5.

    double nsbounds_hit ;   // # of times diagonal modified by sbound.  This
                    // ought to be int64_t, but ndbounds_hit was double in v4
                    // (see above), so nsbounds_hit is made the same type for
                    // consistency.
    float sbound ;  // Same as dbound, but for single precision factorization.
    float other_6 ; // for future expansion

    //--------------------------------------------------------------------------
    // GPU configuration and statistics
    //--------------------------------------------------------------------------

    int useGPU ; // 1 if GPU is requested for CHOLMOD
                 // 0 if GPU is not requested for CHOLMOD
                 // -1 if the use of the GPU is in CHOLMOD controled by the
                 // CHOLMOD_USE_GPU environment variable.

    size_t maxGpuMemBytes ;     // GPU control for CHOLMOD
    double maxGpuMemFraction ;  // GPU control for CHOLMOD

    // for SPQR:
    size_t gpuMemorySize ;      // Amount of memory in bytes on the GPU
    double gpuKernelTime ;      // Time taken by GPU kernels
    int64_t gpuFlops ;          // Number of flops performed by the GPU
    int gpuNumKernelLaunches ;  // Number of GPU kernel launches

    #ifdef SUITESPARSE_CUDA
        // these three types are pointers defined by CUDA:
        #define CHOLMOD_CUBLAS_HANDLE cublasHandle_t
        #define CHOLMOD_CUDASTREAM    cudaStream_t
        #define CHOLMOD_CUDAEVENT     cudaEvent_t
    #else
        // they are (void *) if CUDA is not in use:
        #define CHOLMOD_CUBLAS_HANDLE void *
        #define CHOLMOD_CUDASTREAM    void *
        #define CHOLMOD_CUDAEVENT     void *
    #endif

    CHOLMOD_CUBLAS_HANDLE cublasHandle ;

    // a set of streams for general use
    CHOLMOD_CUDASTREAM  gpuStream [CHOLMOD_HOST_SUPERNODE_BUFFERS] ;

    CHOLMOD_CUDAEVENT   cublasEventPotrf [3] ;
    CHOLMOD_CUDAEVENT   updateCKernelsComplete ;
    CHOLMOD_CUDAEVENT   updateCBuffersFree [CHOLMOD_HOST_SUPERNODE_BUFFERS] ;

    void *dev_mempool ; // pointer to single allocation of device memory
    size_t dev_mempool_size ;

    void *host_pinned_mempool ; // pointer to single alloc of pinned mem
    size_t host_pinned_mempool_size ;

    size_t devBuffSize ;
    int    ibuffer ;
    double syrkStart ;          // time syrk started

    // run times of the different parts of CHOLMOD (GPU and CPU):
    double cholmod_cpu_gemm_time ;
    double cholmod_cpu_syrk_time ;
    double cholmod_cpu_trsm_time ;
    double cholmod_cpu_potrf_time ;
    double cholmod_gpu_gemm_time ;
    double cholmod_gpu_syrk_time ;
    double cholmod_gpu_trsm_time ;
    double cholmod_gpu_potrf_time ;
    double cholmod_assemble_time ;
    double cholmod_assemble_time2 ;

    // number of times the BLAS are called on the CPU and the GPU:
    size_t cholmod_cpu_gemm_calls ;
    size_t cholmod_cpu_syrk_calls ;
    size_t cholmod_cpu_trsm_calls ;
    size_t cholmod_cpu_potrf_calls ;
    size_t cholmod_gpu_gemm_calls ;
    size_t cholmod_gpu_syrk_calls ;
    size_t cholmod_gpu_trsm_calls ;
    size_t cholmod_gpu_potrf_calls ;

    double chunk ;      // chunksize for computing # of OpenMP threads to use.
        // Given nwork work to do, # of threads is
        // max (1, min (floor (work / chunk), nthreads_max))

    int nthreads_max ; // max # of OpenMP threads to use in CHOLMOD.
        // Defaults to SUITESPARSE_OPENMP_MAX_THREADS.

} cholmod_common ;

// size_t BLAS statistcs in Common:
#define CHOLMOD_CPU_GEMM_CALLS      cholmod_cpu_gemm_calls
#define CHOLMOD_CPU_SYRK_CALLS      cholmod_cpu_syrk_calls
#define CHOLMOD_CPU_TRSM_CALLS      cholmod_cpu_trsm_calls
#define CHOLMOD_CPU_POTRF_CALLS     cholmod_cpu_potrf_calls
#define CHOLMOD_GPU_GEMM_CALLS      cholmod_gpu_gemm_calls
#define CHOLMOD_GPU_SYRK_CALLS      cholmod_gpu_syrk_calls
#define CHOLMOD_GPU_TRSM_CALLS      cholmod_gpu_trsm_calls
#define CHOLMOD_GPU_POTRF_CALLS     cholmod_gpu_potrf_calls

// double BLAS statistics in Common:
#define CHOLMOD_CPU_GEMM_TIME       cholmod_cpu_gemm_time
#define CHOLMOD_CPU_SYRK_TIME       cholmod_cpu_syrk_time
#define CHOLMOD_CPU_TRSM_TIME       cholmod_cpu_trsm_time
#define CHOLMOD_CPU_POTRF_TIME      cholmod_cpu_potrf_time
#define CHOLMOD_GPU_GEMM_TIME       cholmod_gpu_gemm_time
#define CHOLMOD_GPU_SYRK_TIME       cholmod_gpu_syrk_time
#define CHOLMOD_GPU_TRSM_TIME       cholmod_gpu_trsm_time
#define CHOLMOD_GPU_POTRF_TIME      cholmod_gpu_potrf_time
#define CHOLMOD_ASSEMBLE_TIME       cholmod_assemble_time
#define CHOLMOD_ASSEMBLE_TIME2      cholmod_assemble_time2

// for supernodal analysis:
#define CHOLMOD_ANALYZE_FOR_SPQR     0
#define CHOLMOD_ANALYZE_FOR_CHOLESKY 1
#define CHOLMOD_ANALYZE_FOR_SPQRGPU  2

//------------------------------------------------------------------------------
// cholmod_start:  first call to CHOLMOD
//------------------------------------------------------------------------------

int cholmod_start (cholmod_common *Common) ;
int cholmod_l_start (cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_finish:  last call to CHOLMOD
//------------------------------------------------------------------------------

int cholmod_finish (cholmod_common *Common) ;
int cholmod_l_finish (cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_defaults: set default parameters
//------------------------------------------------------------------------------

int cholmod_defaults (cholmod_common *Common) ;
int cholmod_l_defaults (cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_maxrank:  return valid maximum rank for update/downdate
//------------------------------------------------------------------------------

size_t cholmod_maxrank      // return validated Common->maxrank
(
    size_t n,               // # of rows of L and A
    cholmod_common *Common
) ;
size_t cholmod_l_maxrank (size_t, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_allocate_work: allocate workspace in Common
//------------------------------------------------------------------------------

// This method always allocates Xwork as double, for backward compatibility
// with CHOLMOD v4 and earlier.  See cholmod_alloc_work for CHOLMOD v5.

int cholmod_allocate_work
(
    size_t nrow,        // size of Common->Flag (nrow int32's)
                        // and Common->Head (nrow+1 int32's)
    size_t iworksize,   // size of Common->Iwork (# of int32's)
    size_t xworksize,   // size of Common->Xwork (# of double's)
    cholmod_common *Common
) ;
int cholmod_l_allocate_work (size_t, size_t, size_t, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_alloc_work: allocate workspace in Common
//------------------------------------------------------------------------------

// Added for CHOLMOD v5: allocates Xwork as either double or single.

int cholmod_alloc_work
(
    size_t nrow,        // size of Common->Flag (nrow int32's)
                        // and Common->Head (nrow+1 int32's)
    size_t iworksize,   // size of Common->Iwork (# of int32's)
    size_t xworksize,   // size of Common->Xwork (# of entries)
    int dtype,          // CHOLMOD_DOUBLE or CHOLMOD_SINGLE
    cholmod_common *Common
) ;
int cholmod_l_alloc_work (size_t, size_t, size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_free_work:  free workspace in Common
//------------------------------------------------------------------------------

int cholmod_free_work (cholmod_common *Common) ;
int cholmod_l_free_work (cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_clear_flag:  clear Flag workspace in Common
//------------------------------------------------------------------------------

// This macro is deprecated; do not use it:
#define CHOLMOD_CLEAR_FLAG(Common)                      \
{                                                       \
    Common->mark++ ;                                    \
    if (Common->mark <= 0 || Common->mark >= INT32_MAX) \
    {                                                   \
        Common->mark = EMPTY ;                          \
        CHOLMOD (clear_flag) (Common) ;                 \
    }                                                   \
}

int64_t cholmod_clear_flag (cholmod_common *Common) ;
int64_t cholmod_l_clear_flag (cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_error:  called when CHOLMOD encounters an error
//------------------------------------------------------------------------------

int cholmod_error
(
    int status,             // Common->status
    const char *file,       // source file where error occurred
    int line,               // line number where error occurred
    const char *message,    // error message to print
    cholmod_common *Common
) ;
int cholmod_l_error (int, const char *, int, const char *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_dbound and cholmod_sbound:  for internal use in CHOLMOD only
//------------------------------------------------------------------------------

// These were once documented functions but no are no longer meant to be used
// by the user application.  They remain here for backward compatibility.

double cholmod_dbound   (double, cholmod_common *) ;
double cholmod_l_dbound (double, cholmod_common *) ;
float  cholmod_sbound   (float,  cholmod_common *) ;
float  cholmod_l_sbound (float,  cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_hypot:  compute sqrt (x*x + y*y) accurately
//------------------------------------------------------------------------------

double cholmod_hypot (double x, double y) ;
double cholmod_l_hypot (double, double) ;

//------------------------------------------------------------------------------
// cholmod_divcomplex:  complex division, c = a/b
//------------------------------------------------------------------------------

int cholmod_divcomplex          // return 1 if divide-by-zero, 0 if OK
(
    double ar, double ai,       // a (real, imaginary)
    double br, double bi,       // b (real, imaginary)
    double *cr, double *ci      // c (real, imaginary)
) ;
int cholmod_l_divcomplex (double, double, double, double, double *, double *) ;

//==============================================================================
// cholmod_sparse: a sparse matrix in compressed-column (CSC) form
//==============================================================================

typedef struct cholmod_sparse_struct
{
    size_t nrow ;   // # of rows of the matrix
    size_t ncol ;   // # of colums of the matrix
    size_t nzmax ;  // max # of entries that can be held in the matrix

    // int32_t or int64_t arrays:
    void *p ;   // A->p [0..ncol], column "pointers" of the CSC matrix
    void *i ;   // A->i [0..nzmax-1], the row indices

    // for unpacked matrices only:
    void *nz ;  // A->nz [0..ncol-1], is the # of nonzeros in each col.
        // This is NULL for a "packed" matrix (conventional CSC).
        // For a packed matrix, the jth column is held in A->i and A->x in
        // postions A->p [j] to A->p [j+1]-1, with no gaps between columns.
        // For an "unpacked" matrix, there can be gaps between columns, so
        // the jth columns appears in positions A-p [j] to
        // A->p [j] + A->nz [j] - 1.

    // double or float arrays:
    void *x ;   // size nzmax or 2*nzmax, or NULL
    void *z ;   // size nzmax, or NULL

    int stype ; // A->stype defines what parts of the matrix is held:
        // 0:  the matrix is unsymmetric with both lower and upper parts stored.
        // >0: the matrix is square and symmetric, with just the upper
        //     triangular part stored.
        // <0: the matrix is square and symmetric, with just the lower
        //     triangular part stored.

    int itype ; // A->itype defines the integers used for A->p, A->i, and A->nz.
        // if CHOLMOD_INT, these arrays are all of type int32_t.
        // if CHOLMOD_LONG, these arrays are all of type int64_t.
    int xtype ;     // pattern, real, complex, or zomplex
    int dtype ;     // x and z are double or single
    int sorted ;    // true if columns are sorted, false otherwise
    int packed ;    // true if packed (A->nz ignored), false if unpacked

} cholmod_sparse ;

//------------------------------------------------------------------------------
// cholmod_allocate_sparse:  allocate a sparse matrix
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_allocate_sparse
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t nzmax,   // max # of entries the matrix can hold
    int sorted,     // true if columns are sorted
    int packed,     // true if A is be packed (A->nz NULL), false if unpacked
    int stype,      // the stype of the matrix (unsym, tril, or triu)
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_allocate_sparse (size_t, size_t, size_t, int, int,
    int, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_free_sparse:  free a sparse matrix
//------------------------------------------------------------------------------

int cholmod_free_sparse
(
    cholmod_sparse **A,         // handle of sparse matrix to free
    cholmod_common *Common
) ;
int cholmod_l_free_sparse (cholmod_sparse **, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_reallocate_sparse: change max # of entries in a sparse matrix
//------------------------------------------------------------------------------

int cholmod_reallocate_sparse
(
    size_t nznew,       // new max # of nonzeros the sparse matrix can hold
    cholmod_sparse *A,  // sparse matrix to reallocate
    cholmod_common *Common
) ;
int cholmod_l_reallocate_sparse (size_t, cholmod_sparse *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_nnz: # of entries in a sparse matrix
//------------------------------------------------------------------------------

int64_t cholmod_nnz             // return # of entries in the sparse matrix
(
    cholmod_sparse *A,          // sparse matrix to query
    cholmod_common *Common
) ;
int64_t cholmod_l_nnz (cholmod_sparse *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_speye:  sparse identity matrix (possibly rectangular)
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_speye
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_speye (size_t, size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_spzeros: sparse matrix with no entries
//------------------------------------------------------------------------------

// Identical to cholmod_allocate_sparse, with packed = true, sorted = true,
// and stype = 0.

cholmod_sparse *cholmod_spzeros     // return a sparse matrix with no entries
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t nzmax,   // max # of entries the matrix can hold
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_spzeros (size_t, size_t, size_t, int,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_transpose: transpose a sparse matrix
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_transpose       // return new sparse matrix C
(
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // <= 0: pattern (with diag)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_transpose (cholmod_sparse *, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_transpose_unsym:  transpose an unsymmetric sparse matrix
//------------------------------------------------------------------------------

// Compute C = A', A (:,f)', or A (p,f)', where A is unsymmetric and C is
// already allocated.  See cholmod_transpose for a routine with a simpler
// interface.

int cholmod_transpose_unsym
(
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // <= 0: pattern (with diag)
    int32_t *Perm,      // permutation for C=A(p,f)', or NULL
    int32_t *fset,      // a list of column indices in range 0:A->ncol-1
    size_t fsize,       // # of entries in fset
    cholmod_sparse *C,  // output matrix, must be allocated on input
    cholmod_common *Common
) ;
int cholmod_l_transpose_unsym (cholmod_sparse *, int, int64_t *, int64_t *,
    size_t, cholmod_sparse *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_transpose_sym:  symmetric permuted transpose
//------------------------------------------------------------------------------

// C = A' or C = A(p,p)' where A and C are both symmetric and C is already
// allocated.  See cholmod_transpose or cholmod_ptranspose for a routine with
// a simpler interface.

int cholmod_transpose_sym
(
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // <= 0: pattern (with diag)
    int32_t *Perm,      // permutation for C=A(p,p)', or NULL
    cholmod_sparse *C,  // output matrix, must be allocated on input
    cholmod_common *Common
) ;
int cholmod_l_transpose_sym (cholmod_sparse *, int, int64_t *, cholmod_sparse *,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_ptranspose: C = A', A(:,f)', A(p,p)', or A(p,f)'
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_ptranspose      // return new sparse matrix C
(
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // <= 0: pattern (with diag)
    int32_t *Perm,      // permutation for C=A(p,f)', or NULL
    int32_t *fset,      // a list of column indices in range 0:A->ncol-1
    size_t fsize,       // # of entries in fset
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_ptranspose (cholmod_sparse *, int, int64_t *,
    int64_t *, size_t, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_sort: sort the indices of a sparse matrix 
//------------------------------------------------------------------------------

int cholmod_sort
(
    cholmod_sparse *A,      // input/output matrix to sort
    cholmod_common *Common
) ;
int cholmod_l_sort (cholmod_sparse *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_band_nnz: # of entries within a band of a sparse matrix
//------------------------------------------------------------------------------

int64_t cholmod_band_nnz    // return # of entries in a band (-1 if error)
(
    cholmod_sparse *A,      // matrix to examine
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    bool ignore_diag,       // if true, exclude any diagonal entries
    cholmod_common *Common
) ;
int64_t cholmod_l_band_nnz (cholmod_sparse *, int64_t, int64_t, bool,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_band:  C = tril (triu (A,k1), k2)
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_band    // return a new matrix C
(
    cholmod_sparse *A,      // input matrix
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    int mode,               // >0: numerical, 0: pattern, <0: pattern (no diag)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_band (cholmod_sparse *, int64_t, int64_t, int,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_band_inplace:  A = tril (triu (A,k1), k2) */
//------------------------------------------------------------------------------

int cholmod_band_inplace
(
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    int mode,               // >0: numerical, 0: pattern, <0: pattern (no diag)
    cholmod_sparse *A,      // input/output matrix
    cholmod_common *Common
) ;
int cholmod_l_band_inplace (int64_t, int64_t, int, cholmod_sparse *,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_aat: C = A*A' or A(:,f)*A(:,f)'
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_aat     // return sparse matrix C
(
    cholmod_sparse *A,  // input matrix
    int32_t *fset,      // a list of column indices in range 0:A->ncol-1
    size_t fsize,       // # of entries in fset
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // 0: pattern (with diag), -1: pattern (remove diag),
                        // -2: pattern (remove diag; add ~50% extra space in C)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_aat (cholmod_sparse *, int64_t *, size_t, int,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_copy_sparse:  C = A, create an exact copy of a sparse matrix
//------------------------------------------------------------------------------

// Creates an exact copy of a sparse matrix.  For making a copy with a change
// of stype and/or copying the pattern of a numerical matrix, see cholmod_copy.
// For changing the xtype and/or dtype, see cholmod_sparse_xtype.

cholmod_sparse *cholmod_copy_sparse  // return new sparse matrix
(
    cholmod_sparse *A,     // sparse matrix to copy
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_copy_sparse (cholmod_sparse *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_copy:  C = A, with possible change of stype
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_copy        // return new sparse matrix
(
    cholmod_sparse *A,  // input matrix, not modified
    int stype,          // stype of C
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // 0: pattern (with diag), -1: pattern (remove diag),
                        // -2: pattern (remove diag; add ~50% extra space in C)
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_copy (cholmod_sparse *, int, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_add: C = alpha*A + beta*B
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_add     // return C = alpha*A + beta*B
(
    cholmod_sparse *A,  // input matrix
    cholmod_sparse *B,  // input matrix
    double alpha [2],   // scale factor for A (two entires used if complex)
    double beta [2],    // scale factor for A (two entires used if complex)
    int values,         // if TRUE compute the numerical values of C
    int sorted,         // ignored; C is now always returned as sorted
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_add (cholmod_sparse *, cholmod_sparse *, double *,
    double *, int, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_sparse_xtype: change the xtype and/or dtype of a sparse matrix
//------------------------------------------------------------------------------

int cholmod_sparse_xtype
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_sparse *A,  // sparse matrix to change
    cholmod_common *Common
) ;
int cholmod_l_sparse_xtype (int, cholmod_sparse *, cholmod_common *) ;

//==============================================================================
// cholmod_factor: symbolic or numeric factorization (simplicial or supernodal)
//==============================================================================

typedef struct cholmod_factor_struct
{
    size_t n ;      // L is n-by-n

    size_t minor ;  // If the factorization failed because of numerical issues
        // (the matrix being factorized is found to be singular or not positive
        // definte), then L->minor is the column at which it failed.  L->minor
        // = n means the factorization was successful.

    //--------------------------------------------------------------------------
    // symbolic ordering and analysis
    //--------------------------------------------------------------------------

    void *Perm ;    // int32/int64, size n, fill-reducing ordering
    void *ColCount ;// int32/int64, size n, # entries in each column of L
    void *IPerm ;   // int32/int64, size n, created by cholmod_solve2;
                    // containing the inverse of L->Perm

    //--------------------------------------------------------------------------
    // simplicial factorization (not supernodal)
    //--------------------------------------------------------------------------

    size_t nzmax ;  // # of entries that L->i, L->x, and L->z can hold

    void *p ;       // int32/int64, size n+1, column pointers
    void *i ;       // int32/int64, size nzmax, row indices
    void *x ;       // float/double, size nzmax or 2*nzmax, numerical values
    void *z ;       // float/double, size nzmax or empty, imaginary values
    void *nz ;      // int32/int64, size ncol, # of entries in each column

    // The row indices of L(:,j) are held in
    // L->i [L->p [j] ... L->p [j] + L->nz [j] - 1].

    // The numeical values of L(:,j) are held in the same positions in L->x
    // (and L->z if L is zomplex)

    // L->next and L->prev hold a link list of columns of L, that tracks the
    // order they appear in the arrays L->i, L->x, and L->z.  The head and tail
    // of the list is n+1 and n, respectively.

    void *next ;    // int32/int64, size n+2
    void *prev ;    // int32/int64, size n+2

    //--------------------------------------------------------------------------
    // supernodal factorization (not simplicial)
    //--------------------------------------------------------------------------

    // L->x is shared with the simplicial structure above.  L->z is not used
    // for the supernodal case since a supernodal factor cannot be zomplex.

    size_t nsuper ;     // # of supernodes
    size_t ssize ;      // # of integers in L->s
    size_t xsize ;      // # of entries in L->x
    size_t maxcsize ;   // size of largest update matrix
    size_t maxesize ;   // max # of rows in supernodes, excl. triangular part

    // the following are int32/int64 and are size nsuper+1:
    void *super ;       // first column in each supernode
    void *pi ;          // index into L->s for integer part of a supernode
    void *px ;          // index into L->x for numeric part of a supernode

    void *s ;           // int32/int64, ssize, integer part of supernodes

    //--------------------------------------------------------------------------
    // type of the factorization
    //--------------------------------------------------------------------------

    int ordering ;      // the fill-reducing method used (CHOLMOD_NATURAL,
        // CHOLMOD_GIVEN, CHOLMOD_AMD, CHOLMOD_METIS, CHOLMOD_NESDIS,
        // CHOLMOD_COLAMD, or CHOLMOD_POSTORDERED).

    int is_ll ;         // true: an LL' factorization; false: LDL' instead
    int is_super ;      // true: supernodal; false: simplicial
    int is_monotonic ;  // true: columns appear in order 0 to n-1 in L, for a
                        // simplicial factorization only

    // Two boolean values above (is_ll, is_super) and L->xtype (pattern or
    // otherwise, define eight types of factorizations, but only 6 are used:

    // If L->xtype is CHOLMOD_PATTERN, then L is a symbolic factor:
    //
    // simplicial LDL': (is_ll false, is_super false).  Nothing is present
    //      except Perm and ColCount.
    //
    // simplicial LL': (is_ll true, is_super false).  Identical to the
    //      simplicial LDL', except for the is_ll flag.
    //
    // supernodal LL': (is_ll true, is_super true).  A supernodal symbolic
    //      factorization.  The simplicial symbolic information is present
    //      (Perm and ColCount), as is all of the supernodal factorization
    //      except for the numerical values (x and z).
    //
    // If L->xtype is CHOLMOD_REAL, CHOLMOD_COMPLEX, or CHOLMOD_ZOMPLEX,
    // then L is a numeric factor:
    //
    //
    // simplicial LDL':  (is_ll false, is_super false).  Stored in compressed
    //      column form, using the simplicial components above (nzmax, p, i,
    //      x, z, nz, next, and prev).  The unit diagonal of L is not stored,
    //      and D is stored in its place.  There are no supernodes.
    //
    // simplicial LL': (is_ll true, is_super false).  Uses the same storage
    //      scheme as the simplicial LDL', except that D does not appear.
    //      The first entry of each column of L is the diagonal entry of
    //      that column of L.
    //
    // supernodal LL': (is_ll true, is_super true).  A supernodal factor,
    //      using the supernodal components described above (nsuper, ssize,
    //      xsize, maxcsize, maxesize, super, pi, px, s, x, and z).
    //      A supernodal factorization is never zomplex.

    int itype ; // integer type for L->Perm, L->ColCount, L->p, L->i, L->nz,
                // L->next, L->prev, L->super, L->pi, L->px, and L->s.
                // These are all int32 if L->itype is CHOLMOD_INT, or all int64
                // if L->itype is CHOLMOD_LONG.

    int xtype ; // pattern, real, complex, or zomplex
    int dtype ; // x and z are double or single

    int useGPU; // if true, symbolic factorization allows for use of the GPU

} cholmod_factor ;

//------------------------------------------------------------------------------
// cholmod_allocate_factor: allocate a numerical factor
//------------------------------------------------------------------------------

// L is returned as double precision

cholmod_factor *cholmod_allocate_factor         // return the new factor L
(
    size_t n,               // L is factorization of an n-by-n matrix
    cholmod_common *Common
) ;
cholmod_factor *cholmod_l_allocate_factor (size_t, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_alloc_factor: allocate a numerical factor (double or single)
//------------------------------------------------------------------------------

cholmod_factor *cholmod_alloc_factor        // return the new factor L
(
    size_t n,               // L is factorization of an n-by-n matrix
    int dtype,              // CHOLMOD_SINGLE or CHOLMOD_DOUBLE
    cholmod_common *Common
) ;
cholmod_factor *cholmod_l_alloc_factor (size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_free_factor:  free a factor
//------------------------------------------------------------------------------

int cholmod_free_factor
(
    cholmod_factor **L,         // handle of sparse factorization to free
    cholmod_common *Common
) ;
int cholmod_l_free_factor (cholmod_factor **, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_reallocate_factor:  change the # entries in a factor
//------------------------------------------------------------------------------

int cholmod_reallocate_factor
(
    size_t nznew,       // new max # of nonzeros the factor matrix can hold
    cholmod_factor *L,  // factor to reallocate
    cholmod_common *Common
) ;
int cholmod_l_reallocate_factor (size_t, cholmod_factor *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_change_factor:  change the type of factor (e.g., LDL' to LL')
//------------------------------------------------------------------------------

int cholmod_change_factor
(
    int to_xtype,       // CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    int to_ll,          // if true: convert to LL'; else to LDL'
    int to_super,       // if true: convert to supernodal; else to simplicial
    int to_packed,      // if true: pack simplicial columns' else: do not pack
    int to_monotonic,   // if true, put simplicial columns in order
    cholmod_factor *L,  // factor to change.
    cholmod_common *Common
) ;
int cholmod_l_change_factor (int, int, int, int, int, cholmod_factor *,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_pack_factor:  pack a factor
//------------------------------------------------------------------------------

// Removes all slack space from all columns in a factor.

int cholmod_pack_factor
(
    cholmod_factor *L,      // factor to pack
    cholmod_common *Common
) ;
int cholmod_l_pack_factor (cholmod_factor *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_reallocate_column:  reallocate a single column L(:,j)
//------------------------------------------------------------------------------

int cholmod_reallocate_column
(
    size_t j,                   // reallocate L(:,j)
    size_t need,                // space in L(:,j) for this # of entries
    cholmod_factor *L,          // L factor modified, L(:,j) resized
    cholmod_common *Common
) ;
int cholmod_l_reallocate_column (size_t, size_t, cholmod_factor *,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_factor_to_sparse:  create a sparse matrix copy of a factor
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_factor_to_sparse    // return a new sparse matrix
(
    cholmod_factor *L,  // input: factor to convert; output: L is converted
                        // to a simplicial symbolic factor
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_factor_to_sparse (cholmod_factor *,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_copy_factor:  create a copy of a factor
//------------------------------------------------------------------------------

cholmod_factor *cholmod_copy_factor     // return a copy of the factor
(
    cholmod_factor *L,      // factor to copy (not modified)
    cholmod_common *Common
) ;
cholmod_factor *cholmod_l_copy_factor (cholmod_factor *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_factor_xtype: change the xtype and/or dtype of a factor
//------------------------------------------------------------------------------

int cholmod_factor_xtype
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_factor *L,  // factor to change
    cholmod_common *Common
) ;
int cholmod_l_factor_xtype (int, cholmod_factor *, cholmod_common *) ;

//==============================================================================
// cholmod_dense: a dense matrix, held by column
//==============================================================================

typedef struct cholmod_dense_struct
{
    size_t nrow ;       // the matrix is nrow-by-ncol
    size_t ncol ;
    size_t nzmax ;      // maximum number of entries in the matrix
    size_t d ;          // leading dimension (d >= nrow must hold)
    void *x ;           // size nzmax or 2*nzmax, if present
    void *z ;           // size nzmax, if present
    int xtype ;         // pattern, real, complex, or zomplex
    int dtype ;         // x and z double or single

} cholmod_dense ;

//------------------------------------------------------------------------------
// cholmod_allocate_dense: allocate a dense matrix (contents not initialized)
//------------------------------------------------------------------------------

cholmod_dense *cholmod_allocate_dense
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t d,       // leading dimension
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_allocate_dense (size_t, size_t, size_t, int,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_zeros: allocate a dense matrix and set it to zero
//------------------------------------------------------------------------------

cholmod_dense *cholmod_zeros
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_zeros (size_t, size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_ones: allocate a dense matrix of all 1's
//------------------------------------------------------------------------------

cholmod_dense *cholmod_ones
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_ones (size_t, size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_eye: allocate a dense identity matrix
//------------------------------------------------------------------------------

cholmod_dense *cholmod_eye      // return a dense identity matrix
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_eye (size_t, size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_free_dense:  free a dense matrix
//------------------------------------------------------------------------------

int cholmod_free_dense
(
    cholmod_dense **X,          // handle of dense matrix to free
    cholmod_common *Common
) ;
int cholmod_l_free_dense (cholmod_dense **, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_ensure_dense:  ensure a dense matrix has a given size and type
//------------------------------------------------------------------------------

cholmod_dense *cholmod_ensure_dense
(
    cholmod_dense **X,  // matrix to resize as needed (*X may be NULL)
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t d,       // leading dimension
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_ensure_dense (cholmod_dense **, size_t, size_t,
    size_t, int, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_sparse_to_dense:  create a dense matrix copy of a sparse matrix
//------------------------------------------------------------------------------

cholmod_dense *cholmod_sparse_to_dense      // return a dense matrix
(
    cholmod_sparse *A,      // input matrix
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_sparse_to_dense (cholmod_sparse *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_dense_nnz: count # of nonzeros in a dense matrix
//------------------------------------------------------------------------------

int64_t cholmod_dense_nnz       // return # of entries in the dense matrix
(
    cholmod_dense *X,       // input matrix
    cholmod_common *Common
) ;
int64_t cholmod_l_dense_nnz (cholmod_dense *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_dense_to_sparse:  create a sparse matrix copy of a dense matrix
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_dense_to_sparse         // return a sparse matrix C
(
    cholmod_dense *X,       // input matrix
    int values,             // if true, copy the values; if false, C is pattern
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_dense_to_sparse (cholmod_dense *, int,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_copy_dense:  create a copy of a dense matrix
//------------------------------------------------------------------------------

cholmod_dense *cholmod_copy_dense   // returns new dense matrix
(
    cholmod_dense *X,   // input dense matrix
    cholmod_common *Common
) ;
cholmod_dense *cholmod_l_copy_dense (cholmod_dense *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_copy_dense2:  copy a dense matrix (pre-allocated)
//------------------------------------------------------------------------------

int cholmod_copy_dense2
(
    cholmod_dense *X,   // input dense matrix
    cholmod_dense *Y,   // output dense matrix (already allocated on input)
    cholmod_common *Common
) ;
int cholmod_l_copy_dense2 (cholmod_dense *, cholmod_dense *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_dense_xtype: change the xtype and/or dtype of a dense matrix
//------------------------------------------------------------------------------

int cholmod_dense_xtype
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_dense *X,   // dense matrix to change
    cholmod_common *Common
) ;
int cholmod_l_dense_xtype (int, cholmod_dense *, cholmod_common *) ;

//=============================================================================
// cholmod_triplet: a sparse matrix in triplet form
//=============================================================================

typedef struct cholmod_triplet_struct
{
    size_t nrow ;   // # of rows of the matrix
    size_t ncol ;   // # of colums of the matrix
    size_t nzmax ;  // max # of entries that can be held in the matrix
    size_t nnz ;    // current # of entries can be held in the matrix

    // int32 or int64 arrays (depending on T->itype)
    void *i ;   // i [0..nzmax-1], the row indices
    void *j ;   // j [0..nzmax-1], the column indices

    // double or float arrays:
    void *x ;   // size nzmax or 2*nzmax, or NULL
    void *z ;   // size nzmax, or NULL

    int stype ; // T->stype defines what parts of the matrix is held:
        // 0:  the matrix is unsymmetric with both lower and upper parts stored.
        // >0: the matrix is square and symmetric, where entries in the lower
        //      triangular part are transposed and placed in the upper
        //      triangular part of A if T is converted into a sparse matrix A.
        // <0: the matrix is square and symmetric, where entries in the upper
        //      triangular part are transposed and placed in the lower
        //      triangular part of A if T is converted into a sparse matrix A.
        //
        // Note that A->stype (for a sparse matrix) and T->stype (for a
        // triplet matrix) are handled differently.  In a triplet matrix T,
        // no entry is ever ignored.  For a sparse matrix A, if A->stype < 0
        // or A->stype > 0, then entries not in the correct triangular part
        // are ignored.

    int itype ; // T->itype defines the integers used for T->i and T->j.
        // if CHOLMOD_INT, these arrays are all of type int32_t.
        // if CHOLMOD_LONG, these arrays are all of type int64_t.
    int xtype ;     // pattern, real, complex, or zomplex
    int dtype ;     // x and z are double or single

} cholmod_triplet ;

//------------------------------------------------------------------------------
// cholmod_allocate_triplet: allocate a triplet matrix
//------------------------------------------------------------------------------

cholmod_triplet *cholmod_allocate_triplet       // return triplet matrix T
(
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t nzmax,   // max # of entries the matrix can hold
    int stype,      // the stype of the matrix (unsym, tril, or triu)
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
) ;
cholmod_triplet *cholmod_l_allocate_triplet (size_t, size_t, size_t, int, int,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_free_triplet: free a triplet matrix
//------------------------------------------------------------------------------

int cholmod_free_triplet
(
    cholmod_triplet **T,        // handle of triplet matrix to free
    cholmod_common *Common
) ;
int cholmod_l_free_triplet (cholmod_triplet **, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_reallocate_triplet: change max # of entries in a triplet matrix
//------------------------------------------------------------------------------

int cholmod_reallocate_triplet
(
    size_t nznew,       // new max # of nonzeros the triplet matrix can hold
    cholmod_triplet *T, // triplet matrix to reallocate
    cholmod_common *Common
) ;
int cholmod_l_reallocate_triplet (size_t, cholmod_triplet *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_sparse_to_triplet:  create a triplet matrix copy of a sparse matrix
//------------------------------------------------------------------------------

cholmod_triplet *cholmod_sparse_to_triplet
(
    cholmod_sparse *A,      // matrix to copy into triplet form T
    cholmod_common *Common
) ;
cholmod_triplet *cholmod_l_sparse_to_triplet (cholmod_sparse *,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_triplet_to_sparse: create a sparse matrix from of triplet matrix
//------------------------------------------------------------------------------

cholmod_sparse *cholmod_triplet_to_sparse       // return sparse matrix A
(
    cholmod_triplet *T,     // input triplet matrix
    size_t nzmax,           // allocate space for max(nzmax,nnz(A)) entries
    cholmod_common *Common
) ;
cholmod_sparse *cholmod_l_triplet_to_sparse (cholmod_triplet *, size_t,
    cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_copy_triplet: copy a triplet matrix
//------------------------------------------------------------------------------

cholmod_triplet *cholmod_copy_triplet   // return new triplet matrix
(
    cholmod_triplet *T,     // triplet matrix to copy
    cholmod_common *Common
) ;
cholmod_triplet *cholmod_l_copy_triplet (cholmod_triplet *, cholmod_common *) ;

//------------------------------------------------------------------------------
// cholmod_triplet_xtype: change the xtype and/or dtype of a triplet matrix
//------------------------------------------------------------------------------

int cholmod_triplet_xtype
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_triplet *T, // triplet matrix to change
    cholmod_common *Common
) ;
int cholmod_l_triplet_xtype (int, cholmod_triplet *, cholmod_common *) ;

//==============================================================================
// memory allocation
//==============================================================================

// These methods act like malloc/calloc/realloc/free, with some differences.
// They are simple wrappers around the memory management functions in
// SuiteSparse_config.  cholmod_malloc and cholmod_calloc have the same
// signature, unlike malloc and calloc.  If cholmod_free is given a NULL
// pointer, it safely does nothing.  cholmod_free must be passed the size of
// the object being freed, but that is just to keep track of memory usage
// statistics. cholmod_realloc does not return NULL if it fails; instead, it
// returns the pointer to the unmodified block of memory.

void *cholmod_malloc    // return pointer to newly allocated memory
(
    size_t n,           // number of items
    size_t size,        // size of each item
    cholmod_common *Common
) ;
void *cholmod_l_malloc (size_t, size_t, cholmod_common *) ;

void *cholmod_calloc    // return pointer to newly allocated memory
(
    size_t n,           // number of items
    size_t size,        // size of each item
    cholmod_common *Common
) ;
void *cholmod_l_calloc (size_t, size_t, cholmod_common *) ;

void *cholmod_free      // returns NULL to simplify its usage
(
    size_t n,           // number of items
    size_t size,        // size of each item
    void *p,            // memory to free
    cholmod_common *Common
) ;
void *cholmod_l_free (size_t, size_t, void *, cholmod_common *) ;

void *cholmod_realloc   // return newly reallocated block of memory
(
    size_t nnew,        // # of items in newly reallocate memory
    size_t size,        // size of each item
    void *p,            // pointer to memory to reallocate (may be NULL)
    size_t *n,          // # of items in p on input; nnew on output if success
    cholmod_common *Common
) ;
void *cholmod_l_realloc (size_t, size_t, void *, size_t *, cholmod_common *) ;

int cholmod_realloc_multiple    // returns true if successful, false otherwise
(
    size_t nnew,    // # of items in newly reallocate memory
    int nint,       // 0: do not allocate I_block or J_block, 1: just I_block,
                    // 2: both I_block and J_block
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    // input/output:
    void **I_block,       // integer block of memory (int32_t or int64_t)
    void **J_block,       // integer block of memory (int32_t or int64_t)
    void **X_block,       // real or complex, double or single, block
    void **Z_block,       // zomplex only: double or single block
    size_t *n,      // current size of I_block, J_block, X_block, and/or Z_block
                    // on input,  changed to nnew on output, if successful
    cholmod_common *Common
) ;
int cholmod_l_realloc_multiple (size_t, int, int, void **, void **, void **,
    void **, size_t *, cholmod_common *) ;

//==============================================================================
// numerical comparisons
//==============================================================================

// These macros were different on Windows for older versions of CHOLMOD.
// They are no longer needed but are kept for backward compatibility.

#define CHOLMOD_IS_NAN(x)       isnan (x)
#define CHOLMOD_IS_ZERO(x)      ((x) == 0.)
#define CHOLMOD_IS_NONZERO(x)   ((x) != 0.)
#define CHOLMOD_IS_LT_ZERO(x)   ((x) < 0.)
#define CHOLMOD_IS_GT_ZERO(x)   ((x) > 0.)
#define CHOLMOD_IS_LE_ZERO(x)   ((x) <= 0.)

#endif

//==============================================================================
// CHOLMOD:Check Module
//==============================================================================

#ifndef NCHECK

/* Routines that check and print the 5 basic data types in CHOLMOD, and 3 kinds
 * of integer vectors (subset, perm, and parent), and read in matrices from a
 * file:
 *
 * cholmod_check_common	    check/print the Common object
 * cholmod_print_common
 *
 * cholmod_check_sparse	    check/print a sparse matrix in column-oriented form
 * cholmod_print_sparse
 *
 * cholmod_check_dense	    check/print a dense matrix
 * cholmod_print_dense
 *
 * cholmod_check_factor	    check/print a Cholesky factorization
 * cholmod_print_factor
 *
 * cholmod_check_triplet    check/print a sparse matrix in triplet form
 * cholmod_print_triplet
 *
 * cholmod_check_subset	    check/print a subset (integer vector in given range)
 * cholmod_print_subset
 *
 * cholmod_check_perm	    check/print a permutation (an integer vector)
 * cholmod_print_perm
 *
 * cholmod_check_parent	    check/print an elimination tree (an integer vector)
 * cholmod_print_parent
 *
 * cholmod_read_triplet	    read a matrix in triplet form (any Matrix Market
 *			    "coordinate" format, or a generic triplet format).
 *
 * cholmod_read_sparse	    read a matrix in sparse form (same file format as
 *			    cholmod_read_triplet).
 *
 * cholmod_read_dense	    read a dense matrix (any Matrix Market "array"
 *			    format, or a generic dense format).
 *
 * cholmod_write_sparse	    write a sparse matrix to a Matrix Market file.
 *
 * cholmod_write_dense	    write a dense matrix to a Matrix Market file.
 *
 * cholmod_print_common and cholmod_check_common are the only two routines that
 * you may call after calling cholmod_finish.
 *
 * Requires the Utility module.  Not required by any CHOLMOD module, except when
 * debugging is enabled (in which case all modules require the Check module).
 *
 * See cholmod_read.c for a description of the file formats supported by the
 * cholmod_read_* routines.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_check_common:  check the Common object */
/* -------------------------------------------------------------------------- */

int cholmod_check_common
(
    cholmod_common *Common
) ;

int cholmod_l_check_common (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_common:  print the Common object */
/* -------------------------------------------------------------------------- */

int cholmod_print_common
(
    /* ---- input ---- */
    const char *name,	/* printed name of Common object */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_common (const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_gpu_stats:  print the GPU / CPU statistics */
/* -------------------------------------------------------------------------- */

int cholmod_gpu_stats   (cholmod_common *) ;
int cholmod_l_gpu_stats (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_sparse:  check a sparse matrix */
/* -------------------------------------------------------------------------- */

int cholmod_check_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_sparse (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_sparse */
/* -------------------------------------------------------------------------- */

int cholmod_print_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to print */
    const char *name,	/* printed name of sparse matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_sparse (cholmod_sparse *, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_dense:  check a dense matrix */
/* -------------------------------------------------------------------------- */

int cholmod_check_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* dense matrix to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_dense (cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_dense:  print a dense matrix */
/* -------------------------------------------------------------------------- */

int cholmod_print_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* dense matrix to print */
    const char *name,	/* printed name of dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_dense (cholmod_dense *, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_factor:  check a factor */
/* -------------------------------------------------------------------------- */

int cholmod_check_factor
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_factor (cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_factor:  print a factor */
/* -------------------------------------------------------------------------- */

int cholmod_print_factor
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to print */
    const char *name,	/* printed name of factor */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_factor (cholmod_factor *, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_triplet:  check a sparse matrix in triplet form */
/* -------------------------------------------------------------------------- */

int cholmod_check_triplet
(
    /* ---- input ---- */
    cholmod_triplet *T,	/* triplet matrix to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_triplet (cholmod_triplet *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_triplet:  print a triplet matrix */
/* -------------------------------------------------------------------------- */

int cholmod_print_triplet
(
    /* ---- input ---- */
    cholmod_triplet *T,	/* triplet matrix to print */
    const char *name,	/* printed name of triplet matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_triplet (cholmod_triplet *, const char *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_subset:  check a subset */
/* -------------------------------------------------------------------------- */

int cholmod_check_subset
(
    /* ---- input ---- */
    int32_t *Set,	/* Set [0:len-1] is a subset of 0:n-1.  Duplicates OK */
    int64_t len,        /* size of Set (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_subset (int64_t *, int64_t, size_t,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_subset:  print a subset */
/* -------------------------------------------------------------------------- */

int cholmod_print_subset
(
    /* ---- input ---- */
    int32_t *Set,	/* Set [0:len-1] is a subset of 0:n-1.  Duplicates OK */
    int64_t len,        /* size of Set (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    const char *name,	/* printed name of Set */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_subset (int64_t *, int64_t, size_t,
    const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_perm:  check a permutation */
/* -------------------------------------------------------------------------- */

int cholmod_check_perm
(
    /* ---- input ---- */
    int32_t *Perm,	/* Perm [0:len-1] is a permutation of subset of 0:n-1 */
    size_t len,		/* size of Perm (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_perm (int64_t *, size_t, size_t, cholmod_common *);

/* -------------------------------------------------------------------------- */
/* cholmod_print_perm:  print a permutation vector */
/* -------------------------------------------------------------------------- */

int cholmod_print_perm
(
    /* ---- input ---- */
    int32_t *Perm,	/* Perm [0:len-1] is a permutation of subset of 0:n-1 */
    size_t len,		/* size of Perm (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    const char *name,	/* printed name of Perm */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_perm (int64_t *, size_t, size_t, const char *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_parent:  check an elimination tree */
/* -------------------------------------------------------------------------- */

int cholmod_check_parent
(
    /* ---- input ---- */
    int32_t *Parent,	/* Parent [0:n-1] is an elimination tree */
    size_t n,		/* size of Parent */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_parent (int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_parent */
/* -------------------------------------------------------------------------- */

int cholmod_print_parent
(
    /* ---- input ---- */
    int32_t *Parent,	/* Parent [0:n-1] is an elimination tree */
    size_t n,		/* size of Parent */
    const char *name,	/* printed name of Parent */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_parent (int64_t *, size_t, const char *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_sparse: read a sparse matrix from a file */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_read_sparse
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_read_sparse (FILE *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_triplet: read a triplet matrix from a file */
/* -------------------------------------------------------------------------- */

cholmod_triplet *cholmod_read_triplet
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_triplet *cholmod_l_read_triplet (FILE *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_dense: read a dense matrix from a file */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_read_dense
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_read_dense (FILE *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_matrix: read a sparse or dense matrix from a file */
/* -------------------------------------------------------------------------- */

void *cholmod_read_matrix
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    int prefer,		/* If 0, a sparse matrix is always return as a
			 *	cholmod_triplet form.  It can have any stype
			 *	(symmetric-lower, unsymmetric, or
			 *	symmetric-upper).
			 * If 1, a sparse matrix is returned as an unsymmetric
			 *	cholmod_sparse form (A->stype == 0), with both
			 *	upper and lower triangular parts present.
			 *	This is what the MATLAB mread mexFunction does,
			 *	since MATLAB does not have an stype.
			 * If 2, a sparse matrix is returned with an stype of 0
			 *	or 1 (unsymmetric, or symmetric with upper part
			 *	stored).
			 * This argument has no effect for dense matrices.
			 */
    /* ---- output---- */
    int *mtype,		/* CHOLMOD_TRIPLET, CHOLMOD_SPARSE or CHOLMOD_DENSE */
    /* --------------- */
    cholmod_common *Common
) ;

void *cholmod_l_read_matrix (FILE *, int, int *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_write_sparse: write a sparse matrix to a file */
/* -------------------------------------------------------------------------- */

int cholmod_write_sparse    // returns the same result as cholmod_symmetry
(
    /* ---- input ---- */
    FILE *f,		    /* file to write to, must already be open */
    cholmod_sparse *A,	    /* matrix to print */
    cholmod_sparse *Z,	    /* optional matrix with pattern of explicit zeros */
    const char *comments,   /* optional filename of comments to include */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_write_sparse (FILE *, cholmod_sparse *, cholmod_sparse *,
    const char *c, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_write_dense: write a dense matrix to a file */
/* -------------------------------------------------------------------------- */

int cholmod_write_dense
(
    /* ---- input ---- */
    FILE *f,		    /* file to write to, must already be open */
    cholmod_dense *X,	    /* matrix to print */
    const char *comments,   /* optional filename of comments to include */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_write_dense (FILE *, cholmod_dense *, const char *,
    cholmod_common *) ;

#endif

//==============================================================================
// CHOLMOD:Cholesky Module
//==============================================================================

#ifndef NCHOLESKY

/* Sparse Cholesky routines: analysis, factorization, and solve.
 *
 * The primary routines are all that a user requires to order, analyze, and
 * factorize a sparse symmetric positive definite matrix A (or A*A'), and
 * to solve Ax=b (or A*A'x=b).  The primary routines rely on the secondary
 * routines, the CHOLMOD Utility module, and the AMD and COLAMD packages.  They
 * make optional use of the CHOLMOD Supernodal and Partition modules, the
 * METIS package, and the CCOLAMD package.
 *
 * Primary routines:
 * -----------------
 *
 * cholmod_analyze		order and analyze (simplicial or supernodal)
 * cholmod_factorize		simplicial or supernodal Cholesky factorization
 * cholmod_solve		solve a linear system (simplicial or supernodal)
 * cholmod_solve2		like cholmod_solve, but reuse workspace
 * cholmod_spsolve		solve a linear system (sparse x and b)
 *
 * Secondary routines:
 * ------------------
 *
 * cholmod_analyze_p		analyze, with user-provided permutation or f set
 * cholmod_factorize_p		factorize, with user-provided permutation or f
 * cholmod_analyze_ordering	analyze a fill-reducing ordering
 * cholmod_etree		find the elimination tree
 * cholmod_rowcolcounts		compute the row/column counts of L
 * cholmod_amd			order using AMD
 * cholmod_colamd		order using COLAMD
 * cholmod_rowfac		incremental simplicial factorization
 * cholmod_rowfac_mask		rowfac, specific to LPDASA
 * cholmod_rowfac_mask2         rowfac, specific to LPDASA
 * cholmod_row_subtree		find the nonzero pattern of a row of L
 * cholmod_resymbol		recompute the symbolic pattern of L
 * cholmod_resymbol_noperm	recompute the symbolic pattern of L, no L->Perm
 * cholmod_postorder		postorder a tree
 *
 * Requires the Utility module, and two packages: AMD and COLAMD.
 * Optionally uses the Supernodal and Partition modules.
 * Required by the Partition module.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_analyze:  order and analyze (simplicial or supernodal) */
/* -------------------------------------------------------------------------- */

/* Orders and analyzes A, AA', PAP', or PAA'P' and returns a symbolic factor
 * that can later be passed to cholmod_factorize. */

cholmod_factor *cholmod_analyze     // order and analyze
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order and analyze */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_analyze (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_analyze_p:  analyze, with user-provided permutation or f set */
/* -------------------------------------------------------------------------- */

/* Orders and analyzes A, AA', PAP', PAA'P', FF', or PFF'P and returns a
 * symbolic factor that can later be passed to cholmod_factorize, where
 * F = A(:,fset) if fset is not NULL and A->stype is zero.
 * UserPerm is tried if non-NULL.  */

cholmod_factor *cholmod_analyze_p
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order and analyze */
    int32_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_analyze_p (cholmod_sparse *, int64_t *,
    int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_analyze_p2:  analyze for sparse Cholesky or sparse QR */
/* -------------------------------------------------------------------------- */

cholmod_factor *cholmod_analyze_p2
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to order and analyze */
    int32_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_analyze_p2 (int, cholmod_sparse *, int64_t *,
    int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_factorize:  simplicial or supernodal Cholesky factorization */
/* -------------------------------------------------------------------------- */

/* Factorizes PAP' (or PAA'P' if A->stype is 0), using a factor obtained
 * from cholmod_analyze.  The analysis can be re-used simply by calling this
 * routine a second time with another matrix.  A must have the same nonzero
 * pattern as that passed to cholmod_analyze. */

int cholmod_factorize       // simplicial or superodal Cholesky factorization
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    /* ---- in/out --- */
    cholmod_factor *L,	/* resulting factorization */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_factorize (cholmod_sparse *, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_factorize_p:  factorize, with user-provided permutation or fset */
/* -------------------------------------------------------------------------- */

/* Same as cholmod_factorize, but with more options. */

int cholmod_factorize_p
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- in/out --- */
    cholmod_factor *L,	/* resulting factorization */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_factorize_p (cholmod_sparse *, double *, int64_t *,
    size_t, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_solve:  solve a linear system (simplicial or supernodal) */
/* -------------------------------------------------------------------------- */

/* Solves one of many linear systems with a dense right-hand-side, using the
 * factorization from cholmod_factorize (or as modified by any other CHOLMOD
 * routine).  D is identity for LL' factorizations. */

#define CHOLMOD_A    0		/* solve Ax=b */
#define CHOLMOD_LDLt 1		/* solve LDL'x=b */
#define CHOLMOD_LD   2		/* solve LDx=b */
#define CHOLMOD_DLt  3		/* solve DL'x=b */
#define CHOLMOD_L    4		/* solve Lx=b */
#define CHOLMOD_Lt   5		/* solve L'x=b */
#define CHOLMOD_D    6		/* solve Dx=b */
#define CHOLMOD_P    7		/* permute x=Px */
#define CHOLMOD_Pt   8		/* permute x=P'x */

cholmod_dense *cholmod_solve	/* returns the solution X */
(
    /* ---- input ---- */
    int sys,		/* system to solve */
    cholmod_factor *L,	/* factorization to use */
    cholmod_dense *B,	/* right-hand-side */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_solve (int, cholmod_factor *, cholmod_dense *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_solve2:  like cholmod_solve, but with reusable workspace */
/* -------------------------------------------------------------------------- */

int cholmod_solve2     /* returns TRUE on success, FALSE on failure */
(
    /* ---- input ---- */
    int sys,		            /* system to solve */
    cholmod_factor *L,	            /* factorization to use */
    cholmod_dense *B,               /* right-hand-side */
    cholmod_sparse *Bset,
    /* ---- output --- */
    cholmod_dense **X_Handle,       /* solution, allocated if need be */
    cholmod_sparse **Xset_Handle,
    /* ---- workspace  */
    cholmod_dense **Y_Handle,       /* workspace, or NULL */
    cholmod_dense **E_Handle,       /* workspace, or NULL */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_solve2 (int, cholmod_factor *, cholmod_dense *, cholmod_sparse *,
    cholmod_dense **, cholmod_sparse **, cholmod_dense **, cholmod_dense **,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_spsolve:  solve a linear system with a sparse right-hand-side */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_spsolve
(
    /* ---- input ---- */
    int sys,		/* system to solve */
    cholmod_factor *L,	/* factorization to use */
    cholmod_sparse *B,	/* right-hand-side */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_spsolve (int, cholmod_factor *, cholmod_sparse *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_etree: find the elimination tree of A or A'*A */
/* -------------------------------------------------------------------------- */

int cholmod_etree
(
    /* ---- input ---- */
    cholmod_sparse *A,
    /* ---- output --- */
    int32_t *Parent,	/* size ncol.  Parent [j] = p if p is the parent of j */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_etree (cholmod_sparse *, int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowcolcounts: compute the row/column counts of L */
/* -------------------------------------------------------------------------- */

int cholmod_rowcolcounts
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int32_t *Parent,	/* size nrow.  Parent [i] = p if p is the parent of i */
    int32_t *Post,	/* size nrow.  Post [k] = i if i is the kth node in
			 * the postordered etree. */
    /* ---- output --- */
    int32_t *RowCount,	/* size nrow. RowCount [i] = # entries in the ith row of
			 * L, including the diagonal. */
    int32_t *ColCount,	/* size nrow. ColCount [i] = # entries in the ith
			 * column of L, including the diagonal. */
    int32_t *First,	/* size nrow.  First [i] = k is the least postordering
			 * of any descendant of i. */
    int32_t *Level,	/* size nrow.  Level [i] is the length of the path from
			 * i to the root, with Level [root] = 0. */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowcolcounts (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, int64_t *,
    int64_t *, int64_t *, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_analyze_ordering:  analyze a fill-reducing ordering */
/* -------------------------------------------------------------------------- */

int cholmod_analyze_ordering
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int ordering,	/* ordering method used */
    int32_t *Perm,	/* size n, fill-reducing permutation to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Parent,	/* size n, elimination tree */
    int32_t *Post,	/* size n, postordering of elimination tree */
    int32_t *ColCount,	/* size n, nnz in each column of L */
    /* ---- workspace  */
    int32_t *First,	/* size nworkspace for cholmod_postorder */
    int32_t *Level,	/* size n workspace for cholmod_postorder */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_analyze_ordering (cholmod_sparse *, int, int64_t *,
    int64_t *, size_t, int64_t *, int64_t *, int64_t *, int64_t *, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_amd:  order using AMD */
/* -------------------------------------------------------------------------- */

/* Finds a permutation P to reduce fill-in in the factorization of P*A*P'
 * or P*A*A'P' */

int cholmod_amd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_amd (cholmod_sparse *, int64_t *, size_t,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_colamd:  order using COLAMD */
/* -------------------------------------------------------------------------- */

/* Finds a permutation P to reduce fill-in in the factorization of P*A*A'*P'.
 * Orders F*F' where F = A (:,fset) if fset is not NULL */

int cholmod_colamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with a coletree postorder */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_colamd (cholmod_sparse *, int64_t *, size_t, int,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowfac:  incremental simplicial factorization */
/* -------------------------------------------------------------------------- */

/* Partial or complete simplicial factorization.  Rows and columns kstart:kend-1
 * of L and D must be initially equal to rows/columns kstart:kend-1 of the
 * identity matrix.   Row k can only be factorized if all descendants of node
 * k in the elimination tree have been factorized. */

int cholmod_rowfac
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    size_t kstart,	/* first row to factorize */
    size_t kend,	/* last row to factorize is kend-1 */
    /* ---- in/out --- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowfac (cholmod_sparse *, cholmod_sparse *, double *, size_t,
    size_t, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowfac_mask:  incremental simplicial factorization */
/* -------------------------------------------------------------------------- */

/* cholmod_rowfac_mask is a version of cholmod_rowfac that is specific to
 * LPDASA.  It is unlikely to be needed by any other application. */

int cholmod_rowfac_mask
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    size_t kstart,	/* first row to factorize */
    size_t kend,	/* last row to factorize is kend-1 */
    int32_t *mask,	/* if mask[i] >= 0, then set row i to zero */
    int32_t *RLinkUp,	/* link list of rows to compute */
    /* ---- in/out --- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowfac_mask (cholmod_sparse *, cholmod_sparse *, double *, size_t,
    size_t, int64_t *, int64_t *, cholmod_factor *,
    cholmod_common *) ;

int cholmod_rowfac_mask2
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    size_t kstart,	/* first row to factorize */
    size_t kend,	/* last row to factorize is kend-1 */
    int32_t *mask,	/* if mask[i] >= maskmark, then set row i to zero */
    int32_t maskmark,
    int32_t *RLinkUp,	/* link list of rows to compute */
    /* ---- in/out --- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowfac_mask2 (cholmod_sparse *, cholmod_sparse *, double *,
    size_t, size_t, int64_t *, int64_t, int64_t *,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_row_subtree:  find the nonzero pattern of a row of L */
/* -------------------------------------------------------------------------- */

/* Find the nonzero pattern of x for the system Lx=b where L = (0:k-1,0:k-1)
 * and b = kth column of A or A*A' (rows 0 to k-1 only) */

int cholmod_row_subtree
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    size_t k,		/* row k of L */
    int32_t *Parent,	/* elimination tree */
    /* ---- output --- */
    cholmod_sparse *R,	/* pattern of L(k,:), n-by-1 with R->nzmax >= n */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_row_subtree (cholmod_sparse *, cholmod_sparse *, size_t,
    int64_t *, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_lsolve_pattern: find the nonzero pattern of x=L\b */
/* -------------------------------------------------------------------------- */

int cholmod_lsolve_pattern
(
    /* ---- input ---- */
    cholmod_sparse *B,	/* sparse right-hand-side (a single sparse column) */
    cholmod_factor *L,	/* the factor L from which parent(i) is derived */
    /* ---- output --- */
    cholmod_sparse *X,	/* pattern of X=L\B, n-by-1 with X->nzmax >= n */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_lsolve_pattern (cholmod_sparse *, cholmod_factor *,
    cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_row_lsubtree:  find the nonzero pattern of a row of L */
/* -------------------------------------------------------------------------- */

/* Identical to cholmod_row_subtree, except that it finds the elimination tree
 * from L itself. */

int cholmod_row_lsubtree
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *Fi, size_t fnz, /* nonzero pattern of kth row of A', not required
			      * for the symmetric case.  Need not be sorted. */
    size_t k,		/* row k of L */
    cholmod_factor *L,	/* the factor L from which parent(i) is derived */
    /* ---- output --- */
    cholmod_sparse *R,	/* pattern of L(k,:), n-by-1 with R->nzmax >= n */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_row_lsubtree (cholmod_sparse *, int64_t *, size_t,
    size_t, cholmod_factor *, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_resymbol:  recompute the symbolic pattern of L */
/* -------------------------------------------------------------------------- */

/* Remove entries from L that are not in the factorization of P*A*P', P*A*A'*P',
 * or P*F*F'*P' (depending on A->stype and whether fset is NULL or not).
 *
 * cholmod_resymbol is the same as cholmod_resymbol_noperm, except that it
 * first permutes A according to L->Perm.  A can be upper/lower/unsymmetric,
 * in contrast to cholmod_resymbol_noperm (which can be lower or unsym). */

int cholmod_resymbol    // recompute symbolic pattern of L
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int pack,		/* if TRUE, pack the columns of L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factorization, entries pruned on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_resymbol (cholmod_sparse *, int64_t *, size_t, int,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_resymbol_noperm:  recompute the symbolic pattern of L, no L->Perm */
/* -------------------------------------------------------------------------- */

/* Remove entries from L that are not in the factorization of A, A*A',
 * or F*F' (depending on A->stype and whether fset is NULL or not). */

int cholmod_resymbol_noperm
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int pack,		/* if TRUE, pack the columns of L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factorization, entries pruned on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_resymbol_noperm (cholmod_sparse *, int64_t *, size_t, int,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rcond:  compute rough estimate of reciprocal of condition number */
/* -------------------------------------------------------------------------- */

double cholmod_rcond	    /* return min(diag(L)) / max(diag(L)) */
(
    /* ---- input ---- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_rcond (cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_postorder: Compute the postorder of a tree */
/* -------------------------------------------------------------------------- */

int32_t cholmod_postorder	/* return # of nodes postordered */
(
    /* ---- input ---- */
    int32_t *Parent,	/* size n. Parent [j] = p if p is the parent of j */
    size_t n,
    int32_t *Weight_p,	/* size n, optional. Weight [j] is weight of node j */
    /* ---- output --- */
    int32_t *Post,	/* size n. Post [k] = j is kth in postordered tree */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_postorder (int64_t *, size_t, int64_t *, int64_t *,
    cholmod_common *) ;

#endif

//==============================================================================
// CHOLMOD:MatrixOps Module
//==============================================================================

#ifndef NMATRIXOPS

/* Basic operations on sparse and dense matrices.
 *
 * cholmod_drop		    A = entries in A with abs. value >= tol
 * cholmod_norm_dense	    s = norm (X), 1-norm, inf-norm, or 2-norm
 * cholmod_norm_sparse	    s = norm (A), 1-norm or inf-norm
 * cholmod_horzcat	    C = [A,B]
 * cholmod_scale	    A = diag(s)*A, A*diag(s), s*A or diag(s)*A*diag(s)
 * cholmod_sdmult	    Y = alpha*(A*X) + beta*Y or alpha*(A'*X) + beta*Y
 * cholmod_ssmult	    C = A*B
 * cholmod_submatrix	    C = A (i,j), where i and j are arbitrary vectors
 * cholmod_vertcat	    C = [A ; B]
 *
 * A, B, C: sparse matrices (cholmod_sparse)
 * X, Y: dense matrices (cholmod_dense)
 * s: scalar or vector
 *
 * Requires the Utility module.  Not required by any other CHOLMOD module.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_drop:  drop entries with small absolute value */
/* -------------------------------------------------------------------------- */

int cholmod_drop
(
    /* ---- input ---- */
    double tol,		/* keep entries with absolute value > tol */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to drop entries from */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_drop (double, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_norm_dense:  s = norm (X), 1-norm, inf-norm, or 2-norm */
/* -------------------------------------------------------------------------- */

double cholmod_norm_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_norm_dense (cholmod_dense *, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_norm_sparse:  s = norm (A), 1-norm or inf-norm */
/* -------------------------------------------------------------------------- */

double cholmod_norm_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_norm_sparse (cholmod_sparse *, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_horzcat:  C = [A,B] */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_horzcat
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to concatenate */
    cholmod_sparse *B,	/* right matrix to concatenate */
    int values,		/* if TRUE compute the numerical values of C */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_horzcat (cholmod_sparse *, cholmod_sparse *, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_scale:  A = diag(s)*A, A*diag(s), s*A or diag(s)*A*diag(s) */
/* -------------------------------------------------------------------------- */

/* scaling modes, selected by the scale input parameter: */
#define CHOLMOD_SCALAR 0	/* A = s*A */
#define CHOLMOD_ROW 1		/* A = diag(s)*A */
#define CHOLMOD_COL 2		/* A = A*diag(s) */
#define CHOLMOD_SYM 3		/* A = diag(s)*A*diag(s) */

int cholmod_scale
(
    /* ---- input ---- */
    cholmod_dense *S,	/* scale factors (scalar or vector) */
    int scale,		/* type of scaling to compute */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to scale */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_scale (cholmod_dense *, int, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_sdmult:  Y = alpha*(A*X) + beta*Y or alpha*(A'*X) + beta*Y */
/* -------------------------------------------------------------------------- */

/* Sparse matrix times dense matrix */

int cholmod_sdmult
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to multiply */
    int transpose,	/* use A if 0, or A' otherwise */
    double alpha [2],   /* scale factor for A */
    double beta [2],    /* scale factor for Y */
    cholmod_dense *X,	/* dense matrix to multiply */
    /* ---- in/out --- */
    cholmod_dense *Y,	/* resulting dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_sdmult (cholmod_sparse *, int, double *, double *,
    cholmod_dense *, cholmod_dense *Y, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_ssmult:  C = A*B */
/* -------------------------------------------------------------------------- */

/* Sparse matrix times sparse matrix */

cholmod_sparse *cholmod_ssmult
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to multiply */
    cholmod_sparse *B,	/* right matrix to multiply */
    int stype,		/* requested stype of C */
    int values,		/* TRUE: do numerical values, FALSE: pattern only */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_ssmult (cholmod_sparse *, cholmod_sparse *, int, int,
    int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_submatrix:  C = A (r,c), where i and j are arbitrary vectors */
/* -------------------------------------------------------------------------- */

/* rsize < 0 denotes ":" in MATLAB notation, or more precisely 0:(A->nrow)-1.
 * In this case, r can be NULL.  An rsize of zero, or r = NULL and rsize >= 0,
 * denotes "[ ]" in MATLAB notation (the empty set).
 * Similar rules hold for csize.
 */

cholmod_sparse *cholmod_submatrix
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to subreference */
    int32_t *rset,	/* set of row indices, duplicates OK */
    int64_t rsize,	/* size of r; rsize < 0 denotes ":" */
    int32_t *cset,	/* set of column indices, duplicates OK */
    int64_t csize,	/* size of c; csize < 0 denotes ":" */
    int values,		/* if TRUE compute the numerical values of C */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_submatrix (cholmod_sparse *, int64_t *,
    int64_t, int64_t *, int64_t, int, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_vertcat:  C = [A ; B] */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_vertcat
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to concatenate */
    cholmod_sparse *B,	/* right matrix to concatenate */
    int values,		/* if TRUE compute the numerical values of C */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_vertcat (cholmod_sparse *, cholmod_sparse *, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_symmetry: determine if a sparse matrix is symmetric */
/* -------------------------------------------------------------------------- */

int cholmod_symmetry
(
    /* ---- input ---- */
    cholmod_sparse *A,
    int option,
    /* ---- output ---- */
    int32_t *xmatched,
    int32_t *pmatched,
    int32_t *nzoffdiag,
    int32_t *nzdiag,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_symmetry (cholmod_sparse *, int, int64_t *, int64_t *, int64_t *,
    int64_t *, cholmod_common *) ;

#endif

//==============================================================================
// CHOLMOD:Modify Module
//==============================================================================

#ifndef NMODIFY

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_modify.h.
 * Copyright (C) 2005-2006, Timothy A. Davis and William W. Hager
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Sparse Cholesky modification routines: update / downdate / rowadd / rowdel.
 * Can also modify a corresponding solution to Lx=b when L is modified.  This
 * module is most useful when applied on a Cholesky factorization computed by
 * the Cholesky module, but it does not actually require the Cholesky module.
 * The Utility module can create an identity Cholesky factorization (LDL' where
 * L=D=I) that can then by modified by these routines.
 *
 * Primary routines:
 * -----------------
 *
 * cholmod_updown	    multiple rank update/downdate
 * cholmod_rowadd	    add a row to an LDL' factorization
 * cholmod_rowdel	    delete a row from an LDL' factorization
 *
 * Secondary routines:
 * -------------------
 *
 * cholmod_updown_solve	    update/downdate, and modify solution to Lx=b
 * cholmod_updown_mark	    update/downdate, and modify solution to partial Lx=b
 * cholmod_updown_mask	    update/downdate for LPDASA
 * cholmod_updown_mask2     update/downdate for LPDASA
 * cholmod_rowadd_solve	    add a row, and update solution to Lx=b
 * cholmod_rowadd_mark	    add a row, and update solution to partial Lx=b
 * cholmod_rowdel_solve	    delete a row, and downdate Lx=b
 * cholmod_rowdel_mark	    delete a row, and downdate solution to partial Lx=b
 *
 * Requires the Utility module.  Not required by any other CHOLMOD module.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_updown:  multiple rank update/downdate */
/* -------------------------------------------------------------------------- */

/* Compute the new LDL' factorization of LDL'+CC' (an update) or LDL'-CC'
 * (a downdate).  The factor object L need not be an LDL' factorization; it
 * is converted to one if it isn't. */

int cholmod_updown          // update/downdate
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;
int cholmod_l_updown (int, cholmod_sparse *, cholmod_factor *, cholmod_common *) ;


/* -------------------------------------------------------------------------- */
/* cholmod_updown_solve:  update/downdate, and modify solution to Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_updown, except that it also updates/downdates the
 * solution to Lx=b+DeltaB.  x and b must be n-by-1 dense matrices.  b is not
 * need as input to this routine, but a sparse change to b is (DeltaB).  Only
 * entries in DeltaB corresponding to columns modified in L are accessed; the
 * rest must be zero. */

int cholmod_updown_solve
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_solve (int, cholmod_sparse *, cholmod_factor *,
    cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_updown_mark:  update/downdate, and modify solution to partial Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_updown_solve, except only part of L is used in
 * the update/downdate of the solution to Lx=b.  This routine is an "expert"
 * routine.  It is meant for use in LPDASA only.  See cholmod_updown.c for
 * a description of colmark. */

int cholmod_updown_mark
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_mark (int, cholmod_sparse *, int64_t *,
    cholmod_factor *, cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_updown_mask:  update/downdate, for LPDASA */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_updown_mark, except has an additional "mask"
 * argument.  This routine is an "expert" routine.  It is meant for use in
 * LPDASA only.  See cholmod_updown.c for a description of mask. */

int cholmod_updown_mask
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    int32_t *mask,	/* size n */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_mask (int, cholmod_sparse *, int64_t *,
    int64_t *, cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

int cholmod_updown_mask2
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    int32_t *mask,	/* size n */
    int32_t maskmark,
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_mask2 (int, cholmod_sparse *, int64_t *,
    int64_t *, int64_t, cholmod_factor *, cholmod_dense *,
    cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowadd:  add a row to an LDL' factorization (a rank-2 update) */
/* -------------------------------------------------------------------------- */

/* cholmod_rowadd adds a row to the LDL' factorization.  It computes the kth
 * row and kth column of L, and then updates the submatrix L (k+1:n,k+1:n)
 * accordingly.  The kth row and column of L must originally be equal to the
 * kth row and column of the identity matrix.  The kth row/column of L is
 * computed as the factorization of the kth row/column of the matrix to
 * factorize, which is provided as a single n-by-1 sparse matrix R. */

int cholmod_rowadd      // add a row to an LDL' factorization
(
    /* ---- input ---- */
    size_t k,		/* row/column index to add */
    cholmod_sparse *R,	/* row/column of matrix to factorize (n-by-1) */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowadd (size_t, cholmod_sparse *, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowadd_solve:  add a row, and update solution to Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowadd, and also updates the solution to Lx=b
 * See cholmod_updown for a description of how Lx=b is updated.  There is on
 * additional parameter:  bk specifies the new kth entry of b. */

int cholmod_rowadd_solve
(
    /* ---- input ---- */
    size_t k,		/* row/column index to add */
    cholmod_sparse *R,	/* row/column of matrix to factorize (n-by-1) */
    double bk [2],	/* kth entry of the right-hand-side b */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowadd_solve (size_t, cholmod_sparse *, double *,
    cholmod_factor *, cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowadd_mark:  add a row, and update solution to partial Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowadd_solve, except only part of L is used in
 * the update/downdate of the solution to Lx=b.  This routine is an "expert"
 * routine.  It is meant for use in LPDASA only.  */

int cholmod_rowadd_mark
(
    /* ---- input ---- */
    size_t k,		/* row/column index to add */
    cholmod_sparse *R,	/* row/column of matrix to factorize (n-by-1) */
    double bk [2],	/* kth entry of the right hand side, b */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowadd_mark (size_t, cholmod_sparse *, double *,
    int64_t *, cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowdel:  delete a row from an LDL' factorization (a rank-2 update) */
/* -------------------------------------------------------------------------- */

/* Sets the kth row and column of L to be the kth row and column of the identity
 * matrix, and updates L(k+1:n,k+1:n) accordingly.   To reduce the running time,
 * the caller can optionally provide the nonzero pattern (or an upper bound) of
 * kth row of L, as the sparse n-by-1 vector R.  Provide R as NULL if you want
 * CHOLMOD to determine this itself, which is easier for the caller, but takes
 * a little more time.
 */

int cholmod_rowdel      // delete a rw from an LDL' factorization
(
    /* ---- input ---- */
    size_t k,		/* row/column index to delete */
    cholmod_sparse *R,	/* NULL, or the nonzero pattern of kth row of L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowdel (size_t, cholmod_sparse *, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowdel_solve:  delete a row, and downdate Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowdel, but also downdates the solution to Lx=b.
 * When row/column k of A is "deleted" from the system A*y=b, this can induce
 * a change to x, in addition to changes arising when L and b are modified.
 * If this is the case, the kth entry of y is required as input (yk) */

int cholmod_rowdel_solve
(
    /* ---- input ---- */
    size_t k,		/* row/column index to delete */
    cholmod_sparse *R,	/* NULL, or the nonzero pattern of kth row of L */
    double yk [2],	/* kth entry in the solution to A*y=b */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowdel_solve (size_t, cholmod_sparse *, double *,
    cholmod_factor *, cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowdel_mark:  delete a row, and downdate solution to partial Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowdel_solve, except only part of L is used in
 * the update/downdate of the solution to Lx=b.  This routine is an "expert"
 * routine.  It is meant for use in LPDASA only.  */

int cholmod_rowdel_mark
(
    /* ---- input ---- */
    size_t k,		/* row/column index to delete */
    cholmod_sparse *R,	/* NULL, or the nonzero pattern of kth row of L */
    double yk [2],	/* kth entry in the solution to A*y=b */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowdel_mark (size_t, cholmod_sparse *, double *,
    int64_t *, cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

#endif

//==============================================================================
// CHOLMOD:Partition Module (CAMD, CCOLAMD, and CSYMAMD)
//==============================================================================

#ifndef NCAMD

/* CHOLMOD Partition module, interface to CAMD, CCOLAMD, and CSYMAMD
 *
 * An interface to CCOLAMD and CSYMAMD, constrained minimum degree ordering
 * methods which order a matrix following constraints determined via nested
 * dissection.
 *
 * These functions do not require METIS.  They are installed unless NCAMD
 * is defined:
 * cholmod_ccolamd		interface to CCOLAMD ordering
 * cholmod_csymamd		interface to CSYMAMD ordering
 * cholmod_camd			interface to CAMD ordering
 *
 * Requires the Utility and Cholesky modules, and two packages: CAMD,
 * and CCOLAMD.  Used by functions in the Partition Module.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_ccolamd */
/* -------------------------------------------------------------------------- */

/* Order AA' or A(:,f)*A(:,f)' using CCOLAMD. */

int cholmod_ccolamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int32_t *Cmember,	/* size A->nrow.  Cmember [i] = c if row i is in the
			 * constraint set c.  c must be >= 0.  The # of
			 * constraint sets is max (Cmember) + 1.  If Cmember is
			 * NULL, then it is interpretted as Cmember [i] = 0 for
			 * all i */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_ccolamd (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_csymamd */
/* -------------------------------------------------------------------------- */

/* Order A using CSYMAMD. */

int cholmod_csymamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    /* ---- output --- */
    int32_t *Cmember,	/* size nrow.  see cholmod_ccolamd above */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_csymamd (cholmod_sparse *, int64_t *,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_camd */
/* -------------------------------------------------------------------------- */

/* Order A using CAMD. */

int cholmod_camd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Cmember,	/* size nrow.  see cholmod_ccolamd above */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_camd (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, cholmod_common *) ;

#endif

//==============================================================================
// CHOLMOD:Partition Module (graph partition methods)
//==============================================================================

// These routines still exist if CHOLMOD is compiled with -DNPARTITION,
// but they return Common->status = CHOLMOD_NOT_INSTALLED in that case.
#if 1

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_partition.h.
 * Copyright (C) 2005-2013, Univ. of Florida.  Author: Timothy A. Davis
 * -------------------------------------------------------------------------- */

/* CHOLMOD Partition module.
 *
 * Graph partitioning and graph-partition-based orderings.  Includes an
 * interface to CCOLAMD and CSYMAMD, constrained minimum degree ordering
 * methods which order a matrix following constraints determined via nested
 * dissection.
 *
 * These functions require METIS:
 * cholmod_nested_dissection	CHOLMOD nested dissection ordering
 * cholmod_metis		METIS nested dissection ordering (METIS_NodeND)
 * cholmod_bisect		graph partitioner (currently based on METIS)
 * cholmod_metis_bisector	direct interface to METIS_ComputeVertexSeparator
 *
 * Requires the Utility and Cholesky modules, and three packages: METIS, CAMD,
 * and CCOLAMD.  Optionally used by the Cholesky module.
 *
 * Note that METIS does not have a version that uses int64_t integers.  If you
 * try to use cholmod_nested_dissection, cholmod_metis, cholmod_bisect, or
 * cholmod_metis_bisector on a matrix that is too large, an error code will be
 * returned.  METIS does have an "idxtype", which could be redefined as
 * int64_t, if you wish to edit METIS or use compile-time flags to redefine
 * idxtype.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_nested_dissection */
/* -------------------------------------------------------------------------- */

/* Order A, AA', or A(:,f)*A(:,f)' using CHOLMOD's nested dissection method
 * (METIS's node bisector applied recursively to compute the separator tree
 * and constraint sets, followed by CCOLAMD using the constraints).  Usually
 * finds better orderings than METIS_NodeND, but takes longer.
 */

int64_t cholmod_nested_dissection	/* returns # of components */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    int32_t *CParent,	/* size A->nrow.  On output, CParent [c] is the parent
			 * of component c, or EMPTY if c is a root, and where
			 * c is in the range 0 to # of components minus 1 */
    int32_t *Cmember,	/* size A->nrow.  Cmember [j] = c if node j of A is
			 * in component c */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_nested_dissection (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_metis */
/* -------------------------------------------------------------------------- */

/* Order A, AA', or A(:,f)*A(:,f)' using METIS_NodeND. */

int cholmod_metis
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with etree or coletree postorder */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_metis (cholmod_sparse *, int64_t *, size_t, int, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_bisect */
/* -------------------------------------------------------------------------- */

/* Finds a node bisector of A, A*A', A(:,f)*A(:,f)'. */

int64_t cholmod_bisect	/* returns # of nodes in separator */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to bisect */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int compress,	/* if TRUE, compress the graph first */
    /* ---- output --- */
    int32_t *Partition,	/* size A->nrow.  Node i is in the left graph if
			 * Partition [i] = 0, the right graph if 1, and in the
			 * separator if 2. */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_bisect (cholmod_sparse *, int64_t *, size_t, int, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_metis_bisector */
/* -------------------------------------------------------------------------- */

/* Find a set of nodes that bisects the graph of A or AA' (direct interface
 * to METIS_ComputeVertexSeperator). */

int64_t cholmod_metis_bisector	/* returns separator size */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to bisect */
    int32_t *Anw,	/* size A->nrow, node weights, can be NULL, */
                        /* which means the graph is unweighted. */
    int32_t *Aew,	/* size nz, edge weights (silently ignored). */
                        /* This option was available with METIS 4, but not */
                        /* in METIS 5.  This argument is now unused, but */
                        /* it remains for backward compatibilty, so as not */
                        /* to change the API for cholmod_metis_bisector. */
    /* ---- output --- */
    int32_t *Partition,	/* size A->nrow */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_metis_bisector (cholmod_sparse *, int64_t *, int64_t *,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_collapse_septree */
/* -------------------------------------------------------------------------- */

/* Collapse nodes in a separator tree. */

int64_t cholmod_collapse_septree
(
    /* ---- input ---- */
    size_t n,		/* # of nodes in the graph */
    size_t ncomponents,	/* # of nodes in the separator tree (must be <= n) */
    double nd_oksep,    /* collapse if #sep >= nd_oksep * #nodes in subtree */
    size_t nd_small,    /* collapse if #nodes in subtree < nd_small */
    /* ---- in/out --- */
    int32_t *CParent,	/* size ncomponents; from cholmod_nested_dissection */
    int32_t *Cmember,	/* size n; from cholmod_nested_dissection */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_collapse_septree (size_t, size_t, double, size_t, int64_t *,
    int64_t *, cholmod_common *) ;

#endif

//==============================================================================
// CHOLMOD:Supernodal Module
//==============================================================================

#ifndef NSUPERNODAL

/* Supernodal analysis, factorization, and solve.  The simplest way to use
 * these routines is via the Cholesky module.  It does not provide any
 * fill-reducing orderings, but does accept the orderings computed by the
 * Cholesky module.  It does not require the Cholesky module itself, however.
 *
 * Primary routines:
 * -----------------
 * cholmod_super_symbolic	supernodal symbolic analysis
 * cholmod_super_numeric	supernodal numeric factorization
 * cholmod_super_lsolve		supernodal Lx=b solve
 * cholmod_super_ltsolve	supernodal L'x=b solve
 *
 * Prototypes for the BLAS and LAPACK routines that CHOLMOD uses are listed
 * below, including how they are used in CHOLMOD.
 *
 * BLAS routines:
 * --------------
 * dtrsv	solve Lx=b or L'x=b, L non-unit diagonal, x and b stride-1
 * dtrsm	solve LX=B or L'X=b, L non-unit diagonal
 * dgemv	y=y-A*x or y=y-A'*x (x and y stride-1)
 * dgemm	C=A*B', C=C-A*B, or C=C-A'*B
 * dsyrk	C=tril(A*A')
 *
 * LAPACK routines:
 * ----------------
 * dpotrf	LAPACK: A=chol(tril(A))
 *
 * Requires the Utility module, and two external packages: LAPACK and the BLAS.
 * Optionally used by the Cholesky module.
 */

/* -------------------------------------------------------------------------- */
/* cholmod_super_symbolic */
/* -------------------------------------------------------------------------- */

/* Analyzes A, AA', or A(:,f)*A(:,f)' in preparation for a supernodal numeric
 * factorization.  The user need not call this directly; cholmod_analyze is
 * a "simple" wrapper for this routine.
 */

int cholmod_super_symbolic
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    cholmod_sparse *F,	/* F = A' or A(:,f)' */
    int32_t *Parent,	/* elimination tree */
    /* ---- in/out --- */
    cholmod_factor *L,	/* simplicial symbolic on input,
			 * supernodal symbolic on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_symbolic (cholmod_sparse *, cholmod_sparse *, int64_t *,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_symbolic2 */
/* -------------------------------------------------------------------------- */

/* Analyze for supernodal Cholesky or multifrontal QR */

int cholmod_super_symbolic2
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to analyze */
    cholmod_sparse *F,	/* F = A' or A(:,f)' */
    int32_t *Parent,	/* elimination tree */
    /* ---- in/out --- */
    cholmod_factor *L,	/* simplicial symbolic on input,
			 * supernodal symbolic on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_symbolic2 (int, cholmod_sparse *, cholmod_sparse *,
    int64_t *, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_numeric */
/* -------------------------------------------------------------------------- */

/* Computes the numeric LL' factorization of A, AA', or A(:,f)*A(:,f)' using
 * a BLAS-based supernodal method.  The user need not call this directly;
 * cholmod_factorize is a "simple" wrapper for this routine.
 */

int cholmod_super_numeric
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* F = A' or A(:,f)' */
    double beta [2],	/* beta*I is added to diagonal of matrix to factorize */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factorization */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_numeric (cholmod_sparse *, cholmod_sparse *, double *,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_lsolve */
/* -------------------------------------------------------------------------- */

/* Solve Lx=b where L is from a supernodal numeric factorization.  The user
 * need not call this routine directly.  cholmod_solve is a "simple" wrapper
 * for this routine. */

int cholmod_super_lsolve
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to use for the forward solve */
    /* ---- output ---- */
    cholmod_dense *X,	/* b on input, solution to Lx=b on output */
    /* ---- workspace   */
    cholmod_dense *E,	/* workspace of size nrhs*(L->maxesize) */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_lsolve (cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_ltsolve */
/* -------------------------------------------------------------------------- */

/* Solve L'x=b where L is from a supernodal numeric factorization.  The user
 * need not call this routine directly.  cholmod_solve is a "simple" wrapper
 * for this routine. */

int cholmod_super_ltsolve
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to use for the backsolve */
    /* ---- output ---- */
    cholmod_dense *X,	/* b on input, solution to L'x=b on output */
    /* ---- workspace   */
    cholmod_dense *E,	/* workspace of size nrhs*(L->maxesize) */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_ltsolve (cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

#endif

#ifdef __cplusplus
}
#endif

//==============================================================================
// CHOLMOD:SupernodalGPU Module
//==============================================================================

//------------------------------------------------------------------------------
// cholmod_score_comp: for sorting descendant supernodes with qsort
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cholmod_descendant_score_t
{
    double score ;
    int64_t d ;
}
descendantScore ;

int cholmod_score_comp   (struct cholmod_descendant_score_t *i,
                          struct cholmod_descendant_score_t *j) ;
int cholmod_l_score_comp (struct cholmod_descendant_score_t *i,
                          struct cholmod_descendant_score_t *j) ;

#ifdef __cplusplus
}
#endif

//------------------------------------------------------------------------------
// remainder of SupernodalGPU Module
//------------------------------------------------------------------------------

#ifdef SUITESPARSE_CUDA

#include "omp.h"
#include <fenv.h>
#ifndef SUITESPARSE_GPU_EXTERN_ON
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/* CHOLMOD_GPU_PRINTF: for printing GPU debug error messages */
/*
#define CHOLMOD_GPU_PRINTF(args) printf args
*/
#define CHOLMOD_GPU_PRINTF(args)

/* define supernode requirements for processing on GPU */
#define CHOLMOD_ND_ROW_LIMIT 256 /* required descendant rows */
#define CHOLMOD_ND_COL_LIMIT 32  /* required descendnat cols */
#define CHOLMOD_POTRF_LIMIT  512  /* required cols for POTRF & TRSM on GPU */

/* # of host supernodes to perform before checking for free pinned buffers */
#define CHOLMOD_GPU_SKIP     3

#define CHOLMOD_HANDLE_CUDA_ERROR(e,s) {if (e) {ERROR(CHOLMOD_GPU_PROBLEM,s);}}

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cholmod_gpu_pointers
{
    double *h_Lx [CHOLMOD_HOST_SUPERNODE_BUFFERS] ;
    double *d_Lx [CHOLMOD_DEVICE_STREAMS] ;
    double *d_C ;
    double *d_A [CHOLMOD_DEVICE_STREAMS] ;
    void   *d_Ls ;
    void   *d_Map ;
    void   *d_RelativeMap ;

} cholmod_gpu_pointers ;

int cholmod_gpu_memorysize   /* GPU memory size available, 1 if no GPU */
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;

int cholmod_l_gpu_memorysize /* GPU memory size available, 1 if no GPU */
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;

int cholmod_gpu_probe   ( cholmod_common *Common ) ;
int cholmod_l_gpu_probe ( cholmod_common *Common ) ;

int cholmod_gpu_deallocate   ( cholmod_common *Common ) ;
int cholmod_l_gpu_deallocate ( cholmod_common *Common ) ;

void cholmod_gpu_end   ( cholmod_common *Common ) ;
void cholmod_l_gpu_end ( cholmod_common *Common ) ;

int cholmod_gpu_allocate   ( cholmod_common *Common ) ;
int cholmod_l_gpu_allocate ( cholmod_common *Common ) ;

#ifdef __cplusplus
}
#endif

#endif

#endif

