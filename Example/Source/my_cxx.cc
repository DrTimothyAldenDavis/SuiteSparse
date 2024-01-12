//------------------------------------------------------------------------------
// SuiteSparse/Example/Source/my_cxx.cc
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example C++ library that relies on SuiteSparse packages

#include <iostream>
#include <string>
#include <cmath>

#include "my_internal.h"

//------------------------------------------------------------------------------
// OK: check a result and return an error if it fails
//------------------------------------------------------------------------------

#define OK(result)                              \
    if (!(result))                              \
    {                                           \
        std::cout << "FAIL file: " << __FILE__  \
            " line: " << __LINE__ << std::endl ;\
        return (-1) ;                           \
    }

//------------------------------------------------------------------------------
// my_version: version of MY library
//------------------------------------------------------------------------------

void my_version (int version [3], char date [128])
{
    // get the version of this library
    strncpy (date, MY_DATE, 127) ;
    version [0] = MY_MAJOR_VERSION ;
    version [1] = MY_MINOR_VERSION ;
    version [2] = MY_PATCH_VERSION ;
}

//------------------------------------------------------------------------------
// my_check_version: check library version
//------------------------------------------------------------------------------

int my_check_version (const char *package, int major, int minor, int patch,
    const char *date, int version [3], long long unsigned int vercode)
{
    // version and date in package header file:
    std::cout <<
        "\n------------------------------------------------------------\n"
        << package << ": v"
        << major << "."
        << minor << "."
        << patch << " "
        << "(" << date << ")" << std::endl ;

    // version in library itself:
    std::cout
        << package << ": v"
        << version [0] << "."
        << version [1] << "."
        << version [2] << " "
        << "(in library)" << std::endl ;

    // make sure the versions match
    int ok = (major == version [0]) &&
             (minor == version [1]) &&
             (patch == version [2]) ;
    if (!ok)
    {
        std::cout << "Versions do not match" << std::endl ;
    }
    std::cout << package << " version code: " << vercode << std::endl ;
    ok = ok && (vercode == SUITESPARSE__VERCODE (major, minor, patch)) ;
    if (!ok)
    {
        std::cout << "version code mismatch: " <<
        vercode << SUITESPARSE__VERCODE (major, minor, patch) << std::endl ;
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// my_function: try each library in SuiteSparse
//------------------------------------------------------------------------------

int my_function (void)      // returns 0 on success, -1 on failure
{

    int version [3] ;

    //--------------------------------------------------------------------------
    // My package
    //--------------------------------------------------------------------------

    char my_date [128] ;
    my_version (version, my_date) ;
    OK (my_check_version ("MY", MY_MAJOR_VERSION, MY_MINOR_VERSION,
        MY_PATCH_VERSION, MY_DATE, version,
        SUITESPARSE__VERCODE (MY_MAJOR_VERSION, MY_MINOR_VERSION,
        MY_PATCH_VERSION))) ;
    std::cout << "MY date: " << my_date << std::endl ;
    OK (strcmp (my_date, MY_DATE) == 0) ;

    //--------------------------------------------------------------------------
    // SuiteSparse_config
    //--------------------------------------------------------------------------

    int v = SuiteSparse_version (version) ;
    OK (my_check_version ("SuiteSparse_config",
        SUITESPARSE_MAIN_VERSION, SUITESPARSE_SUB_VERSION,
        SUITESPARSE_SUBSUB_VERSION, SUITESPARSE_DATE, version,
        SUITESPARSE__VERSION)) ;

    //--------------------------------------------------------------------------
    // CXSparse
    //--------------------------------------------------------------------------

    cxsparse_version (version) ;
    OK (my_check_version ("CXSparse", CS_VER, CS_SUBVER, CS_SUBSUB, CS_DATE,
        version, CXSPARSE__VERSION)) ;

    cs_dl *A = nullptr ;

    // create a dense 2-by-2 matrix
    #define N 2
    #define NNZ 4
    int64_t n = N, nzmax = NNZ ;
    A = cs_dl_spalloc (n, n, nzmax, true, false) ;
    OK (A != nullptr) ;
    int64_t *Ap = A->p ;
    int64_t *Ai = A->i ;
    double  *Ax = A->x ;
    Ap [0] = 0 ;
    Ap [1] = 2 ;
    Ap [2] = 4 ;
    Ai [0] = 0 ; Ax [0] = 11.0 ;    // A(0,0) = 11
    Ai [1] = 1 ; Ax [1] = 21.0 ;    // A(1,0) = 21
    Ai [2] = 0 ; Ax [2] = 12.0 ;    // A(0,1) = 12
    Ai [3] = 1 ; Ax [3] = 22.0 ;    // A(1,1) = 22
    OK (cs_dl_print (A, false)) ;

    //--------------------------------------------------------------------------
    // AMD
    //--------------------------------------------------------------------------

    amd_version (version) ;
    OK (my_check_version ("AMD",
        AMD_MAIN_VERSION, AMD_SUB_VERSION, AMD_SUBSUB_VERSION, AMD_DATE,
        version, AMD__VERSION)) ;

    int64_t P [N] ;
    OK (amd_l_order (n, Ap, Ai, P, nullptr, nullptr) == AMD_OK) ;
    for (int k = 0 ; k < n ; k++)
    {
        std::cout << "P [" << k << "] = " << P [k] << std::endl;
    }

    //--------------------------------------------------------------------------
    // BTF
    //--------------------------------------------------------------------------

    btf_version (version) ;
    OK (my_check_version ("BTF",
        BTF_MAIN_VERSION, BTF_SUB_VERSION, BTF_SUBSUB_VERSION, BTF_DATE,
        version, BTF__VERSION)) ;

    double work ;
    int64_t nmatch ;
    int64_t Q [N], R [N+1], Work [5*N] ;
    int64_t nblocks = btf_l_order (n, Ap, Ai, -1, &work, P, Q, R, &nmatch,
        Work) ;
    OK (nblocks > 0) ;
    for (int k = 0 ; k < n ; k++)
    {
        std::cout << "P [" << k << "] = " << P [k] << std::endl;
    }
    for (int k = 0 ; k < n ; k++)
    {
        std::cout << "Q [" << k << "] = " << Q [k] << std::endl;
    }
    std::cout << "nblocks " << nblocks << std::endl;

    //--------------------------------------------------------------------------
    // CAMD
    //--------------------------------------------------------------------------

    camd_version (version) ;
    OK (my_check_version ("CAMD",
        CAMD_MAIN_VERSION, CAMD_SUB_VERSION, CAMD_SUBSUB_VERSION, CAMD_DATE,
        version, CAMD__VERSION)) ;

    int64_t Cmem [N] ;
    for (int k = 0 ; k < n ; k++)
    {
        Cmem [k] = 0 ;
    }
    OK (camd_l_order (n, Ap, Ai, P, nullptr, nullptr, Cmem) == CAMD_OK) ;
    for (int k = 0 ; k < n ; k++)
    {
        std::cout << "P [" << k << "] = " << P [k] << std::endl;
    }

    //--------------------------------------------------------------------------
    // CCOLAMD
    //--------------------------------------------------------------------------

    ccolamd_version (version) ;
    OK (my_check_version ("CCOLAMD",
        CCOLAMD_MAIN_VERSION, CCOLAMD_SUB_VERSION, CCOLAMD_SUBSUB_VERSION,
        CCOLAMD_DATE, version, CCOLAMD__VERSION)) ;

    int64_t Alen = ccolamd_l_recommended (NNZ, n, n) ;
    int64_t *Awork = (int64_t *) malloc (Alen * sizeof (int64_t)) ;
    OK (Awork != nullptr) ;
    memcpy (Awork, Ai, NNZ * sizeof (int64_t)) ;
    OK (ccolamd_l (n, n, Alen, Awork, P, nullptr, nullptr, Cmem) == CCOLAMD_OK);
    for (int k = 0 ; k < n ; k++)
    {
        std::cout << "P [" << k << "] = " << P [k] << std::endl;
    }
    free (Awork) ;

    //--------------------------------------------------------------------------
    // COLAMD
    //--------------------------------------------------------------------------

    colamd_version (version) ;
    OK (my_check_version ("COLAMD",
        COLAMD_MAIN_VERSION, COLAMD_SUB_VERSION, COLAMD_SUBSUB_VERSION,
        COLAMD_DATE, version, COLAMD__VERSION)) ;

    Alen = ccolamd_l_recommended (NNZ, n, n) ;
    Awork = (int64_t *) malloc (Alen * sizeof (int64_t)) ;
    OK (Awork != nullptr) ;
    memcpy (Awork, Ai, NNZ * sizeof (int64_t)) ;
    OK (colamd_l (n, n, Alen, Awork, P, nullptr, nullptr) == COLAMD_OK) ;
    for (int k = 0 ; k < n ; k++)
    {
        std::cout << "P [" << k << "] = " << P [k] << std::endl;
    }
    free (Awork) ;

    //--------------------------------------------------------------------------
    // CHOLMOD
    //--------------------------------------------------------------------------

    v = cholmod_l_version (version) ;
    OK (my_check_version ("CHOLMOD",
        CHOLMOD_MAIN_VERSION, CHOLMOD_SUB_VERSION, CHOLMOD_SUBSUB_VERSION,
        CHOLMOD_DATE, version, CHOLMOD__VERSION)) ;

    cholmod_common cc ;
    OK (cholmod_l_start (&cc)) ;

    //--------------------------------------------------------------------------
    // GraphBLAS
    //--------------------------------------------------------------------------

    #if ! defined (NO_GRAPHBLAS)
    OK (GrB_init (GrB_NONBLOCKING) == GrB_SUCCESS) ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_VERSION, version) == GrB_SUCCESS) ;
    OK (my_check_version ("GraphBLAS",
        GxB_IMPLEMENTATION_MAJOR, GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB, GxB_IMPLEMENTATION_DATE, version,
        GxB_IMPLEMENTATION)) ;
    OK (GrB_finalize ( ) == GrB_SUCCESS) ;
    #endif

    //--------------------------------------------------------------------------
    // LAGraph
    //--------------------------------------------------------------------------

    #if ! defined (NO_LAGRAPH)
    char msg [LAGRAPH_MSG_LEN], verstring [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Init (msg) == GrB_SUCCESS) ;
    OK (LAGraph_Version (version, verstring, msg) == GrB_SUCCESS) ;
    OK (my_check_version ("LAGraph",
        LAGRAPH_VERSION_MAJOR, LAGRAPH_VERSION_MINOR, LAGRAPH_VERSION_UPDATE,
        LAGRAPH_DATE, version, SUITESPARSE__VERCODE (LAGRAPH_VERSION_MAJOR,
        LAGRAPH_VERSION_MINOR, LAGRAPH_VERSION_UPDATE))) ;
    OK (LAGraph_Finalize (msg) == GrB_SUCCESS) ;
    #endif

    //--------------------------------------------------------------------------
    // KLU
    //--------------------------------------------------------------------------

    klu_version (version) ;
    OK (my_check_version ("KLU",
        KLU_MAIN_VERSION, KLU_SUB_VERSION, KLU_SUBSUB_VERSION,
        KLU_DATE, version, KLU__VERSION)) ;

    double b [N] = {8., 45.} ;
    double xgood [N] = {36.4, -32.7} ;
    double x [N] ;

    klu_l_symbolic *Symbolic ;
    klu_l_numeric *Numeric ;
    klu_l_common Common ;
    OK (klu_l_defaults (&Common)) ;
    Symbolic = klu_l_analyze (n, Ap, Ai, &Common) ;
    OK (Symbolic != nullptr) ;
    Numeric = klu_l_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    OK (Numeric != nullptr) ;
    memcpy (x, b, N * sizeof (double)) ;
    OK (klu_l_solve (Symbolic, Numeric, 5, 1, x, &Common)) ;
    klu_l_free_symbolic (&Symbolic, &Common) ;
    klu_l_free_numeric (&Numeric, &Common) ;
    double err = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        std::cout << "x [" << i << "] = " << x [i] << std::endl;
        err = fmax (err, fabs (x [i] - xgood [i])) ;
    }
    std::cout << "error: " << err << std::endl;
    OK (err < 1e-12) ;

    //--------------------------------------------------------------------------
    // LDL
    //--------------------------------------------------------------------------

    ldl_version (version) ;
    OK (my_check_version ("LDL",
        LDL_MAIN_VERSION, LDL_SUB_VERSION, LDL_SUBSUB_VERSION,
        LDL_DATE, version, LDL__VERSION)) ;

    double x2 [N] ;
    P [0] = 0 ;
    P [1] = 1 ;
    ldl_l_perm (n, x2, xgood, P) ;
    err = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        std::cout << "x2 [" << i << "] = " << x2 [i] << std::endl;
        err = fmax (err, fabs (x2 [i] - xgood [i])) ;
    }
    std::cout << "error: " << err << std::endl;
    OK (err == 0) ;

    //--------------------------------------------------------------------------
    // RBio
    //--------------------------------------------------------------------------

    RBio_version (version) ;
    OK (my_check_version ("RBio",
        RBIO_MAIN_VERSION, RBIO_SUB_VERSION, RBIO_SUBSUB_VERSION,
        RBIO_DATE, version, RBIO__VERSION)) ;

    char mtype [4];
    std::string key {"simple"};
    std::string title {"2-by-2 matrix"};
    mtype [0] = '\0' ;
    int64_t njumbled, nzeros ;
    int result = RBok (n, n, NNZ, Ap, Ai, Ax, nullptr, nullptr, 0,
        &njumbled, &nzeros) ;
    OK (result == RBIO_OK) ;
    std::cout << "njumbled " << njumbled << ", nzeros " << nzeros << std::endl;
    std::string filename {"temp.rb"};
    result = RBwrite (filename.c_str (), title.c_str (), key.c_str (), n, n,
        Ap, Ai, Ax, nullptr, nullptr, nullptr, 0, mtype) ;
    std::cout << "result " << result << std::endl;
    std::cout << "mtype: " << mtype << std::endl;

    // dump out the file
    FILE *f = fopen ("temp.rb", "r") ;
    OK (f != nullptr) ;
    int c ;
    while (1)
    {
        c = fgetc (f) ;
        if (c == EOF) break ;
        fputc (c, stdout) ;
    }
    fclose (f) ;

    //--------------------------------------------------------------------------
    // SPEX
    //--------------------------------------------------------------------------

    OK (SPEX_initialize ( ) == SPEX_OK) ;
    SPEX_version (version) ;
    OK (my_check_version ("SPEX",
        SPEX_VERSION_MAJOR, SPEX_VERSION_MINOR, SPEX_VERSION_SUB, SPEX_DATE,
        version, SPEX__VERSION)) ;
    OK (SPEX_finalize ( ) == SPEX_OK) ;

    //--------------------------------------------------------------------------
    // SPQR
    //--------------------------------------------------------------------------

    SuiteSparseQR_version (version) ;
    OK (my_check_version ("SuiteSparseQR",
        SPQR_MAIN_VERSION, SPQR_SUB_VERSION, SPQR_SUBSUB_VERSION, SPQR_DATE,
        version, SPQR__VERSION)) ;

    cholmod_sparse *A2, A2_struct ;
    cholmod_dense  *B2, B2_struct ;
    cholmod_dense  *X2 ;

    // make a shallow CHOLMOD copy of A
    A2 = &A2_struct ;
    A2->nrow = n ;
    A2->ncol = n ;
    A2->p = Ap ;
    A2->i = Ai ;
    A2->x = Ax ;
    A2->z = nullptr ;
    A2->nzmax = NNZ ;
    A2->packed = true ;
    A2->sorted = true ;
    A2->nz = nullptr ;
    A2->itype = CHOLMOD_LONG ;
    A2->dtype = CHOLMOD_DOUBLE ;
    A2->xtype = CHOLMOD_REAL ;
    A2->stype = 0 ;

    // make a shallow CHOLMOD copy of b
    B2 = &B2_struct ;
    B2->nrow = n ;
    B2->ncol = 1 ;
    B2->x = b ;
    B2->z = nullptr ;
    B2->d = n ;
    B2->nzmax = n ;
    B2->dtype = CHOLMOD_DOUBLE ;
    B2->xtype = CHOLMOD_REAL ;

    X2 = SuiteSparseQR_C_backslash_default (A2, B2, &cc) ;
    OK (X2 != nullptr) ;
    OK (cc.status == CHOLMOD_OK) ;
    cc.print = 5 ;
    OK (cholmod_l_print_dense (X2, "X from QR", &cc)) ;

    //--------------------------------------------------------------------------
    // UMFPACK
    //--------------------------------------------------------------------------

    umfpack_version (version) ;
    OK (my_check_version ("UMFPACK",
        UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION,
        UMFPACK_DATE, version, UMFPACK__VERSION)) ;

    std::cout << UMFPACK_VERSION << std::endl;
    std::cout << UMFPACK_COPYRIGHT;
    std::cout << UMFPACK_LICENSE_PART1;
    std::cout << "BLAS used: " << SuiteSparse_BLAS_library ( ) << std::endl;
    std::cout << "BLAS integer size: "
              << sizeof (SUITESPARSE_BLAS_INT) << " bytes" << std::endl;

    double Control [UMFPACK_CONTROL] ;
    double Info [UMFPACK_INFO] ;
    umfpack_dl_defaults (Control) ;
    Control [UMFPACK_PRL] = 6 ;

    void *Sym, *Num ;
    (void) umfpack_dl_symbolic (n, n, Ap, Ai, Ax, &Sym, Control, Info) ;
    (void) umfpack_dl_numeric (Ap, Ai, Ax, Sym, &Num, Control, Info) ;
    umfpack_dl_free_symbolic (&Sym) ;
    result = umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Num, Control, Info);
    umfpack_dl_free_numeric (&Num) ;
    for (int i = 0 ; i < n ; i++)
    {
        std::cout << "x [" << i << "] = " << x [i] << std::endl;
    }
    err = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        err = fmax (err, fabs (x [i] - xgood [i])) ;
    }
    std::cout << "error: " << err << std::endl;
    OK (err < 1e-12) ;
    umfpack_dl_report_status (Control, result) ;
    umfpack_dl_report_info (Control, Info) ;

    //--------------------------------------------------------------------------
    // Mongoose
    //--------------------------------------------------------------------------

    version [0] = Mongoose::major_version ( ) ;
    version [1] = Mongoose::minor_version ( ) ;
    version [2] = Mongoose::patch_version ( ) ;
    OK (my_check_version ("Mongoose", Mongoose_VERSION_MAJOR,
        Mongoose_VERSION_MINOR, Mongoose_VERSION_PATCH, Mongoose_DATE,
        version, Mongoose__VERSION)) ;
    std::cout << "Mongoose::mongoose_version(): " <<
        Mongoose::mongoose_version ( ) << std::endl;

    //--------------------------------------------------------------------------
    // not used
    //--------------------------------------------------------------------------

    // std::cout << "ParU version code: " << PARU__VERSION << std::endl ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    cs_dl_spfree (A) ;
    A = nullptr ;
    OK (cholmod_l_finish (&cc)) ;
    return (0) ;
}

