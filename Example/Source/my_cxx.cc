// Example library that relies on SuiteSparse packages

#include <iostream>
#include <string>
#include <cmath>

#include "my_internal.h"

#define OK(result)                                            \
    if (!(result))                                            \
    {                                                         \
        std::cout << "abort line " << __LINE__ << std::endl ; \
        abort ( ) ;                                           \
    }

void my_library (int version [3], char date [128])
{
    // get the version of this library
    strncpy (date, MY_DATE, 127) ;
    version [0] = MY_MAJOR_VERSION ;
    version [1] = MY_MINOR_VERSION ;
    version [2] = MY_PATCH_VERSION ;
}

void my_function (void)
{

    //--------------------------------------------------------------------------
    // SuiteSparse_config
    //--------------------------------------------------------------------------

    std::cout << "SuiteSparse: v"
              << SUITESPARSE_MAIN_VERSION << "."
              << SUITESPARSE_SUB_VERSION << "."
              << SUITESPARSE_SUBSUB_VERSION << " "
              << "(" << SUITESPARSE_DATE << ")" << std::endl;
    int version[3];
    int v = SuiteSparse_version (version) ;
    std::cout << "SuiteSparse: v"
              << version[0] << "."
              << version[1] << "."
              << version[2] << " "
              << "(in library)" << std::endl;

    //--------------------------------------------------------------------------
    // CXSparse
    //--------------------------------------------------------------------------

    std::cout << "CXSparse: v"
              << CS_VER << "."
              << CS_SUBVER << "."
              << CS_SUBSUB << " "
              << "(" << CS_DATE << ")" << std::endl;
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

    std::cout << "AMD: v"
              << AMD_MAIN_VERSION << "."
              << AMD_SUB_VERSION << "."
              << AMD_SUBSUB_VERSION << " "
              << "(" << AMD_DATE << ")" << std::endl;
    int64_t P [N] ;
    OK (amd_l_order (n, Ap, Ai, P, nullptr, nullptr) == AMD_OK) ;
    for (int k = 0 ; k < n ; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;

    //--------------------------------------------------------------------------
    // BTF
    //--------------------------------------------------------------------------

    std::cout << "BTF: v"
              << BTF_MAIN_VERSION << "."
              << BTF_SUB_VERSION << "."
              << BTF_SUBSUB_VERSION << " "
              << "(" << BTF_DATE << ")" << std::endl;
    double work ;
    int64_t nmatch ;
    int64_t Q [N], R [N+1], Work [5*N] ;
    int64_t nblocks = btf_l_order (n, Ap, Ai, -1, &work, P, Q, R, &nmatch,
        Work) ;
    OK (nblocks > 0) ;
    for (int k = 0 ; k < n ; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;
    for (int k = 0 ; k < n ; k++)
      std::cout << "Q [" << k << "] = " << Q [k] << std::endl;
    std::cout << "nblocks " << nblocks << std::endl;

    //--------------------------------------------------------------------------
    // CAMD
    //--------------------------------------------------------------------------

    std::cout << "CAMD: v"
              << CAMD_MAIN_VERSION << "."
              << CAMD_SUB_VERSION << "."
              << CAMD_SUBSUB_VERSION << " "
              << "(" << CAMD_DATE << ")" << std::endl;
    int64_t Cmem [N] ;
    for (int k = 0 ; k < n ; k++)
      Cmem [k] = 0 ;
    OK (camd_l_order (n, Ap, Ai, P, nullptr, nullptr, Cmem) == CAMD_OK) ;
    for (int k = 0 ; k < n ; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;

    //--------------------------------------------------------------------------
    // CCOLAMD
    //--------------------------------------------------------------------------

    std::cout << "CCOLAMD: v"
              << CCOLAMD_MAIN_VERSION << "."
              << CCOLAMD_SUB_VERSION << "."
              << CCOLAMD_SUBSUB_VERSION << " "
              << "(" << CCOLAMD_DATE << ")" << std::endl;
    int64_t Alen = ccolamd_l_recommended (NNZ, n, n) ;
    int64_t *Awork = (int64_t *) malloc (Alen * sizeof (int64_t)) ;
    OK (Awork != nullptr) ;
    memcpy (Awork, Ai, NNZ * sizeof (int64_t)) ;
    OK (ccolamd_l (n, n, Alen, Awork, P, nullptr, nullptr, Cmem) == CCOLAMD_OK) ;
    for (int k = 0 ; k < n ; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;
    free (Awork) ;

    //--------------------------------------------------------------------------
    // COLAMD
    //--------------------------------------------------------------------------

    std::cout << "COLAMD: v"
              << COLAMD_MAIN_VERSION << "."
              << COLAMD_SUB_VERSION << "."
              << COLAMD_SUBSUB_VERSION << " "
              << "(" << COLAMD_DATE << ")" << std::endl;
    Alen = ccolamd_l_recommended (NNZ, n, n) ;
    Awork = (int64_t *) malloc (Alen * sizeof (int64_t)) ;
    OK (Awork != nullptr) ;
    memcpy (Awork, Ai, NNZ * sizeof (int64_t)) ;
    OK (colamd_l (n, n, Alen, Awork, P, nullptr, nullptr) == COLAMD_OK) ;
    for (int k = 0 ; k < n ; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;
    free (Awork) ;

    //--------------------------------------------------------------------------
    // CHOLMOD
    //--------------------------------------------------------------------------

    std::cout << "CHOLMOD: v"
              << CHOLMOD_MAIN_VERSION << "."
              << CHOLMOD_SUB_VERSION << "."
              << CHOLMOD_SUBSUB_VERSION << " "
              << "(" << CHOLMOD_DATE << ")" << std::endl;
    v = cholmod_l_version (version) ;
    std::cout << "CHOLMOD: v"
              << version[0] << "."
              << version[1] << "."
              << version[2] << " "
              << "(in library)" << std::endl;
    cholmod_common cc ;
    OK (cholmod_l_start (&cc)) ;

#if ! defined (NO_GRAPHBLAS)
    //--------------------------------------------------------------------------
    // GraphBLAS
    //--------------------------------------------------------------------------

    std::cout << "GraphBLAS: v"
              << GxB_IMPLEMENTATION_MAJOR << "."
              << GxB_IMPLEMENTATION_MINOR << "."
              << GxB_IMPLEMENTATION_SUB << " "
              << "(" << GxB_IMPLEMENTATION_DATE << ")" << std::endl;
    OK (GrB_init (GrB_NONBLOCKING) == GrB_SUCCESS) ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_VERSION, version) == GrB_SUCCESS) ;
    std::cout << "GraphBLAS: v"
              << version[0] << "."
              << version[1] << "."
              << version[2] << " "
              << "(in library)" << std::endl;
    OK (GrB_finalize ( ) == GrB_SUCCESS) ;
#endif

    //--------------------------------------------------------------------------
    // KLU
    //--------------------------------------------------------------------------

    std::cout << "KLU: v"
              << KLU_MAIN_VERSION << "."
              << KLU_SUB_VERSION << "."
              << KLU_SUBSUB_VERSION << " "
              << "(" << KLU_DATE << ")" << std::endl;
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

    std::cout << "LDL: v"
              << LDL_MAIN_VERSION << "."
              << LDL_SUB_VERSION << "."
              << LDL_SUBSUB_VERSION << " "
              << "(" << LDL_DATE << ")" << std::endl;
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

    std::cout << "RBio: v"
              << RBIO_MAIN_VERSION << "."
              << RBIO_SUB_VERSION << "."
              << RBIO_SUBSUB_VERSION << " "
              << "(" << RBIO_DATE << ")" << std::endl;
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

    std::cout << "SPEX: v"
              << SPEX_VERSION_MAJOR << "."
              << SPEX_VERSION_MINOR << "."
              << SPEX_VERSION_SUB << " "
              << "(" << SPEX_DATE << ")" << std::endl;
    OK (SPEX_initialize ( ) == SPEX_OK) ;
    OK (SPEX_finalize ( ) == SPEX_OK) ;

    //--------------------------------------------------------------------------
    // SPQR
    //--------------------------------------------------------------------------

    std::cout << "SPQR: v"
              << SPQR_MAIN_VERSION << "."
              << SPQR_SUB_VERSION << "."
              << SPQR_SUBSUB_VERSION << " "
              << "(" << SPQR_DATE << ")" << std::endl;
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

    std::cout << "UMFPACK: v"
              << UMFPACK_MAIN_VERSION << "."
              << UMFPACK_SUB_VERSION << "."
              << UMFPACK_SUBSUB_VERSION << " "
              << "(" << UMFPACK_DATE << ")" << std::endl;

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
    result = umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Num, Control, Info) ;
    umfpack_dl_free_numeric (&Num) ;
    for (int i = 0 ; i < n ; i++)
      std::cout << "x [" << i << "] = " << x [i] << std::endl;
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
    // free workspace
    //--------------------------------------------------------------------------

    cs_dl_spfree (A) ;
    A = nullptr ;
    OK (cholmod_l_finish (&cc)) ;
}
