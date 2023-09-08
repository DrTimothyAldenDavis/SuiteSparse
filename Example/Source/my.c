// Example library that relies on SuiteSparse packages

// ANSI C include files:
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "my_internal.h"

#define OK(result)                                      \
    if (!(result))                                      \
    {                                                   \
        printf ("abort line %d\n", __LINE__) ;          \
        abort ( ) ;                                     \
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

    printf ("SuiteSparse: v%d.%d.%d (%s)\n",
        SUITESPARSE_MAIN_VERSION, SUITESPARSE_SUB_VERSION,
        SUITESPARSE_SUBSUB_VERSION, SUITESPARSE_DATE) ;
    int version [3] ;
    int v = SuiteSparse_version (version) ;
    printf ("SuiteSparse: v%d.%d.%d (in library)\n",
        version [0], version [1], version [2]) ;

    //--------------------------------------------------------------------------
    // CXSparse
    //--------------------------------------------------------------------------

    printf ("CXSparse: v%d.%d.%d (%s)\n",
        CS_VER, CS_SUBVER, CS_SUBSUB, CS_DATE) ;
    cs_dl *A = NULL ;

    // create a dense 2-by-2 matrix
    #define N 2
    #define NNZ 4
    int64_t n = N, nzmax = NNZ ;
    A = cs_dl_spalloc (n, n, nzmax, true, false) ;
    OK (A != NULL) ;
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

    printf ("AMD: v%d.%d.%d (%s)\n",
        AMD_MAIN_VERSION, AMD_SUB_VERSION, AMD_SUBSUB_VERSION, AMD_DATE) ;
    int64_t P [N] ;
    OK (amd_l_order (n, Ap, Ai, P, NULL, NULL) == AMD_OK) ;
    for (int k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, (int) P [k]) ;

    //--------------------------------------------------------------------------
    // BTF
    //--------------------------------------------------------------------------

    printf ("BTF: v%d.%d.%d (%s)\n",
        BTF_MAIN_VERSION, BTF_SUB_VERSION, BTF_SUBSUB_VERSION, BTF_DATE) ;
    double work ;
    int64_t nmatch ;
    int64_t Q [N], R [N+1], Work [5*N] ;
    int64_t nblocks = btf_l_order (n, Ap, Ai, -1, &work, P, Q, R, &nmatch,
        Work) ;
    OK (nblocks > 0) ;
    for (int k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, (int) P [k]) ;
    for (int k = 0 ; k < n ; k++) printf ("Q [%d] = %d\n", k, (int) Q [k]) ;
    printf ("nblocks %d\n", (int) nblocks) ;

    //--------------------------------------------------------------------------
    // CAMD
    //--------------------------------------------------------------------------

    printf ("CAMD: v%d.%d.%d (%s)\n",
        CAMD_MAIN_VERSION, CAMD_SUB_VERSION, CAMD_SUBSUB_VERSION, CAMD_DATE) ;
    int64_t Cmem [N] ;
    for (int k = 0 ; k < n ; k++) Cmem [k] = 0 ;
    OK (camd_l_order (n, Ap, Ai, P, NULL, NULL, Cmem) == CAMD_OK) ;
    for (int k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, (int) P [k]) ;

    //--------------------------------------------------------------------------
    // CCOLAMD
    //--------------------------------------------------------------------------

    printf ("CCOLAMD: v%d.%d.%d (%s)\n",
        CCOLAMD_MAIN_VERSION, CCOLAMD_SUB_VERSION, CCOLAMD_SUBSUB_VERSION,
        CCOLAMD_DATE) ;
    int64_t Alen = ccolamd_l_recommended (NNZ, n, n) ;
    int64_t *Awork = malloc (Alen * sizeof (int64_t)) ;
    OK (Awork != NULL) ;
    memcpy (Awork, Ai, NNZ * sizeof (int64_t)) ;
    OK (ccolamd_l (n, n, Alen, Awork, P, NULL, NULL, Cmem) == CCOLAMD_OK) ;
    for (int k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, (int) P [k]) ;
    free (Awork) ;

    //--------------------------------------------------------------------------
    // COLAMD
    //--------------------------------------------------------------------------

    printf ("COLAMD: v%d.%d.%d (%s)\n",
        COLAMD_MAIN_VERSION, COLAMD_SUB_VERSION, COLAMD_SUBSUB_VERSION,
        COLAMD_DATE) ;
    Alen = ccolamd_l_recommended (NNZ, n, n) ;
    Awork = malloc (Alen * sizeof (int64_t)) ;
    OK (Awork != NULL) ;
    memcpy (Awork, Ai, NNZ * sizeof (int64_t)) ;
    OK (colamd_l (n, n, Alen, Awork, P, NULL, NULL) == COLAMD_OK) ;
    for (int k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, (int) P [k]) ;
    free (Awork) ;

    //--------------------------------------------------------------------------
    // CHOLMOD
    //--------------------------------------------------------------------------

    printf ("CHOLMOD: v%d.%d.%d (%s)\n",
        CHOLMOD_MAIN_VERSION, CHOLMOD_SUB_VERSION, CHOLMOD_SUBSUB_VERSION,
        CHOLMOD_DATE) ;
    v = cholmod_l_version (version) ;
    printf ("CHOLMOD: v%d.%d.%d (in library)\n",
        version [0], version [1], version [2]) ;
    cholmod_common cc ;
    OK (cholmod_l_start (&cc)) ;

#if ! defined (NO_GRAPHBLAS)
    //--------------------------------------------------------------------------
    // GraphBLAS
    //--------------------------------------------------------------------------

    OK (GrB_init (GrB_NONBLOCKING) == GrB_SUCCESS) ;
    printf ("GraphBLAS: v%d.%d.%d (%s)\n",
        GxB_IMPLEMENTATION_MAJOR, GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB, GxB_IMPLEMENTATION_DATE) ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_VERSION, version) == GrB_SUCCESS) ;
    printf ("GraphBLAS: v%d.%d.%d (in library)\n",
        version [0], version [1], version [2]) ;
    OK (GrB_finalize ( ) == GrB_SUCCESS) ;
#endif

    //--------------------------------------------------------------------------
    // KLU
    //--------------------------------------------------------------------------

    printf ("KLU: v%d.%d.%d (%s)\n",
        KLU_MAIN_VERSION, KLU_SUB_VERSION, KLU_SUBSUB_VERSION, KLU_DATE) ;
    double b [N] = {8., 45.} ;
    double xgood [N] = {36.4, -32.7} ;
    double x [N] ;

    klu_l_symbolic *Symbolic ;
    klu_l_numeric *Numeric ;
    klu_l_common Common ;
    OK (klu_l_defaults (&Common)) ;
    Symbolic = klu_l_analyze (n, Ap, Ai, &Common) ;
    OK (Symbolic != NULL) ;
    Numeric = klu_l_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    OK (Numeric != NULL) ;
    memcpy (x, b, N * sizeof (double)) ;
    OK (klu_l_solve (Symbolic, Numeric, 5, 1, x, &Common)) ;
    klu_l_free_symbolic (&Symbolic, &Common) ;
    klu_l_free_numeric (&Numeric, &Common) ;
    double err = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        printf ("x [%d] = %g\n", i, x [i]) ;
        err = fmax (err, fabs (x [i] - xgood [i])) ;
    }
    printf ("error: %g\n", err) ;
    OK (err < 1e-12) ;

    //--------------------------------------------------------------------------
    // LDL
    //--------------------------------------------------------------------------

    printf ("LDL: v%d.%d.%d (%s)\n",
        LDL_MAIN_VERSION, LDL_SUB_VERSION, LDL_SUBSUB_VERSION, LDL_DATE) ;
    double x2 [N] ;
    P [0] = 0 ;
    P [1] = 1 ;
    ldl_l_perm (n, x2, xgood, P) ;
    err = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        printf ("x2 [%d] = %g\n", i, x2 [i]) ;
        err = fmax (err, fabs (x2 [i] - xgood [i])) ;
    }
    printf ("error: %g\n", err) ;
    OK (err == 0) ;

    //--------------------------------------------------------------------------
    // RBio
    //--------------------------------------------------------------------------

    printf ("RBio: v%d.%d.%d (%s)\n",
        RBIO_MAIN_VERSION, RBIO_SUB_VERSION, RBIO_SUBSUB_VERSION, RBIO_DATE) ;
    char mtype [4], key [8], title [80] ;
    strncpy (key, "simple", 8) ;
    strncpy (title, "2-by-2 matrix", 80) ;
    mtype [0] = '\0' ;
    int64_t njumbled, nzeros ;
    int result = RBok (n, n, NNZ, Ap, Ai, Ax, NULL, NULL, 0,
        &njumbled, &nzeros) ;
    OK (result == RBIO_OK) ;
    printf ("njumbled %d, nzeros %d\n", (int) njumbled, (int) nzeros) ;
    result = RBwrite ("temp.rb", title, key, n, n, Ap, Ai, Ax,
        NULL, NULL, NULL, 0, mtype) ;
    printf ("result %d\n", result) ;
    printf ("mtype: %s\n", mtype) ;

    // dump out the file
    FILE *f = fopen ("temp.rb", "r") ;
    OK (f != NULL) ;
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

    printf ("SPEX: v%d.%d.%d (%s)\n",
        SPEX_VERSION_MAJOR, SPEX_VERSION_MINOR, SPEX_VERSION_SUB, SPEX_DATE) ;
    OK (SPEX_initialize ( ) == SPEX_OK) ;
    OK (SPEX_finalize ( ) == SPEX_OK) ;

    //--------------------------------------------------------------------------
    // SPQR
    //--------------------------------------------------------------------------

    printf ("SPQR: v%d.%d.%d (%s)\n",
        SPQR_MAIN_VERSION, SPQR_SUB_VERSION, SPQR_SUBSUB_VERSION, SPQR_DATE) ;
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
    A2->z = NULL ;
    A2->nzmax = NNZ ;
    A2->packed = true ;
    A2->sorted = true ;
    A2->nz = NULL ;
    A2->itype = CHOLMOD_LONG ;
    A2->dtype = CHOLMOD_DOUBLE ;
    A2->xtype = CHOLMOD_REAL ;
    A2->stype = 0 ;

    // make a shallow CHOLMOD copy of b
    B2 = &B2_struct ;
    B2->nrow = n ;
    B2->ncol = 1 ;
    B2->x = b ;
    B2->z = NULL ;
    B2->d = n ;
    B2->nzmax = n ;
    B2->dtype = CHOLMOD_DOUBLE ;
    B2->xtype = CHOLMOD_REAL ;

    X2 = SuiteSparseQR_C_backslash_default (A2, B2, &cc) ;
    OK (X2 != NULL) ;
    OK (cc.status == CHOLMOD_OK) ;
    cc.print = 5 ;
    OK (cholmod_l_print_dense (X2, "X from QR", &cc)) ;

    //--------------------------------------------------------------------------
    // UMFPACK
    //--------------------------------------------------------------------------

    printf ("UMFPACK: v%d.%d.%d (%s)\n",
        UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION,
        UMFPACK_DATE) ;

    printf ("%s\n", UMFPACK_VERSION) ;
    printf ("%s", UMFPACK_COPYRIGHT) ;
    printf ("%s", UMFPACK_LICENSE_PART1) ;
    printf ("BLAS used: %s\n", SuiteSparse_BLAS_library ( )) ;
    printf ("BLAS integer size: %d bytes\n",
        (int) sizeof (SUITESPARSE_BLAS_INT)) ;

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
    for (int i = 0 ; i < n ; i++) printf ("x [%d] = %g\n", i, x [i]) ;
    err = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        err = fmax (err, fabs (x [i] - xgood [i])) ;
    }
    printf ("error: %g\n", err) ;
    OK (err < 1e-12) ;
    umfpack_dl_report_status (Control, result) ;
    umfpack_dl_report_info (Control, Info) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    cs_dl_spfree (A) ;
    A = NULL ;
    OK (cholmod_l_finish (&cc)) ;
}

