//------------------------------------------------------------------------------
// CHOLMOD/Supernodal/t_cholmod_super_numeric: cholmod_super_numeric template
//------------------------------------------------------------------------------

// CHOLMOD/Supernodal Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

// Template routine for cholmod_super_numeric.  All xtypes supported, except
// that a zomplex A and F result in a complex L (there is no supernodal zomplex
// L).  The GPU is only used for the double cases, not single.  All dtypes are
// supported (A, L and F must have the same dtype).

//------------------------------------------------------------------------------
// complex arithmetic
//------------------------------------------------------------------------------

#undef L_ENTRY
#undef L_CLEAR
#undef L_ASSIGN
#undef L_MULTADD
#undef L_ASSEMBLE
#undef L_ASSEMBLESUB

#ifdef REAL

    //--------------------------------------------------------------------------
    // A, F, and L are all real
    //--------------------------------------------------------------------------

    #define L_ENTRY 1
    #define L_CLEAR(Lx,p)               Lx [p] = 0
    #define L_ASSIGN(Lx,q, Ax,Az,p)     Lx [q] = Ax [p]
    #define L_MULTADD(Lx,q, Ax,Az,p, f) Lx [q] += Ax [p] * f [0]
    #define L_ASSEMBLE(Lx,q,b)          Lx [q] += b [0]
    #define L_ASSEMBLESUB(Lx,q,C,p)     Lx [q] -= C [p]

#else

    //--------------------------------------------------------------------------
    // A and F are complex or zomplex, L and C are complex
    //--------------------------------------------------------------------------

    #define L_ENTRY 2
    #define L_CLEAR(Lx,p)               Lx [2*(p)] = 0 ; Lx [2*(p)+1] = 0
    #define L_ASSEMBLE(Lx,q,b)          Lx [2*(q)] += b [0] ;
    #define L_ASSEMBLESUB(Lx,q,C,p)                 \
        Lx [2*(q)  ] -= C [2*(p)  ] ;               \
        Lx [2*(q)+1] -= C [2*(p)+1] ;

    #ifdef COMPLEX

        //----------------------------------------------------------------------
        // A, F, L, and C are all complex
        //----------------------------------------------------------------------

        #define L_ASSIGN(Lx,q, Ax,Az,p)                 \
            Lx [2*(q)  ] = Ax [2*(p)  ] ;               \
            Lx [2*(q)+1] = Ax [2*(p)+1]

        #define L_MULTADD(Lx,q, Ax,Az,p, f)                                 \
            Lx [2*(q)  ] += Ax [2*(p)  ] * f [0] - Ax [2*(p)+1] * f [1] ;   \
            Lx [2*(q)+1] += Ax [2*(p)+1] * f [0] + Ax [2*(p)  ] * f [1]

    #else

        //----------------------------------------------------------------------
        // A and F are zomplex, L and C is complex
        //----------------------------------------------------------------------

        #define L_ASSIGN(Lx,q, Ax,Az,p)                 \
            Lx [2*(q)  ] = Ax [p] ;                     \
            Lx [2*(q)+1] = Az [p] ;

        #define L_MULTADD(Lx,q, Ax,Az,p, f)                     \
            Lx [2*(q)  ] += Ax [p] * f [0] - Az [p] * f [1] ;   \
            Lx [2*(q)+1] += Az [p] * f [0] + Ax [p] * f [1]

    #endif
#endif

//------------------------------------------------------------------------------
// blas_dump
//------------------------------------------------------------------------------

#undef BLAS_DUMP_TO_FILE
#undef WHICH

#ifdef BLAS_DUMP

    #if (defined (DOUBLE) && defined (REAL))
    #define WHICH 0
    #elif (defined (SINGLE) && defined (REAL))
    #define WHICH 1
    #elif (defined (DOUBLE) && !defined (REAL))
    #define WHICH 2
    #elif (defined (SINGLE) && !defined (REAL))
    #define WHICH 3
    #endif

    #define BLAS_DUMP_TO_FILE(kernel,m,n,k,lda,ldb,ldc)                     \
    {                                                                       \
        if (Common->blas_dump != NULL)                                      \
        {                                                                   \
            fprintf (Common->blas_dump,                                     \
                "%2d %d %d %6d %6d %6d %6d %6d %6d %12.4e\n",               \
                kernel + WHICH,                 /* dgemm, sgemm, etc.    */ \
                (int) sizeof (SUITESPARSE_BLAS_INT),  /* BLAS integer  */   \
                (int) sizeof (Int),                   /* CHOLMOD index */   \
                (int) m, (int) n, (int) k,          /* problem size */      \
                (int) lda, (int) ldb, (int) ldc,    /* leading dimensions */\
                blas_time) ;                    /* time taken */            \
        }                                                                   \
    }

#else

    #define BLAS_DUMP_TO_FILE(kernel,m,n,k,lda,ldb,ldc)

#endif

//------------------------------------------------------------------------------
// t_cholmod_super_numeric
//------------------------------------------------------------------------------

// This function returns FALSE only if integer overflow occurs in the BLAS.
// It returns TRUE otherwise whether or not the matrix is positive definite.

static int TEMPLATE (cholmod_super_numeric_worker)
(
    // input:
    cholmod_sparse *A,  // matrix to factorize
    cholmod_sparse *F,  // F = A' or A(:,f)'
    Real beta [2],      // beta*I is added to diagonal of matrix to factorize
    // input/output:
    cholmod_factor *L,  // factorization
    // workspace:
    cholmod_dense *Cwork,       // size (L->maxcsize)-by-1
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    Real one [2], zero [2] ;
    #ifdef BLAS_TIMER
    double tstart, blas_time ;
    #endif
    Real *Lx, *Ax, *Fx, *Az, *Fz, *C ;
    Int *Super, *Head, *Ls, *Lpi, *Lpx, *Map, *SuperMap, *RelativeMap, *Next,
        *Lpos, *Fp, *Fi, *Fnz, *Ap, *Ai, *Anz, *Iwork, *Next_save, *Lpos_save,
        *Previous;
    Int nsuper, n, s, k1, k2, nscol, psi, psx, psend, nsrow,
        pj, d, kd1, kd2, info, ndcol, ndrow, pdi, pdx, pdend, pdi1, pdi2,
        ndrow1, ndrow2, dancestor, sparent, dnext, nsrow2, ndrow3,
        stype, Apacked, Fpacked, repeat_supernode, nscol2, ss,
        tail, nscol_new = 0;
    info = 0 ;

    ASSERT (L->dtype == A->dtype) ;

    //--------------------------------------------------------------------------
    // declarations for the GPU
    //--------------------------------------------------------------------------

    // these variables are not used if the GPU module is not installed

    #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
    Int ndescendants, mapCreatedOnGpu, supernodeUsedGPU,
        idescendant, dlarge, dsmall, skips ;
    int iHostBuff, iDevBuff, useGPU, GPUavailable ;
    cholmod_gpu_pointers *gpu_p, gpu_pointer_struct ;
    gpu_p = &gpu_pointer_struct ;
    #endif

    //--------------------------------------------------------------------------
    // guard against integer overflow in the BLAS
    //--------------------------------------------------------------------------

    // If integer overflow occurs in the BLAS, Common->status is set to
    // CHOLMOD_TOO_LARGE, and the contents of Lx are undefined.
    Common->blas_ok = TRUE ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    nsuper = L->nsuper ;
    n = L->n ;

    C = Cwork->x ;      // workspace of size L->maxcsize

    one [0] =  1.0 ;    // ALPHA for *syrk, *herk, *gemm, and *trsm
    one [1] =  0. ;
    zero [0] = 0. ;     // BETA for *syrk, *herk, and *gemm
    zero [1] = 0. ;

    // Iwork must be of size 2n + 5*nsuper, allocated in the caller,
    // cholmod_super_numeric.  The memory cannot be allocated here because the
    // cholmod_super_numeric initializes SuperMap, and cholmod_allocate_work
    // does not preserve existing workspace if the space needs to be increase
    // in size.

    // allocate integer workspace
    Iwork = Common->Iwork ;
    SuperMap    = Iwork ;                                   // size n
    RelativeMap = Iwork + n ;                               // size n
    Next        = Iwork + 2*((size_t) n) ;                  // size nsuper
    Lpos        = Iwork + 2*((size_t) n) + nsuper ;         // size nsuper
    Next_save   = Iwork + 2*((size_t) n) + 2*((size_t) nsuper) ;// size nsuper
    Lpos_save   = Iwork + 2*((size_t) n) + 3*((size_t) nsuper) ;// size nsuper
    Previous    = Iwork + 2*((size_t) n) + 4*((size_t) nsuper) ;// size nsuper

    Map  = Common->Flag ;   // size n, use Flag as workspace for Map array
    Head = Common->Head ;   // size n+1, only Head [0..nsuper-1] used

    Ls = L->s ;
    Lpi = L->pi ;
    Lpx = L->px ;

    Super = L->super ;

    Lx = L->x ;

    #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
    // local copy of useGPU
    if ( (Common->useGPU == 1) && L->useGPU)
    {
        // Initialize the GPU.  If not found, don't use it.
        useGPU = TEMPLATE2 (CHOLMOD (gpu_init))
            (C, L, Common, nsuper, n, Lpi[nsuper]-Lpi[0], gpu_p) ;
    }
    else
    {
        useGPU = 0;
    }
    // fprintf (stderr, "local useGPU %d\n", useGPU) ;
    #endif

    #ifdef BLAS_TIMER
    // clear GPU / CPU statistics
    Common->CHOLMOD_CPU_GEMM_CALLS  = 0 ;
    Common->CHOLMOD_CPU_SYRK_CALLS  = 0 ;
    Common->CHOLMOD_CPU_TRSM_CALLS  = 0 ;
    Common->CHOLMOD_CPU_POTRF_CALLS = 0 ;
    Common->CHOLMOD_GPU_GEMM_CALLS  = 0 ;
    Common->CHOLMOD_GPU_SYRK_CALLS  = 0 ;
    Common->CHOLMOD_GPU_TRSM_CALLS  = 0 ;
    Common->CHOLMOD_GPU_POTRF_CALLS = 0 ;
    Common->CHOLMOD_CPU_GEMM_TIME   = 0 ;
    Common->CHOLMOD_CPU_SYRK_TIME   = 0 ;
    Common->CHOLMOD_CPU_TRSM_TIME   = 0 ;
    Common->CHOLMOD_CPU_POTRF_TIME  = 0 ;
    Common->CHOLMOD_GPU_GEMM_TIME   = 0 ;
    Common->CHOLMOD_GPU_SYRK_TIME   = 0 ;
    Common->CHOLMOD_GPU_TRSM_TIME   = 0 ;
    Common->CHOLMOD_GPU_POTRF_TIME  = 0 ;
    Common->CHOLMOD_ASSEMBLE_TIME   = 0 ;
    Common->CHOLMOD_ASSEMBLE_TIME2  = 0 ;
    #endif

    stype = A->stype ;

    if (stype != 0)
    {
        // F not accessed
        Fp = NULL ;
        Fi = NULL ;
        Fx = NULL ;
        Fz = NULL ;
        Fnz = NULL ;
        Fpacked = TRUE ;
    }
    else
    {
        ASSERT (F != NULL) ;
        ASSERT (L->dtype == F->dtype) ;
        Fp = F->p ;
        Fi = F->i ;
        Fx = F->x ;
        Fz = F->z ;
        Fnz = F->nz ;
        Fpacked = F->packed ;
    }

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    Az = A->z ;
    Anz = A->nz ;
    Apacked = A->packed ;

    // clear the Map so that changes in the pattern of A can be detected

    #ifdef _OPENMP
    int nthreads = cholmod_nthreads ((double) n, Common) ;
    #endif

    Int ii ;
    #pragma omp parallel for num_threads(nthreads) \
        if ( n > 128 ) schedule (static)
    for (ii = 0 ; ii < n ; ii++)
    {
        Map [ii] = EMPTY ;
    }

    // If the matrix is not positive definite, the supernode s containing the
    // first zero or negative diagonal entry of L is repeated (but factorized
    // only up to just before the problematic diagonal entry). The purpose is
    // to provide MATLAB with [R,p]=chol(A) ; columns 1 to p-1 of L=R' are
    // required, where L(p,p) is the problematic diagonal entry.  The
    // repeat_supernode flag tells us whether this is the repeated supernode.
    // Once supernode s is repeated, the factorization is terminated.
    repeat_supernode = FALSE ;

    #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
    if ( useGPU )
    {
        // Case of GPU, zero all supernodes at one time for better performance
        int nthreads = cholmod_nthreads ((double) L->xsize, Common) ;
        TEMPLATE2 (CHOLMOD (gpu_clear_memory))(Lx, L->xsize, nthreads) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // supernodal numerical factorization
    //--------------------------------------------------------------------------

    for (s = 0 ; s < nsuper ; s++)
    {

        //----------------------------------------------------------------------
        // get the size of supernode s
        //----------------------------------------------------------------------

        k1 = Super [s] ;            // s contains columns k1 to k2-1 of L
        k2 = Super [s+1] ;
        nscol = k2 - k1 ;           // # of columns in all of s
        psi = Lpi [s] ;             // pointer to first row of s in Ls
        psx = Lpx [s] ;             // pointer to first row of s in Lx
        psend = Lpi [s+1] ;         // pointer just past last row of s in Ls
        nsrow = psend - psi ;       // # of rows in all of s

        PRINT1 (("====================================================\n"
                 "S "ID" k1 "ID" k2 "ID" nsrow "ID" nscol "ID" psi "ID" psend "
                 ""ID" psx "ID"\n", s, k1, k2, nsrow, nscol, psi, psend, psx)) ;

        //----------------------------------------------------------------------
        // zero the supernode s
        //----------------------------------------------------------------------

        ASSERT ((size_t) (psx + nsrow*nscol) <= L->xsize) ;

        #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
        if ( !useGPU )
        #endif
        {
            // Case of no GPU, zero individual supernodes
            Int pend = psx + nsrow * nscol ;        // s is nsrow-by-nscol

            #ifdef _OPENMP
            double work = (double) (pend - psx) * L_ENTRY ;
            int nthreads = cholmod_nthreads (work, Common) ;
            #endif

            Int p ;
            #pragma omp parallel for num_threads(nthreads)   \
                schedule (static) if ( pend - psx > 1024 )
            for (p = psx ; p < pend ; p++)
            {
                L_CLEAR (Lx,p) ;
            }
        }

        //----------------------------------------------------------------------
        // construct the scattered Map for supernode s
        //----------------------------------------------------------------------

        // If row i is the kth row in s, then Map [i] = k.  Similarly, if
        // column j is the kth column in s, then  Map [j] = k.

        #ifdef _OPENMP
        int nthreads = cholmod_nthreads ((double) nsrow, Common) ;
        #endif

        Int k ;
        #pragma omp parallel for num_threads(nthreads)  \
            if ( nsrow > 128 )
        for (k = 0 ; k < nsrow ; k++)
        {
            PRINT1 (("  "ID" map "ID"\n", Ls [psi+k], k)) ;
            Map [Ls [psi + k]] = k ;
        }

        //----------------------------------------------------------------------
        // when using GPU, reorder supernodes by levels.
        // (all supernodes in a level are independent)
        //----------------------------------------------------------------------

        #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
        if ( useGPU )
        {
            TEMPLATE2 (CHOLMOD (gpu_reorder_descendants))
                ( Common, Super, &s, Lpi, Lpos, Head, Next, Previous,
                  &ndescendants, &tail, &mapCreatedOnGpu, gpu_p ) ;
        }
        #endif

        //----------------------------------------------------------------------
        // copy matrix into supernode s (lower triangular part only)
        //----------------------------------------------------------------------

        Int pk = psx ;

        #ifdef _OPENMP
        double work ;
        if (stype != 0)
        {
            Int pfirst = Ap [k1] ;
            Int plast = (Apacked) ? (Ap [k2]) : (pfirst + Anz [k2-1]) ;
            work = (double) (plast - pfirst) ;
        }
        else
        {
            Int pfirst = Fp [k1] ;
            Int plast  = (Fpacked) ? (Fp [k2]) : (pfirst + Fnz [k2-1]) ;
            work = (double) (plast - pfirst) ;
        }
        nthreads = cholmod_nthreads (work, Common) ;
        #endif

        #pragma omp parallel for num_threads(nthreads) \
            if ( k2-k1 > 64 )
        for (k = k1 ; k < k2 ; k++)
        {
            if (stype != 0)
            {

                //--------------------------------------------------------------
                // copy the kth column of A into the supernode
                //--------------------------------------------------------------

                Int p = Ap [k] ;
                Int pend = (Apacked) ? (Ap [k+1]) : (p + Anz [k]) ;
                for ( ; p < pend ; p++)
                {
                    // row i of L is located in row Map [i] of s
                    Int i = Ai [p] ;
                    if (i >= k)
                    {
                        // If the test is false, the numeric factorization of A
                        // is undefined.  The test does not detect all invalid
                        // entries, only some of them (when debugging is
                        // enabled, and Map is cleared after each step, then
                        // all entries not in the pattern of L are detected).
                        Int imap = Map [i] ;
                        if (imap >= 0 && imap < nsrow)
                        {
                            // Lx [Map [i] + pk] = Ax [p]
                            L_ASSIGN (Lx,(imap+(psx+(k-k1)*nsrow)), Ax,Az,p) ;
                        }
                    }
                }
            }
            else
            {

                //--------------------------------------------------------------
                // copy the kth column of A*F into the supernode
                //--------------------------------------------------------------

                Real fjk [2] ;
                Int pf = Fp [k] ;
                Int pfend = (Fpacked) ? (Fp [k+1]) : (pf + Fnz [k]) ;
                for ( ; pf < pfend ; pf++)
                {
                    Int j = Fi [pf] ;

                    // fjk = Fx [pf]
                    L_ASSIGN (fjk,0, Fx,Fz,pf) ;

                    Int p = Ap [j] ;
                    Int pend = (Apacked) ? (Ap [j+1]) : (p + Anz [j]) ;
                    for ( ; p < pend ; p++)
                    {
                        Int i = Ai [p] ;
                        if (i >= k)
                        {
                            // See the discussion of imap above.
                            Int imap = Map [i] ;
                            if (imap >= 0 && imap < nsrow)
                            {
                                // Lx [Map [i] + pk] += Ax [p] * fjk
                                L_MULTADD (Lx,(imap+(psx+(k-k1)*nsrow)),
                                           Ax,Az,p, fjk) ;
                            }
                        }
                    }
                }
            }
        }

        // add beta to the diagonal of the supernode, if nonzero
        if (beta [0] != 0.0)
        {
            // note that only the real part of beta is used
            Int pk = psx ;
            for (Int k = k1 ; k < k2 ; k++)
            {
                // Lx [pk] += beta [0]
                L_ASSEMBLE (Lx,pk, beta) ;
                pk += nsrow + 1 ;       // advance to the next diagonal entry
            }
        }

        PRINT1 (("Supernode with just A: repeat: "ID"\n", repeat_supernode)) ;
        DEBUG (CHOLMOD(dump_super) (s, Super, Lpi, Ls, Lpx, Lx, L->dtype,
            L_ENTRY, Common)) ;
        PRINT1 (("\n\n")) ;

        //----------------------------------------------------------------------
        // save/restore the list of supernodes
        //----------------------------------------------------------------------

        if (!repeat_supernode)
        {

            // Save the list of pending descendants in case s is not positive
            // definite.  Also save Lpos for each descendant d, so that we can
            // find which part of d is used to update s.
            for (d = Head [s] ; d != EMPTY ; d = Next [d])
            {
                Lpos_save [d] = Lpos [d] ;
                Next_save [d] = Next [d] ;
            }
        }
        else
        {
            // restore Lpos from prior failed supernode
            for (d = Head [s] ; d != EMPTY ; d = Next [d])
            {
                Lpos [d] = Lpos_save [d] ;
                Next [d] = Next_save [d] ;
            }
        }

        //----------------------------------------------------------------------
        // update supernode s with each pending descendant d
        //----------------------------------------------------------------------

        #ifndef NDEBUG
        for (d = Head [s] ; d != EMPTY ; d = Next [d])
        {
            PRINT1 (("\nWill update "ID" with Child: "ID"\n", s, d)) ;
            DEBUG (CHOLMOD(dump_super) (d, Super, Lpi, Ls, Lpx, Lx, L->dtype,
                L_ENTRY, Common)) ;
        }
        PRINT1 (("\nNow factorizing supernode "ID":\n", s)) ;
        #endif

        #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
        if ( useGPU )
        {
            // initialize the buffer counter
            Common->ibuffer = 0;
            supernodeUsedGPU = 0;
            idescendant = 0;
            d = Head[s] ;
            dnext = d;
            dlarge = Next[d] ;
            dsmall = tail;
            GPUavailable = 1;
            skips = 0;
        }
        else
        #endif
        {
            // GPU not installed or not used
            dnext = Head [s] ;
        }

        while
            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            ( (!useGPU && (dnext != EMPTY))
               || (useGPU && (idescendant < ndescendants)))
            #else
            ( dnext != EMPTY )
            #endif
        {

            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            if ( useGPU )
            {

                // Conditionally select the next descendant supernode to
                //  assemble.
                //   + first, select the largest descendant
                //   + subsequently, if gpu host buffers are available, select
                //     the largest remaining descendant for assembly on the GPU
                //   + otherwise select the smallest remaining descendant for
                //     assembly on the CPU
                //
                // The objective is to keep the GPU busy assembling the largest
                // descendants, and simultaneously keep the CPU busy assembling
                // the smallest descendants.
                //
                // As this is called for every descendent supernode, moving
                // this code to t_cholmod_gpu incurs substantial overhead -
                // ~20 GF/s on audikw_1 - so it is being left here.

                iHostBuff =
                    (Common->ibuffer) % CHOLMOD_HOST_SUPERNODE_BUFFERS;
                cudaError_t cuErr;

                if ( idescendant > 0 )  {
                    if ( GPUavailable == -1 || skips > 0) {
                        d = dsmall;
                        dsmall = Previous[dsmall] ;
                        skips--;
                    }
                    else {
                        cuErr = cudaEventQuery
                            ( Common->updateCBuffersFree[iHostBuff] ) ;
                        if ( cuErr == cudaSuccess ) {
                            // buffers are available, so assemble a large
                            // descendant (anticipating that this will be
                            // assembled on the GPU)
                            d = dlarge;
                            dlarge = Next[dlarge] ;
                            GPUavailable = 1;
                            skips = 0;
                        }
                        else
                        {
                            // buffers are not available, so the GPU is busy,
                            // so assemble a small descendant (anticipating
                            // that it will be assembled on the host)
                            d = dsmall;
                            dsmall = Previous[dsmall] ;
                            GPUavailable = 0;

                            // if the GPUs are busy, then do this many
                            // supernodes on the CPU before querying GPUs
                            // again.
                            skips = CHOLMOD_GPU_SKIP;
                        }
                    }
                }

                idescendant++;

            }
            else
            #endif
            {
                // GPU not installed or not used
                d = dnext ;
            }

            //------------------------------------------------------------------
            // get the size of supernode d
            //------------------------------------------------------------------

            kd1 = Super [d] ;       // d contains cols kd1 to kd2-1 of L
            kd2 = Super [d+1] ;
            ndcol = kd2 - kd1 ;     // # of columns in all of d
            pdi = Lpi [d] ;         // pointer to first row of d in Ls
            pdx = Lpx [d] ;         // pointer to first row of d in Lx
            pdend = Lpi [d+1] ;     // pointer just past last row of d in Ls
            ndrow = pdend - pdi ;   // # rows in all of d

            PRINT1 (("Child: ")) ;
            DEBUG (CHOLMOD(dump_super) (d, Super, Lpi, Ls, Lpx, Lx, L->dtype,
                L_ENTRY, Common)) ;

            //------------------------------------------------------------------
            // find the range of rows of d that affect rows k1 to k2-1 of s
            //------------------------------------------------------------------

            Int p = Lpos [d] ;      // offset of 1st row of d affecting s
            Int pdi1 = pdi + p ;    // ptr to 1st row of d affecting s in Ls
            Int pdx1 = pdx + p ;    // ptr to 1st row of d affecting s in Lx

            // there must be at least one row remaining in d to update s
            ASSERT (pdi1 < pdend) ;
            PRINT1 (("Lpos[d] "ID" pdi1 "ID" Ls[pdi1] "ID"\n",
                     Lpos[d], pdi1, Ls [pdi1])) ;
            ASSERT (Ls [pdi1] >= k1 && Ls [pdi1] < k2) ;

            for (pdi2 = pdi1 ; pdi2 < pdend && Ls [pdi2] < k2 ; pdi2++) ;
            ndrow1 = pdi2 - pdi1 ;      // # rows in first part of d
            ndrow2 = pdend - pdi1 ;     // # rows in remaining d

            // rows Ls [pdi1 ... pdi2-1] are in the range k1 to k2-1.  Since d
            // affects s, this set cannot be empty.
            ASSERT (pdi1 < pdi2 && pdi2 <= pdend) ;
            PRINT1 (("ndrow1 "ID" ndrow2 "ID"\n", ndrow1, ndrow2)) ;
            DEBUG (for (p = pdi1 ; p < pdi2 ; p++)
                       PRINT1 (("Ls["ID"] "ID"\n", p, Ls[p]))) ;

            //------------------------------------------------------------------
            // construct the update matrix C for this supernode d
            //------------------------------------------------------------------

            // C = L (k1:n-1, kd1:kd2-1) * L (k1:k2-1, kd1:kd2-1)', except
            // that k1:n-1 refers to all of the rows in L, but many of the
            // rows are all zero.  Supernode d holds columns kd1 to kd2-1 of L.
            // Nonzero rows in the range k1:k2-1 are in the list
            // Ls [pdi1 ... pdi2-1], of size ndrow1.  Nonzero rows in the range
            // k2:n-1 are in the list Ls [pdi2 ... pdend], of size ndrow2.  Let
            // L1 = L (Ls [pdi1 ... pdi2-1], kd1:kd2-1), and let
            // L2 = L (Ls [pdi2 ... pdend],  kd1:kd2-1).  C is ndrow2-by-ndrow1.
            // Let C1 be the first ndrow1 rows of C and let C2 be the last
            // ndrow2-ndrow1 rows of C.  Only the lower triangular part of C1
            // needs to be computed since C1 is symmetric.

            // maxcsize is the largest size of C for all pairs (d,s)
            ASSERT (ndrow2 * ndrow1 <= ((Int) L->maxcsize)) ;

            // compute leading ndrow1-by-ndrow1 lower triangular block of C,
            // C1 = L1*L1'

            ndrow3 = ndrow2 - ndrow1 ;  // number of rows of C2
            ASSERT (ndrow3 >= 0) ;

            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            if ( useGPU )
            {
                // set up GPU to assemble new supernode
                if ( GPUavailable == 1) {
                    if ( ndrow2 * L_ENTRY >= CHOLMOD_ND_ROW_LIMIT &&
                         ndcol * L_ENTRY >= CHOLMOD_ND_COL_LIMIT ) {
                        if ( ! mapCreatedOnGpu ) {
                            TEMPLATE2 ( CHOLMOD (gpu_initialize_supernode))
                                ( Common, nscol, nsrow, psi, gpu_p ) ;
                            mapCreatedOnGpu = 1;
                        }
                    }
                    else {
                        // we've reached the limit of GPU-eligible descendants
                        // flag to stop stop performing cudaEventQueries
                        GPUavailable = -1;
                    }
                }
            }
            #endif

            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            if ( !useGPU
                || GPUavailable!=1
                || !TEMPLATE2 (CHOLMOD (gpu_updateC)) (ndrow1, ndrow2, ndrow,
                        ndcol, nsrow, pdx1, pdi1, Lx, C, Common, gpu_p))
            #endif
            {
                // GPU not installed, or not used

                #ifdef BLAS_TIMER
                Common->CHOLMOD_CPU_SYRK_CALLS++ ;
                tstart = SUITESPARSE_TIME ;
                #endif

                #if (defined (DOUBLE) && defined (REAL))
                SUITESPARSE_BLAS_dsyrk ("L", "N",
                    ndrow1, ndcol,              // N, K: L1 is ndrow1-by-ndcol
                    one,                        // ALPHA:  1
                    Lx + L_ENTRY*pdx1, ndrow,   // A, LDA: L1, ndrow
                    zero,                       // BETA:   0
                    C, ndrow2,                  // C, LDC: C1
                    Common->blas_ok) ;

                #elif (defined (SINGLE) && defined (REAL))
                SUITESPARSE_BLAS_ssyrk ("L", "N",
                    ndrow1, ndcol,              // N, K: L1 is ndrow1-by-ndcol
                    one,                        // ALPHA:  1
                    Lx + L_ENTRY*pdx1, ndrow,   // A, LDA: L1, ndrow
                    zero,                       // BETA:   0
                    C, ndrow2,                  // C, LDC: C1
                    Common->blas_ok) ;

                #elif (defined (DOUBLE) && !defined (REAL))
                SUITESPARSE_BLAS_zherk ("L", "N",
                    ndrow1, ndcol,              // N, K: L1 is ndrow1-by-ndcol
                    one,                        // ALPHA:  1
                    Lx + L_ENTRY*pdx1, ndrow,   // A, LDA: L1, ndrow
                    zero,                       // BETA:   0
                    C, ndrow2,                  // C, LDC: C1
                    Common->blas_ok) ;

                #elif (defined (SINGLE) && !defined (REAL))
                SUITESPARSE_BLAS_cherk ("L", "N",
                    ndrow1, ndcol,              // N, K: L1 is ndrow1-by-ndcol
                    one,                        // ALPHA:  1
                    Lx + L_ENTRY*pdx1, ndrow,   // A, LDA: L1, ndrow
                    zero,                       // BETA:   0
                    C, ndrow2,                  // C, LDC: C1
                    Common->blas_ok) ;
                #endif

                #ifdef BLAS_TIMER
                blas_time = SUITESPARSE_TIME - tstart ;
                Common->CHOLMOD_CPU_SYRK_TIME += blas_time ;
                BLAS_DUMP_TO_FILE (0,           // dsyrk ("L", "N", ...)
                    ndrow1, ndcol, 0,           // N, K, 0
                    ndrow,  ndrow2, 0) ;        // LDA, LDC, 0
                #endif

                // compute remaining (ndrow2-ndrow1)-by-ndrow1 block of C,
                // C2 = L2*L1'
                if (ndrow3 > 0)
                {

                    #ifdef BLAS_TIMER
                    Common->CHOLMOD_CPU_GEMM_CALLS++ ;
                    tstart = SUITESPARSE_TIME ;
                    #endif

                    #if (defined (DOUBLE) && defined (REAL))
                    SUITESPARSE_BLAS_dgemm ("N", "C",
                        ndrow3, ndrow1, ndcol,          // M, N, K
                        one,                            // ALPHA:  1
                        Lx + L_ENTRY*(pdx1 + ndrow1),   // A, LDA: L2
                        ndrow,                          // ndrow
                        Lx + L_ENTRY*pdx1,              // B, LDB: L1
                        ndrow,                          // ndrow
                        zero,                           // BETA:   0
                        C + L_ENTRY*ndrow1,             // C, LDC: C2
                        ndrow2,
                        Common->blas_ok) ;

                    #elif (defined (SINGLE) && defined (REAL))
                    SUITESPARSE_BLAS_sgemm ("N", "C",
                        ndrow3, ndrow1, ndcol,          // M, N, K
                        one,                            // ALPHA:  1
                        Lx + L_ENTRY*(pdx1 + ndrow1),   // A, LDA: L2
                        ndrow,                          // ndrow
                        Lx + L_ENTRY*pdx1,              // B, LDB: L1
                        ndrow,                          // ndrow
                        zero,                           // BETA:   0
                        C + L_ENTRY*ndrow1,             // C, LDC: C2
                        ndrow2,
                        Common->blas_ok) ;

                    #elif (defined (DOUBLE) && !defined (REAL))
                    SUITESPARSE_BLAS_zgemm ("N", "C",
                        ndrow3, ndrow1, ndcol,          // M, N, K
                        one,                            // ALPHA:  1
                        Lx + L_ENTRY*(pdx1 + ndrow1),   // A, LDA: L2
                        ndrow,                          // ndrow
                        Lx + L_ENTRY*pdx1,              // B, LDB: L1, ndrow
                        ndrow,
                        zero,                           // BETA:   0
                        C + L_ENTRY*ndrow1,             // C, LDC: C2
                        ndrow2,
                        Common->blas_ok) ;

                    #elif (defined (SINGLE) && !defined (REAL))
                    SUITESPARSE_BLAS_cgemm ("N", "C",
                        ndrow3, ndrow1, ndcol,          // M, N, K
                        one,                            // ALPHA:  1
                        Lx + L_ENTRY*(pdx1 + ndrow1),   // A, LDA: L2
                        ndrow,                          // ndrow
                        Lx + L_ENTRY*pdx1,              // B, LDB: L1, ndrow
                        ndrow,
                        zero,                           // BETA:   0
                        C + L_ENTRY*ndrow1,             // C, LDC: C2
                        ndrow2,
                        Common->blas_ok) ;
                    #endif

                    #ifdef BLAS_TIMER
                    blas_time = SUITESPARSE_TIME - tstart ;
                    Common->CHOLMOD_CPU_GEMM_TIME += blas_time ;
                    BLAS_DUMP_TO_FILE (4,           // dgemm ("N", "C", ...)
                        ndrow3, ndrow1, ndcol,      // M, N, K
                        ndrow,  ndrow,  ndrow2) ;   // LDA, LDB, LDC
                    #endif
                }

                //--------------------------------------------------------------
                // construct relative map to assemble d into s
                //--------------------------------------------------------------

                DEBUG (CHOLMOD(dump_real) ("C", C, L->dtype,
                    ndrow2, ndrow1, TRUE, L_ENTRY, Common)) ;

                #ifdef _OPENMP
                int nthreads = cholmod_nthreads ((double) ndrow2, Common) ;
                #endif

                Int i ;
                #pragma omp parallel for num_threads(nthreads)   \
                    if ( ndrow2 > 64 )
                for (i = 0 ; i < ndrow2 ; i++)
                {
                    RelativeMap [i] = Map [Ls [pdi1 + i]] ;
                    ASSERT (RelativeMap [i] >= 0 && RelativeMap [i] < nsrow) ;
                }

                //--------------------------------------------------------------
                // assemble C into supernode s using the relative map
                //--------------------------------------------------------------

                #ifdef _OPENMP
                double work = (double) ndcol * (double) ndrow2 * L_ENTRY ;
                nthreads = cholmod_nthreads (work, Common) ;
                #endif

                Int j ;
                #pragma omp parallel for num_threads(nthreads) \
                    if (ndrow1 > 64 )
                for (j = 0 ; j < ndrow1 ; j++)              // cols k1:k2-1
                {
                    ASSERT (RelativeMap [j] == Map [Ls [pdi1 + j]]) ;
                    ASSERT (RelativeMap [j] >= 0 && RelativeMap [j] < nscol) ;
                    Int px = psx + RelativeMap [j] * nsrow ;
                    for (Int i = j ; i < ndrow2 ; i++)          // rows k1:n-1
                    {
                        ASSERT (RelativeMap [i] == Map [Ls [pdi1 + i]]) ;
                        ASSERT (RelativeMap [i] >= j && RelativeMap[i] < nsrow);
                        // Lx [px + RelativeMap [i]] -= C [i + pj]
                        Int q = px + RelativeMap [i] ;
                        L_ASSEMBLESUB (Lx,q, C, i+ndrow2*j) ;
                    }
                }

            }

            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            else
            {
                supernodeUsedGPU = 1;   // GPU was used for this supernode
                Common->ibuffer++;      // gpu_updateC is asynchronous, so use
                                        // the next host buffer for the next
                                        // supernode
                Common->ibuffer = Common->ibuffer%
                    (CHOLMOD_HOST_SUPERNODE_BUFFERS*CHOLMOD_DEVICE_STREAMS) ;
            }
            #endif

            //------------------------------------------------------------------
            // prepare this supernode d for its next ancestor
            //------------------------------------------------------------------

            dnext = Next [d] ;

            if (!repeat_supernode)
            {
                // If node s is being repeated, Head [dancestor] has already
                // been cleared (set to EMPTY).  It must remain EMPTY.  The
                // dancestor will not be factorized since the factorization
                // terminates at node s.
                Lpos [d] = pdi2 - pdi ;
                if (Lpos [d] < ndrow)
                {
                    dancestor = SuperMap [Ls [pdi2]] ;
                    ASSERT (dancestor > s && dancestor < nsuper) ;
                    // place d in the link list of its next ancestor
                    Next [d] = Head [dancestor] ;
                    Head [dancestor] = d ;
                }
            }

        }  // end of descendant supernode loop

        #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
        if ( useGPU ) {
            iHostBuff = (Common->ibuffer)%CHOLMOD_HOST_SUPERNODE_BUFFERS;
            iDevBuff = (Common->ibuffer)%CHOLMOD_DEVICE_STREAMS;

            // combine updates assembled on the GPU with updates
            // assembled on the CPU
            TEMPLATE2 ( CHOLMOD (gpu_final_assembly ))
                ( Common, Lx, psx, nscol, nsrow, supernodeUsedGPU,
                  &iHostBuff, &iDevBuff, gpu_p ) ;
        }
        #endif

        PRINT1 (("\nSupernode with contributions A: repeat: "ID"\n",
                 repeat_supernode)) ;
        DEBUG (CHOLMOD(dump_super) (s, Super, Lpi, Ls, Lpx, Lx, L->dtype,
            L_ENTRY, Common)) ;
        PRINT1 (("\n\n")) ;

        //----------------------------------------------------------------------
        // factorize diagonal block of supernode s in LL'
        //----------------------------------------------------------------------

        // The current supernode s is ready to factorize.  It has been updated
        // by all descendant supernodes.  Let S = the current supernode, which
        // holds rows k1:n-1 and columns k1:k2-1 of the updated matrix.   It
        // splits into two parts:  the square diagonal block S1, and the
        // rectangular part S2.  Here, S1 is factorized into L1*L1' and
        // overwritten by L1.
        //
        // If supernode s is being repeated, only factorize it up to but not
        // including the column containing the problematic entry.

        nscol2 = (repeat_supernode) ? (nscol_new) : (nscol) ;

        #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
        if ( !useGPU
            || !supernodeUsedGPU
            || !TEMPLATE2 (CHOLMOD (gpu_lower_potrf))(nscol2, nsrow, psx, Lx,
                                                   &info, Common, gpu_p))
        #endif
        {
            // Note that the GPU will not be used for the triangular solve
            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            supernodeUsedGPU = 0;
            #endif
            #ifdef BLAS_TIMER
            Common->CHOLMOD_CPU_POTRF_CALLS++ ;
            tstart = SUITESPARSE_TIME ;
            #endif

            #if (defined (DOUBLE) && defined (REAL))
            SUITESPARSE_LAPACK_dpotrf ("L",
                nscol2,                     // N: nscol2
                Lx + L_ENTRY*psx, nsrow,    // A, LDA: S1, nsrow
                info,                       // INFO
                Common->blas_ok) ;

            #elif (defined (SINGLE) && defined (REAL))
            SUITESPARSE_LAPACK_spotrf ("L",
                nscol2,                     // N: nscol2
                Lx + L_ENTRY*psx, nsrow,    // A, LDA: S1, nsrow
                info,                       // INFO
                Common->blas_ok) ;

            #elif (defined (DOUBLE) && !defined (REAL))
            SUITESPARSE_LAPACK_zpotrf ("L",
                nscol2,                     // N: nscol2
                Lx + L_ENTRY*psx, nsrow,    // A, LDA: S1, nsrow
                info,                       // INFO
                Common->blas_ok) ;

            #elif (defined (SINGLE) && !defined (REAL))
            SUITESPARSE_LAPACK_cpotrf ("L",
                nscol2,                     // N: nscol2
                Lx + L_ENTRY*psx, nsrow,    // A, LDA: S1, nsrow
                info,                       // INFO
                Common->blas_ok) ;
            #endif

            #ifdef BLAS_TIMER
            blas_time = SUITESPARSE_TIME - tstart ;
            Common->CHOLMOD_CPU_POTRF_TIME += blas_time ;
            BLAS_DUMP_TO_FILE (8,           // dpotrf ("L", ... )
                nscol2, 0, 0,               // N, 0, 0
                nsrow,  0, 0) ;             // LDA, 0, 0
            #endif
        }

        //----------------------------------------------------------------------
        // check if the matrix is not positive definite
        //----------------------------------------------------------------------

        if (repeat_supernode)
        {
            // the leading part has been refactorized; it must have succeeded
            info = 0 ;

            // zero out the rest of this supernode
            Int p = psx + nsrow * nscol_new ;
            Int pend = psx + nsrow * nscol ;            // s is nsrow-by-nscol
            for ( ; p < pend ; p++)
            {
                // Lx [p] = 0
                L_CLEAR (Lx,p) ;
            }
        }

        // info is set to one in SUITESPARSE_LAPACK_*potrf if blas_ok is FALSE.
        // It is set to zero in dpotrf/zpotrf if the factorization was
        // successful.
        CHECK_FOR_BLAS_INTEGER_OVERFLOW ;

        if (info != 0)
        {
            // Matrix is not positive definite.  dpotrf/zpotrf do NOT report an
            // error if the diagonal of L has NaN's, only if it has a zero.
            if (Common->status == CHOLMOD_OK)
            {
                ERROR (CHOLMOD_NOT_POSDEF, "matrix not positive definite") ;
            }

            // L->minor is the column of L that contains a zero or negative
            // diagonal term.
            L->minor = k1 + info - 1 ;

            // clear the link lists of all subsequent supernodes
            for (ss = s+1 ; ss < nsuper ; ss++)
            {
                Head [ss] = EMPTY ;
            }

            // zero this supernode, and all remaining supernodes
            Int pend = L->xsize ;
            for (Int p = psx ; p < pend ; p++)
            {
                // Lx [p] = 0
                L_CLEAR (Lx,p) ;
            }

            // If L is indefinite, it still contains useful information.
            // Supernodes 0 to s-1 are valid, similar to MATLAB [R,p]=chol(A),
            // where the 1-based p is identical to the 0-based L->minor.  Since
            // L->minor is in the current supernode s, it and any columns to the
            // left of it in supernode s are also all zero.  This differs from
            // [R,p]=chol(A), which contains nonzero rows 1 to p-1.  Fix this
            // by setting repeat_supernode to TRUE, and repeating supernode s.
            //
            // If Common->quick_return_if_not_posdef is true, then the entire
            // supernode s is not factorized; it is left as all zero.

            if (info == 1 || Common->quick_return_if_not_posdef)
            {
                // If the first column of supernode s contains a zero or
                // negative diagonal entry, then it is already properly set to
                // zero.  Also, info will be 1 if integer overflow occured in
                // the BLAS.
                Head [s] = EMPTY ;
                #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
                if ( useGPU ) {
                    CHOLMOD (gpu_end) (Common) ;
                }
                #endif
                return (Common->status >= CHOLMOD_OK) ;
            }
            else
            {
                // Repeat supernode s, but only factorize it up to but not
                // including the column containing the problematic diagonal
                // entry.
                repeat_supernode = TRUE ;
                s-- ;
                nscol_new = info - 1 ;
                continue ;
            }
        }

        //----------------------------------------------------------------------
        // compute the subdiagonal block and prepare supernode for its parent
        //----------------------------------------------------------------------

        nsrow2 = nsrow - nscol2 ;
        if (nsrow2 > 0)
        {
            // The current supernode is columns k1 to k2-1 of L.  Let L1 be the
            // diagonal block (factorized by dpotrf/zpotrf above; rows/cols
            // k1:k2-1), and L2 be rows k2:n-1 and columns k1:k2-1 of L.  The
            // triangular system to solve is L2*L1' = S2, where S2 is
            // overwritten with L2.  More precisely, L2 = S2 / L1' in MATLAB
            // notation.

            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            if ( !useGPU
                || !supernodeUsedGPU
                || !TEMPLATE2 (CHOLMOD(gpu_triangular_solve))
                        (nsrow2, nscol2, nsrow, psx, Lx, Common, gpu_p))
                #endif
            {
                #ifdef BLAS_TIMER
                Common->CHOLMOD_CPU_TRSM_CALLS++ ;
                tstart = SUITESPARSE_TIME ;
                #endif

                #if (defined (DOUBLE) && defined (REAL))
                SUITESPARSE_BLAS_dtrsm ("R", "L", "C", "N",
                    nsrow2, nscol2,                 // M, N
                    one,                            // ALPHA: 1
                    Lx + L_ENTRY*psx, nsrow,        // A, LDA: L1, nsrow
                    Lx + L_ENTRY*(psx + nscol2),    // B, LDB, L2, nsrow
                    nsrow,
                    Common->blas_ok) ;

                #elif (defined (SINGLE) && defined (REAL))
                SUITESPARSE_BLAS_strsm ("R", "L", "C", "N",
                    nsrow2, nscol2,                 // M, N
                    one,                            // ALPHA: 1
                    Lx + L_ENTRY*psx, nsrow,        // A, LDA: L1, nsrow
                    Lx + L_ENTRY*(psx + nscol2),    // B, LDB, L2, nsrow
                    nsrow,
                    Common->blas_ok) ;

                #elif (defined (DOUBLE) && !defined (REAL))
                SUITESPARSE_BLAS_ztrsm ("R", "L", "C", "N",
                    nsrow2, nscol2,                 // M, N
                    one,                            // ALPHA: 1
                    Lx + L_ENTRY*psx, nsrow,        // A, LDA: L1, nsrow
                    Lx + L_ENTRY*(psx + nscol2),    // B, LDB, L2, nsrow
                    nsrow,
                    Common->blas_ok) ;

                #elif (defined (SINGLE) && !defined (REAL))
                SUITESPARSE_BLAS_ctrsm ("R", "L", "C", "N",
                    nsrow2, nscol2,                 // M, N
                    one,                            // ALPHA: 1
                    Lx + L_ENTRY*psx, nsrow,        // A, LDA: L1, nsrow
                    Lx + L_ENTRY*(psx + nscol2),    // B, LDB, L2, nsrow
                    nsrow,
                    Common->blas_ok) ;
                #endif

                #ifdef BLAS_TIMER
                blas_time = SUITESPARSE_TIME - tstart ;
                Common->CHOLMOD_CPU_TRSM_TIME += blas_time ;
                BLAS_DUMP_TO_FILE (12,          // dtrsm ("R", "L", "C", "N"...)
                    nsrow2, nscol2, 0,          // M, N
                    nsrow,  nsrow,  0) ;        // LDA, LDB
                #endif
            }

            CHECK_FOR_BLAS_INTEGER_OVERFLOW ;

            if (!repeat_supernode)
            {
                // Lpos [s] is offset of first row of s affecting its parent
                Lpos [s] = nscol ;
                sparent = SuperMap [Ls [psi + nscol]] ;
                ASSERT (sparent != EMPTY) ;
                ASSERT (Ls [psi + nscol] >= Super [sparent]) ;
                ASSERT (Ls [psi + nscol] <  Super [sparent+1]) ;
                ASSERT (SuperMap [Ls [psi + nscol]] == sparent) ;
                ASSERT (sparent > s && sparent < nsuper) ;
                // place s in link list of its parent
                Next [s] = Head [sparent] ;
                Head [sparent] = s ;
            }
        }
        else
        {
            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            TEMPLATE2 ( CHOLMOD (gpu_copy_supernode) )
                ( Common, Lx, psx, nscol, nscol2, nsrow,
                  supernodeUsedGPU, iHostBuff, gpu_p) ;
            #endif
        }

        Head [s] = EMPTY ;  // link list for supernode s no longer needed

        // clear the Map (debugging only, to detect changes in pattern of A)
        DEBUG (for (k = 0 ; k < nsrow ; k++) Map [Ls [psi + k]] = EMPTY) ;
        DEBUG (CHOLMOD(dump_super) (s, Super, Lpi, Ls, Lpx, Lx, L->dtype,
            L_ENTRY, Common)) ;

        if (repeat_supernode)
        {
            // matrix is not positive definite; finished clean-up for supernode
            // containing negative diagonal

            #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
            if ( useGPU )
            {
                CHOLMOD (gpu_end) (Common) ;
            }
            #endif
            return (Common->status >= CHOLMOD_OK) ;
        }
    }

    // success; matrix is positive definite
    L->minor = n ;

    #if (defined (CHOLMOD_HAS_CUDA) && defined (DOUBLE))
    if ( useGPU )
    {
        CHOLMOD (gpu_end) (Common) ;
    }
    #endif

    return (Common->status >= CHOLMOD_OK) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

