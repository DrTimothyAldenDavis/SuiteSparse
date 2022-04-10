//------------------------------------------------------------------------------
// GB_AxB_dot3_cuda: compute C<M> = A'*B in parallel, on the GPU(s)
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This function only computes C<M>=A'*B on the GPUs.  The mask must be
// present, and not complemented.  The mask is always applied.

extern "C"
{
  #include "GB_mxm.h"
}

#include "GB_cuda.h"

#include "GB_jit_cache.h"

#include "jitFactory.hpp"
#include "GB_cuda_type_wrap.hpp"

template<typename T, typename I>
void print_array(void *arr, I size, const char *name) {
    std::cout << "Printing " << name << std::endl;
    for(I i = 0; i < size; ++i) {
        std::cout << static_cast<T*>(arr)[i] << ", ";
    }
    std::cout << std::endl << "Done." << std::endl;
}


#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                               \
{                                                                       \
    if (Nanobuckets != NULL) rmm_wrap_free (Nanobuckets) ; Nanobuckets = NULL ; \
    if (Blockbucket != NULL) rmm_wrap_free (Blockbucket) ; Blockbucket = NULL ; \
    if (Bucket      != NULL) rmm_wrap_free (Bucket);       Bucket      = NULL ; \
    if (Bucketp     != NULL) rmm_wrap_free (Bucketp);      Bucketp     = NULL ; \
    if (offset      != NULL) rmm_wrap_free (offset);       offset      = NULL ; \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                                     \
{                                                                       \
    GB_FREE_WORKSPACE ;                                                 \
    GB_Matrix_free (&C) ;                                               \
}


GrB_Info GB_AxB_dot3_cuda           // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    printf ("HERE IN cuda dot3, mask_struct is %d\n", Mask_struct) ;

    // when CUDA is enabled, no static headers are used in all of GraphBLAS
    GrB_Info info ;
    ASSERT (C != NULL && !(C->static_header)) ;
    ASSERT (M != NULL && !(M->static_header)) ;
    ASSERT (A != NULL && !(A->static_header)) ;
    ASSERT (B != NULL && !(B->static_header)) ;

    ASSERT_MATRIX_OK (M, "M for dot3 cuda A'*B", GB2) ;
    ASSERT_MATRIX_OK (A, "A for dot3 cuda A'*B", GB2) ;
    ASSERT_MATRIX_OK (B, "B for dot3 cuda A'*B", GB2) ;

    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT (!GB_PENDING (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for dot3 numeric A'*B", GB2) ;

    ASSERT (A->vlen == B->vlen) ;
    GBURBLE ("(GPU dot3) ") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t *Nanobuckets = NULL, *Blockbucket = NULL ;
    int64_t *Bucket = NULL;
    int64_t *Bucketp = NULL;
    int64_t *offset = NULL;

    int device = -1;

    CHECK_CUDA_SIMPLE(cudaSetDevice( 0 ));
    CHECK_CUDA_SIMPLE(cudaGetDevice(&device));

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    const int64_t mvlen = M->vlen ;
    const int64_t mvdim = M->vdim ;
    const int64_t mnz = GB_nnz (M) ;
    const int64_t mnvec = M->nvec ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE( M ) ;

    const int64_t anz = GB_nnz (A) ;
    const int64_t anvec = A->nvec ;

    const int64_t bnz = GB_nnz (B) ;
    const int64_t bnvec = B->nvec ;

    //--------------------------------------------------------------------------
    // allocate C, the same size and # of entries as M
    //--------------------------------------------------------------------------

    // FUTURE: ctype need not be the op->ztype
    GrB_Type ctype = semiring->add->op->ztype ;
    int64_t cvlen = mvlen ;
    int64_t cvdim = mvdim ;
    int64_t cnz = mnz ;
    int64_t cnvec = mnvec ;

    int sparsity_M = (M_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE ;
    info = GB_new_bix (&C, // sparse or hyper (from M), existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        sparsity_M, false, M->hyper_switch, cnvec,
        cnz+1,  // add one to cnz for GB_cumsum of Cwork 
        true, /* not iso: */ false, Context) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

    //int64_t *Citemp =  C->i ;        
    //auto *Cxtemp = C->x ;        
    //cudaMalloc ((void**) &(C->i), cnz * sizeof( int64_t) ); 
    //cudaMalloc ((void**) &(C->x), cnz * C->type->size ); 
    CHECK_CUDA_SIMPLE(cudaMemAdvise( C->i, (cnz+1) * sizeof ( int64_t), cudaMemAdviseSetPreferredLocation, device));
    CHECK_CUDA_SIMPLE(cudaMemAdvise( C->x, (cnz+1) * C->type->size , cudaMemAdviseSetPreferredLocation, device));


    //--------------------------------------------------------------------------
    // copy Mp and Mh into C
    //--------------------------------------------------------------------------

    CHECK_CUDA_SIMPLE(cudaMemcpy (C->p, M->p, (cnvec+1) * sizeof (int64_t), cudaMemcpyDefault)) ;
    if (M_is_hyper)
    { 
        // FIXME
        CHECK_CUDA_SIMPLE(cudaMemcpy (C->h, M->h, cnvec * sizeof (int64_t), cudaMemcpyDefault)) ;
    }

    C->magic = GB_MAGIC ;
    C->nvec_nonempty = M->nvec_nonempty ;
    C->nvec = M->nvec ;
    // the dot3 CUDA kernel will produce C->i with jumbled indices
    C->jumbled = true ;

    GBURBLE ("(GPU C created and copied from M) ") ;
    //--------------------------------------------------------------------------
    // stringify the semiring and the mask
    //--------------------------------------------------------------------------

    GB_cuda_semiring_factory mysemiring = GB_cuda_semiring_factory ( ) ;

    // (1) create the semiring code and name
    mysemiring.semiring_factory ( semiring, flipxy,
        ctype, M->type, A->type, B->type, Mask_struct,  // matrix types
        false, GB_sparsity(C), GB_sparsity(M), GB_sparsity(A), GB_sparsity(B) ) ;

    // (2) ensure the jitifier has "GB_semiring_[mysemiring.sr_code].h"
    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (mysemiring) ;

    GBURBLE ("(GPU stringified) ") ;
    //--------------------------------------------------------------------------
    // construct the tasks for phase1 and phase2
    //--------------------------------------------------------------------------

    // on the CPU: nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;
    // on the GPU:
    phase1launchFactory p1lf(mysemiring);
    phase2launchFactory p2lf;
    phase2endlaunchFactory p2elf;


    // # of threads in phase1 and phase2 kernel launches must be the same
    int nthrd = p2lf.get_threads_per_block();
    int ntasks = p2elf.get_number_of_blocks(M);

    int64_t nanobuckets_size = NBUCKETS * nthrd * ntasks;
    int64_t blockbuckets_size = NBUCKETS * ntasks;

    Nanobuckets = (int64_t*)rmm_wrap_malloc(nanobuckets_size * sizeof (int64_t));
    Blockbucket = (int64_t*)rmm_wrap_malloc(blockbuckets_size * sizeof (int64_t));
    Bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));
    Bucket = (int64_t*)rmm_wrap_malloc(mnz * sizeof (int64_t));
    offset = (int64_t*)rmm_wrap_malloc(NBUCKETS * sizeof (int64_t));

    CHECK_CUDA_SIMPLE(cudaMemset(Nanobuckets, 0, nanobuckets_size * sizeof(int64_t)));
    CHECK_CUDA_SIMPLE(cudaMemset(Blockbucket, 0, blockbuckets_size * sizeof(int64_t)));
    CHECK_CUDA_SIMPLE(cudaMemset(Bucketp, 0, (NBUCKETS+1) * sizeof(int64_t)));
    CHECK_CUDA_SIMPLE(cudaMemset(Bucket, 0, mnz * sizeof(int64_t)));
    CHECK_CUDA_SIMPLE(cudaMemset(offset, 0, NBUCKETS * sizeof(int64_t)));

    //--------------------------------------------------------------------------
    // phase1 and phase2: place each C(i,j) in a bucket
    //--------------------------------------------------------------------------

    CHECK_CUDA_SIMPLE(cudaMemAdvise( Bucketp, (NBUCKETS+1) * sizeof ( int64_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA_SIMPLE(cudaMemAdvise( Bucketp, (NBUCKETS+1) * sizeof ( int64_t), cudaMemAdviseSetAccessedBy, device));

    offset = (int64_t*)rmm_wrap_malloc( (NBUCKETS)*sizeof(int64_t)) ;
    CHECK_CUDA_SIMPLE(cudaMemAdvise( offset, NBUCKETS * sizeof ( int64_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA_SIMPLE(cudaMemAdvise( offset, NBUCKETS * sizeof ( int64_t), cudaMemAdviseSetAccessedBy, device));

    memset( offset, 0, NBUCKETS * sizeof(int64_t) );

    //--------------------------------------------------------------------------
    // Pre-fetch arrays that will be used on the device
    //--------------------------------------------------------------------------

    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( M->p, (mnvec+1) * sizeof (int64_t), device, NULL)) ; //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( M->i, mnz * sizeof (int64_t), device, NULL )) ; //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( M->x, mnz * M->type->size, device, NULL )) ; //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( C->i, (cnz+1) * sizeof (int64_t), device, NULL )); //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( C->x, (cnz+1) * C->type->size, device, NULL )); //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( A->p, (anvec+1) * sizeof (int64_t), device, NULL)); // stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( A->i, anz * sizeof (int64_t), device, NULL )) ; //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( A->x, anz * A->type->size, device, NULL )) ; //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( B->p, (bnvec+1) * sizeof (int64_t), device, NULL)); //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( B->i, bnz * sizeof (int64_t), device, NULL )); //stream_data) ;
    CHECK_CUDA_SIMPLE(cudaMemPrefetchAsync( B->x, bnz * B->type->size, device, NULL )); //stream_data) ;

    // The work to compute C(i,j) is held in Ci [p], if C(i,j) appears in
    // as the pth entry in C.
    
    //----------------------------------------------------------------------
    // phase1: assign each C(i,j) to a bucket, and count them
    //----------------------------------------------------------------------

    GBURBLE ("(GPU phase1 start) ") ;

    p1lf.jitGridBlockLaunch(Nanobuckets, Blockbucket, C, M, A, B);

    GBURBLE ("(GPU phase1 done) ") ;

    print_array<int64_t>(Nanobuckets, nanobuckets_size, "Nanobuckets");
    print_array<int64_t>(Blockbucket, blockbuckets_size , "Blockbucket");

    //----------------------------------------------------------------------
    // phase2: cumsum across the blockbuckets, propagate to thread level
    //----------------------------------------------------------------------

    GBURBLE ("(GPU phase1 start) ") ;

    p2lf.jitGridBlockLaunch(Blockbucket, offset, M);

    int64_t s= 0;
    for ( int bucket = 0 ; bucket < NBUCKETS+1; ++bucket)
    {
        Bucketp[bucket] = s;
        s+= offset[bucket];
        printf("bucketp[%d] = %ld, offset=%ld\n", bucket, Bucketp[bucket], offset[bucket]);
    }

    GBURBLE ("(GPU phase2 done) ") ;

    GBURBLE ("(GPU phase2end start) ") ;

    p2elf.jitGridBlockLaunch(Nanobuckets, Blockbucket,
                             Bucketp, Bucket, offset, C, M);

    GBURBLE ("(GPU phase2end done) ") ;

    print_array<int64_t>(Bucket, mnz , "Bucket");
    print_array<int64_t>(M->i, mnz , "M->i");
    print_array<int64_t>(C->i, mnz , "C->i");

    //----------------------------------------------------------------------
    // phase3: do the numerical work
    //----------------------------------------------------------------------

    print_array<int64_t>(Bucketp, NBUCKETS + 1 , "Bucketp");
    C->nzombies = Bucketp[1];  //set pre-zombie counts
    printf("pre-kernel C->nzombies=%ld\n", C->nzombies);

    for ( int bucket = 1 ; bucket < NBUCKETS; ++bucket)
    {
        int64_t start = Bucketp[bucket];
        int64_t end = Bucketp[bucket+1];


        if(end - start > 0) {
            printf("Executing bucket: %d with %ld edges\n", bucket, end-start);
            // TODO: We might want to consider submitting these in different cuda streams (maybe use cuda stream pool?)
            phase3launchFactory p3lf(mysemiring, (GB_bucket_code)bucket);
            p3lf.jitGridBlockLaunch(start, end, Bucketp, Bucket, C,  M, A, B);
        } else {
            printf("Skipping bucket %d, no work to do\n", bucket);
        }

        GBURBLE ("(GPU phase3 done ) ") ;
    }
    C->nzombies += Bucketp[1];
    printf("C->p[0]=%ld\n", C->p[0]);
    printf("C->p[1]=%ld\n", C->p[1]);
    printf("C->nzombies=%ld\n", C->nzombies);

    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS; 
}

