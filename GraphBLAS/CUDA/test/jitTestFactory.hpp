// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
//#include "GB_binary_search.h"
#include "GpuTimer.h"
#include "GB_cuda_buckets.h"
#include "../../rmm_wrap/rmm_wrap.h"
#include <gtest/gtest.h>
#include "test_data.hpp"

extern "C" {
    #include "GB.h"
}

#include "../jitFactory.hpp"
#include "dataFactory.hpp"

////Operations for test results on CPU
//template<typename T> T myOP_plus( T a, T b) { return  a + b;}
//template<typename T> T myOP_min ( T a, T b) { return  a < b ? a : b;}
//template<typename T> T myOP_max ( T a, T b) { return  a > b ? a : b;}
//template<typename T> T myOP_first ( T a, T b) { return  a ;}
//template<typename T> T myOP_second ( T a, T b) { return  b ;}
//template<typename T> T myOP_times ( T a, T b) { return  a * b ;}
//
//template<typename T> T (*myOpPTR)(T a, T b);
//template<typename T> T (*ADD_ptr)(T a, T b);
//template<typename T> T (*MUL_ptr)(T a, T b);

//AxB_dot3_phase1 kernels
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory( int64_t , int64_t , int64_t , int64_t ) ;

//AxB_dot3_phase2 kernels
template <typename T_C>
bool test_AxB_dot3_phase2_factory( int , int64_t , int64_t , int64_t, int64_t ) ;

////AxB_dot3_phase3 kernels
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_dndn_factory( int , int64_t , int64_t , int64_t , std::string&) ;
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_vsvs_factory( int , int64_t , int64_t , int64_t , std::string&) ;
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_spdn_factory( int , int64_t , int64_t , int64_t , std::string&) ;
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_vssp_factory( int , int64_t , int64_t , int64_t , std::string&) ;
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_mp_factory( int , int64_t , int64_t , int64_t , std::string&) ;
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_warp_factory( int , int64_t , int64_t , int64_t , std::string&) ;


//Fixture to generate valid inputs and hold them for tests
class AxB_dot3_Test : public ::testing::Test
{
   void SetUp() {}

   void TearDown() {}
};

template<typename T, typename I>
void print_array(void *arr, I size, const char *name) {
    std::cout << "Printing " << name << std::endl;
    for(I i = 0; i < size; ++i) {
        std::cout << static_cast<T*>(arr)[i] << ", ";
    }
    std::cout << std::endl << "Done." << std::endl;
}

//------------------------------------------------------------------------------
// test_AxB_phase1_factory: test phase1
//------------------------------------------------------------------------------

// Test generator code, to allow parameterized tests
// Uses jitFactory, dataFactory and GB_jit 
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, GrB_Monoid monoid, GrB_BinaryOp binop)
{

    int gpuID;
    cudaGetDevice( &gpuID);

    std::cout<< "found device "<<gpuID<<std::endl;

    /**************************
     * Create reference and input data
     */

    // FIXME: This should be getting set automatically somehow.
    bool flipxy = false;
    bool mask_struct = false;
    bool mask_comp = false;

    SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G(N, N);
    int64_t Annz = N*N;
    int64_t Bnnz = N*N;
    int64_t Cnz = N;
    float Cnzpercent = (float) Cnz/(N*N);

    // TODO: Allocate and fill arrays for buckets and nano buckets
    G.init_A(Annz, GxB_SPARSE, GxB_BY_ROW);
    G.init_B(Bnnz, GxB_SPARSE, GxB_BY_ROW);
    G.init_C(Cnzpercent);
    G.fill_buckets( TB ); // all elements go to testbucket= TB

    GrB_Matrix C = G.getC();
    GrB_Matrix M = G.getM();
    GrB_Matrix A = G.getA();
    GrB_Matrix B = G.getB();

    /************************
     * Create semiring factory
     */

    GB_cuda_semiring_factory mysemiringfactory = GB_cuda_semiring_factory ( ) ;
    GrB_Semiring mysemiring;
    auto grb_info = GrB_Semiring_new(&mysemiring, monoid, binop);
    GRB_TRY (grb_info) ;

    mysemiringfactory.semiring_factory ( mysemiring, flipxy,
                                         C->type,
                                         M->type,
                                         A->type,
                                         B->type,
                                         mask_struct,  // matrix types
                                         mask_comp, GB_sparsity(C),
                                         GB_sparsity(M),
                                         GB_sparsity(A),
                                         GB_sparsity(B));

    /********************
     * Launch kernel
     */

    phase1launchFactory p1lF(mysemiringfactory);

    GpuTimer kernTimer;
    kernTimer.Start();

    int nthrd = p1lF.get_threads_per_block();
    int ntasks = p1lF.get_number_of_blocks(M);

    // TODO: Verify that RMM is checking and throwing exceptions
    int nanobuckets_size = NBUCKETS * nthrd * ntasks;
    int blockbuckets_size = NBUCKETS * ntasks;

    printf("nanobuckets_size: %d\n", nanobuckets_size);
    printf("blockbuckets_size: %d\n", blockbuckets_size);

    int64_t *Nanobuckets = (int64_t*)rmm_wrap_malloc(nanobuckets_size * sizeof (int64_t));
    int64_t *Blockbucket = (int64_t*)rmm_wrap_malloc(blockbuckets_size * sizeof (int64_t));
//
//    std::cout << "INvoking grid block launch for phase1" << std::endl;
    p1lF.jitGridBlockLaunch(Nanobuckets, Blockbucket, C, M, A, B);
    kernTimer.Stop();
    std::cout<<"returned from phase1 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
    print_array<int64_t>(Nanobuckets, nanobuckets_size, "Nanobuckets");
    print_array<int64_t>(Blockbucket, blockbuckets_size, "Blockbucket");
    std::cout<<"==== phase1 done=============================" <<std::endl;
//
    rmm_wrap_free(Nanobuckets);
    rmm_wrap_free(Blockbucket);

    G.del();
//
    return true;
}

//------------------------------------------------------------------------------
// test_AxB_phase2_factory: test phase2 and phase2end
//------------------------------------------------------------------------------

template <typename T_C>
bool test_AxB_phase2_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz)
{

    int gpuID;
    cudaGetDevice( &gpuID);

    std::cout<< "found device "<<gpuID<<std::endl;

    phase2launchFactory p2lF;
    phase2endlaunchFactory p2elF;

    SpGEMM_problem_generator<T_C, T_C, T_C, T_C> G(N, N);
    int64_t Annz = N*N;
    int64_t Bnnz = N*N;
    int64_t Cnz = N;
    float Cnzpercent = (float) Cnz/(N*N);

    G.init_A(Annz, GxB_SPARSE, GxB_BY_ROW);
    G.init_B(Bnnz, GxB_FULL, GxB_BY_ROW);
    G.init_C(Cnzpercent);
    G.fill_buckets( TB ); // all elements go to testbucket= TB
    G.loadCj(); // FIXME: Figure out why this is needed here


    GrB_Matrix C = G.getC();
    GrB_Matrix M = G.getM();       // note: values are not accessed

   GpuTimer kernTimer;
   kernTimer.Start();
    const int64_t mnz = GB_nnz (M) ;

   int nthrd = p2lF.get_threads_per_block();
   int ntasks = p2elF.get_number_of_blocks(M);

    // fabricate data as if it came from phase1:
    int64_t *nanobuckets = (int64_t*)rmm_wrap_malloc(NBUCKETS * nthrd * ntasks * sizeof (int64_t));
    int64_t *blockbucket = (int64_t*)rmm_wrap_malloc(NBUCKETS * ntasks * sizeof (int64_t));
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));
    int64_t *bucket = (int64_t*)rmm_wrap_malloc(mnz * sizeof (int64_t));
    int64_t *offset = (int64_t*)rmm_wrap_malloc(NBUCKETS * sizeof (int64_t));

    std::cout << "nthrd: " << nthrd << ", ntasks: " << ntasks << std::endl;
    fillvector_constant(NBUCKETS * nthrd * ntasks, nanobuckets, (int64_t)1);
    fillvector_constant(NBUCKETS * ntasks, blockbucket, (int64_t)1);
    fillvector_constant(NBUCKETS, bucketp, (int64_t)1);

    print_array<int64_t>(nanobuckets, NBUCKETS*nthrd*ntasks, "nanobuckets");
    print_array<int64_t>(blockbucket, NBUCKETS*ntasks, "blockbucket");
//
//    // launch phase2 (just with p2ntasks as the # of tasks)
    p2lF.jitGridBlockLaunch(blockbucket, offset, M);
//
//    // do the reduction between phase2 and phase2end
    int64_t s= 0;
    for ( int bucket = 0 ; bucket < NBUCKETS+1; ++bucket)
    {
        bucketp[bucket] = s;
        s+= offset[bucket];
        //printf("bucketp[%d] = %ld\n", bucket, Bucketp[bucket]);
    }

    // launch phase2end: note same # of tasks as phase1
    p2elF.jitGridBlockLaunch( nanobuckets, blockbucket,
                              bucketp, bucket, offset, C,
                              M);
//    kernTimer.Stop();
//    std::cout<<"returned from phase2 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//
    print_array<int64_t>(bucketp, NBUCKETS, "bucketp");
    print_array<int64_t>(bucket, mnz, "bucket");
    std::cout<<"phase2 kernel done =================="<<std::endl;
    rmm_wrap_free(nanobuckets);
    rmm_wrap_free(blockbucket);
    rmm_wrap_free(bucketp);
    rmm_wrap_free(bucket);
    rmm_wrap_free(offset);
    G.del();
   return true;
}

template<typename T>
void make_grb_matrix(GrB_Matrix &mat, int64_t n_rows, int64_t n_cols, std::vector<int64_t> &indptr, std::vector<int64_t> &indices, std::vector<T> &data,
                     int gxb_sparsity_control = GxB_SPARSE, int gxb_format = GxB_BY_ROW) {

    GrB_Type type = cuda::jit::to_grb_type<T>();

    GRB_TRY (GrB_Matrix_new (&mat, type, n_rows, n_cols)) ;

    for(int64_t row = 0; row < n_rows; ++row) {
        int64_t start = indptr[row];
        int64_t stop = indptr[row+1];

        for(int64_t offset = start; offset < stop; ++offset) {
            GrB_Index i = (GrB_Index) row;
            GrB_Index j = (GrB_Index) indices[offset];
            T x = data[offset];

            cuda::jit::set_element<T> (mat, x, i, j) ;
        }
    }

    GRB_TRY (GrB_Matrix_wait (mat, GrB_MATERIALIZE)) ;
    GRB_TRY (GB_convert_any_to_non_iso (mat, true, NULL)) ;
    // TODO: Need to specify these
    GRB_TRY (GxB_Matrix_Option_set (mat, GxB_SPARSITY_CONTROL, gxb_sparsity_control)) ;
    GRB_TRY (GxB_Matrix_Option_set(mat, GxB_FORMAT, gxb_format));
    GRB_TRY (GxB_Matrix_fprint (mat, "my mat", GxB_SHORT_VERBOSE, stdout)) ;

    bool iso ;
    GRB_TRY (GxB_Matrix_iso (&iso, mat)) ;
    if (iso)
    {
        printf ("Die! (cannot do iso)\n") ;
        GrB_Matrix_free (&mat) ;
    }

}

template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
bool test_AxB_dot3_full_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz,
                                 GrB_Monoid monoid, GrB_BinaryOp binop) {

    // FIXME: Allow the adaptive tests in this guy

    //Generate test data and setup for using a jitify kernel with 'bucket' interface
    // The testBucket arg tells the generator which bucket we want to exercise
    int64_t Annz;
    int64_t Bnnz;

    switch(TB) {
        case GB_BUCKET_DNDN:
            Annz = N * N;
            Bnnz = N * N;
            break;
        case GB_BUCKET_SPDN:
            Annz = N * N;
            Bnnz = N * 5;
            break;
        case GB_BUCKET_VSSP:
            Annz = N * 2;
            Bnnz = N * 10;
            break;
        case GB_BUCKET_VSVS_4:
        case GB_BUCKET_VSVS_16:
        case GB_BUCKET_VSVS_64:
        case GB_BUCKET_VSVS_256:
            Annz = N * 2;
            Bnnz = N * 4;
            break;
        case GB_BUCKET_MERGEPATH:
            Annz = N * 5;
            Bnnz = N * 2;
            break;
        default:
            printf("Bucket not yet being tested!\n");
            exit(1);
    }
    int64_t Cnz = N;
    float Cnzpercent = (float) Cnz/(N*N);

    // FIXME: make this an argument
    bool Mask_struct = true;

    std::cout << "Getting test data" << std::endl;
    // FIXME: These need to be set based on the bucket being tested
//    TestData<T_A, T_B, T_C, T_M> data = *make_karate_tricount<T_A, T_B, T_C, T_M>();

    std::cout << "Creating problem gen" << std::endl;
//    N = data.A_indptr.size()-1;
    SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G(N, N);
    G.init_C(float(Cnz) / (N * N));

//    GrB_Matrix A;
//    GrB_Matrix B;
//    GrB_Matrix C;
//    GrB_Matrix M;
//
//    GrB_Matrix C_actual = G.getC();

//    make_grb_matrix<T_A>(A, data.A_indptr, data.A_indices, data.A_data, GxB_SPARSE);
//    make_grb_matrix<T_B>(B, data.B_indptr, data.B_indices, data.B_data, GxB_FULL, GxB_BY_ROW);
//    make_grb_matrix<T_C>(C, data.C_indptr, data.C_indices, data.C_data);
//    make_grb_matrix<T_M>(M, data.M_indptr, data.M_indices, data.M_data);


//    std::cout << "Filling A" << std::endl;
    G.init_A(Annz, GxB_SPARSE, GxB_BY_ROW, 543210, 0, 2);
//    std::cout << "Filling B" << std::endl;

    G.init_B(Bnnz, GxB_SPARSE, GxB_BY_ROW, 32, 0, 2);

    /**
     * For testing, we need to create our output C and configure
     * it w/ the necessary sparsity.
     */
    G.fill_buckets( TB); // all elements go to testbucket= TB

    GrB_Matrix C = G.getC();
    GrB_Matrix M = G.getM();
    GrB_Matrix A = G.getA();
    GrB_Matrix B = G.getB();

//    GRB_TRY (GxB_Matrix_fprint (A, "A", GxB_SHORT_VERBOSE, stdout)) ;
//    GRB_TRY (GxB_Matrix_fprint (B, "B", GxB_SHORT_VERBOSE, stdout)) ;
//    GRB_TRY (GxB_Matrix_fprint (M, "M", GxB_SHORT_VERBOSE, stdout)) ;
//    GRB_TRY (GxB_Matrix_fprint (C, "C", GxB_SHORT_VERBOSE, stdout)) ;
//
    std::cout << "Building semiring factgory" << std::endl;
    GB_cuda_semiring_factory mysemiringfactory = GB_cuda_semiring_factory ( ) ;
    GrB_Semiring mysemiring;
    auto grb_info = GrB_Semiring_new(&mysemiring, monoid, binop);
    GRB_TRY (grb_info) ;

    bool flipxy = false;
    bool mask_struct = false;
    bool mask_comp = false;
//    GrB_Matrix C_actual = C;

    mysemiringfactory.semiring_factory ( mysemiring, flipxy,
                                         C->type, M->type,
                                         A->type, B->type,
                                         mask_struct,  // matrix types
                                         mask_comp, GB_sparsity(C),
                                         GB_sparsity(M),
                                         GB_sparsity(A),
                                         GB_sparsity(B) ) ;

    bool result = false;

    /**
     * Run Phase 1: Compute nanobuckets and blockbuckets
     */
    const int64_t mnz = GB_nnz (M) ;

    int chunk_size = 128;

    int number_of_sms = GB_Global_gpu_sm_get (0);
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));

    CHECK_CUDA(cudaMemset(bucketp, 0, (NBUCKETS+1)*sizeof(int64_t)));

    int64_t *bucket = (int64_t*)rmm_wrap_malloc(Cnz * sizeof (int64_t));

    /**
     * Run Phase 3: Execute dot3 on all buckets
     */
    for (int b =0; b < 12; ++b) {// loop on buckets
        if (b == TB) {
            G.fill_buckets(b);
            int64_t *Bucket = G.getBucket();
            int64_t *BucketStart = G.getBucketStart();

            int64_t b_start = BucketStart [b] ;
            int64_t b_end   = BucketStart [b+1] ;
            int64_t nvecs = b_end - b_start ;

            if (nvecs > 0) std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;

            G.loadCj();

           GpuTimer kernTimer;
           kernTimer.Start();

           GB_cuda_mxm_phase3(mysemiringfactory, (GB_bucket_code )b,
                              b_start, b_end, bucketp, Bucket, C, M, B, A);

            print_array<int64_t>(bucketp, NBUCKETS+1, "bucketp");

           kernTimer.Stop();

           std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
           GRB_TRY (GxB_Matrix_fprint (C, "C GPU", GxB_SHORT_VERBOSE, stdout)) ;

            GrB_Matrix C_actual;
            GrB_Type type = cuda::jit::to_grb_type<T_C>();
            GRB_TRY (GrB_Matrix_new (&C_actual, type, N, N)) ;

            // ensure the GPU is not used
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_NEVER)) ;

            // Use GrB_DESC_S for structural because dot3 mask will never be complemented
            GRB_TRY (GrB_mxm(C_actual, M, NULL, mysemiring, A, B,
                Mask_struct ? GrB_DESC_ST1 : GrB_DESC_T1));
//            GRB_TRY (GrB_mxm(C_actual, M, NULL, mysemiring, A, B,
//                             Mask_struct ? GrB_DESC_S : NULL));

            GRB_TRY (GxB_Matrix_fprint (M, "M actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (A, "A actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (B, "B actual", GxB_SHORT_VERBOSE, stdout));

            GRB_TRY(GrB_Matrix_wait(C, GrB_MATERIALIZE));
            GRB_TRY(GrB_Matrix_wait(C_actual, GrB_MATERIALIZE));

            GRB_TRY (GxB_Matrix_fprint (C, "C GPU", GxB_COMPLETE, stdout));
            GRB_TRY (GxB_Matrix_fprint (C_actual, "C_actual", GxB_COMPLETE, stdout));
            // compare
            double tol = 0 ;
            GrB_Index nvals1 = 0, nvals2 = 0 ;
            GRB_TRY (GrB_Matrix_nvals (&nvals1, C)) ;
            GRB_TRY (GrB_Matrix_nvals (&nvals2, C_actual)) ;
            if (nvals1 != nvals2) { printf ("!!\n") ; abort ( ) ; } 
            GrB_Index nrows, ncols ;
            GrB_Matrix_nrows (&nrows, C) ;
            GrB_Matrix_ncols (&ncols, C) ;

            GrB_Matrix T;

            GRB_TRY (GrB_Matrix_new (&T, GrB_BOOL, nrows, ncols)) ;
            GrB_BinaryOp op = NULL;
            GrB_UnaryOp op_abs = NULL ;
            if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
            else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
            else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
            else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
            else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
            else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
            else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
            else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
            else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
            else if (type == GrB_FP32  )
            {
                op = (tol == 0)? GrB_EQ_FP32 : GrB_MINUS_FP32   ;
                op_abs = GrB_ABS_FP32 ;
            }
            else if (type == GrB_FP64  )
            {
                op = (tol == 0)? GrB_EQ_FP64 : GrB_MINUS_FP64   ;
                op_abs = GrB_ABS_FP64 ;
            }
            else if (type == GxB_FC32  )
            {
                op = (tol == 0)? GxB_EQ_FC32 : GxB_MINUS_FC32   ;
                op_abs = GxB_ABS_FC32 ;
            }
            else if (type == GxB_FC64  )
            {
                op = (tol == 0)? GxB_EQ_FC64 : GxB_MINUS_FC64   ;
                op_abs = GxB_ABS_FC64 ;
            }


            // Diff = C - C_actual
            GrB_Matrix Diff ;
            GRB_TRY (GrB_Matrix_new (&Diff, GrB_FP64, nrows, ncols)) ;
            GRB_TRY (GrB_Matrix_apply (Diff, NULL, NULL, GrB_AINV_FP64, C_actual, NULL)) ;
            GRB_TRY (GrB_Matrix_eWiseAdd_BinaryOp (Diff, NULL, NULL, GrB_PLUS_FP64,
                C, Diff, NULL)) ;
            GRB_TRY (GxB_Matrix_fprint (Diff, "Diff actual", GxB_COMPLETE, stdout));
            GRB_TRY (GrB_Matrix_free (&Diff)) ;

            if (tol == 0)
            {
                // check for perfect equality
                GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, op, C, C_actual,
                    NULL)) ;
                GrB_Index nvals3 = 1 ;
                GRB_TRY (GxB_Matrix_fprint (T, "T actual", GxB_SHORT_VERBOSE, stdout));
                GRB_TRY (GrB_Matrix_nvals (&nvals3, T)) ;
                if (nvals1 != nvals3) { printf ("!!\n") ; abort ( ) ; } 
                bool is_same = false ;
                GRB_TRY (GrB_Matrix_reduce_BOOL (&is_same, NULL, GrB_LAND_MONOID_BOOL,
                    T, NULL)) ;
                if (!is_same) { printf ("!!\n") ; abort ( ) ; } 
                GRB_TRY (GrB_Matrix_free (&T)) ;
            }
            else
            {
                // TODO: check with roundoff
                { printf ("!!\n") ; abort ( ) ; } 
            }

            // re-enable the GPU
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_ALWAYS)) ;
         }
        }

    rmm_wrap_free(bucket);
    rmm_wrap_free(bucketp);

    G.del();

    return result;
}

template <typename T>
bool test_reduce_factory(unsigned int N, GrB_Monoid monoid ) {

    //std::cout<<" alloc'ing data and output"<<std::endl;
    std::vector<int64_t> indptr(N+1);
    std::vector<int64_t> index(N);
    std::vector<T> d_data(N);

    indptr[N] = N;
    fillvector_linear<int64_t>((int)N, indptr.data(), (int64_t)0);
    fillvector_constant<int64_t>((int)N, index.data(), (int64_t)1);
    fillvector_linear<T> ( N, d_data.data());

    GrB_Type t = cuda::jit::to_grb_type<T>();

    GrB_Matrix A;
    make_grb_matrix(A, N, N, indptr, index, d_data, GxB_SPARSE, GxB_BY_ROW);

    GRB_TRY (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    GRB_TRY (GxB_Matrix_fprint (A, "A", GxB_COMPLETE, stdout));

    T actual;
    GB_cuda_reduce( A, &actual, monoid );

    GrB_Vector v;
    GrB_Vector_new(&v, t, N);

    // Just sum in place for now (since we are assuming sum)
    int sum = 0;
    for(int i = 0; i < N; ++i) {
        sum+= d_data[i];
        cuda::jit::vector_set_element<T>(v, i, d_data[i]);
    }
    printf("Sum: %d\n", sum);

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_NEVER)) ;

    printf("Invoking grb reduce\n");
    T expected;
    GRB_TRY(cuda::jit::vector_reduce(&expected, v, monoid));
    printf("Done.\n");

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_ALWAYS)) ;
    if(expected != actual) {
        std::cout << "results do not match: reduced=" << expected << ", actual=" << actual << std::endl;
        exit(1);
    } else {
        std::cout << "Results matched!" << std::endl;
    }

    return expected == actual;
}

//bool test_triangle_counting() {
//
//    // Hardcoding int64_t for now
//    TestData<T_A, T_B, T_C, T_M> data = *make_karate_tricount<int64_t, int64_t, int64_t, int64_t>();
//
//    GrB_Monoid monoid = GrB_PLUS_MONOID_INT64;
//    GrB_BinaryOp binop = GrB_TIMES_INT64;
//    std::cout << "Creating problem gen" << std::endl;
//    N = data.A_indptr.size()-1;
//
//    GrB_Matrix A;
//    GrB_Matrix B;
//    GrB_Matrix C;
//    GrB_Matrix M;
//
//    make_grb_matrix<T_A>(A, data.A_indptr, data.A_indices, data.A_data, GxB_SPARSE);
//    make_grb_matrix<T_B>(B, data.B_indptr, data.B_indices, data.B_data, GxB_FULL, GxB_BY_ROW);
//    make_grb_matrix<T_C>(C, data.C_indptr, data.C_indices, data.C_data);
//    make_grb_matrix<T_M>(M, data.M_indptr, data.M_indices, data.M_data);
//
//    GrB_Semiring mysemiring;
//    auto grb_info = GrB_Semiring_new(&mysemiring, monoid, binop);
//    GRB_TRY (grb_info) ;
//
//    mysemiringfactory.semiring_factory ( mysemiring, false,
//                                         C->type, M->type,
//                                         A->type, B->type,
//                                         true,  // matrix types
//                                         false,
//                                         GB_sparsity(C),
//                                         GB_sparsity(M),
//                                         GB_sparsity(A),
//                                         GB_sparsity(B)
//                                       ) ;
//
//    bool result = false;
//
//    /**
//     * Run Phase 1: Compute nanobuckets and blockbuckets
//     */
//    const int64_t mnz = GB_nnz (M) ;
//
//    int chunk_size = 128;
//
//    // Use GrB_DESC_S for structural because dot3 mask will never be complemented
//    GRB_TRY (GrB_mxm(C_actual, M, NULL, mysemiring, A, B, GrB_DESC_ST1));
//
//    GRB_TRY (GxB_Matrix_fprint (M, "M actual", GxB_SHORT_VERBOSE, stdout));
//    GRB_TRY (GxB_Matrix_fprint (A, "A actual", GxB_SHORT_VERBOSE, stdout));
//    GRB_TRY (GxB_Matrix_fprint (B, "B actual", GxB_SHORT_VERBOSE, stdout));
//    GRB_TRY (GxB_Matrix_fprint (C, "C GPU", GxB_SHORT_VERBOSE, stdout));
//    GRB_TRY (GxB_Matrix_fprint (C_actual, "C_actual", GxB_SHORT_VERBOSE, stdout));
//
//    GRB_TRY(GrB_reduce_)
//
//    return result;
//
//}



//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_dndn_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, std::string& SEMI_RING) {
//// Assumes all matrices are square so far, so only N dimension given.
//// Sparsity is dense here so Anz = Bnz = N*N.
//// Generates three randomized matrices, builds buckets and calls a kernel.
//
//
//launchFactory<T_C, T_M, T_A, T_B, T_X, T_Z > lF(SEMI_RING, "dndn");
//
//int testBucket = TB;
//
////unsigned seed = 13372801;
////std::mt19937 r; //random number generator Mersenne Twister
////r.seed(seed);
//int gpuID;
//cudaGetDevice( &gpuID);
//
//std::cout<< "found device "<<gpuID<<std::endl;
//
//T_Z MONOID_IDENTITY;
//if (SEMI_RING == "PLUS_TIMES") {
//   std::cout << "Plus Times (+,*) semiring"<<std::endl;
//   MONOID_IDENTITY = 0;
//   ADD_ptr<T_Z> = myOP_plus<T_Z>;
//   MUL_ptr<T_Z> = myOP_times<T_Z>;
//
//}
//else if(SEMI_RING == "MIN_PLUS") {
//   std::cout << "Min Plus Times (min,+) semiring"<<std::endl;
//   MONOID_IDENTITY = std::numeric_limits<T_Z>::max();
//   ADD_ptr<T_Z> = myOP_min<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//
//}
//else if(SEMI_RING == "MAX_PLUS") {
//   MONOID_IDENTITY = std::numeric_limits<T_Z>::min();
//   std::cout << "Max Plus Times (max,+) semiring"<<std::endl;
//   ADD_ptr<T_Z> = myOP_max<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//}
//
////Generate test data and setup for using a jitify kernel with 'bucket' interface
//SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
//int64_t Annz = N*N;
//int64_t Bnnz = N*N;
//int64_t Cnz = N;
//float Cnzpercent = (float) Cnz/(N*N);
//
//G.init(N, Annz, Bnnz, Cnzpercent);
//
//G.fill_buckets( testBucket); // all elements go to testbucket= TB
//
//matrix<T_C>* C = G.getCptr();
//matrix<T_M>* M = G.getMptr();
//matrix<T_A>* A = G.getAptr();
//matrix<T_B>* B = G.getBptr();
//
//T_C *Cx = C->x;
//T_A *Ax = A->x;
//T_B *Bx = B->x;
//
//// Set clear zombie count
//C->zombie_count = 0;
//
////std::cout<<"got all matrices"<<std::endl;
//int64_t *Bucket = G.getBucket();
//int64_t *BucketStart = G.getBucketStart();
//
//int zc_valid = 0;
//
//bool result = false;
//
//for (int b =0; b < 12; ++b) {// loop on buckets
//
//    int64_t b_start = BucketStart [b] ;
//    int64_t b_end   = BucketStart [b+1] ;
//    int64_t nvecs = b_end - b_start ;
//    if (nvecs > 0) std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;
//
//    T_C *X_valid  = (T_C*) malloc( Cnz*sizeof(T_C));
//    int64_t *i_valid = (int64_t*)malloc( Cnz *sizeof(int64_t));
//    if (b == TB) { //test cases for dense-dense kernels
//       int nthrd = 32;
//       int sz = 4;
//       //int m = 256/sz;
//       int nblck = Cnz;
//       std::cout<< nblck<< " blocks of "<<nthrd<<" threads, "<<b_start<<","<<b_end<<std::endl;
//
//       GpuTimer kernTimer;
//       kernTimer.Start();
//       lF.jitGridBlockLaunch( nblck, nthrd, b_start, b_end, Bucket,
//                                C, M, A, B, sz);
//
//       kernTimer.Stop();
//       std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//       zc_valid = C->zombie_count;
//       C->zombie_count = 0;
//       for (int i =0 ; i< Cnz; ++i) {
//            //std::cout<<"Cx[i] = "<<Cx[i]<<std::endl;
//            X_valid[i] = Cx[i];
//            Cx[i] = 0;
//            i_valid[i] = C->i[i];
//       }
//       G.loadCj();
//
//       for (int64_t pair = b_start ; pair < b_end ; pair++) {
//
//        // get the kth entry in bucket b
//        //std::cout<< " pair ="<<pair<<std::endl;
//        int64_t pC = (Bucket == nullptr) ? pair : Bucket [pair] ;
//        int64_t i = M->i[pC] ;          // row index of C(i,j)
//
//        // get C(i,j)
//        int64_t k = (C->i [pC] >> 4) ;    // col index of C(i,j)
//        //ASSERT ((C->i [pC] & 4) == b) ;
//        int64_t j = (C->h == nullptr) ? k : C->h [k] ; // Mh has been copied into Ch
//        //std::cout<<" found dot "<<pair<<" at ("<<i<<","<<j<<")"<<std::endl;
//
//        // xvp, xvi, xvals:  A(:,i)
//        // xvp is Ap [i] and Ap [i+1]
//        int64_t pA_start = A->p [i] ;
//        int64_t pA_end   = A->p [i+1] ;
//        // indices are in Ai [pA_start ... pA_end-1]
//        // values  are in Ax [pA_start ... pA_end-1]
//
//        // yvp, yvi, yvals:  B(:,j)
//        // yvp is Bp [j] and Bp [j+1]
//        int64_t pB_start = B->p [j] ;
//        int64_t pB_end   = B->p [j+1] ;
//        // indices are in Bi [pB_start ... pB_end-1]
//        // values  are in Bx [pB_start ... pB_end-1]
//        k = pA_start;
//        int64_t l = pB_start;
//        T_Z cij = MONOID_IDENTITY;
//        while( k < pA_end && l < pB_end) {
//           //std::cout<<" A*B="<< (*MUL_ptr<T_Z>) ( (T_Z)Ax[k] , (T_Z) Bx[l]) <<std::endl ;
//           cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)Ax[k] , (T_Z) Bx[l]) ) ;
//           k++;
//           l++;
//           //std::cout<<"Ak = "<< Ax[k]<< " Bl = "<< Bx[l]<< "sum ="<<sum<<std::endl;
//        }
//        //std::cout<< " dot  = "<< sum << std::endl;
//
//        // output for this dot product is
//
//        if (cij == MONOID_IDENTITY) {
//            C->i [pC] = -1;//GB_FLIP (i)
//            C->zombie_count++;
//        }
//        else {
//            Cx [pC] = (T_C)cij;
//            C->i [pC] = i;
//        }
//    }
//       T_C err = 0;
//       for (int j =0 ; j< N; ++j) {
//         for ( int l = C->p[j]; l< C->p[j+1]; ++l) {
//             int64_t i =  C->i[l];
//             //std::cout<<i<<","<<j<<","<<l <<" Cx = "<<Cx[l]<<"x_val="<<X_valid[l]<<std::endl;
//             if (i >= 0)
//                err +=  ( X_valid[l] - Cx[l])*(X_valid[l] - Cx[l]);
//         }
//       }
//       std::cout<< " 2-norm of err ="<< err<<std::endl;
//       std::cout<< " zombie count CPU = "<<C->get_zombie_count()<<" zGPU ="<<zc_valid<<std::endl;
//
//       EXPECT_EQ(err,0);
//       EXPECT_EQ( zc_valid, C->get_zombie_count());
//
//       free(X_valid);
//       free(i_valid);
//     }
//    }
//
//G.del();
//
//return result;
//
//}
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_vsvs_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, std::string& SEMI_RING) {
//// Assumes all matrices are square so far, so only N dimension given.
//// Sparsity is controlled by Anz and Bnz vs N*N.
//// Generates three randomized matrices, builds buckets and calls a kernel.
//
//
//launchFactory<T_C, T_M, T_A, T_B, T_X, T_Z > lF(SEMI_RING, "vsvs");
//
//int testBucket = TB;
//
////unsigned seed = 13372801;
////std::mt19937 r; //random number generator Mersenne Twister
////r.seed(seed);
//int gpuID;
//cudaGetDevice( &gpuID);
//std::cout<< "found device "<<gpuID<<std::endl;
//
////T_Z MONOID_IDENTITY;
//if (SEMI_RING == "PLUS_TIMES") {
//   //MONOID_IDENTITY =(T_Z)0;
//   ADD_ptr<T_Z> = myOP_plus<T_Z>;
//   MUL_ptr<T_Z> = myOP_times<T_Z>;
//
//}
//else if(SEMI_RING == "MIN_PLUS") {
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::max();
//   ADD_ptr<T_Z> = myOP_min<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//
//}
//else if(SEMI_RING == "MAX_PLUS") {
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::min();
//   ADD_ptr<T_Z> = myOP_max<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//}
//
////Generate test data and setup for using a jitify kernel with 'bucket' interface
//SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
//int64_t Cnz = N;
//float Cnzpercent = (float) Cnz/(N*N);
//
//G.init(N, Anz, Bnz, Cnzpercent);
//
//G.fill_buckets( testBucket); // all elements go to testbucket= TB
//
//matrix<T_C>* C = G.getCptr();
//matrix<T_M>* M = G.getMptr();
//matrix<T_A>* A = G.getAptr();
//matrix<T_B>* B = G.getBptr();
//
//T_C *Cx = C->x;
//T_A *Ax = A->x;
//T_B *Bx = B->x;
//int64_t *Ci = C->i;
//int64_t *Mi = M->i;
//int64_t *Ai = A->i;
//int64_t *Bi = B->i;
//int64_t *Ap = A->p;
//int64_t *Bp = B->p;
//
////std::cout<<"got all matrices"<<std::endl;
//int64_t *Bucket = G.getBucket();
//int64_t *BucketStart = G.getBucketStart();
//
//int zc_valid = 0;
//
//bool result = false;
//
//for (int b =0; b < 12; ++b) {// loop on buckets
//
//    int64_t b_start = BucketStart [b] ;
//    int64_t b_end   = BucketStart [b+1] ;
//    int64_t nvecs = b_end - b_start ;
//    if (nvecs > 0) std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;
//
//    T_C *X_valid  = (T_C*) malloc( Cnz*sizeof(T_C));
//    int64_t *i_valid = (int64_t*)malloc( Cnz *sizeof(int64_t));
//    if (b == TB) { //test cases for v.sparse-v.sparse kernels
//       int nthrd = 32;
//       int sz = Anz/N;
//       int m = 256/sz;
//       int nblck = (Cnz -1 + m*nthrd )/(m*nthrd) ;
//       std::cout<< nblck<< " blocks of "<<nthrd<<" threads, "<<b_start<<","<<b_end<<std::endl;
//
//       GpuTimer kernTimer;
//       kernTimer.Start();
//       lF.jitGridBlockLaunch( nblck, nthrd, b_start, b_end, Bucket,
//                                C, M, A, B, sz);
//
//       kernTimer.Stop();
//       std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//       //std::cout<<"returned from kernel"<<std::endl;
//
//       zc_valid = C->zombie_count;
//       C->zombie_count = 0;
//       for (int i =0 ; i< Cnz; ++i) {
//            X_valid[i] = Cx[i];
//            Cx[i] = 0;
//            i_valid[i] = Ci[i];
//       }
//       G.loadCj();
//       for (int64_t pair = b_start ; pair < b_end ; pair++) {
//
//        // get the kth entry in bucket b
//        //std::cout<< " pair ="<<pair<<std::endl;
//        int64_t pC = (Bucket == nullptr) ? pair : Bucket [pair] ;
//        int64_t i = Mi[pC] ;          // row index of C(i,j)
//
//        // get C(i,j)
//        int64_t k = (Ci [pC] >> 4) ;    // col index of C(i,j)
//        //ASSERT ((C->i [pC] & 4) == b) ;
//        int64_t j = (C->h == nullptr) ? k : C->h [k] ; // Mh has been copied into Ch
//        //std::cout<<" found dot "<<pair<<" at ("<<i<<","<<j<<")"<<std::endl;
//
//        // xvp, xvi, xvals:  A(:,i)
//        // xvp is Ap [i] and Ap [i+1]
//        int64_t pA_start = Ap [i] ;
//        int64_t pA_end   = Ap [i+1] ;
//        // indices are in Ai [pA_start ... pA_end-1]
//        // values  are in Ax [pA_start ... pA_end-1]
//
//        // yvp, yvi, yvals:  B(:,j)
//        // yvp is Bp [j] and Bp [j+1]
//        int64_t pB_start = Bp [j] ;
//        int64_t pB_end   = Bp [j+1] ;
//        // indices are in Bi [pB_start ... pB_end-1]
//        // values  are in Bx [pB_start ... pB_end-1]
//        k = pA_start;
//        int64_t l = pB_start;
//        T_Z cij ;
//        bool cij_exists = false;
//        while( k < pA_end && l < pB_end) {
//            if ( Ai[k] < Bi[l]) ++k;
//            else if ( Ai[k] > Bi[l]) ++l;
//            else {
//                if (cij_exists) {
//                   cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( Ax[k] , Bx[l] ) );
//                }
//                else{
//                   cij_exists = true;
//                   cij = (*MUL_ptr<T_Z>)( Ax[k], Bx[l]);
//                }
//                k++;
//                l++;
//            }
//        }
//        //std::cout<< " dot  = "<< sum << std::endl;
//
//        // output for this dot product is
//
//        if (cij_exists) {
//            Ci [pC] = i;
//            Cx[pC] = (T_C)cij;
//        }
//        else {
//            Ci [pC] = -1;//GB_FLIP (i)
//            C->zombie_count++;
//        }
//    }
//       T_C err = 0;
//       for (int j =0 ; j< N; ++j) {
//         for ( int l = C->p[j]; l< C->p[j+1]; ++l) {
//             //std::cout<<i<<","<<j<<","<<l <<" Cx = "<<Cx[l]<<"x_val="<<X_valid[l]<<std::endl;
//             if (Ci[l] > 0)
//                err +=  ( X_valid[l] - Cx[l])*(X_valid[l] - Cx[l]);
//         }
//       }
//       std::cout<< " 2-norm of err ="<< err<<std::endl;
//       std::cout<< " zombie count GPU = "<<C->get_zombie_count()<<" zCPU ="<<zc_valid<<std::endl;
//
//       EXPECT_EQ(err,0);
//       EXPECT_EQ( zc_valid, C->get_zombie_count());
//
//       free(X_valid);
//       free(i_valid);
//     }
//    }
//
//G.del();
//
//return result;
//
//}
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_vssp_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, std::string& SEMI_RING) {
//// Assumes all matrices are square so far, so only N dimension given.
//// Sparsity is controlled by Anz and Bnz vs N*N.
//// Generates three randomized matrices, builds buckets and calls a kernel.
//
//launchFactory<T_C, T_M, T_A, T_B, T_X, T_Z > lF(SEMI_RING, "vssp");
//
//int testBucket = TB;
//
////unsigned seed = 13372801;
////std::mt19937 r; //random number generator Mersenne Twister
////r.seed(seed);
//int gpuID;
//cudaGetDevice( &gpuID);
//std::cout<< "found device "<<gpuID<<std::endl;
//
////T_Z MONOID_IDENTITY;
//if (SEMI_RING == "PLUS_TIMES") {
//   //MONOID_IDENTITY =(T_Z)0;
//   ADD_ptr<T_Z> = myOP_plus<T_Z>;
//   MUL_ptr<T_Z> = myOP_times<T_Z>;
//
//}
//else if(SEMI_RING == "MIN_PLUS") {
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::max();
//   ADD_ptr<T_Z> = myOP_min<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//
//}
//else if(SEMI_RING == "MAX_PLUS") {
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::min();
//   ADD_ptr<T_Z> = myOP_max<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//}
//
////Generate test data and setup for using a jitify kernel with 'bucket' interface
//SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
//
//int64_t Cnz = N;
//float Cnzpercent = (float)( Cnz)/(N*N);
//
//G.init(N, Anz, Bnz, Cnzpercent );
//
//G.fill_buckets( testBucket); // all elements go to testbucket= TB
//
//matrix<T_C>* C = G.getCptr();
//matrix<T_M>* M = G.getMptr();
//matrix<T_A>* A = G.getAptr();
//matrix<T_B>* B = G.getBptr();
//
//T_C *Cx = C->x;
//T_A *Ax = A->x;
//T_B *Bx = B->x;
//int64_t *Ci = C->i;
//int64_t *Mi = M->i;
//int64_t *Ai = A->i;
//int64_t *Bi = B->i;
//int64_t *Ap = A->p;
//int64_t *Bp = B->p;
//
//
////std::cout<<"got all matrices"<<std::endl;
//int64_t *Bucket = G.getBucket();
//int64_t *BucketStart = G.getBucketStart();
//
//int zc_valid = 0;
//int zc = 0;
//
//bool result = false;
//
//for (int b =0; b < 12; ++b) {// loop on buckets
//
//    int64_t b_start = BucketStart [b] ;
//    int64_t b_end   = BucketStart [b+1] ;
//    int64_t nvecs = b_end - b_start ;
//    if (nvecs == 0) continue;
//    std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;
//
//    T_C *X_valid  = (T_C*) malloc( Cnz*sizeof(T_C));
//    int64_t *i_valid = (int64_t*)malloc( Cnz *sizeof(int64_t));
//    if (b == TB) { //test cases for v.sparse-dense kernels
//       int nthrd = 32;
//       int sz = 4;
//       //int m = 256/sz;
//       int nblck = (Cnz -1 + nthrd )/(nthrd) ;
//       std::cout<< nblck<< " blocks of "<<nthrd<<" threads, "<<b_start<<","<<b_end<<std::endl;
//
//       GpuTimer kernTimer;
//       kernTimer.Start();
//       lF.jitGridBlockLaunch( nblck, nthrd, b_start, b_end, Bucket,
//                                C, M, A, B, sz);
//
//       kernTimer.Stop();
//       std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//       //std::cout<<"returned from kernel"<<std::endl;
//
//       zc_valid = C->zombie_count;
//       C->zombie_count = 0;
//       for (int i =0 ; i< Cnz; ++i) {
//            X_valid[i] = Cx[i];
//            Cx[i] = 0;
//            i_valid[i] = C->i[i];
//       }
//       G.loadCj();
//
//
//       for (int64_t pair = b_start ; pair < b_end ; pair++) {
//
//        // get the kth entry in bucket b
//        //std::cout<< " pair ="<<pair<<std::endl;
//        int64_t pC = (Bucket == nullptr) ? pair : Bucket [pair] ;
//
//        int64_t i = Mi[pC] ;          // row index of C(i,j)
//        // get C(i,j)
//        int64_t k = (Ci [pC] >> 4) ;    // col index of C(i,j)
//        //ASSERT ((C->i [pC] & 4) == b) ;
//        int64_t j = (C->h == nullptr) ? k : C->h [k] ; // Mh has been copied into Ch
//        //std::cout<<" found dot "<<pair<<" at ("<<i<<","<<j<<")"<<std::endl;
//
//        int64_t pA      = Ap[i];
//        int64_t pA_end  = Ap[i+1];
//        int64_t nnzA = pA_end - pA;
//
//        int64_t pB      = Bp[j];
//        int64_t pB_end  = Bp[j+1];
//        int64_t nnzB = pB_end - pB;
//
//        //Search for each nonzero in the smaller vector to find intersection
//        bool cij_exists = false;
//
//        T_A aki;
//        T_B bkj;
//        T_Z cij;
//
//        if (nnzA <= nnzB) {
//            //----------------------------------------------------------------------
//            // A(:,i) is very sparse compared to B(:,j)
//            //----------------------------------------------------------------------
//
//            while (pA < pA_end && pB < pB_end)
//            {
//                int64_t ia = Ai [pA] ;
//                int64_t ib = Bi [pB] ;
//                if (ia < ib)
//                {
//                    // A(ia,i) appears before B(ib,j)
//                    pA++ ;
//                }
//                else if (ib < ia)
//                {
//                    // B(ib,j) appears before A(ia,i)
//                    // discard all entries B(ib:ia-1,j)
//                    int64_t pleft = pB + 1 ;
//                    int64_t pright = pB_end - 1 ;
//                    GB_BINARY_TRIM_SEARCH (ia, Bi, pleft, pright) ;
//                    //ASSERT (pleft > pB) ;
//                    pB = pleft ;
//                }
//                else // ia == ib == k
//                {
//                    // A(k,i) and B(k,j) are the next entries to merge
//                    #if defined ( GB_PHASE_1_OF_2 )
//                    cij_exists = true ;
//                    break ;
//                    #else
//                    GB_GETA (aki, Ax, pA) ;             /* aki = A(k,i) */
//                    GB_GETB (bkj, Bx, pB) ;             /* bkj = B(k,j) */
//                    if (cij_exists)
//                    {
//                        cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)aki , (T_Z)bkj ) );
//                        /* cij += aki * bkj */
//                    }
//                    else
//                    {
//                        /* cij = A(k,i) * B(k,j), and add to the pattern */
//                        cij_exists = true ;
//                        cij=  (*MUL_ptr<T_Z>)( (T_Z)aki, (T_Z)bkj) ;
//                        /* cij = aki * bkj */
//                    }
//                    //GB_DOT_TERMINAL (cij) ;         // break if cij == terminal
//                    pA++ ;
//                    pB++ ;
//                    #endif
//                }
//            }
//        }
//        else {
//            //----------------------------------------------------------------------
//            // B(:,j) is very sparse compared to A(:,i)
//            //----------------------------------------------------------------------
//
//            while (pA < pA_end && pB < pB_end)
//            {
//                int64_t ia = Ai [pA] ;
//                int64_t ib = Bi [pB] ;
//                if (ia < ib)
//                {
//                    // A(ia,i) appears before B(ib,j)
//                    // discard all entries A(ia:ib-1,i)
//                    int64_t pleft = pA + 1 ;
//                    int64_t pright = pA_end - 1 ;
//                    GB_BINARY_TRIM_SEARCH (ib, Ai, pleft, pright) ;
//                    //ASSERT (pleft > pA) ;
//                    pA = pleft ;
//                }
//                else if (ib < ia)
//                {
//                    // B(ib,j) appears before A(ia,i)
//                    pB++ ;
//                }
//                else // ia == ib == k
//                {
//                    // A(k,i) and B(k,j) are the next entries to merge
//                    #if defined ( GB_PHASE_1_OF_2 )
//                    cij_exists = true ;
//                    break ;
//                    #else
//                    GB_GETA (aki, Ax, pA) ;             /* aki = A(k,i) */
//                    GB_GETB (bkj, Bx, pB) ;             /* bkj = B(k,j) */
//                    if (cij_exists)
//                    {
//                        cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)aki , (T_Z)bkj ) );
//                        /* cij += aki * bkj */      \
//                    }
//                    else
//                    {
//                        /* cij = A(k,i) * B(k,j), and add to the pattern */
//                        cij_exists = true ;
//                        cij=  (*MUL_ptr<T_Z>)( (T_Z)aki, (T_Z)bkj) ;
//                    }
//                    //GB_DOT_TERMINAL (cij) ;         // break if cij == terminal
//                    pA++ ;
//                    pB++ ;
//                    #endif
//                }
//            }
//
//        }
//        if ( cij_exists){
//           Ci[pair] = i;
//           Cx[pair] = (T_C)cij;
//        }
//        else {
//           zc++;
//           //printf(" %lld, %lld is zombie %d!\n",i,j,zc);
//           Ci[pair] = GB_FLIP( i );
//        }
//
//    }
//       C->zombie_count = zc;
//       T_C err = 0;
//       for (int j =0 ; j< N; ++j) {
//         for ( int l = C->p[j]; l< C->p[j+1]; ++l) {
//             int64_t i = Ci[l];
//             //std::cout<<i<<","<<j<<","<<l <<" Cx = "<<Cx[l]<<"x_val="<<X_valid[l]<<std::endl;
//             if (i > 0){ //not a zombie!
//                 err +=  ( X_valid[l] - Cx[l])*(X_valid[l] - Cx[l]);
//             }
//         }
//       }
//       std::cout<< " 2-norm of err ="<< err<<std::endl;
//       std::cout<< " zombie count GPU = "<<C->get_zombie_count()<<" zCPU ="<<zc_valid<<std::endl;
//
//       EXPECT_EQ(err,0);
//       EXPECT_EQ( zc_valid, C->get_zombie_count());
//
//       free(X_valid);
//       free(i_valid);
//     }
//    }
//
//G.del();
//
//return result;
//
//}
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_spdn_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, std::string& SEMI_RING) {
//// Assumes all matrices are square so far, so only N dimension given.
//// Sparsity is controlled by Anz and Bnz vs N*N.
//// Generates three randomized matrices, builds buckets and calls a kernel.
//
//launchFactory<T_C, T_M, T_A, T_B, T_X, T_Z > lF(SEMI_RING, "spdn");
//
//int testBucket = TB;
//
////unsigned seed = 13372801;
////std::mt19937 r; //random number generator Mersenne Twister
////r.seed(seed);
//int gpuID;
//cudaGetDevice( &gpuID);
//std::cout<< "found device "<<gpuID<<std::endl;
//
////T_Z MONOID_IDENTITY;
//if (SEMI_RING == "PLUS_TIMES") {
//  // MONOID_IDENTITY =(T_Z)0;
//   ADD_ptr<T_Z> = myOP_plus<T_Z>;
//   MUL_ptr<T_Z> = myOP_times<T_Z>;
//
//}
//else if(SEMI_RING == "MIN_PLUS") {
//  // MONOID_IDENTITY = std::numeric_limits<T_Z>::max();
//   ADD_ptr<T_Z> = myOP_min<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//
//}
//else if(SEMI_RING == "MAX_PLUS") {
//  // MONOID_IDENTITY = std::numeric_limits<T_Z>::min();
//   ADD_ptr<T_Z> = myOP_max<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//}
//
////Generate test data and setup for using a jitify kernel with 'bucket' interface
//SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
//
//int64_t Cnz = N;
//float Cnzpercent = (float)( Cnz)/(N*N);
//
////spdn case means B should be dense -> Bnz = N*N;
//G.init(N, Anz, N*N, Cnzpercent );
//
//G.fill_buckets( testBucket); // all elements go to testbucket= TB
//
//matrix<T_C>* C = G.getCptr();
//matrix<T_M>* M = G.getMptr();
//matrix<T_A>* A = G.getAptr();
//matrix<T_B>* B = G.getBptr();
//
//T_C *Cx = C->x;
//T_A *Ax = A->x;
//T_B *Bx = B->x;
//int64_t *Ci = C->i;
//int64_t *Mi = M->i;
//int64_t *Ai = A->i;
//int64_t *Bi = B->i;
//int64_t *Ap = A->p;
//int64_t *Bp = B->p;
//
//
////std::cout<<"got all matrices"<<std::endl;
//int64_t *Bucket = G.getBucket();
//int64_t *BucketStart = G.getBucketStart();
//
//int zc_valid = 0;
//
//bool result = false;
//
//for (int b =0; b < 12; ++b) {// loop on buckets
//
//    int64_t b_start = BucketStart [b] ;
//    int64_t b_end   = BucketStart [b+1] ;
//    int64_t nvecs = b_end - b_start ;
//    if (nvecs == 0) continue;
//    std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;
//
//    T_C *X_valid  = (T_C*) malloc( Cnz*sizeof(T_C));
//    int64_t *i_valid = (int64_t*)malloc( Cnz *sizeof(int64_t));
//    if (b == TB) { //test cases for v.sparse-dense kernels
//       int nthrd = 32;
//       int sz = Anz/N;
//       int m = 256/sz;
//       int nblck = (Cnz -1 + m*nthrd )/(m*nthrd) ;
//       std::cout<< nblck<< " blocks of "<<nthrd<<" threads, "<<b_start<<","<<b_end<<std::endl;
//
//       GpuTimer kernTimer;
//       kernTimer.Start();
//       lF.jitGridBlockLaunch( nblck, nthrd, b_start, b_end, Bucket,
//                                C, M, A, B, sz);
//
//       kernTimer.Stop();
//       std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//       //std::cout<<"returned from kernel"<<std::endl;
//
//       zc_valid = C->zombie_count;
//       C->zombie_count = 0;
//       for (int i =0 ; i< Cnz; ++i) {
//            X_valid[i] = Cx[i];
//            Cx[i] = 0;
//            i_valid[i] = Ci[i];
//       }
//       G.loadCj();
//       for (int64_t pair = b_start ; pair < b_end ; pair++) {
//
//        // get the kth entry in bucket b
//        //std::cout<< " pair ="<<pair<<std::endl;
//        int64_t pC = (Bucket == nullptr) ? pair : Bucket [pair] ;
//        int64_t i = Mi[pC] ;          // row index of C(i,j)
//
//        // get C(i,j)
//        //int64_t k = (Ci [pC] >> 4) ;    // col index of C(i,j)
//        //ASSERT ((C->i [pC] & 4) == b) ;
//        //int64_t j = (C->h == nullptr) ? k : C->h [k] ; // Mh has been copied into Ch
//        //std::cout<<" found dot "<<pair<<" at ("<<i<<","<<j<<")"<<std::endl;
//
//         int64_t pA = Ap[i];
//         int64_t pA_end   = Ap[i+1];
//         int64_t nnzA   = pA_end - pA;
//         int64_t pB = Bp[i];
//         int64_t pB_end   = Bp[i+1];
//         int64_t nnzB   = pB_end - pB;
//         T_A aki;
//         T_B bkj;
//         T_Z cij;
//
//         if( nnzA == A->vlen) // A is dense
//         {
//            int64_t k = Bi [pB] ;               // first row index of B(:,j)
//            // cij = A(k,i) * B(k,j)
//            GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
//            GB_GETB (bkj, Bx, pB  ) ;           // bkj = B(k,j)
//            cij = (*MUL_ptr<T_Z>)( aki, bkj) ;           // cij = aki * bkj
//
//            for (int64_t p = pB+1 ; p < pB_end ; p++)
//            {
//                //GB_DOT_TERMINAL (cij) ;             // break if cij == terminal
//                int64_t k = Bi [p] ;                // next row index of B(:,j)
//                // cij += A(k,i) * B(k,j)
//                GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
//                GB_GETB (bkj, Bx, p   ) ;           // bkj = B(k,j)
//                cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)aki, (T_Z)bkj) );
//            }
//
//         }
//         if( nnzB == B->vlen) // B is dense
//         {
//            int64_t k = Ai [pA] ;               // first row index of A(:,i)
//            // cij = A(k,i) * B(k,j)
//            GB_GETA (aki, Ax, pA  ) ;           // aki = A(k,i)
//            GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
//            cij = (*MUL_ptr<T_Z>)( aki, bkj) ;           // cij = aki * bkj
//
//            for (int64_t p = pA+1 ; p < pA_end ; p++)
//            {
//                //GB_DOT_TERMINAL (cij) ;             // break if cij == terminal
//                int64_t k = Ai [p] ;                // next row index of A(:,i)
//                // cij += A(k,i) * B(k,j)
//                GB_GETA (aki, Ax, p   ) ;           // aki = A(k,i)
//                GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
//                cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)aki, (T_Z)bkj) );
//            }
//         }
//
//         Ci[pair] = i;
//         Cx[pair] = cij;
//
//      }
//       T_C err = 0;
//       for (int j =0 ; j< N; ++j) {
//         for ( int l = C->p[j]; l< C->p[j+1]; ++l) {
//             int64_t i =  Ci[l];
//         //std::cout<<i<<","<<j<<" Cx = "<<Cx[l]<<"x_val="<<X_valid[l]<<std::endl;
//             if (i >=0 )
//                err +=  ( X_valid[l] - Cx[l])*(X_valid[l] - Cx[l]);
//         }
//       }
//       std::cout<< " 2-norm of err ="<< err<<std::endl;
//       std::cout<< " zombie count GPU = "<<C->get_zombie_count()<<" zCPU ="<<zc_valid<<std::endl;
//
//       EXPECT_EQ(err,0);
//       EXPECT_EQ( zc_valid, C->get_zombie_count());
//
//       free(X_valid);
//       free(i_valid);
//     }
//    }
//
//G.del();
//
//return result;
//
//}
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_mp_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, std::string& SEMI_RING) {
//// Assumes all matrices are square so far, so only N dimension given.
//// Sparsity is dense here so Anz = Bnz = N*N.
//// Generates three randomized matrices, builds buckets and calls a kernel.
//
//
//launchFactory<T_C, T_M, T_A, T_B, T_X, T_Z > lF(SEMI_RING, "mp");
//
//int testBucket = TB;
//
////unsigned seed = 13372801;
////std::mt19937 r; //random number generator Mersenne Twister
////r.seed(seed);
////int gpuID;
////cudaGetDevice( &gpuID);
//
////std::cout<< "found device "<<gpuID<<std::endl;
//
////T_Z MONOID_IDENTITY;
//if (SEMI_RING == "PLUS_TIMES") {
//   std::cout << "Plus Times (+,*) semiring"<<std::endl;
//   //MONOID_IDENTITY = 0;
//   ADD_ptr<T_Z> = myOP_plus<T_Z>;
//   MUL_ptr<T_Z> = myOP_times<T_Z>;
//
//}
//else if(SEMI_RING == "MIN_PLUS") {
//   std::cout << "Min Plus Times (min,+) semiring"<<std::endl;
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::max();
//   ADD_ptr<T_Z> = myOP_min<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//
//}
//else if(SEMI_RING == "MAX_PLUS") {
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::min();
//   std::cout << "Max Plus Times (max,+) semiring"<<std::endl;
//   ADD_ptr<T_Z> = myOP_max<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//}
//
////Generate test data and setup for using a jitify kernel with 'bucket' interface
//SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
//int64_t Annz = Anz;
//int64_t Bnnz = Bnz;
//int64_t Cnz = N;
//float Cnzpercent = (float) Cnz/(N*N);
//
//G.init(N, Annz, Bnnz, Cnzpercent);
//
//G.fill_buckets( testBucket); // all elements go to testbucket= TB
//
//matrix<T_C>* C = G.getCptr();
//matrix<T_M>* M = G.getMptr();
//matrix<T_A>* A = G.getAptr();
//matrix<T_B>* B = G.getBptr();
//
//T_C *Cx = C->x;
//T_A *Ax = A->x;
//T_B *Bx = B->x;
//int64_t *Ci = C->i;
//int64_t *Mi = M->i;
//int64_t *Ai = A->i;
//int64_t *Bi = B->i;
//int64_t *Ap = A->p;
//int64_t *Bp = B->p;
//
//// Set clear zombie count
//C->zombie_count = 0;
//
////std::cout<<"got all matrices"<<std::endl;
//int64_t *Bucket = G.getBucket();
//int64_t *BucketStart = G.getBucketStart();
//
//int zc_valid = 0;
//
//bool result = false;
//
//for (int b =0; b < 12; ++b) {// loop on buckets
//
//    int64_t b_start = BucketStart [b] ;
//    int64_t b_end   = BucketStart [b+1] ;
//    int64_t nvecs = b_end - b_start ;
//    if (nvecs > 0) std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;
//
//    T_C *X_valid  = (T_C*) malloc( Cnz*sizeof(T_C));
//    int64_t *i_valid = (int64_t*)malloc( Cnz *sizeof(int64_t));
//    if (b == TB) { //test cases for merge-path kernel
//       int nthrd = 32;
//       int nblck = Cnz;
//       int sz = 0;
//       std::cout<< nblck<< " blocks of "<<nthrd<<" threads, "<<b_start<<","<<b_end<<std::endl;
//
//       GpuTimer kernTimer;
//       kernTimer.Start();
//       lF.jitGridBlockLaunch( nblck, nthrd, b_start, b_end, Bucket,
//                                C, M, A, B, sz);
//
//       kernTimer.Stop();
//       std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//       //std::cout<<"returned from kernel"<<std::endl;
//
//       zc_valid = C->zombie_count;
//       C->zombie_count = 0;
//       for (int i =0 ; i< Cnz; ++i) {
//            //std::cout<<"Cx[i] = "<<Cx[i]<<std::endl;
//            X_valid[i] = Cx[i];
//            i_valid[i] = C->i[i];
//            // clear values for next test
//            Cx[i] = 0;
//       }
//       G.loadCj();
//
//       for (int64_t pair = b_start ; pair < b_end ; pair++) {
//
//        // get the kth entry in bucket b
//        //std::cout<< " pair ="<<pair<<std::endl;
//        int64_t pC = (Bucket == nullptr) ? pair : Bucket [pair] ;
//        int64_t i = Mi[pC] ;          // row index of C(i,j)
//
//        // get C(i,j)
//        int64_t k = (Ci [pC] >> 4) ;    // col index of C(i,j)
//        //ASSERT ((C->i [pC] & 4) == b) ;
//        int64_t j = (C->h == nullptr) ? k : C->h [k] ; // Mh has been copied into Ch
//        //std::cout<<" found dot "<<pair<<" at ("<<i<<","<<j<<")"<<std::endl;
//
//        int64_t pA_start = Ap [i] ;
//        int64_t pA_end   = Ap [i+1] ;
//
//        int64_t pB_start = Bp [j] ;
//        int64_t pB_end   = Bp [j+1] ;
//        // NOTE: this test code is NOT doing merge-path. This is just a
//        // single-threaded linear merge for correctness testing.
//        k = pA_start;
//        int64_t l = pB_start;
//        T_Z cij ;
//        bool cij_exists = false;
//        while( k < pA_end && l < pB_end) {
//           if      ( Ai[k] < Bi[l] ) k += 1;
//           else if ( Ai[k] > Bi[l] ) l += 1;
//           else {
//             if (cij_exists) {
//               //std::cout<<" A*B="<< (*MUL_ptr<T_Z>) ( (T_Z)Ax[k] , (T_Z) Bx[l]) <<std::endl ;
//               cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)Ax[k] , (T_Z) Bx[l]) ) ;
//             }
//             else {
//               cij_exists = true;
//               cij = (*MUL_ptr<T_Z>)( (T_Z)Ax[k], (T_Z)Bx[l] ) ;
//             }
//
//             k++;
//             l++;
//           }
//           //std::cout<<"Ak = "<< Ax[k]<< " Bl = "<< Bx[l]<< "sum ="<<sum<<std::endl;
//        }
//        //std::cout<< " dot  = "<< sum << std::endl;
//
//        // output for this dot product is
//
//        if (cij_exists) {
//            Cx [pC] = (T_C)cij;
//            Ci [pC] = i;
//        }
//        else {
//            C->i [pC] = -1;//GB_FLIP (i)
//            C->zombie_count++;
//        }
//    }
//       T_C err = 0;
//       for (int j =0 ; j< N; ++j) {
//         for ( int l = C->p[j]; l< C->p[j+1]; ++l) {
//
//             if (Ci[l] > 0) {
//                //std::cout<<j<<","<<l <<" Cx = "<<Cx[l]<<"x_val="<<X_valid[l]<<std::endl;
//                err +=  ( X_valid[l] - Cx[l])*(X_valid[l] - Cx[l]);
//             }
//         }
//       }
//       std::cout<< " 2-norm of err ="<< err<<std::endl;
//       std::cout<< " zombie count CPU = "<<C->get_zombie_count()<<" zGPU ="<<zc_valid<<std::endl;
//
//       EXPECT_EQ(err,0);
//       EXPECT_EQ( zc_valid, C->get_zombie_count());
//
//       free(X_valid);
//       free(i_valid);
//     }
//    }
//
//G.del();
//
//return result;
//
//}
//
//template <typename T_C, typename T_M, typename T_A,typename T_B, typename T_X, typename T_Y, typename T_Z>
//bool test_AxB_dot3_warp_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, std::string& SEMI_RING) {
//// Assumes all matrices are square so far, so only N dimension given.
//// Sparsity is dense here so Anz = Bnz = N*N.
//// Generates three randomized matrices, builds buckets and calls a kernel.
//
//
//launchFactory<T_C, T_M, T_A, T_B, T_X, T_Z > lF(SEMI_RING, "warp");
//
//int testBucket = TB;
//
////unsigned seed = 13372801;
////std::mt19937 r; //random number generator Mersenne Twister
////r.seed(seed);
////int gpuID;
////cudaGetDevice( &gpuID);
//
////std::cout<< "found device "<<gpuID<<std::endl;
//
////T_Z MONOID_IDENTITY;
//if (SEMI_RING == "PLUS_TIMES") {
//   std::cout << "Plus Times (+,*) semiring"<<std::endl;
//   //MONOID_IDENTITY = 0;
//   ADD_ptr<T_Z> = myOP_plus<T_Z>;
//   MUL_ptr<T_Z> = myOP_times<T_Z>;
//
//}
//else if(SEMI_RING == "MIN_PLUS") {
//   std::cout << "Min Plus Times (min,+) semiring"<<std::endl;
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::max();
//   ADD_ptr<T_Z> = myOP_min<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//
//}
//else if(SEMI_RING == "MAX_PLUS") {
//   //MONOID_IDENTITY = std::numeric_limits<T_Z>::min();
//   std::cout << "Max Plus Times (max,+) semiring"<<std::endl;
//   ADD_ptr<T_Z> = myOP_max<T_Z>;
//   MUL_ptr<T_Z> = myOP_plus<T_Z>;
//}
//
////Generate test data and setup for using a jitify kernel with 'bucket' interface
//SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
//int64_t Cnz = N;
//float Cnzpercent = (float) Cnz/(N*N);
//
//G.init(N, Anz, Bnz, Cnzpercent);
//
//G.fill_buckets( testBucket); // all elements go to testbucket= TB
//
//matrix<T_C>* C = G.getCptr();
//matrix<T_M>* M = G.getMptr();
//matrix<T_A>* A = G.getAptr();
//matrix<T_B>* B = G.getBptr();
//
//T_C *Cx = C->x;
//T_A *Ax = A->x;
//T_B *Bx = B->x;
//int64_t *Ci = C->i;
//int64_t *Mi = M->i;
//int64_t *Ai = A->i;
//int64_t *Bi = B->i;
//int64_t *Ap = A->p;
//int64_t *Bp = B->p;
//
//// Set clear zombie count
//C->zombie_count = 0;
//
////std::cout<<"got all matrices"<<std::endl;
//int64_t *Bucket = G.getBucket();
//int64_t *BucketStart = G.getBucketStart();
//
//int zc_valid = 0;
//
//bool result = false;
//
//for (int b =0; b < 12; ++b) {// loop on buckets
//
//    int64_t b_start = BucketStart [b] ;
//    int64_t b_end   = BucketStart [b+1] ;
//    int64_t nvecs = b_end - b_start ;
//    if (nvecs > 0) std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;
//
//    T_C *X_valid  = (T_C*) malloc( Cnz*sizeof(T_C));
//    int64_t *i_valid = (int64_t*)malloc( Cnz *sizeof(int64_t));
//    if (b == TB) { //test cases for merge-path kernel
//       int nthrd = 32;
//       int nblck = (Cnz + nthrd -1)/nthrd ;
//       int sz = 0;
//       std::cout<< nblck<< " blocks of "<<nthrd<<" threads, "<<b_start<<","<<b_end<<std::endl;
//
//       GpuTimer kernTimer;
//       kernTimer.Start();
//       lF.jitGridBlockLaunch( nblck, nthrd, b_start, b_end, Bucket,
//                                C, M, A, B, sz);
//
//       kernTimer.Stop();
//       std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//       //std::cout<<"returned from kernel"<<std::endl;
//
//       zc_valid = C->zombie_count;
//       C->zombie_count = 0;
//       for (int i =0 ; i< Cnz; ++i) {
//            //std::cout<<"Cx[i] = "<<Cx[i]<<std::endl;
//            X_valid[i] = Cx[i];
//            i_valid[i] = C->i[i];
//            // clear values for next test
//            Cx[i] = 0;
//       }
//       G.loadCj();
//
//       for (int64_t pair = b_start ; pair < b_end ; pair++) {
//
//        // get the kth entry in bucket b
//        //std::cout<< " pair ="<<pair<<std::endl;
//        int64_t pC = (Bucket == nullptr) ? pair : Bucket [pair] ;
//        int64_t i = Mi[pC] ;          // row index of C(i,j)
//
//        // get C(i,j)
//        int64_t k = (Ci [pC] >> 4) ;    // col index of C(i,j)
//        //ASSERT ((C->i [pC] & 4) == b) ;
//        int64_t j = (C->h == nullptr) ? k : C->h [k] ; // Mh has been copied into Ch
//        //std::cout<<" found dot "<<pair<<" at ("<<i<<","<<j<<")"<<std::endl;
//
//        int64_t pA_start = Ap [i] ;
//        int64_t pA_end   = Ap [i+1] ;
//
//        int64_t pB_start = Bp [j] ;
//        int64_t pB_end   = Bp [j+1] ;
//        // NOTE: this test code is NOT doing merge-path. This is just a
//        // single-threaded linear merge for correctness testing.
//        k = pA_start;
//        int64_t l = pB_start;
//        T_Z cij ;
//        bool cij_exists = false;
//        while( k < pA_end && l < pB_end) {
//           if      ( Ai[k] < Bi[l] ) k += 1;
//           else if ( Ai[k] > Bi[l] ) l += 1;
//           else {
//             if (cij_exists) {
//               //std::cout<<" A*B="<< (*MUL_ptr<T_Z>) ( (T_Z)Ax[k] , (T_Z) Bx[l]) <<std::endl ;
//               cij = (*ADD_ptr<T_Z>)( cij, (*MUL_ptr<T_Z>)( (T_Z)Ax[k] , (T_Z) Bx[l]) ) ;
//             }
//             else {
//               cij_exists = true;
//               cij = (*MUL_ptr<T_Z>)( (T_Z)Ax[k], (T_Z)Bx[l] ) ;
//             }
//
//             k++;
//             l++;
//           }
//           //std::cout<<"Ak = "<< Ax[k]<< " Bl = "<< Bx[l]<< "sum ="<<sum<<std::endl;
//        }
//        //std::cout<< " dot  = "<< sum << std::endl;
//
//        // output for this dot product is
//
//        if (cij_exists) {
//            Cx [pC] = (T_C)cij;
//            Ci [pC] = i;
//        }
//        else {
//            C->i [pC] = -1;//GB_FLIP (i)
//            C->zombie_count++;
//        }
//    }
//       T_C err = 0;
//       for (int j =0 ; j< N; ++j) {
//         for ( int l = C->p[j]; l< C->p[j+1]; ++l) {
//
//             if (Ci[l] > 0) {
//                //std::cout<<j<<","<<l <<" Cx = "<<Cx[l]<<"x_val="<<X_valid[l]<<std::endl;
//                err +=  ( X_valid[l] - Cx[l])*(X_valid[l] - Cx[l]);
//             }
//         }
//       }
//       std::cout<< " 2-norm of err ="<< err<<std::endl;
//       std::cout<< " zombie count CPU = "<<C->get_zombie_count()<<" zGPU ="<<zc_valid<<std::endl;
//
//       EXPECT_EQ(err,0);
//       EXPECT_EQ( zc_valid, C->get_zombie_count());
//
//       free(X_valid);
//       free(i_valid);
//     }
//    }
//
//G.del();
//
//return result;
//
//}
//
//
//template <typename T1,typename T2,typename T3>
//bool test_dndotfactoryUM( unsigned int N, std::string SEMI_RING) {
//
//  dotFactory<T1,T2,T3> dF;
//
//  int block(512);
//  int nblock= (N + 8*block -1)/(8*block);
//  int grid(nblock);
//  T1* x;
//  T2* y;
//  T3* output;
//  CHECK_CUDA( cudaMallocManaged((void**)&x, N*sizeof(T1)) );
//  CHECK_CUDA( cudaMallocManaged((void**)&y, N*sizeof(T2)) );
//  CHECK_CUDA( cudaMallocManaged((void**)&output, nblock*sizeof(T3)) );
//
//  //we will get a triangular sum = N*(N+1)/2 with these inputs
//  fillvector_linear<T1> (N, x);
//  fillvector_constant<T2> (N, y, T2(1));
//
//  dF.jitGridBlockLaunch( grid, block, x, y, output, N, SEMI_RING );
//
//  T3 sum;
//  if (SEMI_RING == "PLUS_TIMES")
//  {
//      myOpPTR<T3> = myOP_plus<T3>;
//      sum = (T3)0;
//  }
//  if (SEMI_RING == "MIN_PLUS")
//  {
//      sum = std::numeric_limits<T3>::max();
//      myOpPTR<T3> = myOP_min<T3>;
//  }
//
//  for (int i =0; i< nblock; ++i) sum = (*myOpPTR<T3>)(sum ,output[i]);
//
//  bool result = false;
//  T3 expect;
//  if (SEMI_RING == "PLUS_TIMES") {
//     expect = (T3)(N*(N-1)/2);
//     T3 temp = (sum -expect) ;
//     if (temp < 0) temp = -temp ;
//     //result = (temp < (T3)1) ; //adjust formula for leading 0
//     EXPECT_LE( temp, (T3)1 );
//  }
//  else if (SEMI_RING == "MIN_PLUS") {
//     expect = (T3) 1;
//     //result = (sum == expect) ;   //min is 1 from the (0,1) pair
//     EXPECT_EQ( sum, expect);
//  }
//  else expect = (T3)0;
//  std::cout <<"test_dotfactoryUM with "<<SEMI_RING<<" semi-ring="<< sum
//                                       <<" expected "<<expect << std::endl;
//
//  cudaFree(x);
//  cudaFree(y);
//  cudaFree(output);
//  return result;
//}
//
//
//template <typename T1,typename T2,typename T3>
//bool test_spdotfactoryUM( unsigned int N, unsigned int xn, unsigned int yn, std::string SEMI_RING) {
//
//#define INTMIN( A, B) ( (A) < (B) ) ?  (A) : (B)
//
//  // N here is the index space that the sparse vectors are drawn from.
//  // Indices in xi and yi are in the range (0,N-1)
//  // We will generate a number of random values in this range for test data
//  std::cout<< " xn,yn= "<<xn<<','<<yn<<"min = "<< std::min( xn, yn) <<std::endl;
//  int n_threads = std::min( xn, yn) / 4;
//  std::cout<< "I think we need "<< n_threads<<" threads to do this."<<std::endl;
//  int pad_threads = 2;
//  while ( pad_threads < n_threads) {
//      pad_threads *= 2;
//  }
//  int block= 32;
//  int nblock= ( pad_threads + block -1)/(block);
//  int grid(nblock);
//  std::cout<<"N="<<N<<" xn ="<<xn<<", yn="<<yn<<" nblock="<<nblock<<" block ="<<block<<std::endl;
//  unsigned int *xi;
//  unsigned int *yi;
//  T1* x;
//  T2* y;
//  T3* output;
//  unsigned int intersection_size = 0; //will be filled in later if needed and xn != yn
//  unsigned seed = 13372801;
//  std::mt19937 r; //random number generator Mersenne Twister
//  r.seed(seed);
//  cudaMallocManaged((void**)&x, xn*sizeof(T1));
//  cudaMallocManaged((void**)&xi, xn*sizeof(int));
//  cudaMallocManaged((void**)&y, yn*sizeof(T2));
//  cudaMallocManaged((void**)&yi, yn*sizeof(int));
//  cudaMallocManaged((void**)&output, nblock*sizeof(T3));
//
//  int inv_sparsity = N/std::max(xn,yn);  //= values not taken per value occupied in index space
//  std::cout<<" Using inv_sparsity value of "<< inv_sparsity<<std::endl;
//  fillvector_constant<T1> (xn, x, T1(1));
//  fillvector_constant<T2> (yn, y, T2(1));
//
//  if( xn == yn){  // test case : all values intersect, generate 1 random number for both
//      intersection_size = xn;
//      std::cout << " all-intersect case..."<<std::endl;
//      for (unsigned int i =0; i < xn; ++i){
//          unsigned int rand_i = inv_sparsity*i+ r() %(inv_sparsity);
//          xi[i] = rand_i; //we will get a count of the intersection size
//          yi[i] = rand_i; //we will get a count of the intersection size
//      }
//      //std::sort (xi, xi + xn);
//      //std::sort (yi, yi + yn);
//  }
//  else { // generate two different sets of indices, no known intersection pattern
//      for (unsigned int i =0; i < xn; ++i){
//          unsigned int rand_i = inv_sparsity*i +r() % (inv_sparsity);
//          xi[i] = rand_i; //we will get a count of the intersection size
//      }
//      for (unsigned int i =0; i < yn; ++i){
//          unsigned int rand_i = inv_sparsity*i +r() % (inv_sparsity);
//          yi[i] = rand_i; //we will get a count of the intersection size
//      }
//      //std::sort (xi, xi + xn);
//      //std::sort (yi, yi + yn);
//      unsigned int xp =0;
//      unsigned int yp =0;
//      while (1){  //find the intersection size by merge of two sorted lists
//          if (xi[xp] < yi[yp]) xp++;
//          else if (xi[xp] > yi[yp]) yp++;
//          else {
//              intersection_size++;
//              xp++;
//              yp++;
//          }
//          if ( ( xp == xn ) || ( yp == yn) )  break;
//      }
//  }
//  if( xn < 128 ) {
//
//    std::cout<< " xi = [";
//    for (unsigned int i = 0 ; i < xn; ++i) {
//        std::cout<< xi[i] << ",";
//    }
//    std::cout<< " ]" <<std::endl;
//
//  }
//  std::cout << " Launching sparseDot CUDA kernel xn = "<<xn<<" yn="<<yn<<std::endl;
//  spdotFactory<T1,T2,T3> spdF;
//  spdF.jitGridBlockLaunch( grid, block, xn, xi, x, yn, yi, y, output, SEMI_RING );
//
//  cudaDeviceSynchronize ( ) ;
//
//  T3 sum;
//  if (SEMI_RING == "PLUS_TIMES")
//  {
//      myOpPTR<T3> = myOP_plus<T3>;
//      sum = (T3)0;
//  }
//  if (SEMI_RING == "MIN_PLUS")
//  {
//      sum = std::numeric_limits<T3>::max();
//      myOpPTR<T3> = myOP_min<T3>;
//  }
//
//  for (int i =0; i< nblock; ++i) sum = (*myOpPTR<T3>)(sum ,output[i]);
//
//  bool result = false;
//  T3 expect;
//  if (SEMI_RING == "PLUS_TIMES") {
//     T3 temp;
//     expect = intersection_size;
//     temp = (sum - expect);
//     if (temp < 0) temp = -temp ;
//     result = (temp < (T3)1) ; //adjust formula for leading 0
//  }
//  else if (SEMI_RING == "MIN_PLUS") {
//     expect = 2;
//     result = (sum== expect) ;   //min is 2 from the (1,1) pair
//  }
//  else expect = (T3) 0;
//
//  std::cout <<"test_spdotfactoryUM with "<<SEMI_RING<<" semi-ring= "
//            << sum << " expected "<<intersection_size<< std::endl;
//  cudaFree(x);
//  cudaFree(xi);
//  cudaFree(y);
//  cudaFree(yi);
//  cudaFree(output);
//  return result;
//}
