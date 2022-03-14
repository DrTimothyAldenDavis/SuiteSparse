// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  Extended example for building on-the-fly kernels with C interface.
  Simple examples demonstrating different ways to load source code
    and call kernels.
 */

#pragma once

#include "GB_jit_launcher.h"
#include "GB_cuda_semiring_factory.hpp"

// FIXME: Is this okay or will it bring in too much (GB.h is brought in transitively)
#include "GraphBLAS.h"
#include "GB_Semiring_new.c"
#include "GrB_Semiring_new.c"
#include "GB_Monoid_new.c"
#include "GrB_Monoid_new.c"
#include "GB_cuda_buckets.h"

#include "type_name.hpp"

#undef  JITIFY_PRINT_INSTANTIATION
#define JITIFY_PRINT_INSTANTIATION 1
#undef  JITIFY_PRINT_SOURCE
#define JITIFY_PRINT_SOURCE 1
#undef  JITIFY_PRINT_LOG
#define JITIFY_PRINT_LOG 1
#undef  JITIFY_PRINT_PTX
#define JITIFY_PRINT_PTX 1
#undef  JITIFY_PRINT_LINKER_LOG
#define JITIFY_PRINT_LINKER_LOG 1
#undef  JITIFY_PRINT_LAUNCH
#define JITIFY_PRINT_LAUNCH 1

#include "test/dataFactory.hpp"
#include "test/semiringFactory.hpp"
// #include "GB_cuda.h"


#if __cplusplus >= 201103L

/**
 * This file is responsible for picking all the parameters and what kernel variaiton we will use for a given instance
 * - data types
 * - semiring types
 * - binary ops
 * - monoids
 *
 * Kernel factory says "Here's the actual instance I want you to build with the given parameters"
 */

//Kernel jitifiers
template<typename T> class reduceFactory ;
template<typename T1, typename T2, typename T3> class dotFactory ;
template<typename T1, typename T2, typename T3> class spdotFactory ;


//AxB_dot3_phase1 kernel launchers
template<  typename T_C, typename T_M, typename T_A, typename T_B, int threads_per_block, int chunk_size> class phase1launchFactory ;

//AxB_dot3_phase3 kernel launchers

template<  typename T_C, typename T_M, 
         typename T_A, typename T_B, typename T_xy, typename T_z> class launchFactory ;


const std::vector<std::string> compiler_flags{
   "-std=c++14",
   "-G",
   "-remove-unused-globals",
   "-w",
   "-D__CUDACC_RTC__",
   "-I.",
   "-I..",
// "-I../../Include",
   "-I../../Source",
   "-I../../Source/Template",
   "-I../local_cub/block",
   "-I../templates",
   "-I/usr/local/cuda/include",
};

const std::vector<std::string> header_names ={};

// FIXME: Need to be able to convert from GrB_Type->std::type to populate these templates
// this isn't going to be known at compile time.
template<  typename T_C, typename T_M, typename T_A, typename T_B, int threads_per_block=32, int chunk_size = 128>
class phase1launchFactory 
{
  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_phase1";

  GB_cuda_semiring_factory &semiring_factory_;

public:

  int get_number_of_blocks(GrB_Matrix M) {
      int number_of_sms = GB_Global_gpu_sm_get (0);
      int nblks = ( GB_nnz (M) + chunk_size - 1)/chunk_size;
      return GB_IMIN( nblks,  128 * number_of_sms);
  }

  int get_threads_per_block() {
      return threads_per_block;
  }

  // This assumes the needed state on the GB_cuda_semiring_factory
  // has already been populated
  phase1launchFactory(GB_cuda_semiring_factory &semiring_factory): semiring_factory_(semiring_factory){}

  bool jitGridBlockLaunch(int64_t *nanobuckets, int64_t *blockBucket,
                          GrB_Matrix C, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B) {

    // Idea is to have each task work on a continguous block of columns of C
    // Note: for small tests, mnz is small so ntasks is be governed by
    // chunksize, not 128*number_of_sms.  For large problems in production,
    // chunksize is less important since ntasks will likely be bounded by
    // 128*number_of_sms (say 128*80 = 10,240 on a V100).

    // Defining dummy instance only so we can introspect type
    T_M dumM;

    std::cout << "A TYpe: " << A->type << std::endl;
    std::cout << "B TYpe: " << B->type << std::endl;
//    // (1) create the semiring code and name

    //    // (2) ensure the jitifier has "GB_semiring_[mysemiring.sr_code].h"
    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (semiring_factory_) ;

    std::stringstream string_to_be_jitted ;
    std::vector<std::string> template_types = {GET_TYPE_NAME(dumM)};

    std::string hashable_name = base_name + "_" + kernel_name;
    string_to_be_jitted << hashable_name << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << semiring_factory_.filename << R"(")" << std::endl <<
    R"(#include ")" << hashable_name << R"(.cu")" << std::endl;
    std::cout << string_to_be_jitted.str();

    bool result = false;

    dim3 grid(get_number_of_blocks(M));
    dim3 block(get_threads_per_block());

    jit::launcher( hashable_name,
                   string_to_be_jitted.str(),
                   header_names,
                   compiler_flags,
                   file_callback)
                 .set_kernel_inst(  kernel_name, template_types)
                 .configure(grid, block)
                 .launch( nanobuckets, blockBucket, C, M, A, B);

      checkCudaErrors( cudaDeviceSynchronize() );
      result = true;

      return result;
     }
};

template<  typename T_C, int threads_per_block = 32, int chunk_size = 128>
class phase2launchFactory
{

  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_phase2";

public:

  int get_threads_per_block() {
        return threads_per_block;
  }

  int get_number_of_blocks(GrB_Matrix M) {
    const int64_t mnz = GB_nnz (M) ;
    int ntasks = ( mnz +chunk_size -1)/chunk_size;
    // Idea is to have each task work on a continguous block of columns of C
    ntasks = GB_IMIN( ntasks,  128*GB_Global_gpu_sm_get (0)) ;    // ntasks will be grid.x
    return (ntasks + threads_per_block - 1) / threads_per_block ;
  }

  bool jitGridBlockLaunch(// parameters to AxB_phase2:
                          int64_t *nanobuckets, int64_t *blockBucket, 
                          int64_t *bucketp, int64_t *bucket, int64_t *offset,
                          GrB_Matrix M) {

    bool result = false;

      dim3 grid(get_number_of_blocks(M));
      dim3 block(get_threads_per_block());

      std::string hashable_name = base_name + "_" + kernel_name;
      std::stringstream string_to_be_jitted ;
      string_to_be_jitted <<
      hashable_name << std::endl << R"(#include ")" << hashable_name << R"(.cu")" << std::endl;

      // dump it:
      std::cout << string_to_be_jitted.str();

      const int64_t mnz = GB_nnz (M) ;
      jit::launcher( hashable_name,
                     string_to_be_jitted.str(),
                     header_names, 
                     compiler_flags,
                     file_callback)
                   .set_kernel_inst( kernel_name, {})
                   .configure(grid, block)
                   // parameters to AxB_phase2:
                   .launch( nanobuckets, blockBucket, bucketp, bucket, offset, mnz);

      checkCudaErrors( cudaDeviceSynchronize() );
      result= true;

      return result;
     }

};

template<  typename T_C, int threads_per_block = 32, int chunk_size = 128>
class phase2endlaunchFactory 
{

  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_phase2end";

public: 

  int get_threads_per_block() {
        return threads_per_block;
  }

  int get_number_of_blocks(GrB_Matrix M) {
    const int64_t mnz = GB_nnz (M) ;
    int ntasks = ( mnz +chunk_size -1)/chunk_size;
    int number_of_sms = GB_Global_gpu_sm_get (0);

    // Idea is to have each task work on a continguous block of columns of C
    return GB_IMIN( ntasks,  128*number_of_sms) ;    // ntasks will be grid.x
  }

  bool jitGridBlockLaunch(int64_t *nanobuckets, int64_t *blockBucket,
                          int64_t *bucketp, int64_t *bucket, int64_t *offset,
                          GrB_Matrix C, GrB_Matrix M)
     {
      
      bool result = false; 

      T_C dumC;

      dim3 grid(get_number_of_blocks(M));
      dim3 block(get_threads_per_block());

      std::string hashable_name = base_name + "_" + kernel_name;
      std::stringstream string_to_be_jitted ;
      string_to_be_jitted <<
      hashable_name << std::endl << R"(#include ")" << hashable_name << R"(.cu")" << std::endl;

      // dump it:
      std::cout << string_to_be_jitted.str();

      jit::launcher( hashable_name,
                     string_to_be_jitted.str(),
                     header_names, 
                     compiler_flags,
                     file_callback)
                   .set_kernel_inst(  kernel_name , {})
                   .configure(grid, block)
                   .launch( nanobuckets, blockBucket, bucketp, bucket, offset, C, GB_nnz (M));

      checkCudaErrors( cudaDeviceSynchronize() );
      result= true;

      return result;
     }

};

template<  typename T_C, typename T_M, typename T_A, typename T_B, typename T_XY, typename T_Z>
class phase3launchFactory
{
  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_dot3";

  GB_cuda_semiring_factory &semiring_factory_;

  GB_bucket_code bucket_code_;
  GB_callback callback_generator;

public:



  /**
   * This assumes the needed state on the GB_cuda_semiring_factory has already been populated.
   * The `bucket_code` determines which kernel is launched
   */
  phase3launchFactory(GB_cuda_semiring_factory &mysemiringfactory, GB_bucket_code bucket_code):
      semiring_factory_(mysemiringfactory), bucket_code_(bucket_code) {}

  bool jitGridBlockLaunch(int64_t start, int64_t end, int64_t *bucketp, int64_t *bucket,
                          GrB_Matrix C,  GrB_Matrix M, GrB_Matrix A, GrB_Matrix B) {
      
      bool result = false; 

      T_C dumC;
      T_M dumM;
      T_A dumA;
      T_B dumB;
      T_XY dumXY;
      T_Z dumZ;


    //----------------------------------------------------------------------
    // phase3: do the numerical work
    //----------------------------------------------------------------------

    C->nzombies = bucketp[1];  //set pre-zombie counts
    const int64_t Cnz = GB_nnz (C) ;
    const int64_t mnvec = M->nvec ;

    int gridsz, blocksz, sz = 4;

    std::stringstream final_kernel_name_ss;
    final_kernel_name_ss << kernel_name << "_";

    /**
     * Configure geometry and kernel function name based on sparsity of C and number of vectors in M
     */
    configure(Cnz, mnvec, final_kernel_name_ss, blocksz, gridsz, sz);

    std::string hashable_name = base_name + "_" + final_kernel_name_ss.str();
    std::stringstream string_to_be_jitted ;

    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (semiring_factory_) ;

    string_to_be_jitted << hashable_name << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << semiring_factory_.filename << R"(")" << std::endl <<
    R"(#include ")" << hashable_name << R"(.cu")" << std::endl;

    std::cout << "String to be jitted: " << string_to_be_jitted.str() << std::endl;

    dim3 grid(gridsz);
    dim3 block(blocksz);

    std::cout<< "program name =" <<hashable_name<<std::endl;
    std::cout << "Final kernel name =" << final_kernel_name_ss.str() << std::endl;
    GBURBLE ("(GPU phase3 launch st,end=%ld,%ld nblocks,blocksize= %d,%d )\n",start,end,gridsz,blocksz) ;
    printf("(GPU phase3 launch st,end=%ld,%ld nblocks,blocksize= %d,%d )\n",start,end,gridsz,blocksz) ;
    jit::launcher( hashable_name,
                   string_to_be_jitted.str(),
                   header_names,
                   compiler_flags,
                   file_callback)
               .set_kernel_inst(final_kernel_name_ss.str(),
                                { GET_TYPE_NAME(dumC),
                                  GET_TYPE_NAME(dumA),
                                  GET_TYPE_NAME(dumB),
                                  GET_TYPE_NAME(dumXY),
                                  GET_TYPE_NAME(dumXY),
                                  GET_TYPE_NAME(dumZ) })
               .configure(grid, block) //if commented, use implicit 1D configure in launch
               .launch(
                        start,             // input/output:
                        end,               // global bucket cumsum, of size NBUCKETS+1
                        bucket,            // global buckets, of size cnz (== mnz)
                        C,                 // final output matrix
                                           // inputs, not modified:
                        M,                 // Mi used for column index
                        A,                 // A matrix
                        B,                 // B matrix
                        sz                 // only used for sparse-sparse cases
                    );

    GBURBLE ("(GPU phase3 done) ") ;

    // do we really want to sync after each kernel launch in production?
    checkCudaErrors( cudaDeviceSynchronize() );
    result= true;

    return result;
  }

private:
    void configure(std::int64_t Cnz, std::int64_t mnvec, std::stringstream &opname,
                   int &blocksz, int &gridsz, int &sz) {
    int number_of_sms = GB_Global_gpu_sm_get (0) ;

    std::string Opname;
    switch (bucket_code_)
    {

        //--------------------------------------------------------------
        // not a bucket ... bring out your dead:
        //--------------------------------------------------------------

        case GB_BUCKET_ZOMBIE : // C(i,j) is a zombie (not a bucket)
            break ;

        //--------------------------------------------------------------
        // CUDA kernel: dndn, handles a single bucket:
        //--------------------------------------------------------------

        // both A(:,i) and B(:,j) are dense
        case GB_BUCKET_DNDN :
            Opname = "phase3_dndn" ;

            blocksz = 32;
            gridsz = ( Cnz -1 + blocksz)/blocksz;
            break ;

        //--------------------------------------------------------------
        // CUDA kernel: spdn, handles 4 buckets:
        //--------------------------------------------------------------

        // A(:,i) is dense and B(:,j) is very sparse (< 256 entries)
        case GB_BUCKET_DNVS :
        // A(:,i) is very sparse (< 256 entries) and B(:,j) is dense
        case GB_BUCKET_VSDN :
            sz = 64 ;
            Opname = "phase3_spdn" ;
            blocksz = 32;
            gridsz = ( Cnz -1 + blocksz)/blocksz;
            break ;

        // A(:,i) is dense and B(:,j) is sparse (>= 256 entries)
        case GB_BUCKET_DNSP :
        // A(:,i) is sparse (>= 256 entries) and B(:,j) is dense
        case GB_BUCKET_SPDN :
            printf("Confiring spdn");
            sz = 256 ;
            Opname = "phase3_spdn" ;
            blocksz = 32;
            gridsz = ( Cnz -1 + blocksz)/blocksz;
            break ;

        //--------------------------------------------------------------
        // CUDA kernel: vssp, handles 1 bucket, uses binary search:
        //--------------------------------------------------------------

        // A(:,i) is very sparse compared to B(:,j), or visa versa
        case GB_BUCKET_VSSP :
            Opname = "phase3_vssp" ;
            blocksz = 32;
            gridsz = ( Cnz -1 + blocksz)/blocksz;
            break ;

        //--------------------------------------------------------------
        // CUDA kernel: vsvs, handles 4 buckets:
        //--------------------------------------------------------------

        // let len = nnz (A (:,i) + nnz (B (:,j)), then:

        case GB_BUCKET_VSVS_256 : sz += 256-64 ;
        case GB_BUCKET_VSVS_64 :  sz += 64-16  ;
        case GB_BUCKET_VSVS_16 :  sz += 16-4   ;
        case GB_BUCKET_VSVS_4 :   sz += 4      ;
            Opname = "phase3_vsvs" ;
            blocksz = 1024;
            gridsz = GB_IMIN( 1024*number_of_sms, ( Cnz  + blocksz -1 )/blocksz);
            gridsz =  ( Cnz  + blocksz -1 )/blocksz;
            break ;

        //--------------------------------------------------------------
        // CUDA kernel: mp, use the merge-path method:
        //--------------------------------------------------------------

        case GB_BUCKET_MERGEPATH :
            Opname = "phase3_mp" ;
            blocksz = 32;
            gridsz = ( Cnz -1 + blocksz)/blocksz;
            break ;

        case GB_BUCKET_WARP_IX :   sz = 32      ;
            Opname = "phase3_warpix" ;
            blocksz = 32;
            gridsz =  GB_IMIN( (mnvec+15)/16, 256*number_of_sms);
            break ;

        default:
            break ;
    }

    opname << Opname;
  }
};

//template<typename T1, typename T2, typename T3>
//class spdotFactory
//{
//  std::string base_name = "GBjit_spDot_";
//public:
//  spdotFactory() {
//  }
//
//  bool jitGridBlockLaunch(int gridsz, int blocksz, unsigned int xn, unsigned int *xi, T1* x,
//                                                   unsigned int yn, unsigned int *yi, T2* y,
//                                                        T3* output, std::string OpName)
//  {
//
//      bool result = false;
//      if (OpName == "PLUS_TIMES") {
//         file_callback = &semiring_plus_times_callback;
//      }
//      else if (OpName == "MIN_PLUS") {
//         file_callback = &semiring_min_plus_callback;
//      }
//
//      T1 dum1;
//      T2 dum2;
//      T3 dum3;
//
//      dim3 grid(gridsz);
//      dim3 block(blocksz);
//
//      jit::launcher( base_name + OpName,
//                     ___templates_sparseDotProduct_cu,
//                     header_names,
//                     compiler_flags,
//                     file_callback)
//
//                   .set_kernel_inst("sparseDotProduct",
//                                    { GET_TYPE_NAME(dum1),
//                                      GET_TYPE_NAME(dum2),
//                                      GET_TYPE_NAME(dum3)})
//                   .configure(grid, block)
//                   .launch(xn, xi, x, yn, yi, y, output);
//
//
//      checkCudaErrors( cudaDeviceSynchronize() );
//      result= true;
//
//      return result;
//  }
//
//};
//
//template<typename T1, typename T2, typename T3>
//class dotFactory
//{
//  std::string base_name = "GBjit_dnDot_";
//public:
//  dotFactory() {
//  }
//
//
//  bool jitGridBlockLaunch(int gridsz, int blocksz, T1* x, T2* y, T3* output, unsigned int N, std::string OpName)
//  {
//
//      bool result = false;
//      if (OpName == "PLUS_TIMES") {
//         file_callback = &semiring_plus_times_callback;
//      }
//      else if (OpName == "MIN_PLUS") {
//         file_callback = &semiring_min_plus_callback;
//      }
//
//      T1 dum1;
//      T2 dum2;
//      T3 dum3;
//
//      dim3 grid(gridsz);
//      dim3 block(blocksz);
//
//      jit::launcher( base_name + OpName,
//                     ___templates_denseDotProduct_cu,
//                     header_names,
//                     compiler_flags,
//                     file_callback)
//
//                   .set_kernel_inst("denseDotProduct",
//                                    { GET_TYPE_NAME(dum1),
//                                      GET_TYPE_NAME(dum2),
//                                      GET_TYPE_NAME(dum3)})
//                   .configure(grid, block)
//                   .launch(x, y, output, N);
//
//      checkCudaErrors( cudaDeviceSynchronize() );
//      result= true;
//
//      return result;
//  }
//
//};
//
//template<typename T>
//class reduceFactory
//{
//  std::string base_name = "GBjit_reduce_";
//
//public:
//  reduceFactory() {
//  }
//
//  bool jitGridBlockLaunch(int gridsz, int blocksz,
//                          T* indata, T* output, unsigned int N,
//                          std::string OpName)
//  {
//      dim3 grid(gridsz);
//      dim3 block(blocksz);
//      bool result = false;
//      T dummy;
//
//      std::cout<<" indata type ="<< GET_TYPE_NAME(dummy)<<std::endl;
//
//      if (OpName == "PLUS") {
//         file_callback = &file_callback_plus;
//      }
//      else if (OpName == "MIN") {
//         file_callback = &file_callback_min;
//      }
//      else if (OpName == "MAX") {
//         file_callback = &file_callback_max;
//      }
//
//
//      jit::launcher( base_name + OpName,
//                     ___templates_reduceUnrolled_cu,
//                     header_names,
//                     compiler_flags,
//                     file_callback)
//                   .set_kernel_inst("reduceUnrolled",
//                                    { GET_TYPE_NAME(dummy) })
//                   .configure(grid, block)
//                   .launch( indata, output, N);
//
//      checkCudaErrors( cudaDeviceSynchronize() );
//
//      result= true;
//
//
//      return result;
//  }
//
//};
//
#endif  // C++11

