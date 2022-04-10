//******************************************************************************
//  Sparse dot products in batch form, sparse - dense case. 
//  Each thread in this kernel is responsible for m vector-pairs(x,y), 
//  m = 256/sz, where sz is in {4, 16, 64, 256}
//  We know each non-zero on the sparse side will hit a dense value.
//  Template on <T_C, T_A, T_B, T_X, T_Y, T_Z >
//  Parameters:

//  int64_t start          <- beginning of bucket  
//  int64_t end            <- end of bucket
//  int64_t *Bucket        <- index of each pair in this bucket
//  matrix<T_C> *C         <- C result matrix 
//  matrix<T_C> *M         <- Mask matrix 
//  matrix<T_A> *A         <- A matrix to multiply, sparse 
//  matrix<T_B> *B         <- B matrix to multiply, dense in sparse format? 
//  int sz                 <- size hint for smaller vector
//******************************************************************************
#pragma once

#include <limits>
#include <cstdint>
#include <stdio.h>
#include "matrix.h"

#include <cooperative_groups.h>

#define tile_sz 32

//#include "local_cub/block/block_reduce.cuh"


using namespace cooperative_groups;

// TODO: Put this in a shared location
template< typename T, int warpSize >
__device__ T reduce_sum(thread_block_tile<warpSize> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        val += g.shfl_down(val,i) ;
    }
    return val; // note: only thread 0 will return full sum
}


template< typename T_C, typename T_A, typename T_B>
__global__ void AxB_dot3_phase3_spdn
( 
  int64_t start, 
  int64_t end,
  int64_t *Bucket, 
  GrB_Matrix C, 
  GrB_Matrix M, 
  GrB_Matrix A, 
  GrB_Matrix B,
  int sz 
)
{
   const T_A *__restrict__ Ax = (T_A *)A->x  ;
   const T_B *__restrict__ Bx = (T_B *)B->x  ;
         T_C *__restrict__ Cx = (T_C *)C->x  ;
         int64_t *__restrict__ Ci = C->i ;
   const int64_t *__restrict__ Mi = M->i ;
   const int64_t *__restrict__ Ai = A->i ;
   const int64_t *__restrict__ Bi = B->i ;
   const int64_t *__restrict__ Ap = A->p ;
   const int64_t *__restrict__ Bp = B->p ;

//   typedef cub::BlockReduce<int, 32> BlockReduce;
//   __shared__ typename BlockReduce::TempStorage temp_storage;

   // sz = expected non-zeros per dot 
   int m = 256/sz;
   int nvec = end - start;
   int dpt = nvec/32;
   m = dpt < m ? dpt : m;
//   if( threadIdx.x ==0)
//      printf("thd:%d %d dots/thrd, nvec = %d blockDim=%d\n",threadIdx.x, sz, nvec, blockDim.x);
//   __syncthreads();
   int dots = (nvec +m -1)/m;

//   printf("dots=%d, m=%d, dpt=%d\n", dots, m, dpt);
   int zc = 0;
     
   for ( int tid= threadIdx.x +blockDim.x*blockIdx.x;
             tid < dots;
             tid += blockDim.x * gridDim.x) {
      int pair_id, im; 
//       if (threadIdx.x ==0)
//         printf("thd%u pi=%lld\n",tid, start+threadIdx.x);
//       __syncthreads();

      for (pair_id = start+tid, im = 0; 
           im < m && pair_id < end;  
           ++im,     pair_id += dots ){

         int64_t i = Mi[pair_id];  // cols from mask

         // TODO: column of Ci / 16?
         int64_t j = Ci[pair_id] >> 4;  // row number of C

         //printf("tid=%d, i=%lu, j=%lu\n", threadIdx.x, i, j);

//      if (threadIdx.x ==0)
//         printf("thd%u i,j=%lld,%lld\n",tid, i,j);
//      __syncthreads();

          // Prime row offsets for both A and B
          int64_t pA       = Ap[i];   // row of C
          int64_t pA_end   = Ap[i+1];
          int64_t nnzA   = pA_end - pA;
          int64_t pB       = Bp[j];   // col of C
          int64_t pB_end   = Bp[j+1];
          int64_t nnzB   = pB_end - pB;
          T_A aki;
          T_B bkj;
          T_C cij;

          int zombie_count = 0;

          if (nnzA == 0 || nnzB == 0)
          {
              i = GB_FLIP (i) ;
              zombie_count +=1;
          }
          else if( nnzA == A->vlen) // A is dense
          {
              /**
               * A is dense, iterate over columns of B, applying monoid and binary op to current
               */
              int64_t k = Bi [pB] ;               // first row index of B(:,j)
              // cij = A(k,i) * B(k,j)
              GB_GETA ( aki=(T_C)Ax[pA+k] ) ;           // aki = A(k,i)
              GB_GETB ( bkj=(T_C)Bx[pB] ) ;           // bkj = B(k,j)


              // TODO: Check tha GB_C_MULT applies the identity automatically since cij has not been initialized
              GB_C_MULT ( cij, aki, bkj ) ;           // cij = aki * bkj

              //printf("A_dense: tid=%d, pair_id=%d, i=%lu, j=%lu, nnzA=%lu, nnzB=%lu, k[B]=%lu, aki=%d, bkj=%d, cij=%d\n", threadIdx.x, pair_id, i, j, nnzA, nnzB, k, aki, bkj, cij);

              /**
               *
               */
              for (int64_t p = pB+1 ; p < pB_end ; ++p)
              {
                  //GB_DOT_TERMINAL (cij) ;             // break if cij == terminal
                  int64_t k = Bi [p] ;                // next row index of B(:,j)
                  // cij += A(k,i) * B(k,j)
                  GB_GETA ( aki=(T_C)Ax[pA+k] ) ;           // aki = A(k,i)
                  GB_GETB ( bkj=(T_C)Bx[p] ) ;           // bkj = B(k,j)
                  GB_MULTADD ( cij, aki, bkj ) ;        // cij += aki * bkj
                  //printf("in_loop: tid=%d, pair_id=%d, i=%lu, j=%lu, nnzA=%lu, nnzB=%lu, k[B]=%lu, aki=%d, bkj=%d, cij=%d\n", threadIdx.x, pair_id, i, j, nnzA, nnzB, k, aki, bkj, cij);
              }

          }
          else if( nnzB == B->vlen) // B is dense
          {
              int64_t k = Ai [pA] ;               // first col index of A(i, :)
              // cij = A(i,k) * B(j,k)
              GB_GETA ( aki=(T_C)Ax[ pA ] ) ;           // aki = A(i,k)

              // Jump straight to position in B vector (since we know it's dense)
              GB_GETB ( bkj=(T_C)Bx[ pB+k ] ) ;           // bkj = B(k,j)

              GB_C_MULT ( cij, aki, bkj) ;           // cij = aki * bkj
              //printf("B_dense: tid=%d, pair_id=%d, i=%lu, j=%lu, nnzA=%lu, nnzB=%lu, k[B]=%lu, aki=%d, bkj=%d, cij=%d\n", threadIdx.x, pair_id, i, j, nnzA, nnzB, k, aki, bkj, cij);

              for (int64_t p = pA+1 ; p < pA_end ; ++p)
              {
                  //GB_DOT_TERMINAL (cij) ;             // break if cij == terminal
                  int64_t k = Ai [p] ;                // next row index of A(:,i)
                  // cij += A(k,i) * B(k,j)
                  GB_GETA ( aki=(T_C)Ax[ p ] ) ;           // aki = A(i,k)
                  GB_GETB ( bkj=(T_C)Bx[ pB+k] ) ;           // bkj = B(j,k)
                  GB_MULTADD ( cij, aki, bkj) ;        // cij += aik * bjk
                  //printf("in_loop: tid=%d, pair_id=%d, i=%lu, j=%lu, nnzA=%lu, nnzB=%lu, k[B]=%lu, aki=%d, bkj=%d, cij=%d\n", threadIdx.x, pair_id, i, j, nnzA, nnzB, k, aki, bkj, cij);
              }
          }
         // C(i,j) = A(:,i) * B(:,j)
//        /**
//         * If A is bitmap, we need to look up offset and nnz of B
//         * and treat Ax as fully dense (size=n*k).
//         */
//
//        // TODO: We probably want to pull this into a separate and
//        //  much smaller kernel just for these formats (e.g. spdn w/ B=bitmap, A=sparse)
//        #if ( GB_A_IS_BITMAP ) // A is dense
//        {
//             int64_t pB = Bp[i];
//             int64_t pB_end   = Bp[i+1];
//             int64_t nnzB   = pB_end - pB;
//             int64_t k = Bi [pB] ;               // first row index of B(:,j)
//            // cij = A(k,i) * B(k,j)
//
////             printf("tid=%d, A is dense, k=%ld, i=%ld\n", threadIdx.x, k, i);
//            GB_GETA ( aki=(T_C)Ax[pA + i] ) ;           // aki = A(k,i)
//            GB_GETB ( bkj=(T_C)Bx[pB] ) ;           // bkj = B(k,j)
//            cij = GB_MULT(aki, bkj ) ;           // cij = aki * bkj
//
//        }
//
//        //TODO: We probably want to pull this into a separate
//        // much smaller kernel just for these formats (e.g. spdn w/ B=full, A=sparse)
//        /**
//         * If A is full, we need to look up offset and nzz of B
//         * and treat Ax as fully dense (size=n*k)
//         */
//        #elif ( GB_A_IS_FULL ) // A is dense
//        {
//             int64_t pB = Bp[i];
//             int64_t pB_end   = Bp[i+1];
//             int64_t nnzB   = pB_end - pB;
//
//            int64_t k = Bi [pB] ;               // first row index of B(:,j)
//            // cij = A(k,i) * B(k,j)
//
////             printf("tid=%d, A is dense, k=%ld, i=%ld\n", threadIdx.x, k, i);
//            GB_GETA ( aki=(T_C)Ax[pA + i] ) ;           // aki = A(k,i)
//            GB_GETB ( bkj=(T_C)Bx[pB] ) ;           // bkj = B(k,j)
//            cij = GB_MULT(aki, bkj ) ;           // cij = aki * bkj
//
//            for (int64_t p = pB+1 ; p < pB_end ; p++)
//            {
//                //GB_DOT_TERMINAL (cij) ;           // break if cij == terminal
//                int64_t k = Bi [p] ;                // next row index of B(:,j)
//                // cij += A(k,i) * B(k,j)
//                GB_GETA ( aki=(T_C)Ax[A->vlen * i + k] ) ;      // aki = A(k,i)
//                GB_GETB ( bkj=(T_C)Bx[p] ) ;                    // bkj = B(k,j)
//                cij = GB_ADD ( cij, GB_MULT(aki, bkj ) ) ;      // cij += aki * bkj
//            }
//        }
//
//        /**
//         * If A is sparse but current row of A is dense, we need to look up
//         * offset of B and offset of A
//         */
//        #elif (GB_B_IS_BITMAP)
//        {
//
//        }
//
//        #elif (GB_B_IS_FULL)
//        {
//
//        }
//
//        /**
//         * If
//         */
//        #else
//        {
//
//
//             int64_t pA = Ap[i];
//             int64_t pA_end   = Ap[i+1];
//             int64_t nnzA   = pA_end - pA;
//
//            int64_t k = Ai [pA] ;               // first row index of A(:,i)
////             printf("tid=%d, B is dense, k=%ld, j=%ld\n", threadIdx.x, k, j);
//            // cij = A(k,i) * B(k,j)
//            GB_GETA ( aki= (T_C)Ax[ pA ] ) ;           // aki = A(k,i)
//            GB_GETB ( bkj=(T_C)Bx[ B->vlen*k+j ] ) ;           // bkj = B(k,j)
//
//            cij =  GB_MULT(aki, bkj) ;           // cij = aki * bkj
////             printf("aki=%d, bkj=%d, cij=%d\n", aki, bkj, cij);
//
//            for (int64_t p = pA+1 ; p < pA_end ; p++)
//            {
//                //GB_DOT_TERMINAL (cij) ;             // break if cij == terminal
//                int64_t k = Ai [p] ;                // next row index of A(:,i)
//                // cij += A(k,i) * B(k,j)
//                GB_GETA ( aki=(T_C)Ax[ p ] ) ;           // aki = A(k,i)
//                GB_GETB ( bkj=(T_C)Bx[ B->vlen*k+j] ) ;           // bkj = B(k,j)
//                cij = GB_ADD ( cij, GB_MULT(aki, bkj) );        // cij += aki * bkj
////                printf("aki=%d, bkj=%d, cij=%d\n", aki, bkj, cij);
//            }
//         } else {
//             if(threadIdx.x == 0 && blockIdx.x == 0) {
//                 printf("ERROR: At least one side must be dense.\n");
//                 break;
//             }
//         }

         GB_PUTC( Ci[pair_id]=i ) ;
         GB_PUTC( Cx[pair_id]=cij ) ;

//         int zc = BlockReduce(temp_storage).Sum(zombie_count);
          thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());
          int zc = reduce_sum<int,tile_sz>(tile, zombie_count);

         if(threadIdx.x == 0 && zc > 0)
            atomicAdd(&(C->nzombies), zc);
      }
  
   }
   
}
