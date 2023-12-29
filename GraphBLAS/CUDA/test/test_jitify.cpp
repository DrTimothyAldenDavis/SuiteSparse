//------------------------------------------------------------------------------
// GraphBLAS/CUDA/test/test_jitify.cpp
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "jitify.hpp"
#include "GB_cuda_jitify_launcher.h"

int main(int argc, char **argv) {

#if 0

BROKEN

    std::string named_program = "GB_jit_AxB_phase2";
    std::string kern_name = "AxB_phase2";


    jitify::experimental::Program& program = *std::get<1>(named_program);
    auto instantiated_kernel = program.kernel(kern_name).instantiate({});

    // hashable name is program name
    // string to be jitted is the actual prgram
    //

    dim3 grid(1);
    dim3 block(1);

//      std::cout<< kernel_name<<" with types " <<GET_TYPE_NAME(dumC)<<std::endl;

    std::string hashable_name = "GB_jit_AxB_phase2";
    std::stringstream string_to_be_jitted ;
    string_to_be_jitted << hashable_name << std::endl <<
    R"(#include "GB_jit_AxB_dot3_phase2.cuh")"; // FIXME: wrong name

    jit::launcher( hashable_name,
                   string_to_be_jitted.str(),
                   header_names,
                   GB_jit_cuda_compiler_flags,
                   file_callback)
            .set_kernel_inst( kernel_name, {})
            .configure(grid, block)
            .launch( nanobuckets, blockBucket, bucketp, bucket, C->mat, cnz);


#endif

}
