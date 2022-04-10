
// Decide branch direction for GPU use for the dot-product MxM
#include "GB_cuda.h"

bool GB_reduce_to_scalar_cuda_branch 
(
    const GrB_Monoid reduce,        // monoid to do the reduction
    const GrB_Matrix A,             // input matrix
    GB_Context Context
)
{

    // work to do
    double work = GB_nnz (A) ;

//    std::cout << "IS_BITMAP: " << GB_IS_BITMAP (A) << "IS_FULL: " << GB_IS_FULL(A) << std::endl;

    int ngpus_to_use = GB_ngpus_to_use (work) ;
    GBURBLE (" work:%g gpus:%d ", work, ngpus_to_use) ;
    printf (" work:%g gpus:%d ", work, ngpus_to_use) ;
    if (ngpus_to_use > 0
        && (reduce->header_size == 0)     // semiring is built-in
        && (A->type->code != GB_UDT_code)
        // FIXME: this is easy
        && !A->iso
    ) {
        return true;
    }
    else
    { 
        return false;
    }

}
