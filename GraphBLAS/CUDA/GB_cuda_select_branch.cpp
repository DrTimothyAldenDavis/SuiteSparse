#include "GB_cuda.hpp"

bool GB_cuda_select_branch
(
    const GrB_Matrix A,
    const GrB_IndexUnaryOp op
)
{
    if (op == NULL)
    {
        return false ;
    }

    if (A->static_header)
    {
        // FIXME: explain this
        return false ;
    }
    
    bool ok = (GB_cuda_type_branch (A->type)) ;

    if (op->xtype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->xtype)) ;
    }
    if (op->ytype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->ytype)) ;
    }
    if (op->ztype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->ztype)) ;
    }

    ok = ok && (op->hash != UINT64_MAX) ;

    return ok ;
}

