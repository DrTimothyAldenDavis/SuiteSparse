//------------------------------------------------------------------------------
// GB_Scalar_wrap: wrap a C scalar inside a GraphBLAS scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This method construct a shallow statically-defined scalar, with no memory
// allocations.

#include "GB.h"
#include "GB_scalar.h"

GxB_Scalar GB_Scalar_wrap   // create a new GxB_Scalar with one entry
(
    GxB_Scalar s,           // GxB_Scalar to create
    GrB_Type type,          // type of GxB_Scalar to create
    int64_t *Sp,            // becomes S->p, an array of size 2
    int64_t *Si,            // becomes S->i, an array of size 1
    void *Sx                // becomes S->x, an array of size 1 * type->size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (s != NULL) ;

    //--------------------------------------------------------------------------
    // create the GxB_Scalar
    //--------------------------------------------------------------------------

    s->magic = GB_MAGIC ;
    s->type = type ;
    s->type_size = type->size ;
    s->hyper_ratio = GB_HYPER_DEFAULT ;
    s->plen = 1 ;
    s->vlen = 1 ;
    s->vdim = 1 ;
    s->nvec = 1 ;
    s->nvec_nonempty = 1 ;
    s->p = Sp ; Sp [0] = 0 ; Sp [1] = 1 ;
    s->h = NULL ;
    s->i = Si ; Si [0] = 0 ;
    s->x = Sx ;
    s->nzmax = 1 ;
    s->hfirst = 0 ;
    s->Pending = NULL ;
    s->nzombies = 0 ;
    s->AxB_method_used = GxB_DEFAULT ;
    s->queue_next = NULL ;  // TODO in 4.0: delete
    s->queue_prev = NULL ;  // TODO in 4.0: delete
    s->enqueued = false ;   // TODO in 4.0: delete
    s->p_shallow = true ;
    s->h_shallow = false ;
    s->i_shallow = true ;
    s->x_shallow = true ;
    s->is_hyper = false ;
    s->is_csc = true ;
    s->is_slice = false ;
    s->mkl = NULL ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (s) ;
}

