//------------------------------------------------------------------------------
// GB_Matrix_extractElement: x = A(row,col)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Extract the value of single scalar, x = A(row,col), typecasting from the
// type of A to the type of x, as needed.

// Returns GrB_SUCCESS if A(row,col) is present, and sets x to its value.
// Returns GrB_NO_VALUE if A(row,col) is not present, and x is unmodified.

// This template constructs GrB_Matrix_extractElement_[TYPE] for each of the
// 13 built-in types, and the _UDT method for all user-defined types.

// FUTURE: tolerate zombies

GrB_Info GB_EXTRACT_ELEMENT     // extract a single entry, x = A(row,col)
(
    GB_XTYPE *x,                // scalar to extract, not modified if not found
    const GrB_Matrix A,         // matrix to extract a scalar from
    GrB_Index row,              // row index
    GrB_Index col               // column index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CONTEXT_RETURN_IF_NULL (A) ;
    GB_CONTEXT_RETURN_IF_FAULTY (A) ;

    // delete any lingering zombies and assemble any pending tuples
    if (GB_PENDING_OR_ZOMBIES (A))
    { 
        GrB_Info info ;
        GB_WHERE (GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Matrix_extractElement") ;
        GB_OK (GB_Matrix_wait (A, Context)) ;
        ASSERT (!GB_ZOMBIES (A)) ;
        ASSERT (!GB_PENDING (A)) ;
        GB_BURBLE_END ;
    }

    GB_CONTEXT_RETURN_IF_NULL (x) ;

    // look for index i in vector j
    int64_t i, j, nrows, ncols ;
    if (A->is_csc)
    { 
        i = row ;
        j = col ;
        nrows = A->vlen ;
        ncols = A->vdim ;
    }
    else
    { 
        i = col ;
        j = row ;
        nrows = A->vdim ;
        ncols = A->vlen ;
    }

    // check row and column indices
    if (row >= nrows)
    { 
        GB_WHERE (GB_WHERE_STRING) ;
        return (GB_ERROR (GrB_INVALID_INDEX, (GB_LOG, "Row index "
            GBu " out of range; must be < " GBd, row, nrows))) ;
    }
    if (col >= ncols)
    { 
        GB_WHERE (GB_WHERE_STRING) ;
        return (GB_ERROR (GrB_INVALID_INDEX, (GB_LOG, "Column index "
            GBu " out of range; must be < " GBd, col, ncols))) ;
    }

    // GB_XCODE and A must be compatible
    GB_Type_code acode = A->type->code ;
    if (!GB_code_compatible (GB_XCODE, acode))
    { 
        GB_WHERE (GB_WHERE_STRING) ;
        return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
            "entry A(i,j) of type [%s] cannot be typecast\n"
            "to output scalar x of type [%s]",
            A->type->name, GB_code_string (GB_XCODE)))) ;
    }

    if (A->nzmax == 0)
    { 
        // quick return
        return (GrB_NO_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // binary search in A->h for vector j
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    bool found ;

    // extract from vector j of a GrB_Matrix
    int64_t k ;
    if (A->is_hyper)
    {
        // look for vector j in hyperlist A->h [0 ... A->nvec-1]
        const int64_t *Ah = A->h ;
        int64_t pleft = 0 ;
        int64_t pright = A->nvec-1 ;
        GB_BINARY_SEARCH (j, Ah, pleft, pright, found) ;
        if (!found)
        { 
            // vector j is empty
            return (GrB_NO_VALUE) ;
        }
        ASSERT (j == Ah [pleft]) ;
        k = pleft ;
    }
    else
    { 
        k = j ;
    }
    int64_t pleft = Ap [k] ;
    int64_t pright = Ap [k+1] - 1 ;

    //--------------------------------------------------------------------------
    // binary search in kth vector for index i
    //--------------------------------------------------------------------------

    // Time taken for this step is at most O(log(nnz(A(:,j))).
    GB_BINARY_SEARCH (i, Ai, pleft, pright, found) ;

    //--------------------------------------------------------------------------
    // extract the element
    //--------------------------------------------------------------------------

    if (found)
    {
        #if !defined ( GB_UDT_EXTRACT )
        if (GB_XCODE == acode)
        { 
            // copy the value from A into x, no typecasting, for built-in
            // types only.
            GB_XTYPE *GB_RESTRICT Ax = ((GB_XTYPE *) (A->x)) ;
            (*x) = Ax [pleft] ;
        }
        else
        #endif
        { 
            // typecast the value from A into x
            size_t asize = A->type->size ;
            GB_cast_array ((GB_void *) x, GB_XCODE,
                ((GB_void *) A->x) +(pleft*asize), acode, asize, 1, 1) ;
        }
        return (GrB_SUCCESS) ;
    }
    else
    { 
        // Entry not found.
        return (GrB_NO_VALUE) ;
    }
}

#undef GB_UDT_EXTRACT
#undef GB_EXTRACT_ELEMENT
#undef GB_XTYPE
#undef GB_XCODE

