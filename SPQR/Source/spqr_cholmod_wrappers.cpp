#include "spqr.hpp"

// For functions which might not be distinguishable by their prototype,
// use plain-old functions.  For the others, use template specializations.

#if SuiteSparse_long_max != INT32_MAX

// If SuiteSparse_long is a different type than int32_t (e.g., on 64-bit
// platforms), use template specializations to select the correct function on
// compile time.


// Specializations for SuiteSparse_long

template <> void *
spqr_malloc <SuiteSparse_long> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_l_malloc (n, size, Common) ;
}

template <> void *
spqr_calloc <SuiteSparse_long> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_l_calloc (n, size, Common) ;
}

template <> void *
spqr_free <SuiteSparse_long> (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return cholmod_l_free (n, size, p, Common) ;
}

template <> void *
spqr_realloc <SuiteSparse_long>
(
    size_t nnew, size_t size, void *p, size_t *n, cholmod_common *Common
)
{
    return cholmod_l_realloc (nnew, size, p, n, Common) ;
}

template <> cholmod_sparse *
spqr_allocate_sparse <SuiteSparse_long>
(
    size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed,
    int stype, int xtype, cholmod_common *Common
)
{
    return cholmod_l_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) ;
}

template <> int
spqr_free_sparse <SuiteSparse_long> (cholmod_sparse **A, cholmod_common *Common)
{
    return cholmod_l_free_sparse (A, Common) ;
}

template <> int
spqr_reallocate_sparse <SuiteSparse_long>
(
    size_t nznew, cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_l_reallocate_sparse (nznew, A, Common) ;
}

template <> cholmod_dense *
spqr_allocate_dense <SuiteSparse_long>
(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *Common
)
{
    return cholmod_l_allocate_dense (nrow, ncol, d, xtype, Common) ;
}

template <> int
spqr_free_dense <SuiteSparse_long>
(
    cholmod_dense **X, cholmod_common *Common
)
{
    return cholmod_l_free_dense (X, Common) ;
}

template <> cholmod_factor *
spqr_allocate_factor <SuiteSparse_long>
(
    size_t n, cholmod_common *Common
)
{
    return cholmod_l_allocate_factor (n, Common) ;
}

template <> int
spqr_free_factor <SuiteSparse_long>
(
    cholmod_factor **L, cholmod_common *Common
)
{
    return cholmod_l_free_factor (L, Common) ;
}

template <> int
spqr_allocate_work <SuiteSparse_long>
(
    size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common
)
{
    return cholmod_l_allocate_work (nrow, iworksize, xworksize, Common) ;
}

template <> int
spqr_amd <SuiteSparse_long> 
(
    cholmod_sparse *A, SuiteSparse_long *fset, size_t fsize,
    SuiteSparse_long *Perm, cholmod_common *Common
)
{
    return cholmod_l_amd (A, fset, fsize, Perm, Common) ;
}

template <> int
spqr_metis <SuiteSparse_long> 
(
    cholmod_sparse *A, SuiteSparse_long *fset, size_t fsize, int postorder,
    SuiteSparse_long *Perm, cholmod_common *Common
)
{
    return cholmod_l_metis (A, fset, fsize, postorder, Perm, Common) ;
}

template <> cholmod_sparse *
spqr_transpose <SuiteSparse_long>
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return cholmod_l_transpose (A, values, Common) ;
}

template <> cholmod_factor *
spqr_analyze_p2 <SuiteSparse_long>
(
    int for_whom, cholmod_sparse *A, SuiteSparse_long *UserPerm,
    SuiteSparse_long *fset, size_t fsize, cholmod_common *Common
)
{
    return cholmod_l_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common) ;
}

template <> int
spqr_colamd <SuiteSparse_long>
(
    cholmod_sparse *A, SuiteSparse_long *fset, size_t fsize, int postorder,
    SuiteSparse_long *Perm, cholmod_common *Common
)
{
    return cholmod_l_colamd (A, fset, fsize, postorder, Perm, Common) ;
}

template <> SuiteSparse_long
spqr_postorder <SuiteSparse_long>
(
    SuiteSparse_long *Parent, size_t n, SuiteSparse_long *Weight_p,
    SuiteSparse_long *Post, cholmod_common *Common
)
{
    return cholmod_l_postorder (Parent, n, Weight_p, Post, Common) ;
}

template <> int64_t
spqr_nnz <SuiteSparse_long>
(
    cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_l_nnz(A, Common) ;
}

template <> cholmod_sparse *
spqr_dense_to_sparse <SuiteSparse_long>
(
    cholmod_dense *X, int values, cholmod_common *Common
)
{
    return cholmod_l_dense_to_sparse(X, values, Common) ;
}

template <> cholmod_dense *
spqr_sparse_to_dense <SuiteSparse_long>
(
    cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_l_sparse_to_dense (A, Common) ;
}

template <> cholmod_sparse *
spqr_speye <SuiteSparse_long>
(
    size_t nrow, size_t ncol, int xtype, cholmod_common *Common
)
{
    return cholmod_l_speye(nrow, ncol, xtype, Common) ;
}


// Specializations for int32_t

template <> void *
spqr_malloc <int32_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_malloc (n, size, Common) ;
}

template <> void *
spqr_calloc <int32_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_calloc (n, size, Common) ;
}

template <> void *
spqr_free <int32_t> (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return cholmod_free (n, size, p, Common) ;
}

template <> void *
spqr_realloc <int32_t>
(
    size_t nnew, size_t size, void *p, size_t *n, cholmod_common *Common
)
{
    return cholmod_realloc (nnew, size, p, n, Common) ;
}

template <> cholmod_sparse *
spqr_allocate_sparse <int32_t>
(
    size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed,
    int stype, int xtype, cholmod_common *Common
)
{
    return cholmod_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) ;
}

template <> int
spqr_free_sparse <int32_t> (cholmod_sparse **A, cholmod_common *Common)
{
    return cholmod_free_sparse (A, Common) ;
}

template <> int
spqr_reallocate_sparse <int32_t>
(
    size_t nznew, cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_reallocate_sparse (nznew, A, Common) ;
}

template <> cholmod_dense *
spqr_allocate_dense <int32_t>
(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *Common
)
{
    return cholmod_allocate_dense (nrow, ncol, d, xtype, Common) ;
}

template <> int
spqr_free_dense <int32_t>
(
    cholmod_dense **X, cholmod_common *Common
)
{
    return cholmod_free_dense (X, Common) ;
}

template <> cholmod_factor *
spqr_allocate_factor <int32_t> (size_t n, cholmod_common *Common)
{
    return cholmod_allocate_factor (n, Common) ;
}

template <> int
spqr_free_factor <int32_t> (cholmod_factor **L, cholmod_common *Common)
{
    return cholmod_free_factor (L, Common) ;
}

template <> int
spqr_allocate_work <int32_t>
(
    size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common
)
{
    return cholmod_allocate_work (nrow, iworksize, xworksize, Common) ;
}

template <> int
spqr_amd <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Perm,
    cholmod_common *Common
)
{
    return cholmod_amd (A, fset, fsize, Perm, Common) ;
}

template <> int
spqr_metis <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder,
    int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_metis (A, fset, fsize, postorder, Perm, Common) ;
}

template <> cholmod_sparse *
spqr_transpose <int32_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return cholmod_transpose (A, values, Common) ;
}

template <> cholmod_factor *
spqr_analyze_p2 <int32_t>
(
    int for_whom, cholmod_sparse *A, int32_t *UserPerm,
    int32_t *fset, size_t fsize, cholmod_common *Common
)
{
    return cholmod_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common) ;
}

template <> int
spqr_colamd <int32_t>
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder,
    int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_colamd (A, fset, fsize, postorder, Perm, Common) ;
}

template <> int32_t
spqr_postorder <int32_t>
(
    int32_t *Parent, size_t n, int32_t *Weight_p,
    int32_t *Post, cholmod_common *Common
)
{
    return cholmod_postorder (Parent, n, Weight_p, Post, Common) ;
}

template <> int64_t
spqr_nnz <int32_t> (cholmod_sparse *A, cholmod_common *Common)
{
    return cholmod_nnz(A, Common) ;
}

template <> cholmod_sparse *
spqr_dense_to_sparse <int32_t>
(
    cholmod_dense *X, int values, cholmod_common *Common
)
{
    return cholmod_dense_to_sparse(X, values, Common) ;
}

template <> cholmod_dense *
spqr_sparse_to_dense <int32_t> (cholmod_sparse *A,  cholmod_common *Common)
{
    return cholmod_sparse_to_dense (A, Common) ;
}

template <> cholmod_sparse *
spqr_speye <int32_t>
(
    size_t nrow, size_t ncol, int xtype, cholmod_common *Common
)
{
    return cholmod_speye(nrow, ncol, xtype, Common) ;
}

#else

// If SuiteSparse_long is the same type as int32_t (e.g., on 32-bit platforms),
// The templates cannot be instantiated for each type.  The correct function
// must be selected on runtime.

template <typename Int>
void *spqr_malloc (size_t n, size_t size, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_malloc (n, size, Common)
            : cholmod_malloc (n, size, Common)) ;
}

template <typename Int>
void *spqr_calloc (size_t n, size_t size, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_calloc (n, size, Common)
            : cholmod_calloc (n, size, Common)) ;
}

template <typename Int>
void *spqr_free (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_free (n, size, p, Common)
            : cholmod_free (n, size, p, Common)) ;
}

template <typename Int>
void *spqr_realloc
(
    size_t nnew, size_t size, void *p, size_t *n, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_realloc (nnew, size, p, n, Common)
            : cholmod_realloc (nnew, size, p, n, Common)) ;
}

template <typename Int>
cholmod_sparse *spqr_allocate_sparse
(
    size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed,
    int stype, int xtype, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common)
            : cholmod_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) );
}

template <typename Int>
int spqr_free_sparse (cholmod_sparse **A, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_free_sparse (A, Common)
            : cholmod_free_sparse (A, Common)) ;
}

template <typename Int>
int spqr_reallocate_sparse 
(
    size_t nznew, cholmod_sparse *A, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_reallocate_sparse (nznew, A, Common)
            : cholmod_reallocate_sparse (nznew, A, Common)) ;
}

template <typename Int>
cholmod_dense *spqr_allocate_dense
(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_allocate_dense (nrow, ncol, d, xtype, Common)
            : cholmod_allocate_dense (nrow, ncol, d, xtype, Common)) ;
}

template <typename Int>
int spqr_free_dense (cholmod_dense **X, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_free_dense (X, Common)
            : cholmod_free_dense (X, Common)) ;
}

template <typename Int>
cholmod_factor *spqr_allocate_factor (size_t n, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_allocate_factor (n, Common)
            : cholmod_allocate_factor (n, Common)) ;
}

template <typename Int>
int spqr_free_factor (cholmod_factor **L, cholmod_common *Common)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_free_factor (L, Common)
            : cholmod_free_factor (L, Common)) ;
}

template <typename Int>
int spqr_allocate_work
(
    size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_allocate_work (nrow, iworksize, xworksize, Common)
            : cholmod_allocate_work (nrow, iworksize, xworksize, Common)) ;
}

template <typename Int>
int spqr_amd
(
    cholmod_sparse *A, Int *fset, size_t fsize, Int *Perm,
    cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_amd (A, fset, fsize, Perm, Common)
            : cholmod_amd (A, fset, fsize, Perm, Common)) ;
}

template <typename Int>
int spqr_metis 
(
    cholmod_sparse *A, Int *fset, size_t fsize, int postorder, Int *Perm,
    cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_metis (A, fset, fsize, postorder, Perm, Common)
            : cholmod_metis (A, fset, fsize, postorder, Perm, Common)) ;
}

template <typename Int>
cholmod_sparse *spqr_transpose
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_transpose (A, values, Common)
            : cholmod_transpose (A, values, Common)) ;
}

template <typename Int>
cholmod_factor *spqr_analyze_p2
(
    int for_whom, cholmod_sparse *A, Int *UserPerm, Int *fset,  size_t fsize,
    cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common)
            : cholmod_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common)) ;
}

template <typename Int>
int spqr_colamd
(
    cholmod_sparse *A, Int *fset, size_t fsize, int postorder, Int *Perm,
    cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_colamd (A, fset, fsize, postorder, Perm, Common)
            : cholmod_colamd (A, fset, fsize, postorder, Perm, Common)) ;
}

template <typename Int>
Int spqr_postorder
(
    Int *Parent, size_t n, Int *Weight_p, Int *Post, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_postorder (Parent, n, Weight_p, Post, Common)
            : cholmod_postorder (Parent, n, Weight_p, Post, Common)) ;
}

template <typename Int>
int64_t spqr_nnz
(
    cholmod_sparse *A,
    cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_nnz(A, Common)
            : cholmod_nnz(A, Common)) ;
}

template <typename Int>
cholmod_sparse *spqr_dense_to_sparse
(
    cholmod_dense *X, int values, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_dense_to_sparse(X, values, Common)
            : cholmod_dense_to_sparse(X, values, Common)) ;
}

template <typename Int>
cholmod_dense *spqr_sparse_to_dense
(
    cholmod_sparse *A, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_sparse_to_dense (A, Common)
            : cholmod_sparse_to_dense (A, Common)) ;
}

template <typename Int>
cholmod_sparse *spqr_speye
(
    size_t nrow, size_t ncol, int xtype, cholmod_common *Common
)
{
    return (Common->itype == CHOLMOD_LONG
            ? cholmod_l_speye(nrow, ncol, xtype, Common)
            : cholmod_speye(nrow, ncol, xtype, Common)) ;
}

// Explicit instantiations for one of the integer types are enough.

template void *
spqr_malloc <int32_t> (size_t n, size_t size, cholmod_common *Common);

template void *
spqr_calloc <int32_t> (size_t n, size_t size, cholmod_common *Common);

template void *
spqr_free <int32_t> (size_t n, size_t size, void *p, cholmod_common *Common);

template void *
spqr_realloc <int32_t>
(
    size_t nnew, size_t size, void *p, size_t *n, cholmod_common *Common
);

template cholmod_sparse *
spqr_allocate_sparse <int32_t>
(
    size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed,
    int stype, int xtype, cholmod_common *Common
);

template int
spqr_free_sparse <int32_t> (cholmod_sparse **A, cholmod_common *Common);

template int spqr_reallocate_sparse <int32_t>
(
    size_t nznew, cholmod_sparse *A, cholmod_common *Common
);

template cholmod_dense *spqr_allocate_dense <int32_t>
(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *Common
);

template int spqr_free_dense <int32_t>
(
    cholmod_dense **X, cholmod_common *Common
);

template cholmod_factor *spqr_allocate_factor <int32_t>
(
    size_t n, cholmod_common *Common
);

template int spqr_free_factor <int32_t>
(
    cholmod_factor **L, cholmod_common *Common
);

template int spqr_allocate_work <int32_t>
(
    size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common
);

template int spqr_amd <int32_t>
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Perm,
    cholmod_common *Common
);

template int spqr_metis <int32_t>
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder,
    int32_t *Perm, cholmod_common *Common
);

template cholmod_sparse *spqr_transpose <int32_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
);

template cholmod_factor *spqr_analyze_p2 <int32_t>
(
    int for_whom, cholmod_sparse *A, int32_t *UserPerm,
    int32_t *fset, size_t fsize, cholmod_common *Common
);

template int spqr_colamd <int32_t>
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder,
    int32_t *Perm, cholmod_common *Common
);

template int32_t spqr_postorder <int32_t>
(
    int32_t *Parent, size_t n, int32_t *Weight_p, int32_t *Post,
    cholmod_common *Common
);

template int64_t
spqr_nnz <int32_t> (cholmod_sparse *A, cholmod_common *Common);

template cholmod_sparse *
spqr_dense_to_sparse <int32_t>
(
    cholmod_dense *X, int values, cholmod_common *Common
);

template cholmod_dense *
spqr_sparse_to_dense <int32_t> (cholmod_sparse *A,  cholmod_common *Common);

template cholmod_sparse *
spqr_speye <int32_t>
(
    size_t nrow, size_t ncol, int xtype, cholmod_common *Common
);

#endif
