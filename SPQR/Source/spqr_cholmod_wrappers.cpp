#include "spqr.hpp"

// template specializations for int32_t and int64_t

template <> void *spqr_malloc <int32_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_malloc (n, size, Common) ;
}
template <> void *spqr_malloc <int64_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_l_malloc (n, size, Common) ;
}

template <> void *spqr_calloc <int32_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_calloc (n, size, Common) ;
}
template <> void *spqr_calloc <int64_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_l_calloc (n, size, Common) ;
}

template <> void *spqr_free <int32_t> (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return cholmod_free (n, size, p, Common) ;
}
template <> void *spqr_free <int64_t> (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return cholmod_l_free (n, size, p, Common) ;
}

template <> void *spqr_realloc <int32_t>
(
    size_t nnew, size_t size, void *p, size_t *n, cholmod_common *Common
)
{
    return cholmod_realloc (nnew, size, p, n, Common) ;
}
template <> void *spqr_realloc <int64_t>
(
    size_t nnew, size_t size, void *p, size_t *n, cholmod_common *Common
)
{
    return cholmod_l_realloc (nnew, size, p, n, Common) ;
}

template <> cholmod_sparse *spqr_allocate_sparse <int32_t>
(
    size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed,
    int stype, int xtype, cholmod_common *Common
)
{
    return cholmod_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) ;
}
template <> cholmod_sparse *spqr_allocate_sparse <int64_t>
(
    size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed,
    int stype, int xtype, cholmod_common *Common
)
{
    return cholmod_l_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) ;
}

template <> int spqr_free_sparse <int32_t>
(
    cholmod_sparse **A, cholmod_common *Common
)
{
    return cholmod_free_sparse (A, Common) ;
}
template <> int spqr_free_sparse <int64_t>
(
    cholmod_sparse **A, cholmod_common *Common
)
{
    return cholmod_l_free_sparse (A, Common) ;
}

template <> int spqr_reallocate_sparse <int32_t>
(
    size_t nznew, cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_reallocate_sparse (nznew, A, Common) ;
}
template <> int spqr_reallocate_sparse <int64_t>
(
    size_t nznew, cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_l_reallocate_sparse (nznew, A, Common) ;
}

template <> cholmod_dense *spqr_allocate_dense <int32_t>
(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *Common
)
{
    return cholmod_allocate_dense (nrow, ncol, d, xtype, Common) ;
}

template <> cholmod_dense *spqr_allocate_dense <int64_t>
(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *Common
)
{
    return cholmod_l_allocate_dense (nrow, ncol, d, xtype, Common) ;
}

template <> int spqr_free_dense <int32_t>
(
    cholmod_dense **X, cholmod_common *Common
)
{
    return cholmod_free_dense (X, Common) ;
}
template <> int spqr_free_dense <int64_t>
(
    cholmod_dense **X, cholmod_common *Common
)
{
    return cholmod_l_free_dense (X, Common) ;
}

template <> cholmod_factor *spqr_allocate_factor <int32_t>
(
    size_t n, cholmod_common *Common
)
{
    return cholmod_allocate_factor (n, Common) ;
}
template <> cholmod_factor *spqr_allocate_factor <int64_t>
(
    size_t n, cholmod_common *Common
)
{
    return cholmod_l_allocate_factor (n, Common) ;
}

template <> int spqr_free_factor <int32_t>
(
    cholmod_factor **L, cholmod_common *Common
)
{
    return cholmod_free_factor (L, Common) ;
}
template <> int spqr_free_factor <int64_t>
(
    cholmod_factor **L, cholmod_common *Common
)
{
    return cholmod_l_free_factor (L, Common) ;
}

template <> int spqr_allocate_work <int32_t>
(
    size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common
)
{
    return cholmod_allocate_work (nrow, iworksize, xworksize, Common) ;
}
template <> int spqr_allocate_work <int64_t>
(
    size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common
)
{
    return cholmod_l_allocate_work (nrow, iworksize, xworksize, Common) ;
}

template <> int spqr_amd <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_amd (A, fset, fsize, Perm, Common) ;
}
template <> int spqr_amd <int64_t> 
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int64_t *Perm, cholmod_common *Common
)
{
    return cholmod_l_amd (A, fset, fsize, Perm, Common) ;
}

template <> int spqr_metis <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder, int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_metis (A, fset, fsize, postorder, Perm, Common) ;
}
template <> int spqr_metis <int64_t> 
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int postorder, int64_t *Perm, cholmod_common *Common
)
{
    return cholmod_l_metis (A, fset, fsize, postorder, Perm, Common) ;
}

template <> cholmod_sparse *spqr_transpose <int32_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return cholmod_transpose (A, values, Common) ;
}
template <> cholmod_sparse *spqr_transpose <int64_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return cholmod_l_transpose (A, values, Common) ;
}

template <>
cholmod_factor *spqr_analyze_p2 <int32_t>
(
    int for_whom, cholmod_sparse *A, int32_t *UserPerm,
    int32_t *fset, size_t fsize, cholmod_common *Common
)
{
    return cholmod_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common) ;
}
template <>
cholmod_factor *spqr_analyze_p2 <int64_t>
(
    int for_whom, cholmod_sparse *A, int64_t *UserPerm,
    int64_t *fset, size_t fsize, cholmod_common *Common
)
{
    return cholmod_l_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common) ;
}

template <> int spqr_colamd <int32_t>
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder,
    int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_colamd (A, fset, fsize, postorder, Perm, Common) ;
}
template <> int spqr_colamd <int64_t>
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int postorder,
    int64_t *Perm, cholmod_common *Common
)
{
    return cholmod_l_colamd (A, fset, fsize, postorder, Perm, Common) ;
}

template <> int32_t spqr_postorder <int32_t>
(
    int32_t *Parent, size_t n, int32_t *Weight_p, int32_t *Post,
    cholmod_common *Common
)
{
    return cholmod_postorder (Parent, n, Weight_p, Post, Common) ;
}
template <> int64_t spqr_postorder <int64_t>
(
    int64_t *Parent, size_t n, int64_t *Weight_p, int64_t *Post,
    cholmod_common *Common
)
{
    return cholmod_l_postorder (Parent, n, Weight_p, Post, Common) ;
}

template <> int64_t spqr_nnz <int32_t>
(
    cholmod_sparse *A,
    cholmod_common *Common
)
{
    return cholmod_nnz(A, Common) ;
}
template <> int64_t spqr_nnz <int64_t>
(
    cholmod_sparse *A,
    cholmod_common *Common
)
{
    return cholmod_l_nnz(A, Common) ;
}

template <> cholmod_sparse *spqr_dense_to_sparse <int32_t>
(
    cholmod_dense *X, int values, cholmod_common *Common
)
{
    return cholmod_dense_to_sparse(X, values, Common) ;
}
template <> cholmod_sparse *spqr_dense_to_sparse <int64_t>
(
    cholmod_dense *X, int values, cholmod_common *Common
)
{
    return cholmod_l_dense_to_sparse(X, values, Common) ;
}

template <> cholmod_dense *spqr_sparse_to_dense <int32_t>
(
    cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_sparse_to_dense (A, Common) ;
}
template <> cholmod_dense *spqr_sparse_to_dense <int64_t>
(
    cholmod_sparse *A, cholmod_common *Common
)
{
    return cholmod_l_sparse_to_dense (A, Common) ;
}

template <> cholmod_sparse *spqr_speye <int32_t>
(
    size_t nrow, size_t ncol, int xtype, cholmod_common *Common
)
{
    return cholmod_speye(nrow, ncol, xtype, Common) ;
}
template <> cholmod_sparse *spqr_speye <int64_t>
(
    size_t nrow, size_t ncol, int xtype, cholmod_common *Common
)
{
    return cholmod_l_speye(nrow, ncol, xtype, Common) ;
}
