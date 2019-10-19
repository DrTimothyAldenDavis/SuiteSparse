#include "btf.h"
#include "btf_internal.h"

int btf_order	    /* returns number of blocks found */
(
    /* input, not modified: */
    int n,	    /* A is n-by-n in compressed column form */
    int Ap [ ],	    /* size n+1 */
    int Ai [ ],	    /* size nz = Ap [n] */

    /* output, not defined on input */
    int P [ ],	    /* size n, row permutation */
    int Q [ ],	    /* size n, column permutation */
    int R [ ],	    /* size n+1.  block b is in rows/cols R[b] ... R[b+1]-1 */
    int *nfound,    /* # nonzeros on diagonal of P*A*Q */

    /* workspace, not defined on input or output */
    int Work [ ]    /* size 5n */
)
{
    int nblocks ;
    *nfound = maxtrans   (n, n, Ap, Ai, Q,    Work) ;
    nblocks = strongcomp (n, Ap, Ai, Q, P, R, Work) ;
    return (nblocks) ;
}
