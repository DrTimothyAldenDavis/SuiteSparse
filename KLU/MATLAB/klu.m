function [L,U,p,q,R,F,r,info] = klu (A,opts)
%KLU sparse left-looking LU factorization, using a block triangular form.
%
%   [L,U,p,q,R,F,r,info] = klu (A,opts) factorizes a square sparse matrix,
%   L*U+F = R\A(p,q), where L and U are the factors of the diagonal blocks of
%   the block, F are the entries above the diagonal blocks.  r corresponds to
%   the 3rd output of dmperm; it specifies where the block boundaries are.  The
%   kth block consists of rows/columns r(k) to r(k+1)-1 of A(p,q).
%
%   opts is an optional struct, with one or more of the following entries.
%   Entries not present are set to their defaults:
%
%	opts.tol	0.001	partial pivoting tolerance
%	opts.growth	1.2	realloc growth size, when L and U need to grow
%	opts.imemamd	1.2	initial size of L and U with AMD is 1.2*nnz(L)+n
%	opts.imem	10	initial size of L and U is 10*nnz(A)+n otherwise
%	opts.btf	1	use BTF if nonzero
%	opts.ordering	0	0: AMD, 1: COLAMD, 2: natural, 3:CHOLMOD(A+A'),
%				4: CHOLMOD (A'*A)
%	opts.scale	-1	0,-1:none (R=I), 1:R=sum(A'), 2:R=max(A')
%
%   info is an output struct with the following entries:
%
%	info.noffdiag	number of off-diagonal pivots chosen (after preordering)
%	info.nrealloc	number of memory reallocations of L and U
%	info.condest	condition number estimate (a lower bound)
%
%   With no outputs, statistics are printed.
%
%   See also LU, DMPERM, CONDEST.

error ('klu mexFunction not found') ;
