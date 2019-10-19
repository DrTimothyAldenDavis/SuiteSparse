function L = cs_updown (L, c, parent, sigma)
%CS_UPDOWN rank-1 update/downdate of a sparse Cholesky factorization.
%   L = cs_updown(L,c,parent) computes the rank-1 update L = chol(L*L'+c*c')',
%   where parent is the elimination tree of L.  c must be a sparse column
%   vector, and find(c) must be a subset of find(L(:,k)) where k = min(find(c)).
%   L = cs_updown(L,c,parent,'-') is the downdate L = chol(L*L'-c*c').
%   L = cs_updown(L,c,parent,'+') is the update L = chol(L*L'+c*c').
%   Updating/downdating is much faster than refactorizing the matrix with
%   cs_chol or chol.  L must not have an entries dropped due to numerical
%   cancellation (use cs_chol(A,0)).
%
%   See also CS_ETREE, CS_CHOL, ETREE, CHOLUPDATE, CHOL.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_updown mexFunction not found') ;
