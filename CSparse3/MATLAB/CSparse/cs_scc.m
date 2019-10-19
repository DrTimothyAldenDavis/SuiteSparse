function [p,r] = cs_scc (A)                                                 %#ok
%CS_SCC strongly-connected components of a square sparse matrix.
%   [p,r] = cs_scc(A) finds a permutation p so that A(p,p) is permuted into
%   block upper triangular form.  The diagonal of A is ignored.  The kth block
%   is given by A (s,s) where s = r(k):r(k+1)-1.  A must be square.
%   For bipartite or rectangular graphs, use cs_scc2.
%
%   Example:
%       Prob = UFget ('HB/arc130') ; A = Prob.A ; [p r] = cs_scc (A) ;
%       cspy (A (p,p)) ;
%
%   See also CS_DMPERM, DMPERM, CS_SCC2.

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_scc mexFunction not found') ;
