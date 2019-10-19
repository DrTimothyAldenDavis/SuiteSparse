function [p, q, r, s] = ccspy (A, bipartite, res)
%CCSPY plot the strongly connected components of a graph.
%   If A is square, [p,q,r,s] = ccspy(A) finds a permutation p so that A(p,q)
%   is permuted into block upper triangular form.  In this case, r=s, p=q and
%   the kth diagonal block is given by A (t,t) where t = r(k):r(k)+1. 
%   The diagonal of A is ignored.
%
%   If A is not square (or for [p,q,r,s] = ccspy(A,1)), then the connected
%   components of the bipartite graph of A are found.  A(p,q) is permuted into
%   block diagonal form, where the diagonal blocks are rectangular.  The kth
%   block is given by A(r(k):r(k)+1,s(k):s(k)+1).  A can be rectangular.
%
%   It then plots the result via cspy, drawing a greenbox around each component.
%   A 3rd input argument (res) controls the resolution (see cspy for a
%   description of the res parameter).
%
%   See also CSPY, CS_DMPERM, DMPERM, CS_SCC, CS_SCC2, CS_DMSPY.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

if (~issparse (A))
    A = sparse (A) ;
end
[m n] = size (A) ;
if (nargin < 3)
    res = 256 ;
end
if (nargin < 2 | isempty (bipartite))
    bipartite = (m ~= n) ;
end

% find the strongly connected components
[p1 q r s] = cs_scc2 (A, bipartite) ;
if (nargout > 0)
    p = p1 ;
end

nb = length (r)-1 ;

% plot the result
S = A (p1,q) ;
if (res == 0)
    spy (S) ;
    e = 1 ;
else
    e = cspy (S,res) ;
end
hold on

title (sprintf ('%d-by-%d, strongly connected commponents: %d\n', m, n, nb)) ;

if (~bipartite)
    plot ([.5 .5 n+.5 n+.5], [.5 .5 n+.5 n+.5], 'r') ;
end

% for k = 1:nb
%     drawbox (r(k), r(k+1), s(k), s(k+1),'g',1,e) ;
% end

if (nb > 1)
    if (e == 1)
	r1 = r (1:nb) - .5 ;
	r2 = r (2:nb+1) - .5 ;
	c1 = s (1:nb) - .5 ;
	c2 = s (2:nb+1) - .5 ;
    else
	r1 = ceil (r (1:nb) / e) - .5 ;
	r2 = ceil ((r (2:nb+1) - 1) / e) + .5 ;
	c1 = ceil (s (1:nb) / e) - .5 ;
	c2 = ceil ((s (2:nb+1) - 1) / e) + .5 ;
    end
    kk = find (diff (c1) > 0 | diff (c2) > 0 | diff (r1) > 0 | diff (r2) > 0) ;
    kk = [1 kk+1] ;
    for k = kk
	plot ([c1(k) c2(k) c2(k) c1(k) c1(k)], ...
	      [r1(k) r1(k) r2(k) r2(k) r1(k)], 'r', 'LineWidth', 1) ;
    end
end

drawbox (1,m+1,1,n+1,'k',1,e) ;
hold off
