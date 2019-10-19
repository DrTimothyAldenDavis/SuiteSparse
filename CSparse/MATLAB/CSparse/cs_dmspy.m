function [p,q,r,s,cc,rr] = cs_dmspy (A,res,seed)
%CS_DMSPY plot the Dulmage-Mendelsohn decomposition of a matrix.
%   [p,q,r,s,cc,rr] = cs_dmspy(A) computes [p,q,r,s,cc,rr] = cs_dmperm(A),
%   does spy(A(p,q)), and then draws boxes around the coarse and fine
%   decompositions.  A 2nd input argument (cs_dmspy(A,res)) changes the
%   resolution of the image to res-by-res (default resolution is 256).
%   If res is zero, spy is used instead of cspy.  If the resolution is low, the
%   picture of the blocks in the figure can overlap.  They do not actually
%   overlap in the matrix.  With 3 arguments, cs_dmspy(A,res,seed),
%   cs_dmperm(A,seed) is used, where seed controls the randomized algorithm
%   used by cs_dmperm.
%
%   Example:
%       Prob = UFget ('HB/arc130') ; cs_dmspy (Prob.A) ;
%
%   See also CS_DMPERM, CS_DMSOL, DMPERM, SPRANK, SPY.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

if (~issparse (A))
    A = sparse (A) ;
end
if (nargin < 2)
    res = 256 ;
end
if (nargin < 3)
    seed = 0 ;
end

% Dulmage-Mendelsohn permutation
[p1,q,r,s,cc,rr] = cs_dmperm (A,seed) ;
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

title (sprintf ( ...
    '%d-by-%d, sprank: %d, fine blocks: %d, coarse blocks: %d-by-%d\n', ...
    size (A), rr(4)-1, nb, length (find (diff (rr))), ...
    length (find (diff (cc))))) ;

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
	      [r1(k) r1(k) r2(k) r2(k) r1(k)], 'k', 'LineWidth', 1) ;
    end
end

[m n] = size (A) ;
drawbox (1,m+1,1,n+1,'k',1,e) ;

drawbox (rr(1), rr(2), cc(1), cc (2), 'r', 2, e) ;
drawbox (rr(1), rr(2), cc(2), cc (3), 'r', 2, e) ;
drawbox (rr(2), rr(3), cc(3), cc (4), 'k', 2, e) ;
drawbox (rr(3), rr(4), cc(4), cc (5), 'r', 2, e) ;
drawbox (rr(4), rr(5), cc(4), cc (5), 'r', 2, e) ;

hold off

