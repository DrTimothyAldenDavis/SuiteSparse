function [p,q,r,s,cc,rr] = cs_dmspy (A,res,seed)
%CS_DMSPY plot the Dulmage-Mendelsohn decomposition of a matrix.
%   [p,q,r,s,cc,rr] = cs_dmspy(A) computes [p,q,r,s,cc,rr] = cs_dmperm(A),
%   does spy(A(p,q)), and then draws boxes around the coarse and fine
%   decompositions.  A 2nd input argument (cs_dmspy(A,res)) changes the
%   resolution of the image to res-by-res (default resolution is 256).
%   If res is zero, spy is used instead of cspy.  If the resolution is low, the
%   picture of the blocks in the figure can overlap.  They do not actually
%   overlap in the matrix.  With 3 arguments, cs_dmspy(A,res,seed),
%   cs_dmperm(A,seed) is used, where seed controls the randomized algorithm.
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

for k = 1:nb
    drawbox (k,k,r,s,'k',1,e) ;
end

drawbox (1,1,rr,cc,'r',2,e) ;
drawbox (1,2,rr,cc,'r',2,e) ;
drawbox (2,3,rr,cc,'k',2,e) ;
drawbox (3,4,rr,cc,'r',2,e) ;
drawbox (4,4,rr,cc,'r',2,e) ;

hold off

%-------------------------------------------------------------------------------

function drawbox (i,j,r,s,color,w,e)
%DRAWBOX draw a box around a submatrix in the figure.

if (r (i) == r (i+1) || s (j) == s (j+1))
    return
end

if (e == 1)
    r1 = r(i)-.5 ;
    r2 = r(i+1)-.5 ;
    c1 = s(j)-.5 ;
    c2 = s(j+1)-.5 ;
else
    r1 = ceil (r(i) / e) - .5 ;
    r2 = ceil ((r(i+1) - 1) / e) + .5 ;
    c1 = ceil (s(j) / e) - .5 ;
    c2 = ceil ((s(j+1) - 1) / e) + .5 ;
end

if (c2 > c1 || r2 > r1)
    plot ([c1 c2 c2 c1 c1], [r1 r1 r2 r2 r1], color, 'LineWidth', w) ;
end
