function A = mesh3d2 (n)
% create an n-by-n-by-n 3D mesh for the 2nd difference operator
nn = 1:n^3 ;
ii = [nn-n^2 ; nn-n ; nn-1 ; nn ; nn+1 ; nn+n ; nn+n^2] ;
jj = repmat (nn, 7, 1) ;
xx = repmat ([-1 -1 -1 6 -1 -1 -1]', 1, n^3) ;
keep = find (ii >= 1 & ii <= n^3 & jj >= 1 & jj <= n^3) ;
A = sparse (ii (keep), jj (keep), xx (keep)) ;
