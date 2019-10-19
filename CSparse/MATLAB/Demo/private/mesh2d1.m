function A = mesh2d1 (n)
% create an n-by-n 2D mesh for the 2nd difference operator
ii = zeros (5*n^2, 1) ;	    % preallocate ii, jj, and xx
jj = zeros (5*n^2, 1) ;
xx = zeros (5*n^2, 1) ;
k = 1 ;
for j = 0:n-1
    for i = 0:n-1
	s = j*n+i + 1 ;
	ii (k:k+4) = [(j-1)*n+i j*n+(i-1) j*n+i j*n+(i+1) (j+1)*n+i ] + 1 ;
	jj (k:k+4) = [s s s s s] ;
	xx (k:k+4) = [-1 -1 4 -1 -1] ;
	k = k + 5 ;
    end
end
% remove entries beyond the boundary
keep = find (ii >= 1 & ii <= n^2 & jj >= 1 & jj <= n^2) ;
A = sparse (ii (keep), jj (keep), xx (keep)) ;
