function test1
%TEST1 test cs_transpose, cs_gaxpy, cs_sparse, cs_sparse2
%
% Example:
%   test1
% See also: testall

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse


index = UFget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:100) ;

for ii = f

    Prob = UFget (ii) ;
    disp (Prob) ;
    for cmplex = 0:double(~ispc)

	A = Prob.A ;
	if (cmplex)
	    A = A + 1i*sprand(A) ;
	end

	B = A' ;
	C = cs_transpose (A) ;
	if (nnz (B-C) ~= 0)
	    error ('!')
	end

	C = cs_transpose (A,0) ;
	if (nnz (A.'-C) ~= 0)
	    error ('!')
	end

	C = cs_transpose (A,1) ;
	if (nnz (A'-C) ~= 0)
	    error ('!')
	end

	[m n] = size (A) ;
	% if (m == n)
	    x = rand (n,1) ;
	    y = rand (m,1) ;
	    z = y+A*x ;
	    q = cs_gaxpy (A,x,y) ;
	    err = norm (z-q,1) / norm (z,1) ;
	    disp (err) ;
	    if (err > 1e-14)
		error ('!')
	    end
	% end

        if (~ispc)
            x = x + 1i*rand (n,1) ;
            y = y + 1i*rand (m,1) ;
            z = y+A*x ;
            q = cs_gaxpy (A,x,y) ;
            err = norm (z-q,1) / norm (z,1) ;
            disp (err) ;
            if (err > 1e-14)
                error ('!')
            end
        end

	[i j x] = find (A) ;
	p = randperm (length (i)) ;
	i = i (p) ;
	j = j (p) ;
	x = x (p) ;
	D = sparse (i,j,x) ;
	E = cs_sparse (i,j,x) ;
	% [i j x]
	F = cs_sparse2 (i,j,x) ;
	if (nnz (D-E) ~= 0)
	    error ('!')
	end
	if (nnz (F-E) ~= 0)
	    error ('!')
	end

	clear A B C D E F
	% pause

    end
end
