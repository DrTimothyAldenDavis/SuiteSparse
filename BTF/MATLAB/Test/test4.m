%TEST4 test script for BTF
% Requires UFget
% Example:
%   test4
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, University of Florida

clear ;
index = UFget ;
f = find (index.nrows == index.ncols) ;
[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;
nmat = length (f) ;

for k = 1:nmat

    Prob = UFget (f (k), index) ;
    A = Prob.A ;

    for tr = [1 -1]

	if (tr == -1)
	    AT = A' ;
	    [m n] = size (A) ;
	    if (m == n)
		if (nnz (spones (AT) - spones (A)) == 0)
		    fprintf ('skip transpose\n') ;
		    continue ;
		end
	    end
	    A = AT ;
	end

	tic
	[p1,q1,r1,work1] = btf (A) ;
	t1 = toc ;
	n1 = length (r1) - 1 ;

	tic
	[p2,q2,r2,work2] = btf (A, 10) ;
	t2 = toc ;
	n2 = length (r2) - 1 ;

	fprintf (...
	'%4d %4d : %10.4f %8d  %8g : %10.4f %8d  %8g : rel %8.4f %8.4f\n', ...
	k, f(k), t1, n1, work1, t2, n2, work2, t1 ./ t2, n2 ./ (max (1, n1))) ;

	if (n1 ~= n2 | work1 ~= work2)					    %#ok
	    disp (Prob) ;
	    fprintf ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n') ;
	end

    end
end

