%TEST5 test script for BTF
% Requires UFget
% Example:
%   test5
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, University of Florida

clear ;
index = UFget ;

[ignore f] = sort (index.nnz) ;

nmat = length(f) ;

for k = 1:nmat

    i = f(k) ;
    Prob = UFget (i, index) ;
    A = Prob.A ;

    for tr = [1 -1]

	if (tr == -1)
	    AT = A' ;
	    [m n] = size (A) ;
	    if (m == n)
		if (nnz (spones (AT) - spones (A)) == 0)
		    fprintf ('skip test with transpose\n') ;
		    continue ;
		end
	    end
	    A = AT ;
	end

	tic
	q1 = maxtrans (A) ;
	t1 = toc ;
	r1 = sum (q1 > 0) ;

	tic
	q2 = maxtrans (A, 10) ;
	t2 = toc ;
	r2 = sum (q2 > 0) ;

	fprintf (...
	    '%4d %4d : %10.4f %8d  : %10.4f %8d  : rel %8.4f %8.4f\n', ...
	    k, f(k), t1, r1, t2, r2, t1 ./ t2, r2 ./ (max (1, r1))) ;

	if (r1 ~= r2)
	    disp (Prob) ;
	    fprintf ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n') ;
	end

    end
end
