% TT: test KLU
% Example:
%   tt

clear

index = UFget ;

[i mat] = sort (index.nnz) ;

if (0)
    jstart = 373 ;
    i = find (mat == jstart) ;
    mat = mat (i:end) ;
    mat = mat (1) ;
end

for j = mat

%  for transpose = 0:1
  for transpose = 0

    Problem = UFget (j) ;
    fprintf ('\n------------- Matrix: %s : transpose %d\n', Problem.name, transpose) ;
    A = Problem.A ;
    [m n] = size (A) ;
    if (m ~= n)
	fprintf ('rectangular (skip)\n') ;
	continue ;
    end
    if (~isreal (A))
	fprintf ('complex (skip)\n') ;
	continue ;
    end

    if (transpose)
	A = A' ;
    end

    q = colamd (A, [0.5 0.5 0]) ;
    A = A (:,q) ;

    [L1, U1, P1] = lu (A) ;
    u = full (min (abs (diag (U1)))) ;
    fprintf ('err: %g min u: %g\n', lu_normest (P1*A, L1, U1), u) ;

    lsize = 2 * nnz (A) ;
    usize = lsize ;
    Control = [lsize, usize] ;

    try
	[L2, U2, P2] = lu2 (A, Control) ;
	u = full (min (abs (diag (U1)))) ;
	fprintf ('err: %g min u: %g\n', lu_normest (P2*A, L2, U2), u) ;
    catch
	fprintf ('singular, or too big\n') ;
    end
    %% pause

    if (j == 44)
	return
    end

  end
end

