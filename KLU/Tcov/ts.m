
clear

warning off MATLAB:nearlySingularMatrix

index = UFget ;

[i mat] = sort (index.nnz) ;

sym = [1e-6 1] ;	% symmetric mode (AMD, tolerance of 1e-6)

hb = [1:4] ;
att = [283 284 286] ;
bomhof = [370:373] ;
grund = [465 466] ;
hamm = [539:544] ;
circuits = [hb bomhof grund hamm att ] ;

% for j = mat
% for j = circuits
for j = 286	% twotone

    Problem = UFget (j) ;
    if (index.isBinary (j))
	% fprintf ('binary (skip)\n') ;
	continue ;
    end
    if (strcmp (index.Group {j}, 'Gset'))
	f% printf ('Gset (skip)\n') ;
	continue ;
    end
    A = Problem.A ;
    [m n] = size (A) ;
    if (m ~= n)
	% fprintf ('rectangular (skip)\n') ;
	continue ;
    end
    if (~isreal (A))
	% fprintf ('complex (skip)\n') ;
	continue ;
    end
    if (sprank (A) < n)
	% fprintf ('structurally singular (skip)\n') ;
	continue
    end

    fprintf ('\n\n============== Matrix: %3d %s ===========================\n', j, Problem.name) ;

%   for transpose = 0:1
    for transpose = 0

	if (transpose)
	    fprintf ('\n------------------- transposed:\n') ;
	    A = A' ;
	end

	for do_btf = 0:1

	    if (do_btf == 1)
		[p,q,r] = dmperm (A) ;
		nblocks = length (r) - 1 ;
		if (nblocks == 1)
		    continue
		end
		fprintf ('-------------btf:\n') ;
	    end

if (0)
	    % UMFPACK v4.2+, defaults
	    solver (@lu_umfpack, do_btf, A) ;

	    % MATLAB's LU, defaults
	    solver (@lu_matlab, do_btf, A) ;

	    % LU2, defaults (no pruning)
	    solver (@lu_2, do_btf, A) ;

	    % LUprune, defaults (with pruning)
	    solver (@lu_prune, do_btf, A) ;

	    fprintf ('-----------------\n') ;

	    % AMD, pure symmetric mode
	    solver (@lu_amd, do_btf, A, [0 1]) ;

	    % MATLAB's LU, symmetric mode
	    solver (@lu_matlab, do_btf, A, sym) ;
end

	    % LU2, symmetric mode (no pruning)
	    solver (@lu_2, do_btf, A, sym) ;

	    % LUprune, symmetric mode (with pruning)
	    solver (@lu_prune, do_btf, A, sym) ;

	end
    end
end

