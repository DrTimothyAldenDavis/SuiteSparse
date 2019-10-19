function do_att
% DO_ATT: test KLU on AT&T matrices
% Example:
%   do_att

warning off MATLAB:nearlySingularMatrix
warning off MATLAB:divideByZero

index = UFget ;
[i mat] = sort (index.nnz) ;

sym = [1e-3 1] ;	% symmetric mode (AMD, tolerance of 1e-3)
unsym = [1 0] ;		% unsymmetric mode (COLAMD, tolerance of 1)
unsym_kundert = [0.1 0] ;

hb = 1:4 ;
att = [283 284 286] ;
bomhof = 370:373 ;
grund = [465 466] ;
hamm = 539:544 ;
circuits = [hb bomhof grund hamm att ] ;

what = 0 ;
circuits = att ;

if (what == 0)
    % UF circuits
    ufcollection = 1 ;
    ufcircuits = 1 ;
elseif (what == 1)
    % David Day's matrices (sandia list)
    ufcollection = 0 ;
    ufcircuits = 0 ;
elseif (what == 2)
    % all matrices in UF collection
    ufcollection = 1 ;
    ufcircuits = 0 ;
end

sandia = get_sandia_matrix_list ;

if (ufcollection)
    if (ufcircuits)
	matrix_list = circuits ;
    else
	matrix_list = mat ;
    end
else
    matrix_list = 1 : length (sandia) ; 
end

% for j = mat
% for j = circuits
% for j = 286	% twotone

for j = matrix_list

  do_sym_test = 1 ;

  if (ufcollection) 
    Problem = UFget (j) ;
    if (index.isBinary (j))
	% fprintf ('binary (skip)\n') ;
	continue ;
    end
    if (strcmp (index.Group {j}, 'Gset'))
	% fprintf ('Gset (skip)\n') ;
	continue ;
    end
    if (strcmp (index.Group {j}, 'ATandT'))
	do_sym_test = 0 ;
    end
    A = Problem.A ;
    name = Problem.name ;
    clear Problem ;
  else
    name = sandia {j} ;
    Problem = UFget (name) ;
    A = Problem.A ;
    clear Problem ;
  end

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

    fprintf ('\n\n=========================== Matrix: %3d %s \n', j, name) ;
    fprintf ('n: %d nnz(A): %d\n', n, nnz (A)) ;

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
		bsize = r (2:nblocks+1) - r (1:nblocks) ;
		fprintf ('\nBTF: ====================== largest block %d\n', max (bsize)) ;
	    end

	    % UMFPACK v4.2+, defaults
	    solver (@lu_umfpack, do_btf, A) ;

	    % ordering only (COLAMD)
	    % solver (@lu_order, do_btf, A) ;

	    % MATLAB's LU, defaults
	    % solver (@lu_matlab, do_btf, A) ;

	    % LU2, defaults (no pruning)
	    % solver (@lu_2, do_btf, A) ;

	    % KLU, unsymmetric mode with no pruning
	    solver (@lu_klu, do_btf, A, [unsym 0]) ;

	    % LUprune, defaults (with pruning)
	    % solver (@lu_prune, do_btf, A) ;

	    % KLU defaults: unsymmetric mode with pruning
	    solver (@lu_klu, do_btf, A, [unsym 1]) ;

	    % LUfact, unsymmetric mode
	    solver (@lu_fact, do_btf, A) ;

	    % Kundert's Sparse1.3, unsymmetric mode
	    solver (@lu_kundert, do_btf, A, unsym_kundert) ;

	    fprintf ('-----------------\n') ;

	    if (do_sym_test)

	    % ordering only (AMD pure symmetric mode)
	    % solver (@lu_order, do_btf, A, sym) ;

	    % MATLAB's LU, symmetric mode
	    % solver (@lu_matlab, do_btf, A, sym) ;

	    % LU2, symmetric mode (no pruning)
	    % solver (@lu_2, do_btf, A, sym) ;

	    % KLU, symmetric mode with no pruning
	    solver (@lu_klu, do_btf, A, [sym 0]) ;

	    % LUprune, symmetric mode (with pruning)
	    % solver (@lu_prune, do_btf, A, sym) ;

	    % KLU, symmetric mode with pruning
	    solver (@lu_klu, do_btf, A, [sym 1]) ;

	    % LUfact, symmetric mode
	    solver (@lu_fact, do_btf, A, sym) ;

	    % Kundert's Sparse1.3, symmetric mode
	    solver (@lu_kundert, do_btf, A, sym) ;

	    end

	end
    end

    % pause
end

