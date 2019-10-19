
clear
rand ('state', 0) ;

files = { 'S_d2q06c', 'adlittle', 'arc130', 'cage3', 'd_dyn', 'galenet', 'matrix1', 'matrix10', 'matrix11', 'matrix12', 'matrix13', 'matrix14', 'matrix15', 'matrix16', 'matrix17', 'matrix18', 'matrix19', 'matrix2', 'matrix20', 'matrix21', 'matrix22', 'matrix23', 'matrix24', 'matrix25', 'matrix26', 'matrix27', 'matrix3', 'matrix4', 'matrix5', 'matrix6', 'matrix7', 'matrix8', 'nug07', 'shl0' }

nmat = length(files) ;

for ii = 1:(nmat + 1)

    if (ii == nmat+1)
	name = 'matrix28' ;
	A = sparse (rand (4) + 1i*rand(4)) ;
	A (1,1) = 0 ;
	A (3,3) = 0 ;
	[nrows ncols] = size (A) ;
	nz = nnz (A) ;
	is_real = 0 ;
    else
	name = files {ii} ;
	f = fopen (name) ;

	s = fscanf (f, '%d %d %d %d\n', 4) ;
	nrows  = s (1) ;
	ncols  = s (2) ;
	nz     = s (3) ;
	is_real = s (4) ;

	if (is_real)
	    ijx = fscanf (f, '%d %d %g\n', 3*nz) ;
	    ijx = reshape (ijx, 3, nz)' ;
	else
	    ijx = fscanf (f, '%d %d %g %g\n', 4*nz) ;
	    ijx = reshape (ijx, 4, nz)' ;
	end

	q = fscanf (f, '%d\n', ncols) ;

	i = ijx (:,1) ;
	j = ijx (:,2) ;
	x = ijx (:,3) ;
	if (~is_real)
	    x = x + 1i * ijx (:,4) ;
	end

	A = sparse (i, j, x, nrows, ncols) ;

	fclose (f) ;
    end

    if (ii == 2)
	Problem = UFget ('LPnetlib/lp_adlittle') ;
    elseif (ii == 3)
	Problem = UFget ('HB/arc130') ;
    elseif (ii == 4)
	Problem = UFget ('vanHeukelum/cage3') ;
    elseif (ii == 5)
	Problem = UFget ('Grund/d_dyn') ;
    elseif (ii == 6)
	Problem = UFget ('LPnetlib/lpi_galenet') ;
    end

    if (ii >= 2 & ii <= 6)
	B = Problem.A ;
	check = norm (A-B,1) ;
	fprintf ('check: %g\n', check) ;
	if (check ~= 0)
	    error ('bad matrix') ;
	end
    end

    if (nrows == ncols)
	d = det (A)
	d2 = det (real (A))
    else
	d = 0 ;
	d2 = 0 ;
    end
    fprintf ('%s d: %g + (%g)i, real: %g\n', name, real(d), imag(d), d2) ;

    % create a new matrix with the determinant at the end of the file

    f2 = fopen (sprintf ('New/%s', name), 'w') ;
    if (ii == nmat+1)
	fprintf (f2, '%d %d %d %d\n', nrows, ncols, nz, is_real) ;
	[i j x] = find (A) ;
	fprintf (f2, '%d %d %30.20e %30.20e\n', [i j real(x) imag(x)]') ;
	fprintf (f2, '1\n2\n3\n4\n') ;
    else
	f = fopen (name) ;
	while (1)
	    s = fgetl (f) ;
	    if (s == -1)
		break
	    end
	    fprintf (f2, '%s\n', s) ;
	end
    end

    if (isreal (A))
	fprintf (f2, '%30.20e\n', d) ;
    else
	fprintf (f2, '%30.20e %30.20e\n', real(d), imag(d)) ;
    end
    fprintf (f2, '%30.20e\n', d2) ;
    fclose (f2) ;
end

