function stats = UFstats (A, kind, nometis, skip_chol, skip_dmperm, Z)
%UFSTATS compute matrix statistics for the UF Sparse Matrix Collection
% Example:
%   stats = UFstats (A, kind, nometis, skip_chol, skip_dmperm, Z)
%
% A: a sparse matrix
% kind: a string with the Problem.kind
% nometis: if nonzero then metis(A,'col') is not used, nor is metis used in the
%       dmperm+ ordering.
% Z: empty, or a sparse matrix the same size as A.  Only used for psym and
%       nzero statistics, described below.
%
% Requires amd, cholmod, metis, RBio, and CSparse.  Computes the following
% statistics, returning them as fields in the stats struct:
%
%   nrows           number of rows
%   ncols           number of columns
%   nnz             number of entries in A
%   RBtype          Rutherford/Boeing type
%   isBinary        1 if binary, 0 otherwise
%   isReal          1 if real, 0 if complex
%   cholcand        1 if a candidate for sparse Cholesky, 0 otherwise
%   nsym            numeric symmetry (0 to 1, where 1=symmetric)
%   psym            pattern symmetry (0 to 1, where 1=symmetric)
%   nnzdiag         nnz (diag (A)) if A is square, 0 otherwise
%   nzero           nnz (Z)
%   amd_lnz         nnz(L) for chol(C(p,p)) where, C=A+A', p=amd(C)
%   amd_flops       flop count for chol(C(p,p)) where, C=A+A', p=amd(C)
%   amd_vnz         nnz in Householder vectors for qr(A(:,colamd(A)))
%   amd_rnz         nnz in R for qr(A(:,colamd(A)))
%   metis_lnz       nnz(L) for chol(C(p,p)) where, C=A+A', p=metis(C)
%   metis_flops     flop count for chol(C(p,p)) where, C=A+A', p=metis(C)
%   metis_vnz       nnz in Householder vectors for qr(A(:,metis(A,'col')))
%   metis_rnz       nnz in R for qr(A(:,metis(A,'col')))
%   nblocks         # of blocks from dmperm
%   sprank          sprank(A)
%   nzoff           # of entries not in diagonal blocks from dmperm
%   ncc             # of strongly connected components
%   dmperm_lnz      nnz(L), using dmperm plus amd or metis
%   dmperm_unz      nnz(U), using dmperm plus amd or metis
%   dmperm_flops    flop count with dperm plus
%   dmperm_vnz      nnz in Householder vectors for dmperm plus
%   dmperm_rnz      nnz in R for dmperm plus
%   posdef          1 if positive definite, 0 otherwise
%   isND	    1 if a 2D/3D problem, 0 otherwise
%
% The *_lnz, *_unz, and *_flops statistics are not computed for rectangular
% or structurally singular matrices.  nzoff and the dmperm_* stats are not
% computed for structurally singular matrices.  If a statistic is not computed,
% it is set to -2.  If an attempt to compute the statistic was made but failed,
% it is set to -1.
%
% See also UFget, UFindex, amd, metis, RBtype, cs_scc, cs_sqr, dmperm.

% Copyright 2006-2007, Timothy A. Davis

% Requires the SuiteSparse set of packages: CHOLMOD, AMD, COLAMD, RBio, CSparse;
% and METIS.

if (nargin < 3)
    nometis = 0 ;
end

%-------------------------------------------------------------------------------
% ensure the matrix is sparse
%-------------------------------------------------------------------------------

if (~issparse (A))
    A = sparse (A) ;
end

%-------------------------------------------------------------------------------
% basic stats
%-------------------------------------------------------------------------------

tic ;
[m n] = size (A) ;
stats.nrows = m ;
stats.ncols = n ;
stats.nnz = nnz (A) ;
stats.RBtype = RBtype (A) ;			% Rutherford/Boeing type
stats.isBinary = (stats.RBtype (1) == 'p') ;
stats.isReal = (stats.RBtype (1) ~= 'c') ;

fprintf ('RBtype: %s time: %g\n', stats.RBtype, toc) ;

%-------------------------------------------------------------------------------
% symmetry and Cholesky candidacy
%-------------------------------------------------------------------------------

% get the symmetry
tic ;
[s xmatched pmatched nzoffdiag nnzdiag] = spsym (A) ;

stats.cholcand = (s >= 6) ; % check if Cholesky candidate

if (m ~= n)
    stats.nsym = 0 ;
    stats.psym = 0 ;
elseif (nzoffdiag > 0)
    stats.nsym = xmatched / nzoffdiag ;
    stats.psym = pmatched / nzoffdiag ;
else
    stats.nsym = 1 ;
    stats.psym = 1 ;
end

fprintf ('cholcand: %d\n', stats.cholcand) ;
fprintf ('nsym: %g psym: %g time: %g\n', stats.nsym, stats.psym, toc) ;
tic ;

stats.nnzdiag = nnzdiag ;

if (nargin > 5)
    stats.nzero = nnz (Z) ;

    % recompute the pattern symmetry with Z included
    if (m == n)
	try
	    AZ = A+Z ;
	    if (nnz (AZ) ~= nnz (A) + nnz (Z))
		error ('A and Z overlap!')
	    end
	    [s xmatched pmatched nzoffdiag] = spsym (AZ) ;
	    clear AZ
	    if (nzoffdiag > 0)
		stats.psym = pmatched / nzoffdiag ;
	    else
		stats.psym = 1 ;
	    end
	catch
	    fprintf ('failed to compute symmetry of pattern of A+Z\n') ;
	end
    end

else
    stats.nzero = 0 ;
end

fprintf ('nsym: %g psym: %g time: %g\n', stats.nsym, stats.psym, toc) ;
tic ;

%-------------------------------------------------------------------------------
% intialize ordering statistics
%-------------------------------------------------------------------------------

% if square, Cholesky of C(p,p) where C=A+A', p = amd(C)
stats.amd_lnz = -1 ;	    % nnz (chol (C))
stats.amd_flops = -1 ;	    % flop counts for chol (C)

% if square or rectangular
stats.amd_vnz = -1 ;	    % nnz (V), upper bound on L, for A(:,colamd(A))
stats.amd_rnz = -1 ;	    % nnz (R), upper bound on U, for A(:,colamd(A))

% if square, Cholesky of C(p,p) where C=A+A', p = metis(C)
stats.metis_lnz = -1 ;	    % nnz (chol (C))
stats.metis_flops = -1 ;    % flop counts for chol (C)

% if square or rectangular
stats.metis_vnz = -1 ;	    % nnz (V), upper bound on L, for A(:,metis(A))
stats.metis_rnz = -1 ;	    % nnz (R), upper bound on U, for A(:,metis(A))

% dmperm analysis
stats.nblocks = -1 ;	    % # of blocks in block-triangular form
stats.sprank = -1 ;	    % structural rank
stats.nzoff = -1 ;	    % # of entries of A in off-diagonal blocks

% cs_scc2
stats.ncc = -1 ;	    % # of strongly connected components

% dmperm: best of amd/metis on each square block, best of colamd/metis
% on rectangular blocks
stats.dmperm_lnz = -1 ;	    % nnz (L), for square struct full rank matrices 
stats.dmperm_unz = -1 ;	    % nnz (U) + nzoff, for square struct full rank mat
stats.dmperm_flops = -1 ;   % Cholesky flop count of each square block
stats.dmperm_vnz = -1 ;	    % nnz (V), upper bound on L
stats.dmperm_rnz = -1 ;	    % nnz (R), upper bound on U

stats.isND = -1 ;	    % 1 if 2D/3D problem, 0 otherwise

d = max (m,n) ;

% if the matrix has a symmetric nonzero pattern, nzoff will always be zero
if (stats.psym == 1)
    stats.nzoff = 0 ;
end

%-------------------------------------------------------------------------------
% determine if positive definite
%-------------------------------------------------------------------------------

if (~stats.cholcand)

    % not a candidate for Cholesky, so it cannot be positive definite
    stats.posdef = 0 ;

elseif (skip_chol)

    % Cholesky was skipped
    fprintf ('skip Cholesky\n') ;
    stats.posdef = -1 ;

else

    % try chol
    try
	[x, cstats] = cholmod2 (A, ones (stats.ncols,1)) ;
	rcond = cstats (1) ;
	fprintf ('rcond: %g\n', rcond) ;
	stats.posdef = (rcond > 0) ;
    catch
	% chol failed
	disp (lasterr) ;
	fprintf ('sparse Cholesky failed\n') ;
	stats.posdef = -1 ;
    end
    clear x cstats
end

fprintf ('posdef: %d time: %g\n', stats.posdef, toc) ;
tic ;

%-------------------------------------------------------------------------------
% transpose A if m < n, for ordering methods
%-------------------------------------------------------------------------------

if (m < n)
    try
	A = A' ;		    % A is now tall and thin, or square
    catch
	disp (lasterr) ;
	fprintf ('transpose failed...\n') ;
	return ;
    end
    [m n] = size (A) ;
end

if (~isreal (A))
    try
	A = spones (A) ;
    catch
	disp (lasterr) ;
	fprintf ('conversion from complex failed...\n') ;
	return ;
    end
end

fprintf ('computed A transpose if needed, time: %g\n', toc) ;
tic ;

%-------------------------------------------------------------------------------
% order entire matrix with AMD and METIS, if square
%-------------------------------------------------------------------------------

if (m == n)

    tic ;
    try
	if (stats.RBtype (2) == 'u')
	    C = A|A' ;
	else
	    C = A ;
	end
    catch
	disp (lasterr) ;
	fprintf ('A+A'' failed\n') ;
    end
    fprintf ('computed A+A'', time: %g\n', toc) ;

    % order the whole matrix with AMD
    tic ;
    try
	p = amd (C) ;
	c = symbfact (C (p,p)) ;
	stats.amd_lnz = sum (c) ;
	stats.amd_flops = sum (c.^2) ;
    catch
	disp (lasterr) ;
	fprintf ('amd failed\n') ;
    end
    clear p c
    fprintf ('AMD   lnz %d flops %g time: %g\n', ...
	stats.amd_lnz, stats.amd_flops, toc) ;

    % order the whole matrix with METIS
    tic ;
    try
	p = metis (C) ;
	c = symbfact (C (p,p)) ;
	stats.metis_lnz = sum (c) ;
	stats.metis_flops = sum (c.^2) ;
    catch
	disp (lasterr) ;
	fprintf ('metis failed\n') ;
    end
    clear p c C
    fprintf ('METIS lnz %d flops %g time: %g\n', ...
	stats.metis_lnz, stats.metis_flops, toc) ;

else

    % not computed if rectangular
    stats.amd_lnz = -2 ;
    stats.amd_flops = -2 ;
    stats.metis_lnz = -2 ;
    stats.metis_flops = -2 ;

end

%-------------------------------------------------------------------------------
% order entire matrix with COLAMD, for LU bounds
%-------------------------------------------------------------------------------

tic ;
try
    % do not ignore any rows, and do not do etree postordering
    q = colamd2mex (A, [d 10]) ;
    [vnz,rnz] = cs_sqr (A (:,q)) ;
    stats.amd_rnz = rnz ;
    stats.amd_vnz = vnz ;
catch
    disp (lasterr) ;
    fprintf ('colamd2 and cs_sqr failed\n') ;
end
clear q
fprintf ('COLAMD rnz %d vnz %d time: %g\n', stats.amd_rnz, stats.amd_vnz, toc) ;
tic ;

%-------------------------------------------------------------------------------
% order entire matrix with METIS, for LU bounds
%-------------------------------------------------------------------------------

if (~nometis)
    try
	q = metis (A, 'col') ;
	[vnz,rnz] = cs_sqr (A (:,q)) ;
	stats.metis_rnz = rnz ;
	stats.metis_vnz = vnz ;
    catch
	disp (lasterr) ;
	fprintf ('metis(A''*A) and cs_sqr failed\n') ;
    end
end
clear q
fprintf ('METIS  rnz %d vnz %d time: %g\n', ...
    stats.metis_rnz, stats.metis_vnz, toc) ;
tic ;

%-------------------------------------------------------------------------------
% strongly connected components
%-------------------------------------------------------------------------------

try
    % find the # of strongly connected components of the graph of a square A,
    % or # of connected components of the bipartite graph of a rectangular A.
    % [p,q,r,s] = cs_scc2 (A) ;
    if (m == n)
	[p r] = cs_scc (A) ;
    else
	[p r] = cs_scc (spaugment (A)) ;
    end
    stats.ncc = length (r) - 1 ;
    clear p r
catch
    disp (lasterr) ;
    fprintf ('cs_scc failed\n') ;
end

fprintf ('scc %d, time: %g\n', stats.ncc, toc) ;
tic ;

%-------------------------------------------------------------------------------
% isND
%-------------------------------------------------------------------------------

s = 0 ;
if (strfind (kind, 'structural'))
    s = 1 ;
elseif (strfind (kind, 'fluid'))
    s = 1 ;
elseif (strfind (kind, '2D'))
    s = 1 ;
elseif (strfind (kind, 'reduction'))
    s = 1 ;
elseif (strfind (kind, 'electromagnetics'))
    s = 1 ;
elseif (strfind (kind, 'semiconductor'))
    s = 1 ;
elseif (strfind (kind, 'thermal'))
    s = 1 ;
elseif (strfind (kind, 'materials'))
    s = 1 ;
elseif (strfind (kind, 'acoustics'))
    s = 1 ;
elseif (strfind (kind, 'vision'))
    s = 1 ;
elseif (strfind (kind, 'robotics'))
    s = 1 ;
end
stats.isND = s ;

fprintf ('isND %d\n', stats.isND) ;

%-------------------------------------------------------------------------------
% Dulmage-Mendelsohn permutation, and order each block
%-------------------------------------------------------------------------------

if (skip_dmperm)
    fprintf ('skip cs_dmperm, known irreducible\n') ;
else
    try
        % find the Dulmage-Mendelsohn decomposition
        [p,q,r,s,cc,rr] = cs_dmperm (A) ;
        nblocks = length (r) - 1 ;
        stats.nblocks = nblocks ;
        stats.sprank = rr(4)-1 ;
    catch
        disp (lasterr) ;
        fprintf ('cs_dmperm failed\n') ;
    end
end

fprintf ('sprank %d, time: %g\n', stats.sprank, toc) ;
fprintf ('nblocks %d\n', stats.nblocks) ;
tic

ok_square = 1 ;
ok_vnz = 1 ;

try

    mm = diff (r) ;
    nn = diff (s) ;
    square = all (mm == nn) ;

    if (~square)

	% not computed if the matrix is rectangular
	stats.dmperm_lnz = -2 ;
	stats.dmperm_unz = -2 ;
	stats.dmperm_flops = -2 ;

    end

    if (stats.sprank < min (m,n))

	% do not report DMPERM results for structurally singular matrices
	stats.nzoff = -2 ;
	stats.dmperm_lnz = -2 ;
	stats.dmperm_unz = -2 ;
	stats.dmperm_flops = -2 ;
	stats.dmperm_vnz = -2 ;
	stats.dmperm_rnz = -2 ;

    elseif (nblocks == n && m == n)

	% square triangular or diagonal
	C = A (p,q) ;
	clear p q r s

	stats.nzoff = nnz (triu (C, 1)) ;
	stats.dmperm_lnz = n ;
	stats.dmperm_unz = n + stats.nzoff ;
	stats.dmperm_flops = n ;
	stats.dmperm_vnz = n ;
	stats.dmperm_rnz = nnz (C) ;

    elseif (nblocks == 1 && m == n)

	% only one block of structural full rank, so don't redo analysis
	clear p q r s

	stats.nzoff = 0 ;
	if (stats.metis_lnz < 0 || stats.amd_lnz < stats.metis_lnz)
	    stats.dmperm_lnz = stats.amd_lnz ;
	    stats.dmperm_unz = stats.amd_lnz ;
	    stats.dmperm_flops = stats.amd_flops ;
	else
	    stats.dmperm_lnz = stats.metis_lnz ;
	    stats.dmperm_unz = stats.metis_lnz ;
	    stats.dmperm_flops = stats.metis_flops ;
	end

	if (stats.metis_vnz < 0 || stats.amd_rnz < stats.metis_rnz)
	    stats.dmperm_vnz = stats.amd_vnz ;
	    stats.dmperm_rnz = stats.amd_rnz ;
	else
	    stats.dmperm_vnz = stats.metis_vnz ;
	    stats.dmperm_rnz = stats.metis_rnz ;
	end

    else

	% analyze each block of the permuted matrix
	C = A (p,q) ;
	clear p q

	nzoff = nnz (C) ;
	lnz = 0 ;
	unz = 0 ;
	flops = 0 ;
	vnz = 0 ;
	rnz = 0 ;

	for k = 1:nblocks
	    i1 = r (k) ;
	    i2 = r (k+1) - 1 ;
	    j1 = s (k) ;
	    j2 = s (k+1) - 1 ;

	    if (i2-i1 == 1 && j2-j1 == 1)
		% singleton case
		nzoff = nzoff - 1 ;
		unz = unz + 1 ;
		lnz = lnz + 1 ;
		flops = flops + 1 ;
		rnz = rnz + 1 ;
		vnz = vnz + 1 ;
		continue ;
	    end

	    % get the kth block
	    S = C (i1:i2, j1:j2) ;
	    [ms ns] = size (S) ;
	    nzoff = nzoff - nnz (S) ;

	    if (ok_square)
		try
		    if (square)

			% all blocks are square, analyze a square block
			% best of amd and metis
			if (nometis)
			    [pblock c] = analyze (S|S', 'sym', 1) ;
			else
			    [pblock c] = analyze (S|S', 'sym', 3) ;
			end
			lnzblock = sum (c) ;
			unz = unz + lnzblock ;
			lnz = lnz + lnzblock ;
			flops = flops + sum (c.^2) ;
			clear c pblock

		    end
		catch
		    % ordering failed, but keep going to compute nzoff
		    ok_square = 0 ;
		end
	    end

	    if (ok_vnz)
		try

		    % analyze a rectangular block, or LU bounds for square block
		    if (ms < ns)
			S = S' ;
		    end
		    % best of amd and metis
		    try
			if (nometis)
			    pblock = analyze (S, 'col', 1) ;
			else
			    pblock = analyze (S, 'col', 3) ;
			end
		    catch
			pblock = colamd2mex (S, [d 10]) ;
		    end
		    [vnz2,rnz2] = cs_sqr (S (:, pblock)) ;
		    rnz = rnz + rnz2 ;
		    vnz = vnz + vnz2 ;

		catch
		    % ordering failed, but keep going to compute nzoff
		    ok_vnz = 0 ;
		end
	    end

	    clear S pblock

	end

	stats.nzoff = nzoff ;

	if (ok_square)
	    if (~square)
		stats.dmperm_unz = -2 ;
		stats.dmperm_lnz = -2 ;
		stats.dmperm_flops = -2 ;
	    else
		stats.dmperm_lnz = lnz ;
		stats.dmperm_unz = unz + nzoff ;
		stats.dmperm_flops = flops ;
	    end
	end

	if (ok_vnz)
	    stats.dmperm_vnz = vnz ;
	    stats.dmperm_rnz = rnz + nzoff ;
	end

	clear C r s

    end

    if (~ok_square)
	disp (lasterr) ;
	fprintf ('cs_dmperm (square: lnz, unz, flops) ordering failed\n') ;
    end
    if (~ok_vnz)
	disp (lasterr) ;
	fprintf ('cs_dmperm (LU bounds: vnz, rnz) ordering failed\n') ;
    end

catch
    disp (lasterr) ;
    fprintf ('cs_dmperm ordering and nzoff failed (or skipped)\n') ;
end

fprintf ('dmperm stats done, time %g\n', toc) ;
fprintf ('UFstats done\n') ;
