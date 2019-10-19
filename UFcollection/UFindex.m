function UF_Index = UFindex (matrixlist)
%UFINDEX create the index for the UF Sparse Matrix Collection
%
% UF_Index = UFindex (matrixlist)
%
% matrixlist: an integer list, in the range of 1 to the length of the
%   UF_Listing.txt file, containing a list of matrices for which to modify
%   the UF_Index entries.  If matrixlist is not present, then the UF_Index
%   is created from scratch.
%
% UF_Index:  a struct containing the index information, with the
%   following fields, assuming that there are n matrices in the collection:
%
%   LastRevisionDate: a string with the date and time the index was updated.
%   DowloadTimeStamp: date and time the index was last downloaded.
%
%   Group: an n-by-1 cell array.  Group {i} is the group for matrix i.
%   Name: n-by-1 cell array.  Name {i} is the name of matrix i.
%
%   The following fields are n-by-1 vectors unless otherwise specified.
%   nrows(id) gives the number of rows of the matrix with id = Problem.id,
%   for example.
%
%   nrows           number of rows
%   ncols           number of columns
%   nnz             number of entries in A
%   RBtype          Rutherford/Boeing type, an n-by-3 char array
%   isBinary        1 if binary, 0 otherwise
%   isReal          1 if real, 0 if complex
%   cholcand        1 if a candidate for sparse Cholesky, 0 otherwise
%   numerical_symmetry  numeric symmetry (0 to 1, where 1=symmetric)
%   pattern_symmetry    pattern symmetry (0 to 1, where 1=symmetric)
%   nnzdiag         nnz (diag (A)) if A is square, 0 otherwise
%   nzero           nnz (Problem.Zeros)
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
% If the statistic is not computed, it is set to -2.  Some statistics are not
% computed for rectangular or structurally singular matrices, for example.
% If an attempt to compute the statistic was made, but failed, it is set to -1.
%
% Example:
%   UFindex
%   UFindex (267:300)
%
% See also UFstats, amd, metis, RBtype, cs_scc, cs_sqr, cs_dmperm.

% Copyright 2006-2007, Timothy A. Davis

% Requires the SuiteSparse set of packages: CHOLMOD, AMD, COLAMD, RBio, CSparse;
% and METIS.

%   10/13/2001: Created by Erich Mirabal
%   12/6/2001, 1/17/2003, 11/16/2006:  modified by Tim Davis

%-------------------------------------------------------------------------------
% initialize an empty index
%-------------------------------------------------------------------------------

% load the filenames
[url topdir] = UFlocation ;
files = textread ([topdir 'mat' filesep 'UF_Listing.txt'], '%s') ;

% if no input, assume we have to do the whole file list
create_new = 0 ;
if (nargin < 1)
    matrixlist = 1:length(files) ;
    create_new = 1 ;
else
    % validate the input : range is limited by the files variable
    if (min (matrixlist) < 1) || (max (matrixlist) > length (files))
        error ('%s: %s', mfilename, 'Invalid input parameter.') ;
    end
end

if (~create_new)
    % load the index from file
    fprintf ('Loading existing UF_Index.mat file\n') ;
    UF_Index = load ('UF_Index.mat') ;
    UF_Index = UF_Index.UF_Index ;
end

% revision tracking device
UF_Index.LastRevisionDate = datestr (now) ;

% the index structure needs a download date for version tracking
UF_Index.DownloadTimeStamp = now ;

% start the index from scratch
if (create_new)

    fprintf ('Creating new UF_Index.mat file\n') ;
    nothing = -ones (1, length (files)) ;

    UF_Index.Group = cell (size (files)) ;
    UF_Index.Name = cell (size (files)) ;       

    UF_Index.nrows = nothing ;
    UF_Index.ncols = nothing ;
    UF_Index.nnz = nothing ;
    UF_Index.nzero = nothing ;
    UF_Index.pattern_symmetry = nothing ;
    UF_Index.numerical_symmetry = nothing ;
    UF_Index.isBinary = nothing ;
    UF_Index.isReal = nothing ;

    % removed has_b, has_guess, has_x, has_Zeros, is_lp, has_coord, zdiag

    UF_Index.nnzdiag = nothing ;

    UF_Index.posdef = nothing ;

    UF_Index.amd_lnz	= nothing ;
    UF_Index.amd_flops	= nothing ;
    UF_Index.amd_vnz	= nothing ;
    UF_Index.amd_rnz	= nothing ;

    UF_Index.metis_lnz	= nothing ;
    UF_Index.metis_flops = nothing ;
    UF_Index.metis_vnz	= nothing ;
    UF_Index.metis_rnz	= nothing ;

    UF_Index.nblocks	= nothing ;
    UF_Index.sprank	= nothing ;
    UF_Index.nzoff	= nothing ;

    UF_Index.dmperm_lnz	= nothing ;
    UF_Index.dmperm_unz	= nothing ;
    UF_Index.dmperm_flops = nothing ;
    UF_Index.dmperm_vnz	= nothing ;
    UF_Index.dmperm_rnz	= nothing ;

    % added RBtype, cholcand, ncc
    UF_Index.RBtype = char (' '*ones (length (files),3)) ;
    UF_Index.cholcand = nothing ;
    UF_Index.ncc = nothing ;

    % added isND
    UF_Index.isND = nothing ;

else

    % make sure we have the right length for the arrays
    if length (UF_Index.nrows) < max (matrixlist)

        len = max (matrixlist) - length (UF_Index.nrows) ;
        nothing = -ones (1, len) ;

	if (len > 0)

	    % don't worry about the cell arrays, only append to numeric arrays
	    UF_Index.nrows = [UF_Index.nrows nothing] ;
	    UF_Index.ncols = [UF_Index.ncols nothing] ;
	    UF_Index.nnz = [UF_Index.nnz nothing] ;
	    UF_Index.nzero = [UF_Index.nzero nothing] ;
	    UF_Index.pattern_symmetry = [UF_Index.pattern_symmetry nothing] ;
	    UF_Index.numerical_symmetry = [UF_Index.numerical_symmetry nothing];
	    UF_Index.isBinary = [UF_Index.isBinary nothing] ;
	    UF_Index.isReal = [UF_Index.isReal nothing] ;

	    UF_Index.nnzdiag = [UF_Index.nnzdiag nothing] ;

	    UF_Index.posdef = [UF_Index.posdef nothing] ;

	    UF_Index.amd_lnz	= [UF_Index.amd_lnz nothing] ;
	    UF_Index.amd_flops	= [UF_Index.amd_flops nothing] ;
	    UF_Index.amd_vnz	= [UF_Index.amd_vnz nothing] ;
	    UF_Index.amd_rnz	= [UF_Index.amd_rnz nothing] ;

	    UF_Index.metis_lnz	= [UF_Index.metis_lnz nothing] ;
	    UF_Index.metis_flops= [UF_Index.metis_flops nothing] ;
	    UF_Index.metis_vnz	= [UF_Index.metis_vnz nothing] ;
	    UF_Index.metis_rnz	= [UF_Index.metis_rnz nothing] ;

	    UF_Index.nblocks	= [UF_Index.nblocks nothing] ;
	    UF_Index.sprank	= [UF_Index.sprank nothing] ;
	    UF_Index.nzoff	= [UF_Index.nzoff nothing] ;

	    UF_Index.dmperm_lnz	= [UF_Index.dmperm_lnz nothing] ;
	    UF_Index.dmperm_unz	= [UF_Index.dmperm_unz nothing] ;
	    UF_Index.dmperm_flops= [UF_Index.dmperm_flops nothing] ;
	    UF_Index.dmperm_vnz	= [UF_Index.dmperm_vnz nothing] ;
	    UF_Index.dmperm_rnz	= [UF_Index.dmperm_rnz nothing] ;

	    UF_Index.RBtype = [UF_Index.RBtype ; char (' '*ones (len,3))] ;
	    UF_Index.cholcand = [UF_Index.cholcand nothing] ;
	    UF_Index.ncc = [UF_Index.ncc nothing] ;

	    UF_Index.isND = [UF_Index.isND nothing] ;
	end

    end
end

fprintf ('Will process %d files\n', length (matrixlist)) ;

nmat = length (UF_Index.nrows) ;
filesize = zeros (nmat,1) ;

%-------------------------------------------------------------------------------
% look through the directory listing, and sort matrixlist by size
%-------------------------------------------------------------------------------

for i = matrixlist

    % note that the matrix is not loaded in this for loop
    ffile = deblank (files {i}) ;

    % group is the first part of the string up to the character before
    % the last file separator
    gi = find (ffile == filesep) ;
    gi = gi (end) ;
    groupN = char (ffile (1:gi-1)) ;

    % name is the last section of the string after the last file separator
    matrixN = char (ffile (gi+1:end)) ;

    % get the directory info of the .mat file
    fileInfo = dir ([topdir 'mat' filesep ffile '.mat']) ;

    % set the file's data into the data arrays
    UF_Index.Name {i} = matrixN ;
    UF_Index.Group {i} = groupN ;

    if (length (fileInfo) > 0)						    %#ok
	filesize (i) = fileInfo.bytes ;
    else
	filesize (i) = 9999999999 ;
    end
    % fprintf ('%s / %s filesize %d\n', groupN, matrixN, fileInfo.bytes) ;

end

fprintf ('\n======================================================\n') ;
fprintf ('Matrices will processed in the following order:\n') ;
for i = matrixlist
    ffile = deblank (files {i}) ;
    fprintf ('Matrix %d: %s filesize %d\n', i, ffile, filesize (i)) ;
    if (filesize (i) == 9999999999)
	fprintf ('skip this file\n') ;
	continue ;
    end
end

%-------------------------------------------------------------------------------
% load the matrices
%-------------------------------------------------------------------------------

% metis (A,'col') fails with a seg fault for these matrices:
skip_metis = [850 858 1257 1258] ;

% these matrices are known to be positive definite, and indefinite,
% respectively, but sparse Cholesky fails (on a 4GB Penitum 4) on some of them:
known_posdef = [ 939 1252 1267 1268 1423 1453 1455 ] ;
known_indef = [ 1348:1368 1586 1411 1901:1905] ;

% these matrices are known to be irreducible, but dmperm fails or takes too
% long
known_irreducible = [ 916 1901:1905 ] ;

for k = 1:length (matrixlist)

    %---------------------------------------------------------------------------
    % get the matrix
    %---------------------------------------------------------------------------

    id = matrixlist (k) ;
    ffile = deblank (files {id}) ;
    fprintf ('\n============================== Matrix %d: %s\n', id, ffile) ;
    if (filesize (id) == 9999999999)
	fprintf ('skip this file\n') ;
	continue ;
    end
    load ([topdir 'mat' filesep ffile]) ;

    % display the Problem struct
    disp (Problem) ;

    %---------------------------------------------------------------------------
    % get all stats
    %---------------------------------------------------------------------------

    nometis = any (id == skip_metis) ;
    if (nometis)
	fprintf ('skip metis - will fail\n') ;
    end

    fprintf ('%s/%s\n', UF_Index.Group {id}, UF_Index.Name {id}) ;

    skip_chol = (any (id == known_posdef) || any (id == known_indef)) ;
    skip_dmperm = any (id == known_irreducible) ;

    if (isfield (Problem, 'Zeros'))
	stats = UFstats (Problem.A, Problem.kind, nometis, skip_chol, ...
            skip_dmperm, Problem.Zeros) ;
    else
	stats = UFstats (Problem.A, Problem.kind, nometis, skip_chol, ...
            skip_dmperm) ;
    end

    %---------------------------------------------------------------------------
    % fix special cases
    %---------------------------------------------------------------------------

    if (stats.posdef == -1)

	if (any (id == known_posdef))
	    fprintf ('known posdef\n') ;
	    stats.posdef = 1 ;
	elseif (any (id == known_indef))
	    fprintf ('known indef\n') ;
	    stats.posdef = 0 ;
	end

    end
    if (any (id == known_irreducible) && stats.sprank < 0)
	% full sprank, and not reducible to block triangular form,
	% but the matrix is to big for dmperm
	fprintf ('known irreducible\n') ;
	stats.sprank = stats.nrows  ;
	stats.nzoff = 0 ;
	stats.nblocks = 1 ;
	stats.ncc = 1 ;
    end

    % display the stats
    disp (stats) ;

    %---------------------------------------------------------------------------
    % save the stats in the index
    %---------------------------------------------------------------------------

    UF_Index.nrows (id) = stats.nrows ;
    UF_Index.ncols (id) = stats.ncols ;
    UF_Index.nnz (id) = stats.nnz ;
    UF_Index.nzero (id) = stats.nzero ;

    UF_Index.pattern_symmetry (id) = stats.psym ;
    UF_Index.numerical_symmetry (id) = stats.nsym ;
    UF_Index.isBinary (id) = stats.isBinary ;
    UF_Index.isReal (id) = stats.isReal ;

    UF_Index.nnzdiag (id) = stats.nnzdiag ;

    UF_Index.posdef (id) = stats.posdef ;

    UF_Index.amd_lnz (id) = stats.amd_lnz ;
    UF_Index.amd_flops (id) = stats.amd_flops ;
    UF_Index.amd_vnz (id) = stats.amd_vnz ;
    UF_Index.amd_rnz (id) = stats.amd_rnz ;

    UF_Index.metis_lnz (id) = stats.metis_lnz ;
    UF_Index.metis_flops (id) = stats.metis_flops ;
    UF_Index.metis_vnz (id) = stats.metis_vnz ;
    UF_Index.metis_rnz (id) = stats.metis_rnz ;

    UF_Index.nblocks (id) = stats.nblocks ;
    UF_Index.sprank (id) = stats.sprank ;
    UF_Index.nzoff (id) = stats.nzoff ;

    UF_Index.dmperm_lnz (id) = stats.dmperm_lnz ;
    UF_Index.dmperm_unz (id) = stats.dmperm_unz ;
    UF_Index.dmperm_flops (id) = stats.dmperm_flops ;
    UF_Index.dmperm_vnz (id) = stats.dmperm_vnz ;
    UF_Index.dmperm_rnz (id) = stats.dmperm_rnz ;

    UF_Index.RBtype (id,:) = stats.RBtype ;
    UF_Index.cholcand (id) = stats.cholcand ;
    UF_Index.ncc (id) = stats.ncc ;

    UF_Index.isND (id) = stats.isND ;

    %---------------------------------------------------------------------------
    % clear the problem and save the index
    %---------------------------------------------------------------------------

    clear Problem
    save UF_Index UF_Index

    % flush the diary
    if (strcmp (get (0, 'Diary'), 'on'))
	diary off
	diary on
    end
end


