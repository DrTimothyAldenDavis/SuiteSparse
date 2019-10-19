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
%   nblocks         # of blocks from dmperm
%   sprank          sprank(A)
%   ncc             # of strongly connected components
%   posdef          1 if positive definite, 0 otherwise
%   isND            1 if a 2D/3D problem, 0 otherwise
%   isGraph         1 if a graph, 0 otherwise
%
% If the statistic is intentionally not computed, it is set to -2.  Some
% statistics are not computed for rectangular or structurally singular
% matrices, for example.  If an attempt to compute the statistic was made, but
% failed, it is set to -1.  If no attempt yet has been made to compute the
% entry, it is set to -3.
%
% Example:
%   UFindex
%   UFindex (267:300)
%
% If updating the UF_Index.mat file, the function first loads it from its
% default location, via UFget.  This function then saves the new UF_Index into
% the UF_Index.mat file in the current working directory (not overwriting the
% old copy, unless it resides in the current working directory).  It creates
% the UFstats.csv file used by UFgui.java and UFkinds.m and places it in the
% current working directory.
%
% See also UFstats.

% Copyright 2006-2011, Timothy A. Davis

% Requires the SuiteSparse set of packages: CHOLMOD, RBio, CSparse

%   10/13/2001: Created by Erich Mirabal

%-------------------------------------------------------------------------------
% initialize an empty index
%-------------------------------------------------------------------------------

% load the filenames
[url topdir] = UFlocation ;
files = textread ([topdir 'mat/UF_Listing.txt'], '%s') ;

% if no input, assume we have to do the whole file list
create_new = 0 ;
if (nargin < 1)
    matrixlist = 1:length(files) ;
    create_new = 1 ;
elseif (~isempty (matrixlist))
    % validate the input : range is limited by the files variable
    if (min (matrixlist) < 1) || (max (matrixlist) > length (files))
        error ('%s: %s', mfilename, 'Invalid input parameter.') ;
    end
end

if (~create_new)
    % load the index from file
    fprintf ('Loading existing UF_Index.mat file\n') ;
    try
        load UF_Index
        fprintf ('loaded UF_Index in current directory:\n%s\n', pwd) ;
        dir
    catch
        fprintf ('loading UF_Index = UFget\n') ;
        UF_Index = UFget ;
    end
end

% get the current UFkinds
if (create_new)
    kinds = cell (length (files),1) ;
else
    try
        kinds = UFkinds ;
    catch
        kinds = cell (2,1) ;
    end
    for i = matrixlist
        kinds {i} = '' ;
    end
end

% revision tracking device
UF_Index.LastRevisionDate = datestr (now) ;

% the index structure needs a download date for version tracking
UF_Index.DownloadTimeStamp = now ;

% start the index from scratch
if (create_new)

    fprintf ('Creating new UF_Index.mat file\n') ;
    nothing = -3 * ones (1, length (files)) ;

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
    UF_Index.nnzdiag = nothing ;
    UF_Index.posdef = nothing ;
    UF_Index.amd_lnz   = nothing ;
    UF_Index.amd_flops = nothing ;
    UF_Index.amd_vnz   = nothing ;
    UF_Index.amd_rnz   = nothing ;
    UF_Index.nblocks   = nothing ;
    UF_Index.sprank    = nothing ;
    UF_Index.RBtype = char (' '*ones (length (files),3)) ;
    UF_Index.cholcand = nothing ;
    UF_Index.ncc = nothing ;
    UF_Index.isND = nothing ;
    UF_Index.isGraph = nothing ;
    UF_Index.lowerbandwidth = nothing ;
    UF_Index.upperbandwidth = nothing ;
    UF_Index.rcm_lowerbandwidth = nothing ;
    UF_Index.rcm_upperbandwidth = nothing ;

else

    % make sure we have the right length for the arrays
    if length (UF_Index.nrows) < max (matrixlist)

        len = max (matrixlist) - length (UF_Index.nrows) ;
        nothing = -ones (1, len) ;

        if (len > 0)
            for i = matrixlist
                UF_Index.Group {i} = '' ;
                UF_Index.Name {i} = '' ;
            end
            UF_Index.nrows      = [UF_Index.nrows nothing] ;
            UF_Index.ncols      = [UF_Index.ncols nothing] ;
            UF_Index.nnz        = [UF_Index.nnz nothing] ;
            UF_Index.nzero      = [UF_Index.nzero nothing] ;
            UF_Index.pattern_symmetry = [UF_Index.pattern_symmetry nothing] ;
            UF_Index.numerical_symmetry = [UF_Index.numerical_symmetry nothing];
            UF_Index.isBinary   = [UF_Index.isBinary nothing] ;
            UF_Index.isReal     = [UF_Index.isReal nothing] ;
            UF_Index.nnzdiag    = [UF_Index.nnzdiag nothing] ;
            UF_Index.posdef     = [UF_Index.posdef nothing] ;
            UF_Index.amd_lnz    = [UF_Index.amd_lnz nothing] ;
            UF_Index.amd_flops  = [UF_Index.amd_flops nothing] ;
            UF_Index.amd_vnz    = [UF_Index.amd_vnz nothing] ;
            UF_Index.amd_rnz    = [UF_Index.amd_rnz nothing] ;
            UF_Index.nblocks    = [UF_Index.nblocks nothing] ;
            UF_Index.sprank     = [UF_Index.sprank nothing] ;
            UF_Index.RBtype     = [UF_Index.RBtype ; char (' '*ones (len,3))] ;
            UF_Index.cholcand   = [UF_Index.cholcand nothing] ;
            UF_Index.ncc        = [UF_Index.ncc nothing] ;
            UF_Index.isND       = [UF_Index.isND nothing] ;
            UF_Index.isGraph    = [UF_Index.isGraph nothing] ;
            UF_Index.lowerbandwidth     = [UF_Index.lowerbandwidth nothing ;
            UF_Index.upperbandwidth     = [UF_Index.upperbandwidth nothing ;
            UF_Index.rcm_lowerbandwidth = [UF_Index.rcm_upperbandwidth nothing ;
            UF_Index.rcm_upperbandwidth = [UF_Index.rcm_upperbandwidth nothing ;
        end
    end
end

if (length (matrixlist) > 0)
    fprintf ('Will process %d files\n', length (matrixlist)) ;
end

nmat = length (UF_Index.nrows) ;
filesize = zeros (nmat,1) ;

%-------------------------------------------------------------------------------
% look through the directory listing
%-------------------------------------------------------------------------------

for i = matrixlist

    % note that the matrix is not loaded in this for loop
    ffile = deblank (files {i}) ;

    % group is the first part of the string up to the character before
    % the last file separator
    gi = find (ffile == '/') ;
    gi = gi (end) ;
    groupN = char (ffile (1:gi-1)) ;

    % name is the last section of the string after the last file separator
    matrixN = char (ffile (gi+1:end)) ;

    % get the directory info of the .mat file
    fileInfo = dir ([topdir 'mat/' ffile '.mat']) ;

    % set the file's data into the data arrays
    UF_Index.Name {i} = matrixN ;
    UF_Index.Group {i} = groupN ;

    if (length (fileInfo) > 0)                                              %#ok
        filesize (i) = fileInfo.bytes ;
    else
        filesize (i) = -1 ;
    end

end

if (length (matrixlist) > 0)
    fprintf ('\n======================================================\n') ;
    fprintf ('Matrices will processed in the following order:\n') ;
    for i = matrixlist
        ffile = deblank (files {i}) ;
        fprintf ('Matrix %d: %s filesize %d\n', i, ffile, filesize (i)) ;
        if (filesize (i) == -1)
            fprintf ('skip this file (not found)\n') ;
            continue ;
        end
    end
end

%-------------------------------------------------------------------------------
% load the matrices
%-------------------------------------------------------------------------------

% known to be positive definite / indefinite:
known_posdef = [ 939 1252 1267 1268 1423 1453 1455 2541:2547 ] ;
known_indef = [ 1348:1368 1586 1411 1901:1905] ;

% known to be irreducible, but dmperm takes too long:
known_irreducible = [ 1902:1905 ] ;
% known_irreducible = [ ] ;

t = tic ;

for k = 1:length (matrixlist)

    %---------------------------------------------------------------------------
    % get the matrix
    %---------------------------------------------------------------------------

    id = matrixlist (k) ;
    ffile = deblank (files {id}) ;
    fprintf ('\n============================== Matrix %d: %s\n', id, ffile) ;
    if (filesize (id) == -1)
	fprintf ('skip this file\n') ;
	continue ;
    end
    load ([topdir 'mat/' ffile]) ;

    % display the Problem struct
    disp (Problem) ;

    %---------------------------------------------------------------------------
    % get all stats
    %---------------------------------------------------------------------------

    kinds {id} = Problem.kind ;

    fprintf ('%s/%s\n', UF_Index.Group {id}, UF_Index.Name {id}) ;

    if (~isequal (Problem.name, [UF_Index.Group{id} '/' UF_Index.Name{id}]))
        error ('name mismatch!') ;
    end
    if (Problem.id ~= id)
        error ('id mismatch!') ;
    end

    skip_chol = (any (id == known_posdef) || any (id == known_indef)) ;
    skip_dmperm = any (id == known_irreducible) ;

    if (isfield (Problem, 'Zeros'))
	stats = UFstats (Problem.A, Problem.kind, skip_chol, ...
            skip_dmperm, Problem.Zeros) ;
    else
	stats = UFstats (Problem.A, Problem.kind, skip_chol, ...
            skip_dmperm) ;
    end

    %---------------------------------------------------------------------------
    % fix special cases
    %---------------------------------------------------------------------------

    if (stats.posdef < 0)
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
	% but dmperm takes too long
	fprintf ('known irreducible\n') ;
	stats.sprank = stats.nrows  ;
	stats.nblocks = 1 ;
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
    UF_Index.pattern_symmetry (id) = stats.pattern_symmetry ;
    UF_Index.numerical_symmetry (id) = stats.numerical_symmetry ;
    UF_Index.isBinary (id) = stats.isBinary ;
    UF_Index.isReal (id) = stats.isReal ;
    UF_Index.nnzdiag (id) = stats.nnzdiag ;
    UF_Index.posdef (id) = stats.posdef ;
    UF_Index.amd_lnz (id) = stats.amd_lnz ;
    UF_Index.amd_flops (id) = stats.amd_flops ;
    UF_Index.amd_vnz (id) = stats.amd_vnz ;
    UF_Index.amd_rnz (id) = stats.amd_rnz ;
    UF_Index.nblocks (id) = stats.nblocks ;
    UF_Index.sprank (id) = stats.sprank ;
    UF_Index.RBtype (id,:) = stats.RBtype ;
    UF_Index.cholcand (id) = stats.cholcand ;
    UF_Index.ncc (id) = stats.ncc ;
    UF_Index.isND (id) = stats.isND ;
    UF_Index.isGraph (id) = stats.isGraph ;
    UF_Index.lowerbandwidth (id) = stats.lowerbandwidth ;
    UF_Index.upperbandwidth (id) = stats.upperbandwidth ;
    UF_Index.rcm_lowerbandwidth (id) = stats.rcm_lowerbandwidth ;
    UF_Index.rcm_upperbandwidth (id) = stats.rcm_upperbandwidth ;

    %---------------------------------------------------------------------------
    % clear the problem and save the index and UFstats.csv
    %---------------------------------------------------------------------------

    clear Problem
    fprintf ('time since last save: %g\n', toc (t)) ;
    if (toc (t) > 20 || k == length (matrixlist))
        t = tic ;
        fprintf ('\n ... saving UF_Index ...\n') ;
        save UF_Index UF_Index

        fprintf ('\nCreating UFstats.csv in current directory:\n')
        fprintf ('%s/UFstats.csv\n', pwd) ;
        f = fopen ('UFstats.csv', 'w') ;
        fprintf (f, '%d\n', nmat) ;
        fprintf (f, '%s\n', UF_Index.LastRevisionDate) ;
        for id = 1:nmat
            fprintf (f,'%s,%s,%d,%d,%d,%d,%d,%d,%d,%.16g,%.16g,%s\n', ...
                UF_Index.Group {id}, ...
                UF_Index.Name {id}, ...
                UF_Index.nrows (id), ...
                UF_Index.ncols (id), ...
                UF_Index.nnz (id), ...
                UF_Index.isReal (id), ...
                UF_Index.isBinary (id), ...
                UF_Index.isND (id), ...
                UF_Index.posdef (id), ...
                UF_Index.pattern_symmetry (id), ...   % formatted with %.16g
                UF_Index.numerical_symmetry (id), ... % formatted with %.16g
                kinds {id}) ;
        end
        fclose (f) ;

        % flush the diary
        if (strcmp (get (0, 'Diary'), 'on'))
            diary off
            diary on
        end
    end
end
