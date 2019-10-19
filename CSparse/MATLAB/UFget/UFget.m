function Problem = UFget (matrix, UF_Index)
%UFget loads a matrix from the UF sparse matrix collection.
%
%   Problem = UFget(matrix) loads a matrix from the UF sparse matrix collection,
%   specified as either a number (1 to the # of matrices in the collection) or
%   as a string (the name of the matrix).  With no input parameters, index=UFget
%   returns an index of matrices in the collection.  A local copy of the matrix
%   is saved (be aware that as of Nov 2005 the entire collection is almost 5GB
%   in size).  If no input or output arguments are provided, the index is
%   printed.  With a 2nd parameter (Problem = UFget (matrix, index)), the index
%   file is not loaded.  This is faster if you are loading lots of matrices.
%   For details on the Problem struct, type the command "type UFget"
%
%   Examples:
%       index = UFget ;
%       Problem = UFget (6)
%       Problem = UFget ('HB/arc130')
%       Problem = UFget (6, index)
%       Problem = UFget ('HB/arc130', index)
%
%   See also UFget_install, UFget_example, UFget_defaults, UFget_java.java.

%   Copyright 2005, Tim Davis, University of Florida.

%   Modification History:
%   10/11/2001: Created by Erich Mirabal
%   3/12/2002:  V1.0 released
%   11/2005: updated for MATLAB version 7.1.

%-------------------------------------------------------------------------------
% get the parameter settings
%-------------------------------------------------------------------------------

params = UFget_defaults ;
indexfile = sprintf ('%sUF_Index.mat', params.dir) ;
indexurl = sprintf ('%s/UF_Index.mat', params.url) ;

%-------------------------------------------------------------------------------
% get the index file (download a new one if necessary)
%-------------------------------------------------------------------------------

% load the existing index file
if (nargin < 2)
    load (indexfile) ;
end

% see if the index file is old; if so, download a fresh copy
if (UF_Index.DownloadTimeStamp + params.refresh < now)
    % a new UF_Index.mat file to get access to new matrices (if any)
    fprintf ('downdloading %s\n', indexurl) ;
    UFget_java.geturl (indexurl, indexfile) ; 
    load (indexfile) ;
    UF_Index.DownloadTimeStamp = now ;
    save (indexfile, 'UF_Index') ;
end

%-------------------------------------------------------------------------------
% return the index file if requested
%-------------------------------------------------------------------------------

% if the user passed in a zero or no argument at all, return the index file
if nargin == 0
    matrix = 0 ;
end

if (matrix == 0)
    if (nargout == 0)
        % no output arguments have been passed, so print the index file
        fprintf ('\nUF sparse matrix collection index:  %s\n', ...
            UF_Index.LastRevisionDate) ;
        fprintf ('\nLegend:\n') ;
        fprintf ('z:         * if the matrix has explicit zero entries\n') ;
        fprintf ('(p,n)sym:  symmetry of the pattern and values\n') ;
        fprintf ('           (0 = unsymmetric, 1 = symmetric, - = not computed)\n') ;
        fprintf ('type:      real\n') ;
        fprintf ('           complex\n') ;
        fprintf ('           binary:  all entries are 0 or 1\n') ;
        fprintf ('           LP:  a linear programming problem\n') ;
        fprintf ('           geo:  has geometric coordinates (2D or 3D)\n') ;
        nmat = length (UF_Index.nrows) ;
        for j = 1:nmat
            if (mod (j, 25) == 1)
                fprintf ('\n') ;
                fprintf ('ID  Group/Name                nrows-by-  ncols  nonzeros z (p,n)sym  type\n') ;
            end
            s = sprintf ('%s/%s', UF_Index.Group {j}, UF_Index.Name {j}) ;
            fprintf ('%3d %-23s %7d-by-%7d %9d ', ...
            j, s, UF_Index.nrows (j), UF_Index.ncols (j), UF_Index.nnz (j)) ;
            if (UF_Index.has_Zeros (j))
                fprintf ('*') ;
            else
                fprintf (' ') ;
            end
            psym = UF_Index.pattern_symmetry (j) ;
            nsym = UF_Index.numerical_symmetry (j) ;
            if (psym < 0)
                fprintf ('  -  ') ;
            else
                fprintf (' %4.2f', psym) ;
            end
            if (nsym < 0)
                fprintf ('  -  ') ;
            else
                fprintf (' %4.2f', nsym) ;
            end
            if (UF_Index.is_lp (j))
                fprintf (' LP') ;
            elseif (UF_Index.isBinary (j))
                fprintf (' binary') ;
            elseif (~UF_Index.isReal (j))
                fprintf (' complex') ;
            else
                fprintf (' real') ;
            end
            if (UF_Index.has_coord (j))
                fprintf (',geo\n') ;
            else
                fprintf ('\n') ;
            end
        end
    else
        Problem = UF_Index ;
    end
    return ;
end

%-------------------------------------------------------------------------------
% determine if the matrix parameter is a matrix index or name
%-------------------------------------------------------------------------------

[group matrix id] = UFget_lookup (matrix, UF_Index) ;

if (id == 0)
    error ('invalid matrix') ;
end

%-------------------------------------------------------------------------------
% download the matrix (if needed) and load it into MATLAB

matdir = sprintf ('%s%s%s%s.mat', params.dir, group) ;
matfile = sprintf ('%s%s%s.mat', matdir, filesep, matrix) ;
maturl = sprintf ('%s/%s/%s.mat', params.url, group, matrix) ;

if (~exist (matdir, 'dir'))
    mkdir (matdir) ;
end

if (exist (matfile, 'file'))
    load (matfile)
else
    fprintf ('downloading %s\n', maturl) ;
    UFget_java.geturl (maturl, matfile) ;
    load (matfile)
    save (matfile, 'Problem') ;
end



%-------------------------------------------------------------------------------
%   Problem: A struct containing the sparse matrix problem:
%       Problem.A       the sparse matrix
%       Problem.name    the name of the matrix ('HB/arc130' for example)
%       Problem.title   a short description of the problem
%       Problem.id      the id number of the matrix.
%       The following fields are not always present:
%       Problem.b       right-hand-side(s)
%       Problem.guess   initial guess (same dimension as b)
%       Problem.x       solution(s)
%       Problem.Zeros   some problems are stated with explicitly zero entries.
%                       The pattern these entries is held in this sparse matrix.
%       Problem.c       For a linear programming problem (minimize c'*x subject
%                       to A*x=b and lo <= x <= hi).
%       Problem.lo      Lower bound on x, for a linear programming problem.
%       Problem.hi      Upper bound on x, for a linear programming problem.
%       Problem.z0      Starting guess for x is z0*ones(size(A,2),1) for
%                       a linear programming problem.
%       Problem.coord   Geometric coordinates of the nodes of the graph of A.
%
%   index:  if there are n matrices in the collection:
%       Group: an n-by-1 cell array.  Group{i} is the group for matrix i.
%       Name: n-by-1 cell array.  Name{i} is the name of matrix i.
%       The following fields are n-by-1 vectors:
%       nrows (i)               number of rows
%       ncols (i)               number of columns
%       nnz (i)                 number of nonzeros (excl. zero entries)
%       nzero (i)               number of explicitly zero entries
%       pattern_symmetry (i)    symmetry of pattern (incl. zero entries)
%       numerical_symmetry (i)  symmetry of numerical values
%       isBinary (i)            1 if values are all 0 or 1.
%       isReal (i)              1 if the matrix is real, as opposed to complex
%       has_b (i)               1 if Problem.b exists, 0 otherwise
%       has_guess (i)           1 if Problem.guess exists, 0 otherwise
%       has_x (i)               1 if Problem.x exists, 0 otherwise
%       has_Zeros (i)           1 if Problem.Zeros exists, 0 otherwise
%       is_lp (i)               1 if b, c, lo, hi, and z0 are in the Problem
%                               (the Problem is a linear program)
%       has_coord (i)           1 if Problem.coord exists, 0 otherwise
%	nnzdiag (i)		number of nonzeros on the diagonal
%	zdiag (i)		number of zeros on the diagonal
%	posdef (i)		1 if positive def, 0 if not, -1 unknown
