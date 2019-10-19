function Problem = UFget (matrix, UF_Index)
%UFGET loads a matrix from the UF Sparse Matrix Collection.
%
%   Problem = UFget(matrix) loads a matrix from the UF sparse matrix collection,
%   specified as either a number (1 to the # of matrices in the collection) or
%   as a string (the name of the matrix).  With no input parameters, index=UFget
%   returns an index of matrices in the collection.  A local copy of the matrix
%   is saved (be aware that as of May 2007 the entire collection is over 8GB
%   in size).  If no input or output arguments are provided, the index is
%   printed.  With a 2nd parameter (Problem = UFget (matrix, index)), the index
%   file is not loaded.  This is faster if you are loading lots of matrices.
%
%   Examples:
%       index = UFget ;
%       Problem = UFget (6)
%       Problem = UFget ('HB/arc130')
%       Problem = UFget (6, index)
%       Problem = UFget ('HB/arc130', index)
%
%   See also UFgrep, UFweb, UFget_example, UFget_defaults, urlwrite.

%   Copyright 2007, Tim Davis, University of Florida.

%-------------------------------------------------------------------------------
% get the parameter settings
%-------------------------------------------------------------------------------

params = UFget_defaults ;
indexfile = sprintf ('%sUF_Index.mat', params.dir) ;
indexurl = sprintf ('%s/UF_Index.mat', params.url) ;

%-------------------------------------------------------------------------------
% get the index file (download a new one if necessary)
%-------------------------------------------------------------------------------

try
    % load the existing index file
    if (nargin < 2)
	load (indexfile) ;
    end
    % see if the index file is old; if so, download a fresh copy
    refresh = (UF_Index.DownloadTimeStamp + params.refresh < now) ;
catch
    % oops, no index file.  download it.
    refresh = 1 ;
end

if (refresh)
    % a new UF_Index.mat file to get access to new matrices (if any)
    fprintf ('downloading %s\n', indexurl) ;
    fprintf ('to %s\n', indexfile) ;
    urlwrite (indexurl, indexfile) ;
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
        fprintf ('(p,n)sym:  symmetry of the pattern and values\n') ;
        fprintf ('           (0 = unsymmetric, 1 = symmetric, - = not computed)\n') ;
        fprintf ('type:      real\n') ;
        fprintf ('           complex\n') ;
        fprintf ('           binary:  all entries are 0 or 1\n') ;
        nmat = length (UF_Index.nrows) ;
        for j = 1:nmat
            if (mod (j, 25) == 1)
                fprintf ('\n') ;
                fprintf ('ID   Group/Name                nrows-by-  ncols  nonzeros  (p,n)sym  type\n') ;
            end
            s = sprintf ('%s/%s', UF_Index.Group {j}, UF_Index.Name {j}) ;
            fprintf ('%4d %-30s %7d-by-%7d %9d ', ...
            j, s, UF_Index.nrows (j), UF_Index.ncols (j), UF_Index.nnz (j)) ;
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
            if (UF_Index.isBinary (j))
                fprintf (' binary\n') ;
            elseif (~UF_Index.isReal (j))
                fprintf (' complex\n') ;
            else
                fprintf (' real\n') ;
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
    fprintf ('to %s\n', matfile) ;
    urlwrite (maturl, matfile) ;
    load (matfile)
    save (matfile, 'Problem') ;
end
