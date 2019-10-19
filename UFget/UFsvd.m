function S = UFsvd (matrix, UF_Index)
%UFsvd singular values of a matrix in the UF collection.
%
% As of Nov 2012, only matrices for which min(size(A)) <= 30401
% have their singular values computed.
%
% Examples:
%   S = UFsvd ('HB/arc130')
%   S = UFsvd (6)
%   index = UFget
%   S = UFsvd (6, index)
%
% S is a struct containing:
%   s       the singular values (a column vector of size min(size(A)))
%   how     a string

if (nargin < 2)
    % load the UF index
    UF_Index = UFget ;
end

% look up the matrix in the UF index
[group matrix id] = UFget_lookup (matrix, UF_Index) ;
if (id == 0)
    error ('invalid matrix') ;
end

% determine where the files go
params = UFget_defaults ;
svddir = sprintf ('%ssvd%s%s', params.topdir, filesep, group) ;
svdfile = sprintf ('%s%s%s_SVD.mat', svddir, filesep, matrix) ;
svdurl = sprintf ('%s/svd/%s/%s_SVD.mat', params.topurl, group, matrix) ;

% make sure the mat/Group directory exists
if (~exist (svddir, 'dir'))
    mkdir (svddir) ;
end

% download the *_SVD.mat file, if not already downloaded
if (~exist (svdfile, 'file'))
    fprintf ('downloading %s\n', svdurl) ;
    fprintf ('to %s\n', svdfile) ;
    tmp = tempname ;                        % download to a temp file first
    try
        urlwrite (svdurl, tmp) ;
    catch me                                %#ok
        error ('SVD not yet computed for this matrix (or URL not found)') ;
    end
    movefile (tmp, svdfile, 'f') ;          % move the new matrix into place
end

% load the SVD
load (svdfile) ;

