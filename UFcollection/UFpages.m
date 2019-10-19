function UFpages (figures, list)
%UFPAGES create web page for each matrix in UF Sparse Matrix Collection
% Usage: UFpages (figures, list)
%
% figures: 1 if figures are to be created, 0 otherwise
% list: list of matrix id's to process.  All matrices in the collection are
%   processed if not present.
%
% Example:
%
%   UFpages             % create all the pages
%   UFpages (0)         % create all the pages, but not the figures
%   UFpages (1,1:10)    % create pages for just matrices 1 to 10
%   UFpages (0,1:10)    % ditto, but do not create the figures
%
% See also UFpage, UFget.

% Copyright 2006-2007, Timothy A. Davis

if (nargin < 1)
    figures = 1 ;
end
index = UFget ;
if (nargin < 2)
    list = 1:length (index.nrows) ;
end
for i = list
    UFpage (i, index, figures) ;
end

