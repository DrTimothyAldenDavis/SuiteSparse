function UFallpages (figures)
%UFALLPAGES create all web pages for the UF Sparse Matrix Collection
% with the exception of the top-level index.html file (which is created
% manually).
%
% Example:
%
%   UFallpages
%   UFallpages(0)       % does not create figures (assumes they already exist)
%
% See also UFget, UFlists, UFpages.

% Copyright 2006-2007, Timothy A. Davis

if (nargin < 1)
    figures = 1 ;
end
UFlists ;
UFpages (figures) ;

