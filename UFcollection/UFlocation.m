function [url, topdir] = UFlocation
%UFLOCATION URL and top-level directory of the UF Sparse Matrix Collection
%
% Example:
%   [url, topdir] = UFlocation
%
% See also UFget.

% Copyright 2006-2007, Timothy A. Davis

params = UFget_defaults ;
t = find (params.dir == filesep) ;
topdir = regexprep (params.dir (1:t(end-1)), '[\/\\]', filesep) ;

t = find (params.url == '/') ;
url = params.url (1:t(end)) ;

