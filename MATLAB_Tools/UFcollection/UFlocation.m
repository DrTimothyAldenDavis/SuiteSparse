function [url, topdir] = UFlocation
%UFLOCATION URL and top-level directory of the UF Sparse Matrix Collection
%
% Example:
%   [url, topdir] = UFlocation
%
% See also UFget.

% Copyright 2006-2007, Timothy A. Davis

params = UFget_defaults ;
url = [ params.topurl '/'] ;
topdir = params.topdir ;

