function UFcollection_install (nlargefile)
%UFCOLLECTION_INSTALL install the UFcollection toolbox
%
% Example:
%   UFcollection_install
%
% See also UFget.

% Copyright 2006-2007, Timothy A. Davis

if (nargin < 1)
    % try with large-file I/O
    nlargefile = 0 ;
end

if (nlargefile)
    fprintf ('Trying to compile without large file support...\n') ;
    mex -I../UFconfig -DNLARGEFILE UFfull_write.c
else
    try
	mex -I../UFconfig UFfull_write.c
    catch
	fprintf ('Trying to compile without large file support...\n') ;
	mex -I../UFconfig -DNLARGEFILE UFfull_write.c
    end
end

addpath (pwd) ;
fprintf ('UFcollection toolbox successfully compiled.\n') ;
