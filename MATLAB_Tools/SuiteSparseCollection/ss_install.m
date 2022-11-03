function ss_install (nlargefile)
%SS_INSTALL install the SuiteSparseCollection toolbox
%
% Example:
%   ss_install
%
% See also ssget.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (nargin < 1)
    % try with large-file I/O
    nlargefile = 0 ;
end

if (nlargefile)
    fprintf ('Trying to compile without large file support...\n') ;
    mex -DNLARGEFILE ssfull_write.c
else
    try
	mex ssfull_write.c
    catch
	fprintf ('Trying to compile without large file support...\n') ;
	mex --DNLARGEFILE ssfull_write.c
    end
end

addpath (pwd) ;
fprintf ('sscollection toolbox successfully compiled.\n') ;
