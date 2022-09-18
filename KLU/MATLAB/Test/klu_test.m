function klu_test (nmat)
%klu_test KLU test
% Example:
%   klu_test
%
% See also klu

% KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
% Authors: Timothy A. Davis and Ekanathan Palamadai.
% SPDX-License-Identifier: LGPL-2.1+

if (nargin < 1)
    nmat = 200 ;
end

test1 (nmat) ;
test2 (nmat) ;
test3 ;
test4 (nmat) ;
test5  ;
