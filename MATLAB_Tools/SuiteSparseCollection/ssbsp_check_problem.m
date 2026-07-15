function ssbsp_check_problem (Problem, bspfile)
%SSBSP_CHECK_PROBLEM compare a Binsparse file with a MATLAB Problem struct
%
%   ssbsp_check_problem (Problem, bspfile)
%
% Checks the primary matrix, explicit zero pattern, b, x, and supported aux
% components.  This is the component-level checker used by the collection BSP
% experiment; it does not reconstruct or compare the complete Problem metadata.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (nargin ~= 2)
    error ('SuiteSparse:ssbsp_check_problem:InvalidArguments', ...
        'usage: ssbsp_check_problem (Problem, bspfile)') ;
end
ssbsp_test_collection ('check_problem', Problem, bspfile) ;
