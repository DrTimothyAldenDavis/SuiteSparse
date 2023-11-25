function test27
%TEST27 test nesdis with one matrix (HB/west0479)
% Example:
%   test27
% See also cholmod_test

% Copyright 2006-2023, Timothy A. Davis, All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

fprintf ('=================================================================\n');
fprintf ('test27: test nesdis\n') ;

Prob = ssget ('HB/west0479') ;
testnd (Prob.A) ;

fprintf ('test27 passed\n') ;
