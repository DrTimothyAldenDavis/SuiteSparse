function reset_rand
%RESET_RAND resets the state of rand
%
% Example
%   reset_rand
%
% See also RandStream, rand, rng

% Factorize, Copyright (c) 2011-2012, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

if (verLessThan ('matlab', '7.12'))
    rand ('seed', 0) ;                                                      %#ok
else
    rng ('default') ;
end
