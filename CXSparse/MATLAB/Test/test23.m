function test23
%TEST23 test cs_dmspy
%
% Example:
%   test23
% See also: testall

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

clear functions

randn ('state', 0) ;
rand ('state', 0) ;

clf

for trials = 1:1000

    % m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    m = n ;
    % d = 0.1 * rand (1) ;
    d = rand (1) * 4 * max (m,n) / max (m*n,1) ;
    A = sprandn (m,n,d) ;
    % S = sprandn (m,m,d) + speye (m) ;

    if (~ispc)
        if (rand ( ) > .5)
            A = A + 1i * sprand (A) ;
        end
    end

    cs_dmspy (A) ;
    drawnow

    % pause
end
