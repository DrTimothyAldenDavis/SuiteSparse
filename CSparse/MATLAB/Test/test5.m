function test5
%TEST5 test cs_add
%
% Example:
%   test5
% See also: testall

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

rand ('state', 0) ;

for trial = 1:100
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = rand (1) ;
    A = sprandn (m,n,d) ;
    B = sprandn (m,n,d) ;

    C = A+B ;
    D = cs_add (A,B) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    C = pi*A+B ;
    D = cs_add (A,B,pi) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    C = pi*A+3*B ;
    D = cs_add (A,B,pi,3) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end

end
