function amd_demo
% AMD DEMO
%
% A demo of AMD for MATLAB.
%
% --------------------------------------------------------------------------
% AMD Version 2.0, Copyright (c) 2006 by Timothy A. Davis,
% Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.
% email: davis at cise.ufl.edu    CISE Department, Univ. of Florida.
% web: http://www.cise.ufl.edu/research/sparse/amd
% --------------------------------------------------------------------------
%
% See also: amd, amd_make

% This orders the same matrix as the ANSI C demo, amd_demo.c.  It includes an
% additional analysis of the matrix via MATLAB's symbfact routine.

% First, print the help information for AMD
help amd

% Get the Harwell/Boeing can_24 matrix.

load can_24
A = spconvert (can_24) ;

n = size (A,1) ;

figure (1)
clf
hold off
subplot (2,2,1) ;
spy (A)
title ('HB/can24 matrix') ;

% print the details during AMD ordering and SYMBFACT
spparms ('spumoni', 1) ;

% order the matrix.  Note that the Info argument is optional.
fprintf ('\nIf the next step fails, then you have\n') ;
fprintf ('not yet compiled the AMD mexFunction.\n') ;
[p, Info] = amd (A) ;

% order again, but this time print some statistics
[p, Info] = amd (A, [10 1 1]) ;

fprintf ('Permutation vector:\n') ;
fprintf (' %2d', p) ;
fprintf ('\n\n') ;

subplot (2,2,2) ;
spy (A (p,p))
title ('Permuted matrix') ;

% The amd_demo.c program stops here.

fprintf ('Analyze A(p,p) with MATLAB''s symbfact routine:\n') ;
[cn, height, parent, post, R] = symbfact (A (p,p)) ;

subplot (2,2,3) ;
spy (R') ; 
title ('Cholesky factor, L') ;

subplot (2,2,4) ;
treeplot (parent) ;
title ('elimination tree') ;

% results from symbfact
lnz = sum (cn) ;                % number of nonzeros in L, incl. diagonal
cn = cn - 1 ;                   % get the count of off-diagonal entries
fl = n + sum (cn.^2 + 2*cn) ;   % flop count for chol (A (p,p)
fprintf ('number of nonzeros in L (including diagonal):      %d\n', lnz) ;
fprintf ('floating point operation count for chol (A (p,p)): %d\n', fl) ;

% approximations from amd:
lnz2 = n + Info (10) ;
fl2 = n + Info (11) + 2 * Info (12) ;
fprintf ('\nResults from AMD''s approximate analysis:\n') ;
fprintf ('number of nonzeros in L (including diagonal):      %d\n', lnz2) ;
fprintf ('floating point operation count for chol (A (p,p)): %d\n\n', fl2) ;

if (lnz2 ~= lnz | fl ~= fl2)
    fprintf ('Note that the nonzero and flop counts from AMD are slight\n') ;
    fprintf ('upper bounds.  This is due to the approximate minimum degree\n');
    fprintf ('method used, in conjunction with "mass elimination".\n') ;
    fprintf ('See the discussion about mass elimination in amd.h and\n') ;
    fprintf ('amd_2.c for more details.\n') ;
end

% turn off diagnostic output in MATLAB's sparse matrix routines
spparms ('spumoni', 0) ;
