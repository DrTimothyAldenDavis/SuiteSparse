% ktest: test KLU
% Example:
%   ktest

matrices = {'Asmall', 'Amiddle', 'Alarge'} ;
symtol = 1e-6 ;

for i = 1:length(matrices)

    fprintf ('\n\nLoading matrix: %s\n', matrices {i}) ;
    load (matrices {i}) ;
    A=A' ;
    pack

    % solve A*x = y with KLU + BTF + AMD

% symtol: partial pivoting tolerance.  Use 1e-3 (for example) to prefer diagonal
% pivoting.

    fprintf ('calling klu\n') ;
    tic
    [x, Info] = klus (A, y, symtol, 'print info') ;
    t = toc ;

%
% Info (1): n
% Info (2): nz in off diagonal part
% Info (3): # of blocks
% Info (4): max nz in diagonal blocks of A
% Info (5): dimension of largest block
% Info (6): estimated nz in L, incl. diagonal, excludes off-diagonal entries
% Info (7): estimated nz in U, incl. diagonal, excludes off-diagonal entries
%
% Info (8): nz in L, including diagonal, excludes off-diagonal entries
% Info (9): nz in U, including diagonal, excludes off-diagonal entries
% Info (10): analyze cputime
% Info (11): factor cputime
% Info (12): solve cputime
% Info (13): refactorize cputime (if computed)
% Info (14): # off-diagonal pivots chosen
%
% b may be n-by-m with m > 1.  It must be dense.
%

    fprintf ('total wallclock time %g\n', t) ;
    fprintf ('Info (2): %g   nz in offdiag part\n', Info (2)) ;
    fprintf ('Info (3): %g # of blocks\n', Info (3)) ;
    fprintf ('Info (4): %g max nz in diagonal blocks of A\n', Info (4)) ;
    fprintf ('Info (5): %g dimension of largest block\n', Info (5)) ;
    fprintf ('Info (6): %g estimated nz in L, incl. diagonal, excludes off-diagonal entries\n', Info (6)) ;
    fprintf ('Info (7): %g estimated nz in U, incl. diagonal, excludes off-diagonal entries\n', Info (7)) ;
    fprintf ('Info (8): %g nz in L, including diagonal, excludes off-diagonal entries\n', Info (8)) ;
    fprintf ('Info (9): %g nz in U, including diagonal, excludes off-diagonal entries\n', Info (9)) ;
    fprintf ('Info (10): %g analyze cputime\n', Info (10)) ;
    fprintf ('Info (11): %g factor cputime\n', Info (11)) ;
    fprintf ('Info (12): %g solve cputime\n', Info (12)) ;
    fprintf ('Info (13): %g refactorize cputime (if computed)\n', Info (13)) ;
    fprintf ('Info (14): %g # off-diagonal pivots chosen\n', Info (14)) ;

end

    
