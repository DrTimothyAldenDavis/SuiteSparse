function spqr_install (metis_path, tbb)
%SPQR_INSTALL compile and install SuiteSparseQR
%
% Example:
%   spqr_install                        % compiles using ../../metis-4.0, no TBB
%   spqr_install ('/my/metis')          % using non-default path to METIS
%   spqr_install ('no metis')           % do not use METIS at all
%   spqr_install ('../../metis-4.0',1)  % with METIS, and TBB multithreading
%
% SuiteSparseQR relies on CHOLMOD, AMD, and COLAMD, and can optionally use
% CCOLAMD, CAMD, and METIS as well.  By default, CCOLAMD, CAMD,
% and METIS are used.  METIS is assumed to be in the ../../metis-4.0 directory.
%
% See http://www-users.cs.umn.edu/~karypis/metis for a copy of METIS 4.0.1.
%
% You can only use spqr_install while in the SuiteSparseQR/MATLAB directory.
%
% Multithreading based on Intel Threading Building Blocks (TBB) is optionally
% used. It is not enabled by default since it conflicts with multithreading in
% the BLAS.
%
% See also spqr, spqr_solve, spqr_qmult.

%   Copyright 2008, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse

if (nargin < 1)
    metis_path = '../../metis-4.0' ;
end

if (nargin < 2)
    tbb = 0 ;
end

% compile SuiteSparseQR and add to the path
spqr_make (metis_path, tbb) ;
spqr_path = pwd ;
addpath (spqr_path)

fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', spqr_path) ;

fprintf ('\nTo try your new mexFunctions, try this command:\n') ;
fprintf ('spqr_demo\n') ;


