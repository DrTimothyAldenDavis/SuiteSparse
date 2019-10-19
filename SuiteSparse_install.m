function SuiteSparse_install
%SuiteSparse_install: compiles and installs all of SuiteSparse for use in
%   MATLAB.  SuiteSparse is a Suite of Sparse matrix packages.
%
%   Your current working directory must be SuiteSparse in order to use this
%   function.  Directories are added temporarily your path.
%   You should add them permanently, using the PATHTOOL.
%
%   Example:
%	SuiteSparse_install
%
%   See also AMD, COLAMD, CAMD, CCOLAMD, CHOLMOD, UMFPACK, CSPARSE, UFget,
%       RBio, UFcollection, SuiteSparse, PATHTOOL, PATH, STARTUP.
%
%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

have_metis = exist ('metis-4.0', 'dir') ;

% print the introduction
help SuiteSparse_install
SuiteSparse = pwd ;
addpath (SuiteSparse) ;
help SuiteSparse

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

% compile and install AMD and UMFPACK
fprintf ('\n') ;
fprintf ('==========================================================\n') ;
fprintf ('=== UMFPACK and AMD ======================================\n') ;
fprintf ('==========================================================\n') ;
cd ([SuiteSparse '/UMFPACK/MATLAB']) ;
umfpack_make

% compile and install AMD, COLAMD, CCOLAMD, CAMD, and CHOLMOD:
fprintf ('\n') ;
fprintf ('==========================================================\n') ;
cd ('../../CHOLMOD/MATLAB') ;
if (have_metis)
   fprintf ('=== CHOLMOD (with METIS), AMD, COLAMD, CAMD, CCOLAMD =====\n');
   fprintf ('==========================================================\n');
   cholmod_make
else
   fprintf ('=== CHOLMOD (without METIS), AMD, COLAMD, CAMD, CCOLAMD ==\n');
   fprintf ('==========================================================\n');
   cholmod_make ('no metis') ;
end

% compile and install CSparse and UFget
fprintf ('\n') ;
fprintf ('============================================================\n') ;
fprintf ('=== CSPARSE and UFGET ======================================\n') ;
fprintf ('============================================================\n') ;
cd ([SuiteSparse '/CSparse/MATLAB']) ;
cs_install

% compile and install LDL
fprintf ('\n') ;
fprintf ('============================================================\n') ;
fprintf ('=== LDL ====================================================\n') ;
fprintf ('============================================================\n') ;
cd ([SuiteSparse '/LDL']) ;
addpath ([SuiteSparse '/LDL']) ;
ldldemo

% compile and install RBio
fprintf ('\n') ;
fprintf ('============================================================\n') ;
fprintf ('=== RBio ===================================================\n') ;
fprintf ('============================================================\n') ;
cd ([SuiteSparse '/RBio']) ;
try
    RBinstall ;
catch
    fprintf ('Unable to install RBio (requires a Fortran compiler)\n') ;
end

% compile and install UFcollection
fprintf ('\n') ;
fprintf ('============================================================\n') ;
fprintf ('=== UFcollection ===========================================\n') ;
fprintf ('============================================================\n') ;
cd ([SuiteSparse '/UFcollection']) ;
UFcollection_install ;

% post-install wrapup
cd (SuiteSparse)
fprintf ('\n=============================================================\n') ;
fprintf ('SuiteSparse is now installed.  Run pathtool and save your path\n') ;
fprintf ('for future sessions.\n') ;


