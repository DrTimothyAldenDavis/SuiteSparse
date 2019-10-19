function SuiteSparse_install
%SuiteSparse_install: compiles and installs all of SuiteSparse for use in
%   MATLAB.  The SuiteSparse is a Suite of Sparse matrix packages.
%
%   Your current working directory must be SuiteSparse in order to use this
%   function.  Directories are added temporarily your path and javaclasspath.
%   You should add them permanently, using the PATHTOOL.  Add the Java directory
%   for UFget to your classpath.txt, or add a JAVAADDPATH command to your
%   STARTUP M-file.
%
%   See also AMD, COLAMD, CAMD, CCOLAMD, CHOLMOD, UMFPACK, CSPARSE, UFget,
%       SuiteSparse, PATHTOOL, PATH, JAVACLASSPATH, JAVAADDPATH, STARTUP.
%
%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

have_metis = exist ('metis-4.0', 'dir') ;

% print the introduction
help SuiteSparse_install
SuiteSparse = pwd ;
addpath (SuiteSparse) ;
help SuiteSparse
input ('Hit enter to install the SuiteSparse: ') ;

% compile and install AMD and UMFPACK
fprintf ('\n=============================================================\n') ;
fprintf ('=== UMFPACK and AMD =========================================\n') ;
fprintf ('=============================================================\n') ;
cd ([SuiteSparse '/UMFPACK/MATLAB']) ;
umfpack_make

% compile and install AMD, COLAMD, CCOLAMD, CAMD, and CHOLMOD:
fprintf ('\n=============================================================\n') ;
cd ('../../CHOLMOD/MATLAB') ;
if (have_metis)
    fprintf ('=== CHOLMOD (with METIS), AMD, COLAMD, CAMD, CCOLAMD ========\n') ;
    fprintf ('=============================================================\n') ;
    cholmod_make
else
    fprintf ('=== CHOLMOD (without METIS), AMD, COLAMD, CAMD, CCOLAMD =====\n') ;
    fprintf ('=============================================================\n') ;
    cholmod_make ('no metis') ;
end

% compile and install CSparse and UFget
fprintf ('\n=============================================================\n') ;
fprintf ('=== CSPARSE and UFGET =======================================\n') ;
fprintf ('=============================================================\n') ;
cd ([SuiteSparse '/CSparse/MATLAB']) ;
cs_install

% compile and install LDL
fprintf ('\n=============================================================\n') ;
fprintf ('=== LDL =====================================================\n') ;
fprintf ('=============================================================\n') ;
cd ([SuiteSparse '/LDL']) ;
addpath ([SuiteSparse '/LDL']) ;
ldldemo

% post-install wrapup
cd (SuiteSparse)
fprintf ('\n=============================================================\n') ;
fprintf ('SuiteSparse is now installed.  Run pathtool and save your path\n') ;
fprintf ('for future sessions.  Add the directory\n') ;
fprintf ('%s/CSparse/MATLAB/UFget\n', SuiteSparse) ;
fprintf ('to your classpath.txt file:\n') ;
which classpath.txt
fprintf ('or add the command:\n') ;
fprintf ('javaaddpath (''%s/CSparse/MATLAB/UFget'') ;\n', SuiteSparse) ;
fprintf ('to your startup.m file.  Type "doc startup" for more details.\n') ;

