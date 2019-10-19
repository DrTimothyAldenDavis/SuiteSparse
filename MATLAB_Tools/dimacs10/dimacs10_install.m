function dimacs10_install
%DIMACS10_INSTALL compiles and installs dimacs10 for use in MATLAB.
% Your current working directory must be the one containing this
% file (dimacs10/dimacs10_install.m).  Also runs a few short tests.
%
% Example
%
%   dimacs10_install
%
% See also dimacs10, metis_graph_read, UFget.

%   Copyright 2011, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

% compile the mexFunctions
here = pwd ;
addpath (here) ;
cd private
me = [ ] ;
try
    if (~isempty (strfind (computer, '64')))
        fprintf ('Compiling dimacs10 (64-bit)\n') ;
        mex -largeArrayDims convert_to_graph.c
        mex -largeArrayDims metis_graph_read_mex.c
    else
        fprintf ('Compiling dimacs10 (32-bit)\n') ;
        mex convert_to_graph.c
        mex metis_graph_read_mex.c
    end
catch me
end

% run some tests and add dimacs10 to the path
if (isempty (me))
    metis_graph_test ;
    lastwarn ('') ;
    savepath
    ok = isempty (lastwarn) ;
    cd (here) ;
    dimacs10_demo (-1) ;
    dimacs10_demo ([7 15 5 6 109 23 82 57 24]) ;
    if (~ok)
        fprintf ('\nUnable to save path.  Add this line to your startup.m:\n') ;
        fprintf ('\n    addpath %s\n', here) ;
    end
else
    cd (here) ;
    rethrow (me) ;
end

