%METIS_GRAPH_INSTALL installs metis_graph for use in MATLAB
% Your current directory must be metis_graph/private.  metis_graph is compiled,
% and the metis_graph/ directory is added to your path.
%
% Example
%
%   metis_graph_install
%
% See also metis_graph_read, metis_graph_test

% Copyright 2011, Tim Davis

mex metis_graph_read_mex.c
here = pwd ;
cd ..
addpath (pwd) ;
try
    savepath (pwd) ;
catch me
    disp (me.message) ;
    fprintf ('Unable to save path.  Add this to your startup.m instead:\n') ;
    fprintf ('addpath (%s) ;\n', pwd) ;
end
cd (here) ;
metis_graph_test
