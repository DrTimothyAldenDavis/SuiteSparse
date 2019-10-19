function ssmult_make
%SSMULT_MAKE compiles ssmult
% Note that the "lcc" compiler provided with MATLAB for Windows can generate
% slow code; use another compiler if possible.  Your current directory must be
% SSMULT for ssmult_make to work properly.  If you use Linux/Unix/Mac, I
% recommend that you use COPTIMFLAGS='-O3 -DNDEBUG' in your mexopts.sh file.
%
% Example:
%   ssmult_make
%
% See also ssmult, ssmultsym, ssmult_unsorted, sstest, sstest2, mtimes.

% Copyright 2007, Timothy A. Davis, University of Florida

d = '' ;
if (~isempty (strfind (computer, '64')))
    % 64-bit MATLAB
    d = ' -largeArrayDims -DIS64' ;
end

v = getversion ;
if (v < 6.5)
    % mxIsDouble is false for a double sparse matrix in MATLAB 6.1 or earlier
    d = [d ' -DMATLAB_6p1_OR_EARLIER'] ;
end

cmd = sprintf ('mex -O%s ssmult.c', d) ;
% disp (cmd) ;
eval (cmd) ;

cmd = sprintf ('mex -O%s ssmultsym.c', d) ;
% disp (cmd) ;
eval (cmd) ;

cmd = sprintf ('mex -O%s -DUNSORTED ssmult.c -output ssmult_unsorted', d) ;
% disp (cmd) ;
eval (cmd) ;

fprintf ('SSMULT successfully compiled\n') ;

%-------------------------------------------------------------------------------
function v = getversion
% determine the MATLAB version, and return it as a double.
v = sscanf (version, '%d.%d.%d') ;
v = 10.^(0:-1:-(length(v)-1)) * v ;

