function btf_make
%BTF_MAKE compile BTF for use in MATLAB
% Your current working directory must be BTF/MATLAB for this function to work.
%
% Example:
%   btf_make
%
% See also btf, maxtrans, stroncomp, dmperm.

% Copyright 2004-2007, Tim Davis, University of Florida

details = 0 ;       % if 1, print details of each command

mexcmd = 'mex -O -DDLONG -I../Include -I../../UFconfig ' ;
if (~isempty (strfind (computer, '64')))
    mexcmd = [mexcmd '-largeArrayDims '] ;
end

s = [mexcmd 'maxtrans.c ../Source/btf_maxtrans.c'] ;
if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;

s = [mexcmd 'strongcomp.c ../Source/btf_strongcomp.c'] ;
if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;

s = [mexcmd 'btf.c ../Source/btf_maxtrans.c ../Source/btf_strongcomp.c ../Source/btf_order.c'] ;
if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;

fprintf ('BTF successfully compiled.\n') ;
