function kinds = UFkinds
%UFKINDS: get 'kind' of each problem in the UF collection.
%
% A problem 'kind' is the domain from which the problem arises.  Examples
% include 'circuit simulation', 'optimization', 'least squares', and 'robotics'.
% See UFget/README.txt for a complete list.
%
% A 'kind' is related to the matrix 'Group'.  The latter is determined by
% person, persons, or organization that created the matrix.  'Group' may also
% refer to another matrix collection that has been included in this collection
% (such as 'HB' for the Harwell/Boeing collection).
%
% kinds = UFkinds returns a cell array with one row per matrix.  kinds{id}
% is the same as Problem.kind where Problem=UFget(id).  If an error occurs,
% an empty cell array is returned.
%
% Example:
%
%   kinds = UFkinds ;
%   Problem = UFget (42)
%   kinds{42}
%
% results in the following output:
%
%   Problem = 
%        title: 'SYMMETRIC STIFFNESS MATRIX - FRAME WITHIN A SUSPENSION BRIDGE'
%            A: [485x485 double]
%         name: 'HB/bcsstk20'
%           id: 42
%         date: '1984'
%       author: 'J. Lewis'
%           ed: 'I. Duff, R. Grimes, J. Lewis'
%         kind: 'structural problem'
%
%   ans =
%       'structural problem'
%
% To find a list t of the id's all matrices from material science:
%
%   kinds = UFkinds ;
%   t = zeros (1, length (kinds)) ;
%   for id = 1:length(kinds)
%       t (id) = ~isempty (strfind (kinds {id}, 'material')) ;
%   end
%   t = find (t) ;
%   for id = t
%       Problem = UFget (id)
%   end
%
% See also UFget.

% Copyright 2009-2012, Timothy A. Davis, http://www.suitesparse.com

params = UFget_defaults ;
statfile = sprintf ('%smatrices/UFstats.csv', params.topdir) ;

try
    f = fopen (statfile, 'r') ;
    nmat = str2double (fgetl (f)) ;     % number of matrices
    s = fgetl (f) ;                     % date the UFstats.csv file was created
    kinds = cell (nmat,1) ;
    for id = 1:nmat
        s = fgetl (f) ;
        t = find (s == ',', 1, 'last') ;
        r = s ((t+1):end) ;
        kinds {id} = r ;
    end
catch
    kinds = cell (0,0) ;                % return empty string on error
end

