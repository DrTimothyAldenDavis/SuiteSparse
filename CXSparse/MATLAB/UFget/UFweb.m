function stats = UFweb (matrix, opts)
%UFWEB opens the URL for a matrix.
%
%   UFweb(matrix) opens the URL for a matrix.  This parameter can be a string,
%   or an integer.  If it is a string with no "/" character, the web page for a
%   matrix group is displayed.  With no arguments, a list of all the matrix
%   groups is displayed.
%
%   Example:
%
%   If Problem = UFget ('HB/arc130'), the first four examples display
%   the same thing, the web page for the HB/arc130 matrix:
%
%       UFweb (6)
%       UFweb ('HB/arc130')
%       stats = UFweb (6)
%
%   The latter also returns statistics about the matrix or matrix group.
%   To display the web page for the HB (Harwell-Boeing) group:
%
%       UFweb ('HB')
%
%   To display the home page for the UF sparse matrix collection:
%
%       UFweb
%       UFweb (0)
%       UFweb ('')
%
%   The latter two are useful if a second optional parameter is specified.
%   The second optional argument is a string passed as additional parameters to
%   the MATLAB web command.  To use the system web browser instead of the MATLAB
%   browser, for example, use UFweb ('HB/arc130', '-browser').
%
%   See also web, UFget, UFget_defaults.

%   Copyright 2008, Tim Davis, University of Florida.

params = UFget_defaults ;
UF_Index = UFget ;

if (nargin < 1)
    matrix = '' ;
end
if (nargin < 2)
    opts = '' ;
end

% get the matrix group, name, and id
[group name id] = UFget_lookup (matrix, UF_Index) ;

url = params.url ;
len = length (url) ;
if (strcmp (url ((len-3):len), '/mat'))
    % remove the trailing '/mat'
    url = url (1:(len-4)) ;
end

% open the web page for the matrix, group, or whole collection
if (id == 0)
    if (isempty (group))
        eval (['web ' url '/matrices/index.html ' opts])
    else
        eval (['web ' url '/matrices/' group '/index.html ' opts])
    end
else
    eval (['web ' url '/matrices/' group '/' name '.html ' opts])
end

% return stats
if (nargout > 0)

    if (id == 0)

        if (isempty (group))

            % return stats about the whole collection
            stats.nmatrices = length (UF_Index.nrows) ;
            stats.LastRevisionDate = UF_Index.LastRevisionDate ;
            stats.DownloadTime = datestr (UF_Index.DownloadTimeStamp) ;

        else

            % return stats about one matrix group
            nmat = length (UF_Index.nrows) ;
            ngroup = 0 ;
            for i = 1:nmat
                if (strcmp (group, UF_Index.Group {i}))
                    ngroup = ngroup + 1 ;
                end
            end
            stats.nmatrices = ngroup ;
            stats.LastRevisionDate = UF_Index.LastRevisionDate ;
            stats.DownloadTime = datestr (UF_Index.DownloadTimeStamp) ;

        end
    else

        % look up the matrix statistics
        stats.Group = group ;
        stats.Name = name ;
        stats.nrows = UF_Index.nrows (id) ;
        stats.ncols = UF_Index.ncols (id) ;
        stats.nnz = UF_Index.nnz (id) ;
        stats.nzero = UF_Index.nzero (id) ;
        stats.pattern_symmetry = UF_Index.pattern_symmetry (id) ;
        stats.numerical_symmetry = UF_Index.numerical_symmetry (id) ;
        stats.isBinary = UF_Index.isBinary (id) ;
        stats.isReal = UF_Index.isReal (id) ;

        stats.nnzdiag = UF_Index.nnzdiag (id) ;
        stats.posdef = UF_Index.posdef (id) ;

        stats.amd_lnz = UF_Index.amd_lnz (id) ;
        stats.amd_flops = UF_Index.amd_flops (id) ;
        stats.amd_vnz = UF_Index.amd_vnz (id) ;
        stats.amd_rnz = UF_Index.amd_rnz (id) ;
        stats.metis_lnz = UF_Index.metis_lnz (id) ;
        stats.metis_flops = UF_Index.metis_flops (id) ;
        stats.metis_vnz = UF_Index.metis_vnz (id) ;
        stats.metis_rnz = UF_Index.metis_rnz (id) ;
        stats.nblocks = UF_Index.nblocks (id) ;
        stats.sprank = UF_Index.sprank (id) ;
        stats.nzoff = UF_Index.nzoff (id) ;
        stats.dmperm_lnz = UF_Index.dmperm_lnz (id) ;
        stats.dmperm_unz = UF_Index.dmperm_unz (id) ;
        stats.dmperm_flops = UF_Index.dmperm_flops (id) ;
        stats.dmperm_vnz = UF_Index.dmperm_vnz (id) ;
        stats.dmperm_rnz = UF_Index.dmperm_rnz (id) ;

        stats.RBtype = UF_Index.RBtype (id,:) ;
        stats.cholcand = UF_Index.cholcand (id) ;
        stats.ncc = UF_Index.ncc (id) ;

    end
end
