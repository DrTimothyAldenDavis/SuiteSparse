function UFpage (matrix, index, figures)
%UFPAGE create web page for a matrix in UF Sparse Matrix Collection
%
% Usage:
%      UFpage (matrix, index, figures)
%
% matrix: id or name of matrix to create the web page for.
% index: the UF index, from UFget.
% figures: 1 if the figures are to be created, 0 otherwise
%
% Example:
%
%   UFpage (267)
%   UFpage ('HB/west0479')
%
% See also UFget, cspy, UFgplot, UFint.

% This function assumes that the mat/, MM/, and RB/ directories all reside in
% the same parent directory, given by the download directory specified by
% UFget_defaults.

% Copyright 2006-2007, Timothy A. Davis

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargin < 2)
    index = UFget ;
end
if (nargin < 3)
    figures = 1 ;
end

%-------------------------------------------------------------------------------
% get the Problem and its contents
%-------------------------------------------------------------------------------

Problem = UFget (matrix,index) ;
disp (Problem) ;
fullname = Problem.name ;
s = strfind (fullname, '/') ;
grp = fullname (1:s-1) ;
name = fullname (s+1:end) ;
id = Problem.id ;

% create the primary directory
[url topdir] = UFlocation ;
matrices = [topdir 'matrices'] ;
if (~exist (matrices, 'dir'))
    mkdir (matrices) ;
end

% create the group directory
if (~exist ([matrices filesep grp], 'dir'))
    mkdir ([matrices filesep grp]) ;
end

% determine the full path of the problem
fullpath = regexprep ([matrices filesep fullname], '[\/\\]', filesep) ;

ptitle = Problem.title ;

z = 0 ;
if (isfield (Problem, 'Zeros'))
    z = nnz (Problem.Zeros) ;
    Problem = rmfield (Problem, 'Zeros') ;
end

nblocks = index.nblocks (id) ;
ncc = index.ncc (id) ;

has_b = isfield (Problem, 'b') ;
has_x = isfield (Problem, 'x') ;
has_aux = isfield (Problem, 'aux') ;

if (has_b)
    b = Problem.b ;
    Problem = rmfield (Problem, 'b') ;
    if (iscell (b))
	b = sprintf ('cell %d-by-%d\n', size (b)) ;
    elseif (issparse (b))
	b = sprintf ('sparse %d-by-%d\n', size (b)) ;
    else
	b = sprintf ('full %d-by-%d\n', size (b)) ;
    end
end

if (has_x)
    x = Problem.x ;
    Problem = rmfield (Problem, 'x') ;
    if (iscell (x))
	x = sprintf ('cell %d-by-%d\n', size (x)) ;
    elseif (issparse (x))
	x = sprintf ('sparse %d-by-%d\n', size (x)) ;
    else
	x = sprintf ('full %d-by-%d\n', size (x)) ;
    end
end

nodename = [ ] ;
if (has_aux)
    aux = Problem.aux ;
    Problem = rmfield (Problem, 'aux') ;
    auxfields = fields (aux) ;
    has_coord = isfield (aux, 'coord') ;
    has_nodename = isfield (aux, 'nodename') ;
    auxs = cell (1, length (auxfields)) ;
    for k = 1:length(auxfields)
	siz = size (aux.(auxfields{k})) ;
	if (iscell (aux.(auxfields{k})))
	    auxs {k} = sprintf ('cell %d-by-%d\n', siz) ;
	elseif (issparse (aux.(auxfields{k})))
	    auxs {k} = sprintf ('sparse %d-by-%d\n', siz) ;
	else
	    auxs {k} = sprintf ('full %d-by-%d\n', siz) ;
	end
    end
    if (has_coord)
	xyz = aux.coord ;
    end
    if (has_nodename)
	nodename = aux.nodename ;
    end
    clear aux
else
    has_coord = 0 ;
end

kind = Problem.kind ;
if (isfield (Problem, 'notes'))
    notes = Problem.notes ;
else
    notes = '' ;
end

au = Problem.author ;
ed = Problem.ed ;
da = Problem.date ;

m = index.nrows (id) ;
n = index.ncols (id) ;
nz = index.nnz (id) ;
nnzdiag = index.nnzdiag (id) ;

if (strfind (kind, 'graph'))
    bipartite = ~isempty (strfind (kind, 'bipartite')) ;
    directed = ~isempty (regexp (kind, '\<directed', 'once')) ;
else
    bipartite = (m ~= n) ;
    directed = (index.pattern_symmetry (id) < 1) ;
end

%-------------------------------------------------------------------------------
% create the pictures
%-------------------------------------------------------------------------------

if (figures)

    try
	A = Problem.A ;
    catch
	fprintf ('failed to extract A from Problem struct\n') ;
	A = sparse (0) ;
    end
    clear Problem

    %---------------------------------------------------------------------------
    % create the gplot
    %---------------------------------------------------------------------------

    do_gplot = has_coord ;
    if (do_gplot)
	UFgplot (A, xyz, directed, nodename) ;
	print (gcf, '-dpng', '-r128', [fullpath '_gplot.png']) ;
	print (gcf, '-dpng', '-r512', [fullpath '_gplot_big.png']) ;
    end

    %---------------------------------------------------------------------------
    % create the thumbnail picture
    %---------------------------------------------------------------------------

    cspy (A, 16) ;
    print (gcf, '-dpng', '-r12', [fullpath '_thumb.png']) ;

    %---------------------------------------------------------------------------
    % create the regular picture
    %---------------------------------------------------------------------------

    cspy (A, 128) ;
    print (gcf, '-dpng', '-r64', [fullpath '.png']) ;

    %---------------------------------------------------------------------------
    % create the dmperm figure, but not for graphs
    %---------------------------------------------------------------------------

    do_dmspy = (nblocks > 1) & (isempty (strfind (kind, 'graph'))) ;
    if (do_dmspy)
	try
	    cs_dmspy (A, 128) ;
	    title ('Dulmage-Mendelsohn permutation') ;
	catch
	    fprintf ('dmspy failed\n') ;
	    delete ([fullpath '_dmperm.png']) ;
	    do_dmspy = 0 ;
	end
	print (gcf, '-dpng', '-r64', [fullpath '_dmperm.png']) ;
    end

    %---------------------------------------------------------------------------
    % create the ccspy figure
    %---------------------------------------------------------------------------

    do_scc = (ncc > 1) ;
    if (do_dmspy && m == n && nnzdiag == n)
	% don't do scc for a square matrix with zero-free diagonal
	do_scc = 0 ;
    end
    if (do_scc)
	try
	    ccspy (A, bipartite, 128) ;
	    if (bipartite)
		title ('connected components of the bipartite graph') ;
	    else
		title ('strongly connected components of the graph') ;
	    end
	    print (gcf, '-dpng', '-r64', [fullpath '_scc.png']) ;
	catch
	    fprintf ('ccspy failed\n') ;
	    delete ([fullpath '_cc.png']) ;
	    do_scc = 0 ;
	end
    end

else

    %---------------------------------------------------------------------------
    % the plots already exist - check the files
    %---------------------------------------------------------------------------

    do_scc   = exist ([fullpath '_scc.png'], 'file') ;
    do_dmspy = exist ([fullpath '_dmperm.png'], 'file') ;
    do_gplot = exist ([fullpath '_gplot.png'], 'file') ;

end

clear Problem

%-------------------------------------------------------------------------------
% create the web page for the matrix
%-------------------------------------------------------------------------------

f = fopen ([fullpath '.html'], 'w') ;
if (f < 0)
    error ('unable to create matrix web page') ;
end

% add the header
fprintf (f,'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n');
fprintf (f, '<html lang="EN"><head>\n') ;
fprintf (f, '<meta http-equiv="content-type" content="text/html; charset=') ;
fprintf (f, 'iso-8859-1"><title>%s sparse matrix</title></head>\n', fullname) ;
fprintf (f, '<body bgcolor="#ffffff" link="#0021a5">\n') ;

% Yifan Hu's medium-sized graph plot, for all matrices
yifan_graphs = 'http://www.research.att.com/~yifanhu/GALLERY/GRAPHS/' ;
yifan_thumb = [yifan_graphs 'GIF_THUMBNAIL/'] ;
yifan_medium = [yifan_graphs 'GIF_SMALL/'] ;
yname = strrep (fullname, '/', '@') ;
fprintf (f, '\n<p>') ;
fprintf (f, ...
    '<a href="%s%s.html"><img alt="%s graph" src="%s%s.gif"></a>\n\n', ...
    yifan_medium, yname, fullname, yifan_medium, yname) ;

% small gifs go first:
% fprintf (f, '\n\n<p><img alt="%s" src="%s_thumb.png"></a>\n', fullname, name) ;
% 
% 
% fprintf (f, ...
    % '<a href="%s%s.html"><img alt="%s graph" src="%s%s.gif"></a>\n', ...
    % yifan_medium, yname, fullname, yifan_thumb, yname) ;

% matrix name and description
fprintf (f, '<p>Matrix: %s\n', fullname) ;
fprintf (f, '<p>Description: %s<p><hr>\n', ptitle) ;

% link to UF collection
fprintf (f, '<li><a href="..">UF Sparse Matrix Collection</a>\n') ;

% link to group
fprintf (f, '<li><a href="./index.html">Matrix group: %s</a>\n', grp) ;

% add link to mat/<group>/README.txt
fprintf (f, '<li><a href="../../mat/%s/README.txt">', grp) ;
fprintf (f, 'Click here for a description of the %s group.</a>\n', grp) ;

% link to all matrices
fprintf (f, '<li><a href="../list_by_id.html">') ;
fprintf (f, 'Click here for a list of all matrices</a>\n') ;

% link to all groups
fprintf (f, '<li><a href="../groups.html">') ;
fprintf (f, 'Click here for a list of all matrix groups</a>\n') ;

% download link for MATLAB format
fprintf (f, ...
    '<li><a href="../../mat/%s.mat">download as a MATLAB mat-file</a>',...
    fullname) ;
fsize (f, [topdir 'mat/' fullname '.mat']) ; 

fprintf (f, 'Use <a href="%smat/UFget.html">UFget</a>(%d)', url, id) ;
fprintf (f, ' or UFget(''%s'') in MATLAB.\n', fullname) ;

% download link for Matrix Market format
fprintf (f, ...
'<li><a href="../../MM/%s.tar.gz">download in Matrix Market format</a>',...
fullname) ;
fsize (f, [topdir 'MM/' fullname '.tar.gz']) ;

% download link for Rutherford/Boeing format
fprintf (f, ...
'<li><a href="../../RB/%s.tar.gz">download in Rutherford/Boeing format</a>',...
fullname) ;
fsize (f, [topdir 'RB/' fullname '.tar.gz']) ;

%-------------------------------------------------------------------------------
% link to images
%-------------------------------------------------------------------------------

fprintf (f, '\n\n<p><img alt="%s" src="%s.png"></a>\n', fullname, name) ;

% dmspy, if it exists
if (do_dmspy)
    fprintf (f, '\n<p><img alt="dmperm of %s" src="%s_dmperm.png">\n', ...
    fullname, name) ;
end

% ccspy, if it exists
if (do_scc)
    fprintf (f, '<p><img alt="scc of %s" src="%s_scc.png">\n', ...
    fullname, name) ;
end

% gplot, if it exists
if (do_gplot)
    fprintf (f, '<p>') ;
    fprintf (f, ...
'<a href="%s_gplot_big.png"><img alt="%s graph" src="%s_gplot.png"></a>\n', ...
    name, fullname, name) ;
end

%-------------------------------------------------------------------------------
% table of matrix properties
%-------------------------------------------------------------------------------

fprintf (f, '<p><table border=1>\n') ;
stat (f, '<i><a href="../legend.html">Matrix properties</a></i>', '%s', ' ') ;
stat (f, 'number of rows', '%s', UFint (m)) ;
stat (f, 'number of columns', '%s', UFint (n)) ;
stat (f, 'nonzeros', '%s', UFint (nz)) ;

srank = index.sprank (id) ;
if (srank == min (m,n))
    stat (f, 'structural full rank?', '%s', 'yes') ;
else
    stat (f, 'structural full rank?', '%s', 'no') ;
end
stat (f, 'structural rank', '%s', UFint (srank)) ;

stat (f, '# of blocks from dmperm', '%s', UFint (nblocks)) ;
stat (f, '# strongly connected comp.', '%s', UFint (ncc)) ;

if (srank == min (m,n))
    stat (f, 'entries not in dmperm blocks', '%s', ...
	UFint (index.nzoff (id))) ;
end

stat (f, 'explicit zero entries', '%s', UFint (z)) ;

s = index.pattern_symmetry (id) ;
if (s == 1)
    stat (f, 'nonzero pattern symmetry', '%s', 'symmetric') ;
else
    stat (f, 'nonzero pattern symmetry', '%8.0f%%', s*100) ;
end

s = index.numerical_symmetry (id) ;
if (s == -1)
    stat (f, 'numeric value symmetry', '%s', 'unknown') ;
elseif (s == 1)
    stat (f, 'numeric value symmetry', '%s', 'symmetric') ;
else
    stat (f, 'numeric value symmetry', '%8.0f%%', s*100) ;
end

% print the Rutherford/Boeing type
mtype = index.RBtype (id,:) ;
ss = '-' ;
if (mtype (1) == 'r')
    ss = 'real' ;
elseif (mtype (1) == 'c')
    ss = 'complex' ;
elseif (mtype (1) == 'i')
    ss = 'integer' ;
elseif (mtype (1) == 'p')
    ss = 'binary' ;
end
stat (f, 'type', '%s', ss) ;

ss = '-' ;
if (mtype (2) == 'r')
    ss = 'rectangular' ;
elseif (mtype (2) == 'u')
    ss = 'unsymmetric' ;
elseif (mtype (2) == 's')
    ss = 'symmetric' ;
elseif (mtype (2) == 'h')
    ss = 'Hermitian' ;
elseif (mtype (2) == 'z')
    ss = 'skew-symmetric' ;
end
stat (f, 'structure', '%s', ss) ;

if (index.cholcand (id) == 1)
    ss = 'yes' ;
elseif (index.cholcand (id) == 0)
    ss = 'no' ;
else
    ss = '?' ;
end
stat (f, 'Cholesky candidate?', '%s', ss) ;

s = index.posdef (id) ;
if (s == 0)
    ss = 'no' ;
elseif (s == 1)
    ss = 'yes' ;
else
    ss = 'unknown' ;
end
stat (f, 'positive definite?', '%s', ss) ;

fprintf (f, '</table><p>\n') ;

%-------------------------------------------------------------------------------
% problem author, ed, kind
%-------------------------------------------------------------------------------

fprintf (f, '<p><table border=1>\n') ;
fprintf (f, '<tr><td>author</td><td align=left>%s</td>\n', au) ;
fprintf (f, '<tr><td>editor</td><td align=left>%s</td>\n', ed) ;
fprintf (f, '<tr><td>date</td><td align=left>%s</td>\n', da) ;
fprintf (f, '<tr><td><a href=../kind.html>kind</a></td><td align=left>%s</td>\n', kind);
s = index.isND (id) ;
if (s == 0)
    ss = 'no' ;
else
    ss = 'yes' ;
end
fprintf (f, '<tr><td>2D/3D problem?</td><td align=left>%s</td>\n', ss) ;
fprintf (f, '</table><p>\n') ;

%-------------------------------------------------------------------------------
% fields
%-------------------------------------------------------------------------------

if (has_b || has_x || has_aux)
    fprintf (f, '<p><table border=1>\n') ;
    stat (f, 'Additional fields', '%s', 'size and type') ;
    if (has_b)
	stat (f, 'b', '%s', b) ;
    end
    if (has_x)
	stat (f, 'x', '%s', x) ;
    end
    if (has_aux)
	for k = 1:length(auxfields)
	    stat (f, auxfields{k}, '%s', char (auxs{k})) ;
	end
    end
    fprintf (f, '</table><p>\n') ;
end

%-------------------------------------------------------------------------------
% Notes
%-------------------------------------------------------------------------------

if (~isempty (notes))
    fprintf (f, '<p>Notes:<p><pre>\n') ;
    for k = 1:size(notes,1)
	fprintf (f, '%s\n', notes (k,:)) ;
    end
    fprintf (f, '</pre>\n') ;
end

%-------------------------------------------------------------------------------
% ordering statistics
%-------------------------------------------------------------------------------

fprintf (f, '<p><table border=1>\n') ;
if (nblocks == 1 || index.nzoff (id) == -2)

    stat (f, ...
    '<i><a href="../legend.html">Ordering statistics:</a></i>', ...
    '%s', '<i>AMD</i>', '<i>METIS</i>') ;

    if (index.amd_lnz (id) > -2)
	stat (f, 'nnz(chol(P*(A+A''+s*I)*P''))', '%s', ...
	    UFint (index.amd_lnz (id)), ...
	    UFint (index.metis_lnz (id))) ;
	stat (f, 'Cholesky flop count', '%7.1e', ...
	    index.amd_flops (id), ...
	    index.metis_flops (id)) ;
	stat (f, 'nnz(L+U), no partial pivoting', '%s', ...
	    UFint (2*index.amd_lnz (id) - min(m,n)), ...
	    UFint (2*index.metis_lnz (id) - min(m,n))) ;
    end

    stat (f, 'nnz(V) for QR, upper bound nnz(L) for LU', '%s', ...
	UFint (index.amd_vnz (id)), ...
	UFint (index.metis_vnz (id))) ;
    stat (f, 'nnz(R) for QR, upper bound nnz(U) for LU', '%s', ...
	UFint (index.amd_rnz (id)), ...
	UFint (index.metis_rnz (id))) ;

else

    stat (f, ...
    '<i><a href="../legend.html">Ordering statistics:</a></i>', ...
    '%s', '<i>AMD</i>', '<i>METIS</i>', '<i>DMPERM+</i>') ;

    if (index.amd_lnz (id) > -2)
	stat (f, 'nnz(chol(P*(A+A''+s*I)*P''))', '%s', ...
	    UFint (index.amd_lnz (id)), ...
	    UFint (index.metis_lnz (id)), ...
	    UFint (index.dmperm_lnz (id))) ;
	stat (f, 'Cholesky flop count', '%7.1e', ...
	    index.amd_flops (id), ...
	    index.metis_flops (id), ...
	    index.dmperm_flops (id)) ;
	stat (f, 'nnz(L+U), no partial pivoting', '%s', ...
	    UFint (2*index.amd_lnz (id) - min(m,n)), ...
	    UFint (2*index.metis_lnz (id) - min(m,n)), ...
	    UFint (index.dmperm_lnz (id) + index.dmperm_unz (id)-min(m,n))) ;
    end

    stat (f, 'nnz(V) for QR, upper bound nnz(L) for LU', '%s', ...
	UFint (index.amd_vnz (id)), ...
	UFint (index.metis_vnz (id)), ...
	UFint (index.dmperm_vnz (id))) ;
    stat (f, 'nnz(R) for QR, upper bound nnz(U) for LU', '%s', ...
	UFint (index.amd_rnz (id)), ...
	UFint (index.metis_rnz (id)), ...
	UFint (index.dmperm_rnz (id))) ;

end
fprintf (f, '</table><p>\n') ;

%-------------------------------------------------------------------------------
% note regarding orderings
%-------------------------------------------------------------------------------

if (z > 0)
    fprintf (f, '<p><i>Note that all matrix statistics (except nonzero');
    fprintf (f, ' pattern symmetry) exclude the %d explicit zero entries.\n',z);
    fprintf (f, '<i>\n') ;
end

%-------------------------------------------------------------------------------
% etc ...
%-------------------------------------------------------------------------------

fprintf (f, '<p><p><i>Maintained by Tim Davis</a>, last updated %s.', date) ;
fprintf (f, '<br>Matrix pictures by <a href=') ;
fprintf (f, '"%sCSparse/CSparse/MATLAB/CSparse/cspy.m">cspy</a>, a ', url) ;
fprintf (f, 'MATLAB function in the <a href="%sCSparse">CSparse</a>', url) ;
fprintf (f, ' package.<br>\n') ;
fprintf (f, 'Matrix graphs by Yifan Hu, AT&T Labs Visualization Group.\n') ;
fprintf (f, '</body>\n</html>\n') ;

fclose (f) ;


%-------------------------------------------------------------------------------

function fsize (f, filename)
% fsize: print the filesize
d = dir (regexprep (filename, '[\/\\]', filesep)) ;
if (isempty (d))
    fprintf ('\n') ;
elseif (d.bytes < 1024)
    fprintf (f, ', file size: %4d bytes.\n', d.bytes) ;
elseif (d.bytes > 2^20)
    fprintf (f, ', file size: %8.0f MB.\n', d.bytes / 2^20) ;
else
    fprintf (f, ', file size: %8.0f KB.\n', d.bytes / 2^10) ;
end


%-------------------------------------------------------------------------------

function stat (f, what, format, value1, value2, value3)
% stat: print one row of a table
s = val (format, value1) ;
fprintf (f, '<tr><td>%s</td><td align=right>%s</td>\n', what, s) ;
if (nargin > 4)
    fprintf (f, '<td align=right>%s</td>\n', val (format, value2)) ;
end
if (nargin > 5)
    fprintf (f, '<td align=right>%s</td>\n', val (format, value3)) ;
end
fprintf (f, '</tr>\n') ;


%-------------------------------------------------------------------------------

function s = val (format, value)
% val: print a value in a table
if (~ischar (value) && value < 0)
    s = '-' ;
else
    s = sprintf (format, value) ;
end
