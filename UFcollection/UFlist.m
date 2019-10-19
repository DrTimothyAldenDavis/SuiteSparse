function UFlist (what, group)
%UFLIST create a web page index for the UF Sparse Matrix Collection
%
% Usage: UFlist (what)
%
%  what:
%       'group'     sort by group, then filename
%       'name'      sort by name
%       'dimension' sort by max row or col dimension
%       'id'        sort by id (default, if "what" is not present)
%       'number of nonzeros'   sort by number of nonzeros
%       'type'      sort by type, then dimension
%       'symmetry'  sort by symmetry.  Rectangular matrices first, sorted by
%                   min(nrow,ncol)-max(nrow,ncol), then numerical symmetry
%                   0 to just less than one.  Then numerical symmetry = 1
%                   but not spd (sorted by dimension).  Then spd sorted
%                   by dimension.
%
% If two arguments are present, only that group is created.
% In this case, "what" must be "group".
%
% Example:
%
%   UFlist ('id')
%   UFlist ('group', 'HB')
%
% See also UFget, UFint.

% Copyright 2006-2007, Timothy A. Davis

index = UFget ;

if (nargin < 1)
    what = 'id' ;
end

by_group = (nargin > 1) ;

% create the primary directory
[url topdir] = UFlocation ;
matrices = [topdir 'matrices'] ;
if (~exist (matrices, 'dir'))
    mkdir (matrices) ;
end

if (by_group)
    fprintf ('group: %s\n', group) ;
    loc = '../' ;
    if (~exist ([matrices filesep group], 'dir'))
	mkdir ([matrices filesep group]) ;
    end
    f = fopen ([matrices filesep group filesep 'index.html'], 'w') ;
else
    fprintf ('list: %s\n', what) ;
    f = fopen ([matrices filesep 'list_by_' what '.html'], 'w') ;
    loc = '' ;
end
if (f < 0)
    error ('unable to create html file\n') ;
end

nmat = length (index.nrows) ;

% add the header
fprintf (f, ...
    '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n') ;
fprintf (f, '<html lang="EN"><head>\n') ;
fprintf (f, '<meta http-equiv="content-type" content="text/html; charset=') ;
fprintf (f, 'iso-8859-1">') ;
if (by_group)
    fprintf (f, '<title>UF Sparse Matrix Collection - %s group', group) ;
else
    fprintf (f, '<title>UF Sparse Matrix Collection - sorted by ') ;
    if (strcmp (what, 'nnz'))
	fprintf (f, 'number of nonzeros') ;
    else
	fprintf (f, '%s', what) ;
    end
end
fprintf (f, '</title></head>\n') ;
fprintf (f, '<body bgcolor="#ffffff" link="#0021a5">\n') ;
fprintf (f, ...
    '<li><a href="%sindex.html">UF Sparse Matrix Collection</a><hr>\n', loc) ;
if (by_group)
    fprintf (f, '<p>UF Sparse Matrix Collection: %s group.<p>\n', group) ;
else
    fprintf (f, '<p>UF Sparse Matrix Collection, sorted by %s.\n', what) ;
    fprintf (f, '  Click on a column header to sort by that column.<p>\n') ;
end

% add link to mat/<group>/README.txt
if (by_group)
    fprintf (f, '<li><a href="../../mat/%s/README.txt">', group) ;
    fprintf (f, 'Click here for a description of the %s group.</a>\n', group) ;
end

fprintf (f, '<li><a href="%sgroups.html">', loc) ;
fprintf (f, 'Click here for a list of all matrix groups.</a>\n') ;

fprintf (f, '<li><a href="%slist_by_id.html">', loc) ;
fprintf (f, 'Click here for a list of all matrices.</a>\n') ;

fprintf (f, '<p><table>\n') ;

% sort by filename
[ignore, iname] = sort (lower (index.Name)) ;

if (by_group)

    list = [ ] ;
    for i = 1:nmat
	if (strcmp (group, index.Group {i}))
	    list = [list i] ;						    %#ok
	end
    end
    [ignore i] = sort (lower (index.Name (list))) ;
    list = list (i) ;

    if (isempty (list))
	error ('empty group!') ;
    end

elseif (strcmp (what, 'group'))

    % sort by filename, then stable sort by group
    [ignore, i] = sort (lower (index.Group (iname))) ;
    list = iname (i) ;

elseif (strcmp (what, 'name'))

    % sort by filename only
    list = iname' ;

elseif (strcmp (what, 'dimension'))

    % sort by filename, then stable sort by max dimension
    [ignore, i] = sort (max (index.nrows (iname), index.ncols (iname))) ;
    list = iname (i) ;

elseif (strcmp (what, 'id'))

    list = 1:nmat ;

elseif (strcmp (what, 'nnz'))

    % sort by filename, then stable sort by nnz
    [ignore, i] = sort (index.nnz (iname)) ;
    list = iname (i) ;

elseif (strcmp (what, 'symmetry'))

%	'symmetry'  sort by symmetry.  Rectangular matrices first, sorted by
%			min(nrow,ncol)-max(nrow,ncol), then numerical symmetry
%			0 to just less than one.  Then numerical symmetry = 1
%			but not spd (sorted by dimension).  Then spd sorted
%			by dimension.


    s1 = min (index.nrows, index.ncols) - max (index.nrows, index.ncols) ;
    s2 = index.numerical_symmetry ;
    s2 (find (s2) == -1) = 1 ;
    s3 = index.posdef ;
    s3 (find (s3) == -1) = 2 ;
    s4 = max (index.nrows, index.ncols) ;

    [ignore list] = sortrows ([s1' s2' s3' s4'], [1 2 3 4]) ;

elseif (strcmp (what, 'type'))

    [ignore i1] = sort (max (index.nrows, index.ncols)) ;
    s = index.RBtype (i1,:) ;
    [ignore i2] = sortrows (s) ;
    list = i1 (i2) ;

else
    error ('unknown list') ;
end

% ensure list is a row vector
list = list (:)' ;

% print the header
fprintf (f, '<tr>\n') ;

if (by_group)
    fprintf (f, '<th>matrix</th>\n') ;
    fprintf (f, '<th>graph</th>\n') ;
    fprintf (f, '<th>Group/Name</th>\n') ;
    fprintf (f, '<th>id</th>\n') ;
    fprintf (f, '<th>download</th>\n') ;
    fprintf (f, '<th># rows</th>\n') ;
    fprintf (f, '<th># cols</th>\n') ;
    fprintf (f, '<th>nonzeros</th>\n') ;
    fprintf (f, '<th>type</th>\n') ;
    fprintf (f, '<th>sym</th>\n') ;
    fprintf (f, '<th>spd?</th>\n') ;
else
    fprintf (f, '<th>matrix</th>\n') ;
    fprintf (f, '<th>graph</th>\n') ;
    fprintf (f, '<th><a href=%slist_by_group.html>Group</a>\n', loc) ;
    fprintf (f, 'and <a href=%slist_by_name.html>Name</a></th>\n', loc) ;
    fprintf (f, '<th><a href=%slist_by_id.html>id</a></th>\n', loc) ;
    fprintf (f, '<th>download</th>\n') ;
    fprintf (f, '<th><a href=%slist_by_dimension.html># rows</a></th>\n', loc) ;
    fprintf (f, '<th><a href=%slist_by_dimension.html># cols</a></th>\n', loc) ;
    fprintf (f, '<th><a href=%slist_by_nnz.html>nonzeros</a></th>\n', loc) ;
    fprintf (f, '<th><a href=%slist_by_type.html>type</a></th>\n', loc) ;
    fprintf (f, '<th><a href=%slist_by_symmetry.html>sym</a></th>\n', loc) ;
    fprintf (f, '<th><a href=%slist_by_symmetry.html>spd?</a></th>\n', loc) ;
end

yifan_graphs = 'http://www.research.att.com/~yifanhu/GALLERY/GRAPHS/' ;
yifan_thumb = [yifan_graphs 'GIF_THUMBNAIL/'] ;

for id = list

    group = index.Group {id} ;
    name = index.Name {id} ;
    nrows = index.nrows (id) ;
    ncols = index.ncols (id) ;
    nz = index.nnz (id) ;
    sym = index.numerical_symmetry (id) ;
    mtype = index.RBtype (id,:) ;

    s = index.posdef (id) ;
    if (s == 0)
	ss = 'no' ;
    elseif (s == 1)
	ss = 'yes' ;
    else
	ss = '?' ;
    end

    fprintf (f, '<tr>\n') ;

    % thumbnail link to the matrix page
    fprintf (f, '<td>\n') ;
    w = 'width="96" height="72"' ;
    yname = [group '@' name] ;
    if (by_group)

	fprintf (f, '<a href="%s.html"><img %s alt="%s/%s"', ...
            name, w, group, name) ;
	fprintf (f, ' src="%s_thumb.png"></a>\n', name) ;

        fprintf (f, '</td><td>\n') ;
	fprintf (f, '<a href="%s.html"><img alt="%s/%s"', ...
            name, group, name) ;
	fprintf (f, ' src="%s%s.gif"></a>\n', yifan_thumb, yname) ;

    else

	fprintf (f, '<a href="%s/%s.html"><img %s alt="%s/%s"', ...
	    group, name, w, group, name) ;
	fprintf (f, ' src="%s/%s_thumb.png"></a>\n', group, name) ;

        fprintf (f, '</td><td>\n') ;
	fprintf (f, '<a href="%s/%s.html"><img alt="%s/%s"', ...
	    group, name, group, name) ;
	fprintf (f, ' src="%s%s.gif"></a>\n', yifan_thumb, yname) ;

    end

    fprintf (f, '</td>\n') ;

    % group
    if (by_group)
	fprintf (f, '<td>%s/', group) ;
    else
	fprintf (f, '<td><a href="%s/index.html">%s</a>/', group, group);
    end
    
    % name
    if (by_group)
	fprintf (f, '<a href="%s.html">%s</a></td>\n', name, name) ;
    else
	fprintf (f, '<a href="%s/%s.html">%s</a></td>\n', group, name, name) ;
    end

    % id
    fprintf (f, '<td>%d</td>\n', id) ;

    % download links
    fprintf (f, '<td>\n') ;
    fprintf (f, '<a href="%s../mat/%s/%s.mat">MAT</a>', loc, group, name) ;
    fprintf (f, ', <a href="%s../MM/%s/%s.tar.gz">MM</a>', loc, group, name) ;
    fprintf (f, ', <a href="%s../RB/%s/%s.tar.gz">RB</a>', loc, group, name) ;
    fprintf (f, '</td>\n') ;

    % nrow
    fprintf (f, '<td align=right>%s</td>\n', UFint (nrows)) ;

    % ncols
    fprintf (f, '<td align=right>%s</td>\n', UFint (ncols)) ;

    % nz
    fprintf (f, '<td align=right>%s</td>\n', UFint (nz)) ;

    % print the Rutherford/Boeing type
    mattype = '' ;
    if (mtype (1) == 'r')
	mattype = 'real' ;
    elseif (mtype (1) == 'c')
	mattype = 'complex' ;
    elseif (mtype (1) == 'i')
	mattype = 'integer' ;
    elseif (mtype (1) == 'p')
	mattype = 'binary' ;
    end
    if (mtype (2) == 'r')
	mattype = [mattype ' rectangular'] ;				    %#ok
    elseif (mtype (2) == 'u')
	mattype = [mattype ' unsymmetric'] ;				    %#ok
    elseif (mtype (2) == 's')
	mattype = [mattype ' symmetric'] ;				    %#ok
    elseif (mtype (2) == 'h')
	mattype = [mattype ' Hermitian'] ;				    %#ok
    elseif (mtype (2) == 'z')
	mattype = [mattype ' skew-symmetric'] ;				    %#ok
    end
    fprintf (f, '<td>%s</td>\n', mattype) ;

    % numerical symmetry (as a percentage)
    if (sym == -1)
	fprintf (f, '<td align=right>?</td>\n') ;
    elseif (sym == 1)
	fprintf (f, '<td align=right>yes</td>\n') ;
    elseif (nrows ~= ncols)
	fprintf (f, '<td align=right>-</td>\n') ;
    else
	if (sym > 0 && sym < 0.01)
	    fprintf (f, '<td align=right>%5.2f%%</td>\n', sym * 100) ;
	else
	    fprintf (f, '<td align=right>%5.0f%%</td>\n', sym * 100) ;
	end
    end

    % positive definite?
    fprintf (f, '<td>%s</td>\n', ss) ;

    fprintf (f, '</tr>\n\n') ;
end

fprintf (f, '</table>\n\n') ;

fprintf (f, '<p><p><i>Maintained by Tim Davis, last updated %s.', date) ;
fprintf (f, '<br>Matrix pictures by <a href=') ;
fprintf (f, '"%sCSparse/CSparse/MATLAB/CSparse/cspy.m">cspy</a>, a MATLAB', ...
    url) ;
fprintf (f, ' function in the <a href="%sCSparse">CSparse</a> package.\n', ...
    url) ;
fprintf (f, '<br>See <a href="%smat/UFget">UFget</a> to download directly', ...
    url) ;
fprintf (f, ' into MATLAB.') ;
fprintf (f, '</body>\n') ;
fprintf (f, '</html>\n') ;

fclose (f) ;
