SuiteSparseCollection, Copyright 2007-2019, Timothy A. Davis,
http://www.suitesparse.com

SuiteSparseCollection is a MATLAB toolbox for managing the SuiteSparse Matrix
Collection.  If you are a MATLAB user of the collection, you would not normally
need to use this toolbox.  It contains code for creating the index for the
collection (the ss_index.mat file in the ssget package), for creating the web
pages for the collection, and creating the Matrix Market and Rutherford/Boeing
versions of the matrices.  This code is posted here primarily so that users of
the collection can see how the matrices and their statistics were generated.

This software (ssread, specifically) also allows the user to keep a single copy
of the collection for use both inside MATLAB and outside MATLAB.  The MM/ and
RB/ versions of the collection can be read into MATLAB via ssread, even without
explicitly extracting the tar files.  They can also be read by non-MATLAB
programs.  ssread is much slower than ssget, however.

--------------------------------------------------------------------------------
MATLAB help for the SuiteSparseCollection toolbox:
--------------------------------------------------------------------------------

SuiteSparseCollection: software for managing the SuiteSparse Matrix Collection

  To create the index:

    ssindex    - create the index for the SuiteSparse Matrix Collection
    ssstats    - compute matrix statistics for the SuiteSparse Matrix Collection

  To create images for the web pages:

    ssgplot    - draw a plot of the graph of a sparse matrix
    ssint      - print an integer to a string, adding commas every 3 digits
    sslocation - URL and top-level directory of SuiteSparse Matrix Collection
    sspage     - create images for a matrix in SuiteSparse Matrix Collection
    sspages    - create images for matrices SuiteSparse Matrix Collection
    dsxy2figxy - Transform point or position from axis to figure coords

  To create the Matrix Market and Rutherford/Boeing versions of the collection:

    ssexport     - export to Matrix Market and Rutherford/Boeing formats
    ssread       - read a Problem in Matrix Market or Rutherford/Boeing format
    sswrite      - write a Problem in Matrix Market or Rutherford/Boeing format
    ssfull_read  - read a full matrix using a subset of Matrix Market format
    ssfull_write - write a full matrix using a subset of Matrix Market format

  Requires ssget, CSparse, CHOLMOD, AMD, COLAMD, RBio, and METIS.


--------------------------------------------------------------------------------
Files:
--------------------------------------------------------------------------------

    Doc/ChangeLog           changes to the package
    Contents.m		    MATLAB help
    dsxy2figxy.m	    convert XY points for plot annotations
    README.txt		    this file
    ssexport.m		    export to MM and RB
    ss_install.m            installation
    ssfull_read.m	    read a full matrix
    ssfull_write.c	    write a full matrix
    ssfull_write.m	    MATLAB help for ssfull_write
    ssgplot.m		    plot a graph
    ssindex.m		    create ss_index.mat
    sslocation.m	    URL and directory for the collection
    sspage.m		    create a web page for a matrix
    sspages.m		    create web pages for all matrices
    ssread.m		    read a Problem
    ssstats.m		    compute statistics about a matrix
    sswrite.m		    write a Problem

See MATLAB_Tools/Doc/License.txt for the license

--------------------------------------------------------------------------------
To add a matrix to the collection:
--------------------------------------------------------------------------------

These instructions are for the maintainer of the collection (that is, just
notes to myself), but they also indicate how the above software is used.
Requires most of SuiteSparse (ssget, CHOLMOD, AMD, COLAMD, CSparse, RBio, and
SuiteSparseCollection), and METIS.

The location of the primary archive of the collection is
backslash.cse.tamu.edu:/archive/davis/SuiteSparseCollection,
which is only accessible to the maintainers of the collection.

A local copy of ssget should be installed where the mat, MM, RB, and svd
directories are replaced with symbolic links to the directories in
backslash.cse.tamu.edu:/archive/davis/SuiteSparseCollection.  The
ssget/files directory should not be a symbolic link.

1) Get the matrix into MATLAB (method depending on how the matrix was
    submitted).  Use load and sparse2, RBread, mread, or specialized code
    written just for that matrix.

2) Add the full matrix name to the end of the file:
    backslash:/archive/davis/SuiteSparseCollection/files/ss_listing.txt
    Each line in the file has the form Group/Name.  The line number in
    ss_listing.txt must match the matrix Problem.id number, and full name
    must match the Problem.name in the MATLAB struct.

3) Create a new directory
    backslash:/archive/davis/SuiteSparseCollection/mat/Group,
    where Group is the new matrix group.  Add a README.txt file to this
    directory, the first line of which is a one-line summary that will appear
    in the top-level web page for the collection.  Skip this step if adding a
    matrix to an existing group.

4) Create the Problem struct; type "help sswrite" for details.  Required fields:

    Problem.name    full name of the matrix (Group/Name)
    Problem.title   short descriptive title
    Problem.A	    the sparse matrix
    Problem.id	    integer line number in ss_listing.txt
    Problem.date    date the matrix was created, or added to the collection
    Problem.author  matrix author
    Problem.ed	    matrix editor/collector
    Problem.kind    a string.  For a description, see below.

    optional fields:

    Problem.Zeros   binary pattern of explicit zero entries
    Problem.b	    right-hand-side
    Problem.x	    solution
    Problem.notes   a char array
    Problem.aux	    auxiliary matrices (contents are problem dependent)

    Save to a MATLAB mat-file.  In the mat directory, do this in MATLAB:

    save (Problem.name, 'Problem', '-v7') ;

    or for very large problems (ids 1903 and 1905, for example):

    save (Problem.name, 'Problem', '-v7.3') ;

    Move the new *.mat files into
    backslash:/archive/davis/SuiteSparseCollection/mat/Group.

5) Compute matrix statistics and extend the ss_index. Do this in MATLAB:

    ssindex (ids)

    where ids is a list of the new matrix id's.  Updated ss_index.mat and
    ssstats.csv files are placed in the current working directory.
    Move the new ss_index.mat and ssstats.csv files to 
    backslash:/archive/davis/SuiteSparseCollection/files,
    overwriting the old copies there.  Also copy them into
    [path to ssget]/files/ so they will be found by the MATLAB ssget.m.

6) Export the matrix in Matrix Market and Rutherford/Boeing formats.

    ssexport (ids)

    or

    ssexport (ids, 'check')

    then tar and compress the resulting MM/Group/Name and RB/Group/Name
    directories, one per Problem (if ssexport has not already done so).  The
    2nd option with 'check' is slower, but it reads back in the created MM and
    RB formats and ensures they are identical to the MATLAB *.mat files.
    The tar files can be too big for MATLAB so they must be tar.gz'ed
    manually with:

        tar -v -cf - $problem | gzip -9 > $problem.tar.gz

    where $problem is a single problem folder, such as MM/HB/west0067.
    This creates the MM/HB/west0067.tar.gz file.

7) Create the images for the web pages:

    sspages (ids)

8) Make the collection world-readable.  In 
    backslash:/archive/davis/SuiteSparseCollection/ do:

    chmod -R og+rX mat files MM RB

9) reconstruct the sparse.tamu.edu web pages to include the new matrices

--------------------------------------------------------------------------------
Problem.kind
--------------------------------------------------------------------------------

Problems with 2D/3D geometry

    2D/3D problem
    acoustics problem
    computational fluid dynamics problem
    computer graphics/vision problem
    electromagnetics problem
    materials problem
    model reduction problem
    robotics problem
    semiconductor device problem
    structural problem
    thermal problem 

Problems that normally do not have 2D/3D geometry

    chemical process simulation problem
    circuit simulation problem
    counter-example problem: Some of these may have 2D/3D geometry.
    economic problem
    frequency-domain circuit simulation problem
    least squares problem
    linear programming problem
    optimization problem
    power network problem
    statistical/mathematical problem
    theoretical/quantum chemistry problem
    combinatorial problem

Graph problems

    This problem includes the graph or multigraph keyword. It is a network or
    graph. A graph may or may not have 2D/3D geometry (typically it does not).
    Several secondary phrases can be included in Problem.kind:

        directed or undirected: A graph is either directed, undirected, or
        bipartite. (bipartite graphs are always undirected). If not bipartite,
        the matrix will always be square. Unsymmetric permutations of the
        matrix have no meaning. If directed, the edge (i,j) is not the same as
        (j,i), and the matrix will normally be unsymmetric. If undirected, the
        edges (i,j) and (j,i) are the same, and the matrix is always symmetric.

        weighted: If the graph has edge weights, this word will appear. The
        edge weight of edge (i,j) is the value of A(i,j). This phrase is used
        for a graph only, never for a multigraph. If the graph is not weighted,
        the matrix is binary.

        bipartite: If the rows and columns of the matrix reflect different sets
        of nodes. The matrix A is normally rectangular, but can be square. Any
        permutation (unsymmetric or symmetric) of the matrix is meaningful.

        random: This Problem has been randomly generated. It is included in the
        SuiteSparse Matrix Collection only because it has been used as a
        standard benchmark. Randomly generated problems are otherwise excluded
        from the collection.  multigraph or graph: If the matrix represents a
        multigraph, then A(i,j) reflects the number of edges (i,j). The edges
        themselves are always unweighted. If the matrix represents a graph,
        then A(i,j) is either 0 or 1 for an unweighted graph, or the weight of
        edge (i,j) otherwise. 

        temporal:  If the edges have time stamps.  The edges are usually
        held in Problem.aux.temporal_edges, an e-by-3 or by-4 dense matrix,
        if there are e temporal edges.  Each row is a single edge, with
        [source target time] or [source target weight time].

        multigraph:  If there can be multiple edges (i,j) in the graph.
        If the graph is unweighted, then A(i,j) is the number of (i,j) edges.

