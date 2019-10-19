SJget:  a MATLAB interface to the SJSU singular matrix collection.
MATLAB 7.0 or later is required.

Date: March 20, 2008.

Availability: http://www.math.sjsu.edu/singular/matrices/software/SJget

Derived from the ssget toolbox on March 18, 2008:
  Copyright 2005-2007, Tim Davis, University of Florida.
  Authors: Tim Davis and Erich Mirable.
  Availability: http://www.suitesparse.com

See http://www.math.sjsu.edu/singular/matrices/software/SJget.tar.gz
for a single archive file with all the files listed below.

Files of most interest:
    SJget/SJget.m		    primary user interface,
                            SJget will get summary information
                            for all matrices or detailed
                            information for a single matrix
    SJget/SJget_install.m	installation for SJget

additional files of interest
    SJget/Contents.m		help for SJget
    SJget/README.txt		this file
    SJget/SJget_example.m	demo for SJget
    SJget/SJplot.m          picture a plot of the full or partial
                            singular value spectrum
    SJget/SJweb.m		    opens the URL for a matrix or collection
    SJget/SJrank.m          calculates numerical rank for a specified
                            tolerance using precomputed singular values

some additional utilities:
    SJget/SJget_defaults.m	default parameter settings for SJget
    SJget/SJget_lookup.m	get the group, name, and id of a matrix
    SJget/SJgrep.m          search for matrices in the SJSU Singular 
                            Matrix Collection.


To install the package, after unzipping the files type
           SJget_install
Remember to save your path so that it includes SJget directory.
Type "pathtool" or "help savepath" in MATLAB for more details.

For an example of use, type this command in MATLAB:

    SJget_example

The MATLAB statement

    Problem = SJget ('HB/ash292')

(for example), will download a singular matrix called HB/ash292 (a least
squares problem) and load it into MATLAB.  You don't need to use your
web browser to load the matrix.  The statement

    Problem = SJget (52)

will also load same the HB/ash292 matrix.  Each matrix in the collection has
a unique identifier, in the range 1 to the number of matrices.  As new
matrices are added, the id's of existing matrices will not change.

To view an index of the entire collection, just type

    SJget

in MATLAB.  To modify your download directory, edit the SJget_defaults.m file.

To open a URL of the entire collection, just type

    SJweb

To open the URL of a group of matrices in the collection:

    SJweb ('HB')

To open the web page for one matrix, use either of these formats:

    SJweb ('HB/ash292')
    SJweb (52)

For more information on how some of the index entries were created, see
http://www.math.sjsu.edu/singular/matrices/SJsingular.html
and http://www.suitesparse.com.
