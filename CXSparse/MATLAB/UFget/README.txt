UFget:  MATLAB and Java interfaces to the UF sparse matrix collection.
Copyright 2005-2012, Timothy A. Davis, http://www.suitesparse.com

REQUIREMENTS:

    Java JRE 1.6.0 or later is required for the UFgui Java program.
    UFget requires MATLAB 7.0.  A few of the largest matrices require MATLAB
    7.3 or later.

See http://www.suitesparse.com
for a single archive file with all the files listed below:

    UFget/README.txt            this file
    UFget/UFsettings.txt        default settings for Java and MATLAB

    for Java:
    UFget/UFgui.java            a stand-alone Java interface to the collection
    UFget/UFgui.jar             the compiled UFgui program
    UFget/UFhelp.html           help for UFgui
    UFget/Makefile              for compiling UFgui.java into UFgui.jar
    UFget/matrices/UFstats.txt  matrix statistics file for UFgui.java

    for MATLAB:
    UFget/Contents.m            help for UFget in MATLAB
    UFget/UFget_defaults.m      default parameter settings for UFget
    UFget/UFget_example.m       demo for UFget
    UFget/UFget_lookup.m        get the group, name, and id of a matrix
    UFget/UFget.m               primary user interface
    UFget/UFgrep.m              searches for matrices by name
    UFget/UFkinds.m             returns the 'kind' for all matrices
    UFget/UFweb.m               opens the URL for a matrix or collection
    UFget/mat/UF_Index.mat      index to the UF sparse matrix collection

    download directories:
    UFget/MM                    for Matrix Market files
    UFget/RB                    for Rutherford/Boeing files
    UFget/mat                   for *.mat files
    UFget/matrices              for *.png icon images of the matrices

For the Java UFgui program:

    To run the UFgui on Windows or Mac OS X, just double-click the UFgui.jar
    file.  Or, on any platform, type the following command in your command
    window:

        java -jar UFgui.jar

    If that doesn't work, then you need to install the Java JDK or JRE and add
    it to your path.  See http://java.sun.com/javase/downloads/index.jsp and
    http://java.sun.com/javase/6/webnotes/install/index.html for more
    information.  For Ubuntu, you can install Java by running the Synaptics
    package manager.

    The UFgui.jar file is the compiled version of UFgui.java.  If you modify
    UFgui.java, you can recompile it (for Unix/Linux/Mac/Solaris) by typing
    the command:

        make

    To run the UFgui in Linux/Solaris/Unix, your window manager might support
    double-clicking the UFgui.jar file as well.  If not, type either of
    the following commands:

        java -jar UFgui.jar

    or

        make run
        

For the UFget.m MATLAB interface:

    To install the MATLAB package, just add the path containing the UFget
    directory to your MATLAB path.  Type "pathtool" in MATLAB for more details.

    For a simple example of use, type this command in MATLAB:

        UFget_example

    The MATLAB statement

        Problem = UFget ('HB/arc130')

    (for example), will download a sparse matrix called HB/arc130 (a laser
    simulation problem) and load it into MATLAB.  You don't need to use your
    web browser to load the matrix.  The statement

        Problem = UFget (6)

    will also load same the HB/arc130 matrix.  Each matrix in the collection
    has a unique identifier, in the range 1 to the number of matrices.  As new
    matrices are added, the id's of existing matrices will not change.

    To view an index of the entire collection, just type

        UFget

    in MATLAB.  To modify your download directory, edit the UFget_defaults.m
    file (note that this will not modify the download directory for the
    UFgui java program, however).

    To open a URL of the entire collection, just type

        UFweb

    To open the URL of a group of matrices in the collection:

        UFweb ('HB')

    To open the web page for one matrix, use either of these formats:

        UFweb ('HB/arc130')
        UFweb (6)

    To download a new index, to get access to new matrices:

        UFget ('refresh')

    (by default, using UFget downloads the index every 90 days anyway).

    To search for matrices

For more information on how the matrix statistics were created, see
http://www.suitesparse.com.
