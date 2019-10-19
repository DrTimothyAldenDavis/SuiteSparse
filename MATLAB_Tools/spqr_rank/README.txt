QUICK INSTALL:

    (1) Download all of SuiteSparse at http://www.suitesparse.com.

    (2) Extract SuiteSparse from the SuiteSparse.tar.gz archive file.

    (3) In MATLAB, type the following commands while in the SuiteSparse
        directory (you will need a C/C++ compiler installed):

            SuiteSparse_install
            savepath

    (4) If 'savepath' fails it means you do not have permission to modify
        the system file that defines your MATLAB path.  In this case, see
        'doc startup' to create a startup.m file with addpath commands to
        include all the paths for SuiteSparse.  Type 'path' to see a list
        of paths.  spqr_rank is located in SuiteSparse/MATLAB_TOOLS/spqr_rank.

    (5) Type this command for a quick demo:
        
            quickdemo_spqr_rank

You're done!  For more details:

See Contents.m for a summary of the purpose of the SPQR_RANK package, 
a description of the files in the spqr_rank folder and installation
instructions.

After installing the package you may wish to run the demonstration
   demo_spqr_rank
which should take less than a minute to run.
Additional tests for the package are done by
   test_spqr_coverage
which takes a few minutes to run and
   test_spqr_rank
which takes half an hour or less to run.

The folders in the package contain:

spqr_rank:  the core user callable routines
private:    utilities called by the core routines.  These routines are not
            designed to be directly called by the user.
SJget:      a subset, adequate to run the demonstration and testing routines
            (demo_spqr_rank, test_spqr_rank and test_spqr_coverage) with
            default settings, of matrices from the San Jose State 
            University Singular Matrix Database.  Also SJget has utilities
            to download additional matrices from the database.
            
Copyright 2011-2012, Leslie Foster and Timothy A Davis.

ChangeLog:

Jan 19, 2012: version 1.1.0, backport to MATLAB R2008a (RandStream and '~' arg).
Dec  7, 2011: version 1.0.1 added to SuiteSparse; added quickdemo_spqr_rank.
May 19, 2011: version 1.0.0 released.
