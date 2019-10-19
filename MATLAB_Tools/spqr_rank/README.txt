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
            
Copyright 2011, Leslie Foster and Timothy A Davis.

ChangeLog:

Dec 7, 2011: version 1.0.1 added to SuiteSparse; added quickdemo_spqr_rank.
May 19, 2011: version 1.0.0 released.

