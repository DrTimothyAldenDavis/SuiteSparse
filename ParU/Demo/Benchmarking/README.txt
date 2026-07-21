This folder contains sample scripts to run the paru_benchmark program, and the
MUMPS and SuperLU benchmarks, on the test matrices used for the ACM TOMS
submission for ParU.  Linux is required.

To get the matrices (in Matrix Market format) from the sparse.tamu.edu website,
use the following, in this folder (assuming ~/SuiteSparse is in your home
folder; revise the first command accordingly):

    cd ~/SuiteSparse/ParU/Demo/Benchmarking
    chmod +x get_matrices
    ./get_matrices

The above script will download the matrices into your /tmp/matrices folder.
About 17GB is required.  These are used for all solvers except SuperLU.

--------------------------------------------------------------------------------
Benchmarking ParU
--------------------------------------------------------------------------------

    First, compile ParU and its demos/benchmark programs (where "SuiteSparse")
    is your top-level SuiteSparse repository (assuming it is in your home
    directory):

        cd ~/SuiteSparse
        make
        cd ParU
        make demos

    Finally, run the benchmarks for ParU and UMFPACK in this folder, with:

        cd ~/SuiteSparse/ParU/Demo/Benchmarking
        chmod +x run_benchmarks
        script
        ./run_benchmarks

    This will run all the benchmarks for ParU and UMFPACK for the ACM TOMS
    paper, and will save the results in the file "typescript"; the results will
    also be displayed on your screen.

    Note that we used the following scripts to benchmark ParU and UMFPACK for
    the ACM TOMS paper submission for ParU, but they are specific to our two
    systems.  We include them for reference; they are optional:

        do_paru_and_umf_hyper
        do_paru_and_umf.slurm

--------------------------------------------------------------------------------
Benchmarking MUMPS
--------------------------------------------------------------------------------

    To benchmark MUMPS, first obtain a copy of MUMPS 5.7.3.  After
    uncompressing the original MUMPS 5.7.3 into (say) a ~/MUMPS folder in your
    home directory, make the following modifications;

        cp -f mumps_573_benchmarking/Makefile.inc ~/MUMPS/
        cp -f mumps_573_benchmarking/examples/* ~/MUMPS/examples

    Then edit your ~/MUMPS/Makefile.inc to select the appropriate libraries. 
    You will likely need to revise the location of the metis-5.1.0 library;
    it is not included in MUMPS.  You can obtain it at one of these links:

        https://github.com/KarypisLab/METIS
        https://karypis.github.io/glaros/software/metis/overview.html
        https://karypis.github.io/glaros/files/sw/metis/metis-5.1.0.tar.gz

    Place a copy in ~/metis-5.1.0 (for example), and revise your
    MUMPS/Makefile.inc file accordingly.  Then build MUMPS, following the MUMPS
    instructions.  Next, use the following to run MUMPS on the test matrices
    (again, assuming your top-level SuiteSparse folder with ParU is
    ~/SuiteSparse):

        cd ~/SuiteSparse/ParU/Demo/Benchmarking/mumps_573_benchmarking
        script
        ./run_mumps

    where the results will be in the file "typescript" created by the script
    command.

--------------------------------------------------------------------------------
Benchmarking SuperLU_MT
--------------------------------------------------------------------------------

    To benchmark SuperLU_MT 4.0.1, first obtain a copy of superlu_mt_401 and
    (suppose it appears as ~/superlu_mt) and copy a few revised files into the
    original distribution:

        cp -f superlu_mt_401_benchmarking/SRC/* ~/superlu_mt/SRC
        cp -f superlu_mt_401_benchmarking/EXAMPLE/* ~/superlu_mt/EXAMPLE
        cp -f build_with_* ~/superlu_mt/
        cp -f CMakeLists.txt ~/superlu_mt/

    Then revise the build_with_gcc_and_mkl to match your system (you will need
    to tell it where to find the Intel MKL library).  Then build SuperLU_MT
    with:

        ./build_with_gcc_and_mkl

    download the matrices for SuperLU_MT with:

        ./get_RB_matrices

    (requires about 15GB).  Next, run the SuperLU_MT benchmarks with:

        cd ~/SuiteSparse/ParU/Demo/Benchmarking/superlu_mt_401_benchmarking
        script
        ./run_superlu

--------------------------------------------------------------------------------
Analyzing the results
--------------------------------------------------------------------------------

    The output files from all of these benchmarks vary from program to program.
    To collect the run times for import into a CSV file, use the following on
    each the output files:

        grep TABLE typescript

    sample outputs are listed below.  For UMFPACK and ParU, the 3rd column
    is the name of the matrix.  The next 3 columns give the umfpack
    and paru strategies (1: unsym, 2: symmetric), and the ordering
    (1: amd/colamd, 3: metis).  The sym_time is the symbolic analysis
    time, the num_times are the run times for each # of threads used
    (from high to low), followed by the solve times.

        TABLE,  UMF, TSOPF_RS_b39_c30.mtx, 1, 1, 1, sym_time:, 7.406790e-02, num_times:,  1.018119e-01,  1.018070e-01,  1.014700e-01,  9.615564e-02,  9.654265e-02,  9.607372e-02,  9.630437e-02,  sol_times:,  2.128671e-02,  2.123689e-02,  1.612758e-02,  1.608125e-02,  1.602140e-02,  1.599254e-02,  1.601883e-02, 
        TABLE,  UMF, TSOPF_RS_b39_c30.mtx, 1, 1, 3, sym_time:, 3.292655e-01, num_times:,  1.517104e-01,  1.516826e-01,  1.518991e-01,  1.511355e-01,  1.515573e-01,  1.528105e-01,  1.514552e-01,  sol_times:,  2.080022e-02,  2.066906e-02,  2.067235e-02,  2.076371e-02,  2.066134e-02,  2.082458e-02,  2.074122e-02, 
        TABLE, ParU, TSOPF_RS_b39_c30.mtx, 1, 1, 1, sym_time:, 7.695978e-02, num_times:,  1.453122e-01,  1.216802e-01,  1.230776e-01,  1.284256e-01,  1.216281e-01,  1.164964e-01,  9.921592e-02,  sol_times:,  8.401886e-03,  7.752119e-03,  8.355235e-03,  8.849248e-03,  8.315628e-03,  7.050963e-03,  5.527283e-03, 
        TABLE, ParU, TSOPF_RS_b39_c30.mtx, 1, 1, 3, sym_time:, 3.495287e-01, num_times:,  2.116326e-01,  1.406327e-01,  1.310422e-01,  1.301144e-01,  1.134511e-01,  1.313425e-01,  1.388268e-01,  sol_times:,  1.492923e-02,  1.442635e-02,  1.443325e-02,  1.284580e-02,  1.152180e-02,  1.207555e-02,  9.344153e-03, 


    An example MUMPS output is listed below.  It has the same format as the
    ParU and UMFPACK outputs, except the run times are in order of low to high
    # of threads.  The 4th column is the ordering (1: amd, 2: metis on A+A').

        TABLE, MUMPS, /tmp/matrices/TSOPF_RS_b39_c30/TSOPF_RS_b39_c30.mtx, 1, sym_time:, 3.204014e-01, num_times:,  6.222303e-02,  7.103832e-02,  4.108610e-02,  4.489215e-02,  4.803422e-02,  6.339786e-02,  sol_times:,  1.091543e-02,  2.187732e-02,  8.204759e-03,  9.458208e-03,  1.547582e-02,  1.731181e-02, 

    SuperLU is similar, except that the matrix name is not listed (use awk to
    find both "TABLE" and "Matrix:" if preferred).

        TABLE, SuperLU_MT, threads:, 32, ordering:,  3,  analyze_time:,  4.29381728e-02,  4.40463973e-02,  4.21937061e-02,  4.61900234e-02,  4.15172875e-02,  4.73920098e-02,  factor_time:,  8.40138663e-02,  6.18000347e-02,  5.73010538e-02,  5.09567745e-02,  5.99720702e-02,  7.43006011e-02,  solve_time:,  6.62509473e-02,  7.43965395e-02,  5.66021195e-02,  7.09967716e-02,  4.94663576e-02,  6.65261745e-02,

    For the ACM TOMS submissions, we then copied these run times from a
    spreadsheet into a MATLAB script that generated the plots in the figures in
    the paper.  This step is a bit tedious so we have omitted the details.
    However, the final results for our two systems are in these files in this
    folder:

        analyze_grace.m         plot the results on grace.hprc.tamu.edu
        analyze_hyper.m         plot the results on a 24-core desktop
        plot_one_matrix.m       used by analyze_*.m
        subplot_one_matrix.m    used by analyze_*.m

