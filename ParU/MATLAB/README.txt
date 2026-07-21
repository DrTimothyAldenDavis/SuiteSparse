ParU MATLAB interface

Use paru_make in the MATLAB command window to compile the MATLAB interface
to ParU.  This will also run two demos, paru_tiny and paru_demo.

If you see an error like this in Linux:

Invalid MEX-file '/whatever/paru/SuiteSparse/ParU/MATLAB/paru.mexa64':
/usr/local/MATLAB/R2026a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version
`CXXABI_1.3.15' not found (required by
/whatever/paru/SuiteSparse/ParU/MATLAB/paru.mexa64)

then it means your version of MATLAB is using a different version of libstdc++
than is available on your system.  You are likely using a version of the gcc
compiler not supported by MATLAB.

To fix this, use one of these four fixes:

(1) use a supported compiler.  See
    https://www.mathworks.com/support/requirements/supported-compilers-linux.html

(2) each time you launch MATLAB, preface it with a definition of LD_PRELOAD.
    In your terminal, use this to start up MATLAB, each time:

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab   

(3) Add the following line to your ~/.bashrc or ~/.zshrc to set this
    environment variable permanently:

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

(4) Edit MATLAB itself and remove the old libstdc++.so.6 file, and also
    libgcc_s.so.1 if it is also causing problems:

    cd /usr/local/MATLAB/R2026a/sys/os/glnxa64
    sudo mkdir unused
    sudo mv libstdc++.so.6* unused
    sudo mv libgcc_s.so.1*  unused # If necessary   

