#!/bin/bash
echo '========================================================================='
echo '======================== on the GPU:'
echo 'First analysis/factorize takes longer, to "warmup" the GPU, which is the'
echo 'time to allocate and pin GPU/CPU memory pools.  That only needs to be'
echo 'done once for the entire application, however.'
echo '========================================================================='
CHOLMOD_USE_GPU=1 ../build/cholmod_dl_demo < ~/nd6k.mtx
echo ''
echo '========================================================================='
echo '======================== on the CPU:'
echo '========================================================================='
CHOLMOD_USE_GPU=0 ../build/cholmod_dl_demo < ~/nd6k.mtx
