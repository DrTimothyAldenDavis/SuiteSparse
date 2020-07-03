#!/bin/csh
        tcov -x cm.profile SLIP_*.c slip_*.c >& /dev/null
        echo -n "statments not yet tested: "
        ./covs > covs.out
        grep "#####" *tcov | wc -l
