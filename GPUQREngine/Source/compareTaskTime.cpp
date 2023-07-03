#ifdef SUITESPARSE_CUDA
#include "GPUQREngine_Scheduler.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"

int compareTaskTime (const void * a, const void * b)
{
    TaskDescriptor *ta = (TaskDescriptor*) a;
    TaskDescriptor *tb = (TaskDescriptor*) b;

    int64_t aFlops = getWeightedFlops(ta);
    int64_t bFlops = getWeightedFlops(tb);

    // revised
    if (aFlops == bFlops) return (0) ;
    if (aFlops <  bFlops) return (-1) ;
    if (aFlops >  bFlops) return (1) ;

    // old:
    // return bFlops - aFlops;
}
#endif
