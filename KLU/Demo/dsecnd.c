#include "dsecnd.h"

#ifdef USE_NANOTIME

/* from http://www.ncsa.uiuc.edu/UserInfo/Resources/Hardware/IA32LinuxCluster/Doc/timing.html */

unsigned long long int nanotime_ia32(void)
{
    unsigned long long int val;
    __asm__ __volatile__("rdtsc" : "=A" (val) : );
    return(val);
}

#define SPARSE    3192963000.		/* persimmon */
/* #define SPARSE 1994171000. */	/* Dell Latitude C840 ("sparse") */
/* #define SPARSE 1395738000. */	/* IBM Thinkpad */

/*
static long int CPS;
static double iCPS;
static unsigned start=0;
*/

double dsecnd_ (void) /* Include an '_' if you will be calling from Fortan */
{
    return (((double) nanotime_ia32 ( )) / SPARSE) ;
}


#if 0
    double foo;
    if (!start)
    {
	/* CPU Clock Freq. in Hz from routine in /usr/lib/librt.a */
	/* CPS=__get_clockfreq(); */
	/* CPU Clock Freq. in Hz taken from /proc/cpuinfo */
	CPS=1994171000 ;
	iCPS=1.0/(double)CPS;
	start=1;
    }
    /* Uncomment one of the following */
    foo=iCPS*nanotime_ia32();	    /* If running on IA32 machine */
    /* foo=iCPS*nanotime_ia64(); */   /* If running on IA64 machine */

    return(foo);
}
#endif

#else

/* generic ANSI C version */
#include <time.h>
double dsecnd_ (void)
{
    double x = clock ( ) ;
    return (x / CLOCKS_PER_SEC) ;
}

#endif
