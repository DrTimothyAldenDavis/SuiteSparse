// Example user program

#include <stdio.h>
#include <string.h>

#include "my.h"

int main (void)
{
    printf ("My demo\n") ;
    int version [3] ;
    char date [128] ;
    my_library (version, date) ;
    printf ("Date from #include 'my.h':  %s\n", MY_DATE) ;
    printf ("Date from compiled library: %s\n", date) ;
    printf ("version from #include 'my.h.': %d.%d.%d\n",
        MY_MAJOR_VERSION, MY_MINOR_VERSION, MY_PATCH_VERSION) ;
    printf ("version from compiled library: %d.%d.%d\n",
        version [0], version [1], version [2]) ;

    my_function ( ) ;
    return (0) ;
}

