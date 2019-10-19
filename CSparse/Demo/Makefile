CC = cc
CFLAGS = -O

I = -I../Include

CS = ../Lib/libcsparse.a

all: lib cs_demo1 cs_demo2 cs_demo3
	- ./cs_demo1 < ../Matrix/t1
	- ./cs_demo2 < ../Matrix/t1
	- ./cs_demo2 < ../Matrix/ash219
	- ./cs_demo2 < ../Matrix/bcsstk01
	- ./cs_demo2 < ../Matrix/fs_183_1
	- ./cs_demo2 < ../Matrix/mbeacxc
	- ./cs_demo2 < ../Matrix/west0067
	- ./cs_demo2 < ../Matrix/lp_afiro
	- ./cs_demo2 < ../Matrix/bcsstk16
	- ./cs_demo3 < ../Matrix/bcsstk01
	- ./cs_demo3 < ../Matrix/bcsstk16

lib:
	( cd ../Lib ; $(MAKE) )

cs_demo1: lib cs_demo1.c Makefile
	$(CC) $(CFLAGS) $(I) -o cs_demo1 cs_demo1.c $(CS) -lm

cs_demo2: lib cs_demo2.c cs_demo.c cs_demo.h Makefile
	$(CC) $(CFLAGS) $(I) -o cs_demo2 cs_demo2.c cs_demo.c $(CS) -lm

cs_demo3: lib cs_demo3.c cs_demo.c cs_demo.h Makefile
	$(CC) $(CFLAGS) $(I) -o cs_demo3 cs_demo3.c cs_demo.c $(CS) -lm

clean:
	rm -f *.o

purge: distclean

distclean: clean
	rm -f cs_demo1 cs_demo2 cs_demo3 *.a
