#===============================================================================
# RBio/Tcov/Makefile
#===============================================================================

include ../../UFconfig/UFconfig.mk

# valgrind is not used
V =

# with test coverage
CFLAGS = -O0 -g --coverage

# Tcov requires gcc
CC = gcc

C = $(CC) $(CFLAGS)
TAR = tar -O -xvvzf

SRC = RBio.c UFconfig.c
OBJ = RBio.o UFconfig.o
INC = RBio.h UFconfig.h

all: RBdemo RBtest
	- ./RBdemo ../RBio/private/bcsstk01.rb
	- ./RBdemo ../RBio/private/farm.rb
	- ./RBdemo ../RBio/private/lap_25.pse
	- ./RBdemo ../RBio/private/lap_25.rb
	- ./RBdemo ../RBio/private/west0479.rb
	- ./RBdemo ../RBio/private/west0479.rua
	- $(TAR) matrices/dwg961a.tar.gz  dwg961a/dwg961a.rb     | ./RBdemo
	- $(TAR) matrices/mhd1280a.tar.gz mhd1280a/mhd1280a.rb   | ./RBdemo
	- $(TAR) matrices/mhd1280b.tar.gz mhd1280b/mhd1280b.rb   | ./RBdemo
	- $(TAR) matrices/qc324.tar.gz    qc324/qc324.rb         | ./RBdemo
	- $(TAR) matrices/plskz362.tar.gz plskz362/plskz362.rb   | ./RBdemo
	- $(TAR) matrices/west0067.tar.gz west0067/west0067.rb   | ./RBdemo
	- $(TAR) matrices/Tina_DisCog.tar.gz Tina_DisCog/Tina_DisCog.rb | ./RBdemo
	- ./RBdemo matrices/m4.rb
	- ./RBdemo matrices/s4.rb
	- ./RBdemo matrices/m4b.rb
	- ./RBdemo mangled/1.rb
	- ./RBdemo mangled/2.rb
	- ./RBdemo mangled/3.rb
	- ./RBdemo mangled/4.rb
	- ./RBdemo mangled/5.rb
	- ./RBdemo mangled/6.rb
	- ./RBdemo mangled/7.rb
	- ./RBdemo mangled/8.rb
	- ./RBdemo mangled/9.rb
	- ./RBdemo mangled/10.rb
	- ./RBdemo mangled/11.rb
	- ./RBdemo mangled/12.rb
	- ./RBdemo mangled/13.rb
	- ./RBdemo mangled/14.rb
	- ./RBdemo mangled/15.rb
	- ./RBtest
	- gcov *.c > gcov.out
	- grep "#####" RBio.c.gcov | wc -l

RBdemo.c:
	ln -s ../Demo/RBdemo.c

RBio.c:
	ln -s ../Source/RBio.c

RBio.h:
	ln -s ../Include/RBio.h

UFconfig.c:
	ln -s ../../UFconfig/UFconfig.c

UFconfig.h:
	ln -s ../../UFconfig/UFconfig.h

RBdemo: RBdemo.c $(OBJ) $(INC) Makefile
	$(C) -o RBdemo RBdemo.c $(OBJ)

RBtest: RBtest.c $(OBJ) $(INC) Makefile
	$(C) -o RBtest RBtest.c $(OBJ)

RBio.o: RBio.c $(INC)
	$(C) -c RBio.c

UFconfig.o: UFconfig.c $(INC)
	$(C) -c UFconfig.c

code: RBdemo RBtest

clean:
	- $(RM) *.gcov *.gcda *.gcno temp*.rb *.o

purge: clean
	- $(RM) $(SRC) $(INC) RBdemo *.gcov gcov.out RBtest RBdemo.c
	- $(RM) -rf RBdemo.dSYM RBtest.dSYM

