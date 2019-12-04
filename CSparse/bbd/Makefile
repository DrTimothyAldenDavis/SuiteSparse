CF = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -Og -g
I = -I../Include

LDLIBS += -lm
CS = $(LDFLAGS) ../Lib/libcsparse.a $(LDLIBS)

EXE = test
SRC = test.c mmio.c

all: lib $(EXE)

lib:
	( cd ../Lib ; $(MAKE) )

$(EXE): lib Makefile $(SRC)
	$(CC) $(CF) $(I) -o $(EXE) $(SRC) $(CS)

clean:
	- -rm *.o $(EXE)


