default: all

ccode: all

include ../../UFconfig/UFconfig.mk

# for testing only:
# TEST = -DTESTING

C = $(CC) $(CFLAGS)

INC = ../Include/btf.h ../Include/btf_internal.h

I = -I../Include

all: library

library: libbtf.a

OBJ = btf_order.o maxtrans.o strongcomp.o

libbtf.a: $(OBJ)
	$(AR) libbtf.a $(OBJ)
	$(RANLIB) libbtf.a

$(OBJ): $(INC)

#-------------------------------------------------------------------------------

btf_order.o: ../Source/btf_order.c
	$(C) -c $(I) $< -o $@

maxtrans.o: ../Source/maxtrans.c
	$(C) -c $(I) $< -o $@

strongcomp.o: ../Source/strongcomp.c
	$(C) -c $(I) $< -o $@

#-------------------------------------------------------------------------------

purge: distclean

distclean: clean
	- $(RM) libbtf.a 

clean:
	- $(RM) $(CLEAN)
