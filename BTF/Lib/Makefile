default: all

ccode: all

include ../../UFconfig/UFconfig.mk

# for testing only:
# TEST = -DTESTING

C = $(CC) $(CFLAGS)

INC = ../Include/btf.h ../Include/btf_internal.h

I = -I../Include -I../../UFconfig

all: library

library: libbtf.a

OBJ = btf_order.o btf_maxtrans.o btf_strongcomp.o \
    btf_l_order.o btf_l_maxtrans.o btf_l_strongcomp.o

libbtf.a: $(OBJ)
	$(AR) libbtf.a $(OBJ)
	$(RANLIB) libbtf.a

$(OBJ): $(INC)

#-------------------------------------------------------------------------------

btf_order.o: ../Source/btf_order.c
	$(C) -c $(I) $< -o $@

btf_maxtrans.o: ../Source/btf_maxtrans.c
	$(C) -c $(I) $< -o $@

btf_strongcomp.o: ../Source/btf_strongcomp.c
	$(C) -c $(I) $< -o $@

#-------------------------------------------------------------------------------

btf_l_order.o: ../Source/btf_order.c
	$(C) -c $(I) -DDLONG $< -o $@

btf_l_maxtrans.o: ../Source/btf_maxtrans.c
	$(C) -c $(I) -DDLONG $< -o $@

btf_l_strongcomp.o: ../Source/btf_strongcomp.c
	$(C) -c $(I) -DDLONG $< -o $@

#-------------------------------------------------------------------------------

purge: distclean

distclean: clean
	- $(RM) libbtf.a 

clean:
	- $(RM) $(CLEAN)
