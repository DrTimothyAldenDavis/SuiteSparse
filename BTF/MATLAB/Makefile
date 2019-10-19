
include ../../UFconfig/UFconfig.mk

I = -I../Include -I../../UFconfig

MX = $(MEX) $(I) -DDLONG

all: maxtrans.mexglx strongcomp.mexglx btf.mexglx

recursive: strongcomp_recursive.mexglx

maxtrans.mexglx: ../Source/btf_maxtrans.c ../Include/btf.h maxtrans.c \
		../Include/btf_internal.h
	$(MX) maxtrans.c ../Source/btf_maxtrans.c

strongcomp.mexglx: ../Source/btf_strongcomp.c ../Include/btf.h \
		strongcomp.c ../Include/btf_internal.h
	$(MX) strongcomp.c ../Source/btf_strongcomp.c

strongcomp_recursive.mexglx: ../Source/btf_strongcomp.c ../Include/btf.h \
		strongcomp.c ../Include/btf_internal.h
	$(MX) -DRECURSIVE -output strongcomp_recursive \
		../Source/btf_strongcomp.c strongcomp.c

btf.mexglx: ../Source/btf_strongcomp.c ../Include/btf.h btf.c \
		../Include/btf_internal.h \
		../Source/btf_maxtrans.c ../Source/btf_order.c
	$(MX) btf.c ../Source/btf_maxtrans.c \
		../Source/btf_strongcomp.c ../Source/btf_order.c

distclean: purge

purge: clean
	- $(RM) *.o *.mex*

clean:
	- $(RM) $(CLEAN)
