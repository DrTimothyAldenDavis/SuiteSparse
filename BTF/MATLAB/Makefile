include ../../UFconfig/UFconfig.mk

all: dp.mexglx maxtrans.mexglx maxtrans_recursive.mexglx strongcomp.mexglx \
	strongcomp_recursive.mexglx btf.mexglx

just: strongcomp.mexglx strongcomp_recursive.mexglx

maxtrans.mexglx: ../Source/maxtrans.c ../Include/btf.h maxtrans_mex.c \
		../Include/btf_internal.h
	mex -I../Include ../Source/maxtrans.c maxtrans_mex.c

dp.mexglx: ../Source/maxtrans.c dmperm.c ../Include/btf.h \
		dp_mex.c ../Include/btf_internal.h
	mex -I../Include dmperm.c dp_mex.c ../Source/maxtrans.c \
		-output dp

maxtrans_recursive.mexglx: ../Source/maxtrans.c ../Include/btf.h \
		maxtrans_mex.c ../Include/btf_internal.h
	mex -I../Include -DRECURSIVE -output maxtrans_recursive \
		../Source/maxtrans.c maxtrans_mex.c

strongcomp.mexglx: ../Source/strongcomp.c ../Include/btf.h strongcomp_mex.c \
		../Include/btf_internal.h
	mex -I../Include ../Source/strongcomp.c strongcomp_mex.c

strongcomp_recursive.mexglx: ../Source/strongcomp.c ../Include/btf.h \
		strongcomp_mex.c ../Include/btf_internal.h
	mex -I../Include -DRECURSIVE -output strongcomp_recursive \
		../Source/strongcomp.c strongcomp_mex.c


btf.mexglx: ../Source/strongcomp.c ../Include/btf.h ../Include/btf_internal.h \
		../Source/maxtrans.c btf_mex.c ../Source/btf_order.c
	mex -I../Include -output btf ../Source/maxtrans.c \
		../Source/strongcomp.c btf_mex.c ../Source/btf_order.c

distclean: purge

purge: clean
	- $(RM) *.o *.mex*

clean:
	- $(RM) $(CLEAN)
