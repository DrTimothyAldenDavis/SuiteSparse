MEX = mex -O
AR = ar cr
RANLIB = ranlib

all: cs_sparse2.mexglx \
	cs_ipvec.mexglx \
	cs_pvec.mexglx \
	cs_reach.mexglx \
	cs_maxtransr.mexglx \
	cs_reachr.mexglx \
	cs_rowcnt.mexglx \
	cs_frand.mexglx

mexcsparse:
	( cd ../CSparse ; make mexcsparse.a )

I = -I../../Include -I../CSparse

cs_ipvec.mexglx: cs_ipvec_mex.c mexcsparse
	$(MEX) -output cs_ipvec $< $(I) ../CSparse/mexcsparse.a

cs_pvec.mexglx: cs_pvec_mex.c mexcsparse
	$(MEX) -output cs_pvec $< $(I) ../CSparse/mexcsparse.a

cs_reach.mexglx: cs_reach_mex.c mexcsparse
	$(MEX) -output cs_reach $< $(I) ../CSparse/mexcsparse.a

cs_sparse2.mexglx: cs_sparse2_mex.c mexcsparse
	$(MEX) -output cs_sparse2 $< $(I) ../CSparse/mexcsparse.a

cs_maxtransr.mexglx: cs_maxtransr_mex.c mexcsparse
	$(MEX) -output cs_maxtransr $< $(I) ../CSparse/mexcsparse.a

cs_reachr.mexglx: cs_reachr_mex.c mexcsparse
	$(MEX) -output cs_reachr $< $(I) ../CSparse/mexcsparse.a

cs_rowcnt.mexglx: cs_rowcnt_mex.c mexcsparse
	$(MEX) -output cs_rowcnt $< $(I) ../CSparse/mexcsparse.a

cs_frand.mexglx: cs_frand_mex.c mexcsparse
	$(MEX) -output cs_frand $< $(I) ../CSparse/mexcsparse.a



clean:
	rm -f *.o

distclean: clean
	rm -f *.mex* *.dll *.a

purge: distclean
