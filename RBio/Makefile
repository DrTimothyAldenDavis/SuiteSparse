all: RBread RBtype RBwrite RBraw

clean:
	- rm -f *.mexglx *.o

purge: clean

MEX = mex -O 

RBread: RBread.f RBrread.f RBcread.f RBread_mex.f RBcsplit.f RBint.f
	$(MEX) RBread.f RBrread.f RBcread.f RBread_mex.f RBcsplit.f RBint.f

RBtype:  RBtype.f RBwrite.f RBint.f
	$(MEX) RBtype.f RBwrite.f RBint.f

RBwrite:  RBwrite.f RBwrite_mex.f RBint.f
	$(MEX) RBwrite.f RBwrite_mex.f RBint.f

RBraw: RBraw.f RBread.f RBint.f
	$(MEX) RBraw.f RBread.f RBint.f

