all: ssmult.mexglx sptranspose.mexglx ssmultsym.mexglx

OBJ = \
	ssmult.o \
	ssmult_dot.o \
	ssmult_saxpy.o \
	ssmult_transpose.o

$(OBJ): ssmult.h

.c.o:
	mex -O -c $<

ssmult.mexglx: ssmult_mex.c $(OBJ) ssmult.h ssmult_template.c
	mex -O -output ssmult ssmult_mex.c $(OBJ)

ssmultsym.mexglx: ssmultsym.c ssmult.h
	mex -O -output ssmultsym ssmultsym.c

sptranspose.mexglx: sptranspose.c ssmult_transpose.o ssmult.h
	mex -O -output sptranspose sptranspose.c ssmult_transpose.o

clean:
	rm -f *.o

purge: clean
	rm *.mexglx
