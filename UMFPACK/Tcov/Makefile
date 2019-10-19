all: go

include UFconfig/UFconfig.mk

go: prog
	- ut > ut.out
	- tail ut.out
	#- $(RM) ut.out
	- ( cd UMFPACK/Source ; gcovs umf*.c )
	- ( cd AMD/Source     ; gcovs amd*.c )

prog:
	( cd UMFPACK ; make library )
	( cd AMD ; make library )
	$(CC) -DDINT $(CFLAGS) $(UMFPACK_CONFIG) -IUMFPACK/Source -IUMFPACK/Include -IAMD/Source -IAMD/Include -o ut ut.c UMFPACK/Lib/libumfpack.a AMD/Lib/libamd.a $(LIB)

utcov:
	- ( cd UMFPACK/Source ; gcovs umf*.c )
	- ( cd AMD/Source     ; gcovs amd*.c )


purge:
	( cd UMFPACK ; make purge )
	( cd AMD ; make purge )
