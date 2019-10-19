
include ../../UFconfig/UFconfig.mk

linux: distclean
	./DO.linux

sol: distclean
	./DO.solaris

distclean:
	- ( cd .. ; make purge )
	- ( cd ../../AMD ; make purge )
	- $(RM) -rf Out/*

purge: distclean

