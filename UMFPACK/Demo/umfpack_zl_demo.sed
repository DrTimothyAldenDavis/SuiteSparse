/::/d
1,$s/_xx_/_zl_/g
1,$s/Int/int64_t/g
1,$s/WSIZE/10/
1,$s/%ld/%"PRId64"/g
/define ABS/ {
	s/ABS/ABS(x,z) ((x) >= 0 ? (x) : -(x)) + ((z) >= 0 ? (z) : -(z))/
	}
