function s = signum (x)
s = ones (length (x),1) ;
s (find (x < 0)) = -1 ;
s
