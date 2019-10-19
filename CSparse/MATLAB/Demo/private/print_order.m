function print_order (order)
% print_order(order) prints the ordering determined by the order parameter
switch (fix (order))
    case 0
	fprintf ('natural    ') ;
    case 1
	fprintf ('amd(A+A'')  ') ;
    case 2
	fprintf ('amd(S''*S)  ') ;
    case 3
	fprintf ('amd(A''*A)  ') ;
    otherwise
	fprintf ('undefined  ') ;
end
