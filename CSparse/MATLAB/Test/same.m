function same (p1,p2)
    if (isempty (p1))
	if (~isempty (p2))
	    p1
	    p2
	    error ('empty!') ;
	end
    elseif (any (p1 ~= p2))
	p1
	p2
	error ('!') ;
    end

