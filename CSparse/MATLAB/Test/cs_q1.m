function Q = cs_qmake1 (V, Beta, p)
[m n] = size (V) ;
Q = speye (m) ;
if (nargin > 2)
    Q = Q (:,p) ;
end
for i = 1:m
    for k = 1:n
	Q (i,:) = Q (i,:) - ((Q(i,:) * V(:,k)) * Beta(k)) * V(:,k)' ;
    end
end
