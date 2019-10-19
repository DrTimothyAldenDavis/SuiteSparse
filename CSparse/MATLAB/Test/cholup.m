function L = cholup (Lold,w)
% given Lold and w, compute L so that L*L' = Lold*Lold' + w*w'

n = size (Lold,1) ;
L = [Lold w] ;

for k = 1:n

    g = givens (L(k,k), L(k,n+1)) ;

    L (:, [k n+1]) = L (:, [k n+1]) * g' ;


    L
    pause
end

L = L (:,1:n)
