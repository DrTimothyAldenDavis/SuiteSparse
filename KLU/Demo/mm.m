m3 ;

figure (1)
clf
plot (s, 'r-x')
hold on
plot (t, 'b-o')
plot (0)

figure (2)
clf
plot (s ./(1:24), 'r-x')
hold on
plot (t ./(1:24), 'b-o')
plot (0)

% incremental cost of solve
ds = [s(1) s(2:24)-s(1:23)] ;
dt = [t(1) t(2:24)-t(1:23)] ;

figure (3)
clf
plot (ds, 'r-x')
hold on
plot (dt, 'b-o')
plot (0)
