function control = default_control
% control = default_control
%
% returns default klu control
control = zeros (1,20) ;
control (1) = 0 ;     % printing (unused)
control (2) = 1 ;     % 0: no BTF, 1: do BTF
control (3) = 1 ;     % 0: no scale, 1: sum, 2: max
control (4) = 0 ;     % ordering (0: amd, 1: colamd, 2: user)
control (5) = 0.001 ; % tol
control (6) = 1.5 ;   % growth
control (7) = 1.2 ;   % init mem AMD
control (8) = 10. ;   % init mem COLAMD
