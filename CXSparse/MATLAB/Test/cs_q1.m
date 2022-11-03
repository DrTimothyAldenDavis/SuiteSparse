function Q = cs_q1 (V, Beta, p)
%CS_Q1 construct Q from Householder vectors
% Example:
%   Q = cs_q1 (V, beta, p)
% See also: testall

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

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
