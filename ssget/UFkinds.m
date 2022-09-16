function kinds = UFkinds
%UFKINDS get 'kind' of each problem in the SuiteSparse Matrix Collection.
% UFkinds is deprecated; use sskinds instead

% Copyright (c) 2009-2019, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

warning ('UFget:deprecated', 'UFget is deprecated; use ssget instead') ;
kinds = sskinds ;

