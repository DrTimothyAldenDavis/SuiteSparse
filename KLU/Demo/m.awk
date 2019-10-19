/KLU  solve/ {
    nrhs = $5
    t = $8
    printf "s(%2d) = %g ; \n", nrhs, t
}

/KLU tsolve \(/ {
    nrhs = $5
    t = $8
    printf "t(%2d) = %g ; \n", nrhs, t
}
