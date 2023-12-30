export MAT_PATH="../Matrix"

echo "========tumorAntiAngiogenesis_2.mtx========"
./x_quick_test <$MAT_PATH/tumorAntiAngiogenesis_2.mtx
echo "========olm500.mtx========"
./x_quick_test <$MAT_PATH/olm500.mtx
./cov

echo "========adder_dcop_05.mtx========"
./x_quick_test <$MAT_PATH/adder_dcop_05.mtx
echo "========bayer10.mtx========"
./x_quick_test <$MAT_PATH/bayer10.mtx
echo "========rajat01.mtx========"
./x_quick_test 104 <$MAT_PATH/rajat01.mtx
echo "========rajat19.mtx========"
./x_quick_test <$MAT_PATH/rajat19.mtx
echo "========west0497.mtx========"
./x_brutal_test <$MAT_PATH/west0497.mtx
echo "========gent113.mtx========"
./x_quick_test 104 <$MAT_PATH/gent113.mtx
./cov

echo "========hangGlider_2.mtx========"
./x_quick_test <$MAT_PATH/hangGlider_2.mtx
echo "========Tina_AskCal.mtx========"
./x_brutal_test 104 <$MAT_PATH/Tina_AskCal.mtx
echo "========GD98_a.mtx========"
./x_brutal_test 104 <$MAT_PATH/GD98_a.mtx
echo "========a0.mtx========"
./x_brutal_test 104 <$MAT_PATH/a0.mtx
echo "========a1.mtx========"
./x_brutal_test 104 <$MAT_PATH/a1.mtx
echo "========Tina_AskCal_perm.mtx========"
./x_brutal_test 104 <$MAT_PATH/Tina_AskCal_perm.mtx
./cov

echo "========ash219.mtx========"
./x_brutal_test 104 <$MAT_PATH/ash219.mtx
echo "========lp_e226.mtx========"
./x_brutal_test 104 <$MAT_PATH/lp_e226.mtx
echo "========r2.mtx========"
./x_brutal_test 104 <$MAT_PATH/r2.mtx
echo "========LFAT5.mtx========"
./x_brutal_test <$MAT_PATH/LFAT5.mtx
echo "========west0067.mtx========"
./x_paru_c <$MAT_PATH/west0067.mtx
./cov

echo "========arrow.mtx========"
./x_quick_test <$MAT_PATH/arrow.mtx
./cov

echo "========a2.mtx========"
./x_brutal_test <$MAT_PATH/a2.mtx
echo "========az88.mtx========"
./x_brutal_test <$MAT_PATH/az88.mtx
echo "========young1c.mtx========"
./x_quick_test 102 <$MAT_PATH/young1c.mtx
echo "========s32.mtx========"
./x_brutal_test 104 <$MAT_PATH/s32.mtx
echo "========lp_share1b.mtx========"
./x_brutal_test 104 <$MAT_PATH/lp_share1b.mtx
./cov

echo "========cage3.mtx========"
./x_brutal_test -14 <$MAT_PATH/cage3.mtx
echo "========b1_ss.mtx========"
./x_brutal_test -15 <$MAT_PATH/b1_ss.mtx
echo "========lfat5b.mtx========"
./x_brutal_test -15 <$MAT_PATH/lfat5b.mtx
echo "========c32.mtx========"
./x_brutal_test 104 <$MAT_PATH/c32.mtx
echo "========bfwa62.mtx========"
./x_brutal_test -15 <$MAT_PATH/bfwa62.mtx
echo "========Ragusa16.mtx========"
./x_brutal_test 104 <$MAT_PATH/Ragusa16.mtx
echo "========lp_e226_transposed.mtx========"
./x_brutal_test 104 <$MAT_PATH/lp_e226_transposed.mtx
./cov

echo "========row0.mtx========"
./x_brutal_test 105 <$MAT_PATH/row0.mtx

echo "========Groebner_id2003_aug.mtx========"
./x_brutal_test 104 <$MAT_PATH/Groebner_id2003_aug.mtx
echo "========c2.mtx========"
./x_brutal_test 104 <$MAT_PATH/c2.mtx
echo "========a4.mtx========"
./x_brutal_test 104 <$MAT_PATH/a4.mtx
echo "========problem.mtx========"
./x_brutal_test 104 <$MAT_PATH/problem.mtx
echo "========permuted_b1_ss.mtx========"
./x_brutal_test -14 <$MAT_PATH/permuted_b1_ss.mtx
echo "========pwr01b.mtx========"
./x_brutal_test <$MAT_PATH/pwr01b.mtx
./cov

echo "========Franz6_id1959_aug.mtx========"
./x_brutal_test 104 <$MAT_PATH/Franz6_id1959_aug.mtx

echo "========Ragusa16_pattern.mtx========"
# square, but structurally singular
./x_brutal_test 104 <$MAT_PATH/Ragusa16_pattern.mtx

echo "========temp.mtx========"
./x_brutal_test <$MAT_PATH/temp.mtx
echo "========cage5.mtx========"
./x_brutal_test -15 <$MAT_PATH/cage5.mtx

echo "========a04.mtx========"
# 0-by-4
./x_paru_c 104 <$MAT_PATH/a04.mtx

echo "========lpi_galenet.mtx========"
# rectangular
./x_paru_c 104 <$MAT_PATH/lpi_galenet.mtx

echo "========nnc1374.mtx========"
./x_quick_test <$MAT_PATH/nnc1374.mtx
./cov

# Long tests
echo "========494_bus.mtx========"
./x_brutal_test <$MAT_PATH/494_bus.mtx
./cov

# echo "========dwt_878.mtx========"
# time ./x_paru_c <$MAT_PATH/dwt_878.mtx

echo "========xenon1.mtx========"
./x_quick_test <$MAT_PATH/xenon1.mtx
./cov

echo "========c-62.mtx========"
./x_quick_test <$MAT_PATH/c-62.mtx
./cov

