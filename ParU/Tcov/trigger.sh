export MAT_PATH="../Matrix"

echo "========tumorAntiAngiogenesis_2.mtx========"
./paru_quick_test <$MAT_PATH/tumorAntiAngiogenesis_2.mtx
echo "========olm500.mtx========"
./paru_quick_test <$MAT_PATH/olm500.mtx
./cov

echo "========adder_dcop_05.mtx========"
./paru_quick_test <$MAT_PATH/adder_dcop_05.mtx
echo "========bayer10.mtx========"
./paru_quick_test <$MAT_PATH/bayer10.mtx
echo "========rajat01.mtx========"
./paru_quick_test 104 <$MAT_PATH/rajat01.mtx
echo "========rajat19.mtx========"
./paru_quick_test <$MAT_PATH/rajat19.mtx
echo "========west0497.mtx========"
./paru_brutal_test <$MAT_PATH/west0497.mtx
echo "========gent113.mtx========"
./paru_quick_test 104 <$MAT_PATH/gent113.mtx
./cov

echo "========hangGlider_2.mtx========"
./paru_quick_test <$MAT_PATH/hangGlider_2.mtx
echo "========Tina_AskCal.mtx========"
./paru_brutal_test 104 <$MAT_PATH/Tina_AskCal.mtx
echo "========GD98_a.mtx========"
./paru_brutal_test 104 <$MAT_PATH/GD98_a.mtx
echo "========a0.mtx========"
./paru_brutal_test 104 <$MAT_PATH/a0.mtx
echo "========a1.mtx========"
./paru_brutal_test 104 <$MAT_PATH/a1.mtx
echo "========Tina_AskCal_perm.mtx========"
./paru_brutal_test 104 <$MAT_PATH/Tina_AskCal_perm.mtx
./cov

echo "========ash219.mtx========"
./paru_brutal_test 104 <$MAT_PATH/ash219.mtx
echo "========lp_e226.mtx========"
./paru_brutal_test 104 <$MAT_PATH/lp_e226.mtx
echo "========r2.mtx========"
./paru_brutal_test 104 <$MAT_PATH/r2.mtx
echo "========LFAT5.mtx========"
./paru_brutal_test <$MAT_PATH/LFAT5.mtx
echo "========west0067.mtx========"
./paru_c_test <$MAT_PATH/west0067.mtx
./cov

echo "========arrow.mtx========"
./paru_quick_test <$MAT_PATH/arrow.mtx
./cov

echo "========a2.mtx========"
./paru_brutal_test <$MAT_PATH/a2.mtx
echo "========az88.mtx========"
./paru_brutal_test <$MAT_PATH/az88.mtx
echo "========young1c.mtx========"
./paru_quick_test 102 <$MAT_PATH/young1c.mtx
echo "========s32.mtx========"
./paru_brutal_test 104 <$MAT_PATH/s32.mtx
echo "========lp_share1b.mtx========"
./paru_brutal_test 104 <$MAT_PATH/lp_share1b.mtx
./cov

echo "========cage3.mtx========"
./paru_brutal_test -14 <$MAT_PATH/cage3.mtx
echo "========b1_ss.mtx========"
./paru_brutal_test -15 <$MAT_PATH/b1_ss.mtx
echo "========lfat5b.mtx========"
./paru_brutal_test -15 <$MAT_PATH/lfat5b.mtx
echo "========c32.mtx========"
./paru_brutal_test 104 <$MAT_PATH/c32.mtx
echo "========bfwa62.mtx========"
./paru_brutal_test -15 <$MAT_PATH/bfwa62.mtx
echo "========Ragusa16.mtx========"
./paru_brutal_test 104 <$MAT_PATH/Ragusa16.mtx
echo "========lp_e226_transposed.mtx========"
./paru_brutal_test 104 <$MAT_PATH/lp_e226_transposed.mtx
./cov

echo "========row0.mtx========"
./paru_brutal_test 105 <$MAT_PATH/row0.mtx

echo "========Groebner_id2003_aug.mtx========"
./paru_brutal_test 104 <$MAT_PATH/Groebner_id2003_aug.mtx
echo "========c2.mtx========"
./paru_brutal_test 104 <$MAT_PATH/c2.mtx
echo "========a4.mtx========"
./paru_brutal_test 104 <$MAT_PATH/a4.mtx
echo "========problem.mtx========"
./paru_brutal_test 104 <$MAT_PATH/problem.mtx
echo "========permuted_b1_ss.mtx========"
./paru_brutal_test -14 <$MAT_PATH/permuted_b1_ss.mtx
echo "========pwr01b.mtx========"
./paru_brutal_test <$MAT_PATH/pwr01b.mtx
./cov

echo "========Franz6_id1959_aug.mtx========"
./paru_brutal_test 104 <$MAT_PATH/Franz6_id1959_aug.mtx

echo "========Ragusa16_pattern.mtx========"
# square, but structurally singular
./paru_brutal_test 104 <$MAT_PATH/Ragusa16_pattern.mtx

echo "========temp.mtx========"
./paru_brutal_test <$MAT_PATH/temp.mtx
echo "========cage5.mtx========"
./paru_brutal_test -15 <$MAT_PATH/cage5.mtx

echo "========a04.mtx========"
# 0-by-4
./paru_c_test 104 <$MAT_PATH/a04.mtx

echo "========lpi_galenet.mtx========"
# rectangular
./paru_c_test 104 <$MAT_PATH/lpi_galenet.mtx

echo "========nnc1374.mtx========"
./paru_quick_test <$MAT_PATH/nnc1374.mtx
./cov

# Long tests
echo "========494_bus.mtx========"
./paru_brutal_test <$MAT_PATH/494_bus.mtx
./cov

# echo "========dwt_878.mtx========"
# time ./paru_c_test <$MAT_PATH/dwt_878.mtx

echo "========xenon1.mtx========"
./paru_quick_test <$MAT_PATH/xenon1.mtx
./cov

echo "========c-62.mtx========"
./paru_quick_test <$MAT_PATH/c-62.mtx
./cov

