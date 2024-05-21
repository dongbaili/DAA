n_train=1800 # n_train means the number of samples in EACH source domain
n_test=1200
test_angle=90
train_angles="30 60 120 150"
python images.py $train_angles --n_train $n_train --n_test $n_test --test_angle $test_angle