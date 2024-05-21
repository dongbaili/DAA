task="ACSEmployment"
test_state="HI"
lr=0.001
k=6
T=2000

lambdda=0.001
dis="mmd"
python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis

lambdda=0.000001
dis="cosine"
python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis

lambdda=0.000001
dis="mse"
python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis
