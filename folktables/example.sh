task="ACSEmployment"
test_state="HI"
lr=0.001
k=5
T=2000

lambdda=0.0001
dis="mmd"
python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis

lambdda=0.000001
dis="cosine"
python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis

lambdda=0.000001
dis="mse"
python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis
