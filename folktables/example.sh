task="ACSEmployment"
test_state="HI"
lr=0.001
k=5
T_list=("2000")
for j in "${!T_list[@]}"; do
    T="${T_list[j]}"
    lambdda=0.01
    dis="mmd"
    python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis

    lambdda=0.000001
    dis="cosine"
    python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis

    lambdda=0.000001
    dis="mse"
    python main.py --test_state $test_state --T $T --k $k --lr $lr --lambdda $lambdda --task $task --dis $dis
done