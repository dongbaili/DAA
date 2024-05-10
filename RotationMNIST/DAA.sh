txtdir="domainbed/txtlist"
dataset="RMnist"
source="30 60 120 150"
target="90"
n=1800 # n means the total sample budget
visible_n_test=100
k=6
dis="mse"

python DAA.py --txtdir $txtdir --dataset $dataset --source $source --target $target --n $n --k $k --visible_n_test $visible_n_test --dis $dis