txtdir="domainbed/txtlist"
dataset="RMnist"
algorithm="ERM"
target="90"
 
source="DAA_cosine"
python main.py --txtdir $txtdir --dataset $dataset --source $source --target $target --my_algorithm $algorithm

source="DAA_mmd"
python main.py --txtdir $txtdir --dataset $dataset --source $source --target $target --my_algorithm $algorithm

source="DAA_mse"
python main.py --txtdir $txtdir --dataset $dataset --source $source --target $target --my_algorithm $algorithm

python changeSize.py --n 900
# this 900 means 900 samples from each source, in order to set training sample size the same
# 900 * 2 = 1800
source="60 120"
python main.py --txtdir $txtdir --dataset $dataset --source $source --target $target --my_algorithm $algorithm

python changeSize.py --n 450
# 450 * 4 = 1800
source="30 60 120 150"
python main.py --txtdir $txtdir --dataset $dataset --source $source --target $target --my_algorithm $algorithm 