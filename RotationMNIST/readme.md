## 1. Image Preparation
- images.py

        The file used to download MNIST dataset, rotate them, and generate the corresponding .csv and .txt files.
- images.sh

        The script file to run images.py. You are free to choose the number of samples, the target rotation degree and the source rotation degrees combinations. 

## 2. DAA selection
- DAA.py 

        The file that can run DAA algorithm. It will generate .txt file that contains the selected samples, record number of samples collected from each source, and print running time.

        The selected samples are in domainbed/txtlist/RMnist/xxx.txt

- DAA.sh

        The script file to run DAA.py. You can choose k, visible_n_test, source, target and so on. Make sure to replace txtdir correctly and n should be no larger than n_train which you choosed in images.sh

## 3. Training and testing the network
- main.py

        The main file that contains the whole procedure of training and testing.

- train.sh

        The script file to run main.py, you can choose algorithms, source and target here. Once a source domain is choosed, all data in the .txt file will be used (you cannot change sample size here). If you want to change n, go back to the first two steps.

## Quick Start
run this command to generate images.
```bash
./images.sh
```
(We take total budget T = 1800 as example. So in images.sh n_train should be set as 1800)

run this command to select samples using DAA method.
```bash
./DAA.sh
```
(n should be set as 1800)

then in "results/samples.txt", there will be something like this:

        dis = mmd, n = 1800, K = 6, [82, 522, 1072, 124]
        dis = cosine, n = 1800, K = 6, [83, 443, 1188, 86]
        dis = mse, n = 1800, K = 6, [311, 533, 842, 114]
Four numbers represents the number of samples that DAA selected from four sources respectively.

run command to train and test
```bash
./train.sh
```
then "results/accuracies.txt" will look like this:

        ERM['DAA_cosine'],DAA_cosine: 0.9139,overall: 0.9139,90: 0.8725,overall: 0.8725,
        ERM['DAA_mse'],DAA_mse: 0.9306,overall: 0.9306,90: 0.8525,overall: 0.8525,
        ERM['DAA_mmd'],DAA_mmd: 0.8861,overall: 0.8861,90: 0.8525,overall: 0.8525,
        ERM['60', '120'],60: 0.9444,120: 0.9500,overall: 0.9472,90: 0.8625,overall: 0.8625,
        ERM['30', '60', '120', '150'],30: 0.9000,60: 0.9111,120: 0.8556,150: 0.8556,overall: 0.8806,90: 0.7892,overall: 0.7892,

You only need to look at the last number in a row, that is the test accuracy on test domain.
