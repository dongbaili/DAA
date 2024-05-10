## Download and preprocess data
- data.py

        The file used to download folktable dataset and process it into many seperate .npy file (in order to loader faster later. this may take some time). Make sure you have the permission to download that data.

## DAA + train and test
- main.py

        The main file that contains data selection methods including DAA, top3, top10 and uniform. The training and testing process is also in this file.

- example.sh

        The script file to run main.py. You are free to choose task, test_state, k, distance_metrics and other hyper-parameters.

## Quick Start
run this command to download and prepare data
```bash
python3 data.py
```
run this commance to run the algorithm
```bash
./example.sh
```

then in dir "results/ACSEmployment_HI",
you will get 3 files: erm.txt, reweight.txt and self_supervise.txt.

each of them will look like this:

        mmd, k=5, T=2000
        0.7620138888888889, 0.7602500000000001, 0.7562777777777777, 0.7605416666666667 
        cosine, k=5, T=2000
        0.7653472222222223, 0.7573055555555556, 0.7592777777777778, 0.758361111111111 
        mse, k=5, T=2000
        0.7641388888888889, 0.7604861111111111, 0.7626111111111111, 0.7584166666666667

4 numbers in a row means the test accuracy of "ours, top3, top10, uniform" respectively.

