
# PatchLinear for Single-step Forecasting and Anomaly Detection

Model for univariate time series single-step forecasting and anomaly detection.


## Requirements

* Python==3.9.0
* numpy==1.26.3
* pandas==2.1.4
* scikit_learn==1.4.1
* scipy==1.11.4
* matplotlib==3.7.5
* torch==2.1.1

## Approach Directory

* data_provider: Dataset preprocessing
* logs: Log directory, storing result logs of each dataset
* util: Utility library
  * index.py: Calculate the position of the most similar historical subsequence to the current subsequence
  * tools.py: Anomaly label adjustment, plotting, rolling calculation of subsequence mean and variance
* models: Models of different Methods
* layers: Layers that execute the model
* scripts: Executable files directory
* exp: Main files directory

## Usage

1. Dataset
   
   1. Place all datasets in the same directory, such as the `./dataset/` directory
   2. The organization of each dataset file is as follows:
      1. KPI
         1. train/*.csv
         2. test/*.csv
         3. The same name sequence in the train and test directories corresponds to the same sequence, and the sequence consists of timestamp, value, label
      2. Yahoo
         1. real/*.csv
         2. synthetic/*.csv
         3. A3/*.csv
         4. A4/*.csv
      3. MSLU: As it is part of the NASA MSL dataset, it can be directly accessed from the NASA dataset
         1. train/*.npy   Storing training set sequences
         2. test/*.npy    Storing test set sequences
         3. test_label/*.csv Storing corresponding label data
      4. NAB
         1. *.csv Directly place the corresponding sequence data

2. Adapt the environment with `requirement.txt`:
   
   ```bash
   pip install -r requirement.txt
   ```

3. Run the executable files in the corresponding executable files directory to start the algorithm.

## Running

You can run the following command to perform single-step forecasting or anomaly detection on the sequences in the corresponding dataset:

```bash
sh ./scripts/PatchLinear/Yahoo.sh
```

The settable parameters are explained in the `run_longExp.py` file.

## Acknowledgement
We appreciate the following github repo very much for the valuable code base and datasets:

[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

[https://github.com/yuqinie98/PatchTST/](https://github.com/yuqinie98/PatchTST/)

[https://github.com/ts-kim/RevIN](https://github.com/ts-kim/RevIN)

## Contact

If you have any questions or concerns, please contact me: zhengmh1229@stu.xmu.edu.cn or submit an issue

