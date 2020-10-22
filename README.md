>📋  README.md for code

# My Paper Title

This repository is the official implementation of [Fair Hierarchical Clustering](https://arxiv.org/abs/2006.10221). 

Author: Y. Wang

Email: yuyanw@andrew.cmu.edu

## Requirements

To install required python packages:

```setup
pip install -r requirements.txt
```

>📋  Download data at the following links:

census - https://archive.ics.uci.edu/ml/datasets/census+income  filename: adult.data

bank - https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  filename: bank-full in bank.zip

Put these two files in the same directory as the .py files.

## Data Preprocessing

For data preprocessing, run the following files:

data_preprocess.py


>📋  In the main file, change the dataset names depending on which dataset you want to use, and output dataset names and output directory according to user preferrence.

See the file for more detailed instructions

Important functions: "load_data" for preprocessing data with two colors, "load_data_multi_color" for preprocessing data with multiple colors.

The user can also explore more settings of fairness by changing the parameter values in these two functions.

Default output filenames we use:

Two colors:

Dataset  Protected Feature  Fairness (b:r)  Output File Name
Census   Gender             1:3             adult.csv
Census   Race               1:7             adult_r.csv
Bank     Marital Status     1:2             bank.csv
Bank     Age                2:3             bank_a.csv

Multiple colors:

Dataset  Protected Feature  Fairness (1/c)  Output File Name
Census   Age            	1/3           	adult_4_color.csv
Bank     Age                1/3             bank__4_color.csv



## Running Time Test

For run time test, run the following file:

test_script_run_time.py - for value objective
test_script_run_time_moseley_wang.py - for revenue objective

>📋  The file saves data to a chosen directory. See the main function and create the directory in the same folder as the .py file in advance. The user can 

also customize the name of this directory, but remember to also customize it later when tidying up or plotting the results.



## Running Time Plot

To plot the growth of running time against sample size and see the contrast between the fairlet local search algorithm and vaniila average linkage, run

the following file:

run_time_plot.py

The code will produce a picture in ".pdf" format in the current repository.


## Validation: two colors, value objective 

Run the following file:

test_script_random_validation.py - for validating with a random fairlet decomposition

The other stats are collected while running:

test_script_run_time.py



## Validation: two colors, revenue objective

Run the following files:

test_script_random_validation_moseley_wang.py - for validating with a random fairlet decomposition

The other stats are collected while running:

test_script_run_time.py


## Validation: multiple colors, value objective

Run the following file:

test_script_validation_multi_color.py


## Validation: multiple colors, revenue objective

Run the following file:

test_script_validation_multi_color_moseley_wang.py


## Result Collection

Run the following file:

data_tidy_up.py - two colors

data_tidy_up_multi_color.py - multi colors

>📋  The user will need to manually change the result directory in the main function. The code outputs the mean of every data over the 5 instances.


## Plot the growth of two objective: fairness objective function for fairlets and value objective

Run the following file:

plot_two_objs.py

The user can change the size of the subsample if needed.


