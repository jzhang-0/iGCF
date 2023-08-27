# iGCF Source Code

This repository contains the source code for the paper titled "Interactive Graph Convolutional Filtering". 

## Datasets

The three datasets used in the paper can be downloaded from the following links:

- Dataset 1: [KuaiRec](https://kuairec.com/)
- Dataset 2: [Movielens-1m](https://grouplens.org/datasets/movielens/)
- Dataset 3: [EachMovie](http://www.gatsby.ucl.ac.uk/~chuwei/data/EachMovie/eachmovie.html)

After downloading the datasets, change the data paths accordingly in the file [src/script/load_data.py](src/script/load_data.py).
<!-- ## How to Run the Code
1. Install the necessary dependencies (it's recommended to do this in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
2. Set model parameter, run the model:
    ```bash
    python run.py
    ``` -->

## Core Code
The core model is implemented in the file [iGCF](src/model/GCNICF_Meta_V2_dir/GCNICF_Meta_V2_f.py).

## Dependencies
See `requirements.txt` for details
