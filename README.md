## How to run DeepCoFFEA 

0. - Download our new data from [here](https://drive.google.com/file/d/1ZYFXfESD15SAR4Q8hsoVYdTHpTD8Orys/view?usp=sharing).
   The folder has files which are traces whose names are *circuit-index*_*site-index*. This mean, if files have same circuit_index in their names, they’re collected using the same circuit.
   
   Each line in the file consists of "*time_stamp*\t*packet_size*”. So you can split each line based on ‘\t’. For example, line.spilt(‘\t’)[0] is Time_stamp and line.spilt(‘\t’)[1] is pkt size. 
   
   [Here](https://drive.google.com/drive/folders/1PG0sF6AHHn_2LxyoIztwjpoxDmB7r39z?usp=sharing) is window-separated data we already preprocessed and used in the paper (so you can skip the following 1 and 2 steps)
   
   
   - Download timegap data from [here](https://drive.google.com/drive/folders/1JUC-KBghWX42yg19gYDcrospyuE16d6X?usp=sharing).
   
   - Download DeepCorr data from [here](https://drive.google.com/drive/folders/1Z4PyMCX99xME3T_LLvURejSfisP9jy4n?usp=sharing). 


1. Gather flow pairs whose packet counts > threshold per window. Before running code, modify line 82-86 as follows.

```
data_path = '/data/website-fingerprinting/datasets/CrawlE_Proc_100/' #input data path
out_file_path = '/data/seoh/CrawlE_Proc_100_files.txt' #output text file path to record flow pairs (i.e.,file_names)
threshold=20 # min number of packets per window in both ends
create_overlap_window_csv(data_path, out_file_path, threshold, 5, 11, 2) # We're using 11 windows and each window lasts 5 sec
```
Then, run the code
```
python filter.py
```
2. Create formatted-input pickles to feed them to triplet network.  Before running code, modify line 206-209 as follows.
```
data_path = '/data/website-fingerprinting/datasets/CrawlE_Proc/' #input data path
file_list_path = '/data/seoh/greaterthan50_final_burst.txt' # the path to txt file (we got from filter.py)
prefix_pickle_output = '/data/website-fingerprinting/datasets/new_dcf_data/crawle_new_overlap_interval' #output path
create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 5, 11, 2) # We're using 11 windows and each window lasts 5 sec
```
Then, run the code. This code will create 11 pickle files in which each file carries the partial trace for each window.
```
python new_dcf_parse.py
```

3. Train FENs using pickle files created by new_dcf_parse.py. Configure the arguments as needed. For example,
```
--input: path to pickle files
--model: path to save the trained models
--test: path to save a testing set
```

```bash
python train_fens.py (--input <your_input_path> --model <your_model_path> --test <test_set_path>)
```

This script will save testing npz file in <test_set_path>, and trained models in <your_model_path>.

We stopped training when loss = 0.006 (DeepCorr set) and loss = 0.002 (our new set).

4. Evaluate trained FENs using trained two models and test dataset (eval_dcf.py). Configure arguments as needed. For example, 
```
--flow: number of flow pairs in the testing set (by default, use 2094)
--test: path to the testing set
--model1: path to the model 1 (tor embedding network)
--model2: path to the model 2 (exit embedding network)
--output: path to save the result ( it's formatted as [threshold|tpr|fpr|bdr])
```

```bash
python eval_dcf.py (--input <your_input_path> --model1 <your_model1_path> --model2 <your_model2_path> --output <your_output_path>)
```

The script above will generate TPRs, FPRs, and BDRs when using 9 out 11 window results. 
