import tensorflow as tf
from new_model import create_model
import numpy as np
import os
# import keras.backend.tensorflow_backend as ktf
from sklearn.metrics.pairwise import cosine_similarity
import csv
import time
import argparse


total_emb = 0
total_vot = 0
total_cos = 0

# if you don't use GPU, comment out the following
os.environ["CUDA_VISIBLE_DEVICES"] = "0";


parser = argparse.ArgumentParser ()
parser.add_argument('--test', default='./data/datasets/test_data/10k_test.npz')
parser.add_argument ('-flow', default=10000)
parser.add_argument ('-tor_len', default=500)
parser.add_argument ('-exit_len', default=800)
parser.add_argument('--model1',
                    default='./data/model/crawle_overlap_new2021_11_interval5_addn2_model1_w_superpkt_0.003036')
parser.add_argument('--model2',
                    default='./data/model/crawle_overlap_new2021_11_interval5_addn2_model2_w_superpkt_0.003036')
parser.add_argument ('-output', default="./data/results/crawle_dcf0.002_"+str(11)+"_0.02_interval5_addn2_238_800.csv")
# 使用np.savez保存多个数组
# np.savez('my_arrays.npz', big_float_array=big_float_array, array1=array1, array2=array2)

args = parser.parse_args ()

def get_session(gpu_fraction=0.85):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def ini_cosine_output(single_output_l, input_number):
    for pairs in range (0, (input_number * input_number)):
        single_output_l.append (0)  # ([0])
    # print("......done!")


def Cosine_Similarity_eval(tor_embs, exit_embs, similarity_threshold, single_output_l, evaluating_window, last_window,
                           correlated_shreshold, cosine_similarity_all_list, muti_output_list):

    global total_vot
    # print('single_output_l ',np.array(single_output_l).shape)
    number_of_lines = tor_embs.shape[0]
    start_emd = time.time ()
    for tor_emb_index in range (0, number_of_lines):
        t = similarity_threshold[tor_emb_index]
        constant_num = int (tor_emb_index * number_of_lines)
        for exit_emb_index in range (0, number_of_lines):
            if (cosine_similarity_all_list[tor_emb_index][exit_emb_index] >= t):
                # print('single_output_l[constant_num + exit_emb_index] ',single_output_l[constant_num + exit_emb_index])
                single_output_l[constant_num + exit_emb_index] = single_output_l[constant_num + exit_emb_index] + 1

    if (evaluating_window == last_window):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        confusion_matrix = []
        # now begin to evaluate
        # print("evaluating .......")
        for tor_eval_index in range (0, tor_embs.shape[0]):
            confusion_matrix.append([0 for _ in range(tor_embs.shape[0])])
            for exit_eval_index in range (0, tor_embs.shape[0]):
                cos_condithon_a = (tor_eval_index == exit_eval_index)
                number_of_ones = (single_output_l[(tor_eval_index * (tor_embs.shape[0])) + exit_eval_index])
                cos_condition_b = (number_of_ones >= correlated_shreshold)
                cos_condition_c = (number_of_ones < correlated_shreshold)
                
                if cos_condition_b:
                    confusion_matrix[tor_eval_index][exit_eval_index] += 1

                if (cos_condithon_a and cos_condition_b):
                    TP = TP + 1
                if (cos_condithon_a and cos_condition_c):
                    FN = FN + 1
                if ((not (cos_condithon_a)) and cos_condition_b):
                    FP = FP + 1
                if ((not (cos_condithon_a)) and cos_condition_c):
                    TN = TN + 1

        if ((TP + FN) != 0):
            TPR = (float) (TP) / (TP + FN)
        else:
            TPR = -1

        if ((FP + TN) != 0):
            FPR = (float) (FP) / (FP + TN)
        else:
            FPR = -1

        muti_output_list.append (TPR)
        muti_output_list.append (FPR)
        muti_output_list.append (calculate_bdr (TPR, FPR))
        np.save("D:\\Desktop\\cm.npy",confusion_matrix)
        print(TPR,FPR,calculate_bdr (TPR, FPR))

    # print(".....done!")
    end_time = time.time ()
    total_vot = total_vot + (end_time - start_emd)


def calculate_bdr(tpr, fpr):
    TPR = tpr
    FPR = fpr
    c = 1 / int(args.flow)
    u = (int(args.flow)-1) / int(args.flow)
    if ((TPR * c) + (FPR * u)) != 0:
        BDR = (TPR * c) / ((TPR * c) + (FPR * u))
    else:
        BDR = -1
    return BDR


def preprocessing_new_test_data(win, number_of_interval):
    npz_path = '../data/obfs_new/obfs4_new_interval' + str (number_of_interval) + '_win' + str (win) + '.npz'
    np_array = np.load (npz_path, encoding='latin1', allow_pickle=True)
    tor_seq = np_array["tor"]
    exit_seq = np_array["exit"]
    number_of_traces = tor_seq.shape[0]
    print(f"number_of_traces={number_of_traces}")
    print(f"type(tor_seq[0])={type(tor_seq[0])}")
    print(f"len(tor_seq[0])={len(tor_seq[0])}")
    print(f"tor_seq[0][1]={tor_seq[0][1]}")
    '''
    [ [{'ipd': x, 'size': y},{}.....{}]
      [{},{}.....{}]
      [{},{}.....{}] ]
    '''

    for i in range (0, number_of_traces):
        tor_seq[i] = [float (pair["ipd"]) * 1000.0 for pair in tor_seq[i]] + [float (pair["size"]) / 1000.0 for pair in
                                                                              tor_seq[i]]
        if len (tor_seq[i]) < (500 * 2):
            tor_seq[i] = tor_seq[i] + ([0] * ((500 * 2) - (len (tor_seq[i]))))
        elif len (tor_seq[i]) > (500 * 2):
            tor_seq[i] = tor_seq[i][0:(500 * 2)]

        exit_seq[i] = [float (pair["ipd"]) * 1000.0 for pair in exit_seq[i]] + [float (pair["size"]) / 1000.0 for pair
                                                                                in exit_seq[i]]
        if len (exit_seq[i]) < (800 * 2):
            exit_seq[i] = exit_seq[i] + ([0] * ((800 * 2) - (len (exit_seq[i]))))
        elif len (exit_seq[i]) > (800 * 2):
            exit_seq[i] = exit_seq[i][0:(800 * 2)]

    tor_test = np.reshape (np.array (list (tor_seq)), (10000, 1000, 1))
    exit_test = np.reshape (np.array (list (exit_seq)), (10000, 1600, 1))
    print (tor_test[0][1])
    return (tor_test, exit_test)


# Every tor flow will have a unique threshold
def threshold_finder(input_similarity_list, curr_win, gen_ranks, thres_seed, use_global):
    output_shreshold_list = []
    for simi_list_index in range (0, len (input_similarity_list)):
        correlated_similarity = input_similarity_list[simi_list_index][simi_list_index]
        temp = list (input_similarity_list[simi_list_index])
        temp.sort (reverse=True)

        cut_point = int ((len (input_similarity_list[simi_list_index]) - 1) * ((thres_seed) / 100))
        if use_global == 1:
            output_shreshold_list.append (thres_seed)  # temp[cut_point]
        elif use_global != 1:
            output_shreshold_list.append (temp[cut_point])  # temp[cut_point]
    return output_shreshold_list


def eval_model(full_or_half, five_or_four, use_new_data, model1_path, model2_path, test_path, thr, use_global,
               muti_output_list, soft_muti_output_list):
    global total_emb
    global total_vot
    global total_cos


    test_data = np.load (test_path, allow_pickle=True)
    # print(test_data['tor'])
    # print("-------ex-----------")
    # print(test_data['exit'])
    print(test_data['tor'][0].shape)
    print(test_data['exit'][0].shape)
    pad_t = int(args.tor_len)*2#500*2 #238*2
    pad_e = int(args.exit_len)*2#800*2 #100*2

    tor_model = create_model (input_shape=(pad_t, 1), emb_size=64, model_name='tor')
    exit_model = create_model (input_shape=(pad_e, 1), emb_size=64, model_name='exit')

    # load triplet models for tor and exit traffic
    tor_model.load_weights (model1_path + ".h5")
    exit_model.load_weights (model2_path + ".h5")

    # print('Get logits for 5 windows')

    # This list will be the output list of cosine similarity approach.
    # should have 2093*2093 sub-sublists which are corrsponding to the number of pairs like t0e0, t0e1...
    # And each sub-sub-sublist should have 5 elements which are corrsponding to the status of correlation
    single_output = []

    # This list should have 2093 sub-sublists for each tor flow.
    # Each sublist should have 2093 elements which corrsponding to 2093 similarities of each tor flow
    # No need to use 5 list for each window. We can just overwrite the old table
    # used by threshold_finder()
    cosine_similarity_table = []
    threshold_result = []

    # below are the code that are used for controlling the behavior of the program

    activated_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    last_activated_window = 10
    correlated_shreshold_value = five_or_four
    thres_seed = thr

    for win in range (11):
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ We are in window %d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" % win)
        pos_dist = []
        neg_dist = []
        # Get feature embeddings for each window
        # For example, test_data['tor'][0] means testing tor traffic in window1
        #              test_data['exit'][win] means testing exit traffic in window1

        if use_new_data == 1:
            (test_data_tor, test_data_exit) = preprocessing_new_test_data (win, 5)
        elif use_new_data != 1:
            test_data_tor = test_data['tor'][win][:full_or_half]
            test_data_exit = test_data['exit'][win][:full_or_half]
        start_emd = time.time ()
        tor_embs = tor_model.predict (test_data_tor)
        exit_embs = exit_model.predict (test_data_exit)
        end_emd = time.time ()
        # print('[#####] Time for embedding: ', end_emd - start_emd, 'sec')
        total_emb = total_emb + (end_emd - start_emd)

        # >>>>>>>>>> below are the code for cosine similarity

        if win == 0:
            # print("init the final cosine similarity output now.....")
            ini_cosine_output (single_output, tor_embs.shape[0])
        # print("getting cosine similarity results......")
        start_cos = time.time ()
        cosine_similarity_table = cosine_similarity (tor_embs, exit_embs)
        end_cos = time.time ()
        # print('[#####] Time for cosine: ', end_cos - start_cos, 'sec')
        total_cos = total_cos + (end_cos - start_cos)
        threshold_result = threshold_finder (cosine_similarity_table, win, 0, thres_seed, use_global)

        if win in activated_windows:
            Cosine_Similarity_eval (tor_embs, exit_embs, threshold_result, single_output, win, last_activated_window,
                                    correlated_shreshold_value, cosine_similarity_table, muti_output_list)



if __name__ == "__main__":
    # if you don't use GPU, comment out the following
    tf.compat.v1.keras.backend.set_session(get_session())
    start_time = time.time()
    test_path = args.test
    model1_path = args.model1
    model2_path = args.model2

    # For time complexity analysis, use only one threshold (e.g., [60])
    #rank_thr_list = [60,50,47,43,40,37,33,28,24,20,16.667,14,12.5,11,10,9,8.333,7,6.25,5,4.545,3.846,2.941,1.667,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]#[10]
    rank_thr_list = [10]

    num_of_thr = len (rank_thr_list)


    flow_length = int (args.flow)

    five_or_four = 9

    rank_multi_output = []
    five_rank_multi_output = []
    for i in range (0, num_of_thr):
        rank_multi_output.append ([(rank_thr_list[i])])
        five_rank_multi_output.append ([(rank_thr_list[i])])

    epoch_index = 0
    use_global = 0
    use_new_data = 0

    for thr in rank_thr_list:
        eval_model (flow_length, five_or_four, use_new_data, model1_path, model2_path, test_path, thr, use_global,
                    rank_multi_output[epoch_index], [])
        epoch_index = epoch_index + 1
    end_time = time.time ()
    
    with open(args.output, "w", newline="") as rank_f:
        writer = csv.writer(rank_f)
        writer.writerows(rank_multi_output)

    print ("total: ", str (end_time-start_time))



