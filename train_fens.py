from __future__ import print_function

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gc
from keras.callbacks import LambdaCallback
from new_model import create_model, create_model_2d
# import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import os
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
import sys
import numpy as np
import argparse
import random
import pickle
import keras.backend as K
#### Stop the model training when 0.002 to get the best result in the paper!!!!

os.environ["CUDA_VISIBLE_DEVICES"] = "0";
parser = argparse.ArgumentParser()

def get_params():
    parser.add_argument ('--tor_len', required=False, default=500)
    parser.add_argument ('--exit_len', required=False, default=800)
    parser.add_argument ('--win_interval', required=False, default=5)
    parser.add_argument ('--num_window', required=False, default=11)
    parser.add_argument ('--alpha', required=False, default=0.1)
    parser.add_argument ('--input', required=False, default='./data/datasets/train_data/crawle_new_overlap_interval')
    parser.add_argument ('--test', required=False, default='./data/datasets/test_data/crawle_overlap_new2021_interval')
    parser.add_argument ('--model', required=False, default="./data/model/crawle_overlap_new2021_")
    args = parser.parse_args ()
    return args

def get_session(gpu_fraction=0.85):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def load_whole_seq_new(tor_seq,exit_seq,circuit_labels,test_c,train_c,model_gb):
    train_window1=[]
    train_window2=[]
    test_window1=[]
    test_window2=[]
    window_tor=[]
    window_exit=[]

    window_tor_size = []
    window_exit_size = []
    window_tor_ipd = []
    window_exit_ipd = []
    print("extract both ipd and size features...")
    for i in range(len(tor_seq)):
        window_tor_size.append([float(pair["size"])/1000.0 for pair in tor_seq[i]])
        window_exit_size.append([float(pair["size"]) / 1000.0 for pair in exit_seq[i]])
        window_tor_ipd.append ([float(pair["ipd"])* 1000.0 for pair in tor_seq[i]])
        window_exit_ipd.append ([float(pair["ipd"])* 1000.0 for pair in exit_seq[i]])
    print('window_tor_size', np.array(window_tor_size, dtype=object).shape)
    print('window_exit_size', np.array(window_exit_size, dtype=object).shape)
    print('window_tor_ipd', np.array(window_tor_ipd, dtype=object).shape)
    print('window_exit_ipd', np.array(window_exit_ipd, dtype=object).shape)
    window_tor_ipd = np.array(window_tor_ipd, dtype=object)
    window_exit_ipd = np.array(window_exit_ipd, dtype=object)

    # Change the first idp to 0 across all windows.
    new_window_tor_ipd = []
    new_window_exit_ipd = []
    for trace in window_tor_ipd:
        new_trace = [0]+list(trace[1:])
        new_window_tor_ipd.append([ipd for ipd in new_trace])
    for trace in window_exit_ipd:
        new_trace = [0]+list(trace[1:])
        new_window_exit_ipd.append([ipd for ipd in new_trace])

    window_tor_ipd = new_window_tor_ipd
    window_exit_ipd = new_window_exit_ipd
    print('window_tor_ipd',window_tor_ipd[10][:10])
    print('window_exit_ipd',window_exit_ipd[10][:10])

    for i in range(len(window_tor_ipd)):
        window_tor.append(np.concatenate((window_tor_ipd[i], window_tor_size[i]), axis=None))
        window_exit.append(np.concatenate((window_exit_ipd[i], window_exit_size[i]), axis=None))

    window_tor = np.array(window_tor, dtype=object)
    window_exit = np.array(window_exit, dtype=object)
    print('window_tor', window_tor.shape)
    print('window_exit', window_exit.shape)

    for w, c in zip (window_tor, circuit_labels):
        if c in train_c:
            train_window1.append(w)
        elif c in test_c:
            test_window1.append(w)

    for w, c in zip (window_exit, circuit_labels):
        if c in train_c:
            train_window2.append(w)
        elif c in test_c:
            test_window2.append(w)

    print ('train_window1', np.array(train_window1, dtype=object).shape)
    print ('train_window2', np.array(train_window1, dtype=object).shape)

    return np.array(train_window1, dtype=object), np.array(train_window2, dtype=object), np.array(test_window1, dtype=object), np.array(test_window2, dtype=object), np.array(test_window1, dtype=object), np.array(test_window2, dtype=object)


if __name__ == '__main__':
    args = get_params()
    tf.compat.v1.keras.backend.set_session(get_session())

    model_gb = 'cnn1d'

    ## Params for time-based window
    interval = args.win_interval#5
    t_flow_size = int(args.tor_len)#500#400#238  # 238#150#184  # 238#264
    e_flow_size = int(args.exit_len)#800#330#140
    num_windows = int(args.num_window)#11#21#5
    window_index_list = np.arange(num_windows)

    pad_t = t_flow_size * 2
    pad_e = e_flow_size * 2

    alpha_value = float(args.alpha)#0.1

    train_windows1 = []
    valid_windows1 = []
    test_windows1 = []
    train_windows2 = []
    valid_windows2 = []
    test_windows2 = []
    train_labels = []
    test_labels = []
    valid_labels = []

    for i, window_index in enumerate(window_index_list):
        print(f"Processing {i} of {len(window_index_list)}...")
        addn = 2
        pickle_path = args.input+str(interval)+'_win'+ str(window_index) +'_addn'+ str(addn) +'_w_superpkt.pickle'


        with open (pickle_path, 'rb') as handle:
            traces = pickle.load (handle)
            tor_seq = traces["tor"]
            exit_seq = traces["exit"]
            labels = traces["label"]
            circuit_labels = np.array ([int (labels[i].split ('_')[0]) for i in range (len (labels))])

            print (tor_seq[0])

            circuit = {}
            for i in range(len(labels)):
                if labels[i].split ('_')[0] not in circuit.keys ():
                    circuit[labels[i].split ('_')[0]] = 1
                else:
                    circuit[labels[i].split ('_')[0]] += 1

            # No overlapping circuits between training and testing sets
            global test_c
            global train_c
            if window_index == 0:
                test_c = []
                train_c = []
                sum_ins = 2093
                keys = list (circuit.keys ())
                random.shuffle (keys)
                for key in keys:
                    if sum_ins > 0:
                        sum_ins -= circuit[key]
                        test_c.append (key)
                    else:
                        train_c.append (key)
                test_c = np.array (test_c).astype ('int')
                train_c = np.array (train_c).astype ('int')
            # print (train_c)
            print ('test_c', test_c)
            print ('train_c', train_c)
        ###########
        train_set_x1, train_set_x2, test_set_x1, test_set_x2, valid_set_x1, valid_set_x2 = load_whole_seq_new(tor_seq,exit_seq,circuit_labels,test_c,train_c,model_gb)

        temp_test1 = []
        temp_test2 = []

        print(train_set_x1.shape)
        print(valid_set_x1.shape)
        print(test_set_x1.shape)

        print('train_set_x1', train_set_x1.shape)
        for x in train_set_x1:
            train_windows1.append(np.reshape(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'), [-1, 1]))

        for x in valid_set_x1:
            valid_windows1.append(np.reshape(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'), [-1, 1]))

        for x in test_set_x1:
            temp_test1.append(np.reshape(np.pad(x[:pad_t], (0, pad_t - len(x[:pad_t])), 'constant'), [-1, 1]))

        for x in train_set_x2:
            train_windows2.append(np.reshape(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'), [-1, 1]))

        for x in valid_set_x2:
            valid_windows2.append(np.reshape(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'), [-1, 1]))

        for x in test_set_x2:
            temp_test2.append(np.reshape(np.pad(x[:pad_e], (0, pad_e - len(x[:pad_e])), 'constant'), [-1, 1]))

        print('temp_test1: ', np.array(temp_test1).shape)
        print('temp_test2: ', np.array(temp_test2).shape)
        test_windows1.append(np.array(temp_test1))
        test_windows2.append(np.array(temp_test2))

    np.savez_compressed(args.test+str(interval)+'_test' + str(num_windows) + 'addn'+str(addn)+'_w_superpkt.npz',
             tor=np.array(test_windows1),
             exit=np.array(test_windows2))

    train_windows1 = np.array(train_windows1)
    valid_windows1 = np.array(valid_windows1)

    train_windows2 = np.array(train_windows2)
    valid_windows2 = np.array(valid_windows2)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    valid_labels = np.array(valid_labels)

    print('train_windows1: ', np.array(train_windows1).shape)
    print('train_windows2: ', np.array(train_windows2).shape)
    print('test_windows1: ', np.array(test_windows1).shape)
    print('test_windows2: ', np.array(test_windows2).shape)

    input_shape1 = (pad_t, 1)
    input_shape2 = (pad_e, 1)

    shared_model1 = create_model(input_shape=input_shape1, emb_size=64, model_name='tor')  ##
    shared_model2 = create_model(input_shape=input_shape2, emb_size=64, model_name='exit')  ##

    anchor = Input(input_shape1, name='anchor')
    positive = Input(input_shape2, name='positive')
    negative = Input(input_shape2, name='negative')

    a = shared_model1(anchor)
    p = shared_model2(positive)
    n = shared_model2(negative)

    print('a shape', a.shape)
    print('p shape', p.shape)
    print('n shape', n.shape)
    pos_sim = Dot(axes=-1, normalize=True)([a, p])
    neg_sim = Dot(axes=-1, normalize=True)([a, n])
    print('pos_sim shape', pos_sim.shape)
    print('neg_sim shape', neg_sim.shape)


    # customized loss
    def cosine_triplet_loss(X):
        _alpha = alpha_value
        positive_sim, negative_sim = X

        losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
        # if similarity is based on the distance functions, use below
        # losses = K.maximum(0.0, positive_sim - negative_sim + _alpha)
        return K.mean(losses)


    loss = Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim, neg_sim])

    model_triplet = Model(
        inputs=[anchor, positive, negative],
        outputs=loss)
    print(model_triplet.summary())

    opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


    def identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)


    model_triplet.compile(loss=identity_loss, optimizer=opt)

    batch_size = 128  # batch_size_value


    def intersect(a, b):
        return list(set(a) & set(b))


    def build_similarities(conv1, conv2, tor_t, exit_t):

        tor_embs = conv1.predict(tor_t)
        exit_embs = conv2.predict(exit_t)
        all_embs = np.concatenate((tor_embs, exit_embs), axis=0)

        del tor_t
        del exit_t
        del tor_embs
        del exit_embs

        all_embs = all_embs / np.linalg.norm(all_embs, axis=-1, keepdims=True)
        mid = int(len(all_embs) / 2)
        all_sims = np.dot(all_embs[:mid], all_embs[mid:].T)
        
        del all_embs
        gc.collect()

        return all_sims


    def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
        # If no similarities were computed, return a random negative
        if similarities is None:
            # print(neg_imgs_idx)
            # print(anc_idxs)
            anc_idxs = list(anc_idxs)
            valid_neg_pool = neg_imgs_idx  # .difference(anc_idxs)
            print('valid_neg_pool', valid_neg_pool.shape)
            return np.random.choice(valid_neg_pool, len(anc_idxs), replace=False)
        final_neg = []
        # for each positive pair
        for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
            anchor_class = anc_idx
            # print('anchor_class',anchor_class)
            valid_neg_pool = neg_imgs_idx  # .difference(np.array([anchor_class]))
            # positive similarity
            sim = similarities[anc_idx, pos_idx]
            # find all negatives which are semi(hard)
            possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
            possible_ids = intersect(valid_neg_pool, possible_ids)
            appended = False
            for iteration in range(num_retries):
                if len(possible_ids) == 0:
                    break
                idx_neg = np.random.choice(possible_ids, 1)[0]
                if idx_neg != anchor_class:
                    final_neg.append(idx_neg)
                    appended = True
                    break
            if not appended:
                final_neg.append(np.random.choice(valid_neg_pool, 1)[0])
            
        del similarities
        gc.collect()

        return final_neg


    class SemiHardTripletGenerator():
        def __init__(self, Xa_train, Xp_train, batch_size, neg_traces_train_idx, Xa_train_all, Xp_train_all, conv1, conv2):
            self.batch_size = batch_size  # 128

            self.Xa = Xa_train
            self.Xp = Xp_train
            self.Xa_all = Xa_train_all
            self.Xp_all = Xp_train_all
            self.cur_train_index = 0
            self.num_samples = Xa_train.shape[0]
            self.neg_traces_idx = neg_traces_train_idx

            if conv1:
                self.similarities = build_similarities(conv1, conv2, self.Xa_all, self.Xp_all)  # compute all similarities including cross pairs
                
            else:
                self.similarities = None

        def next_train(self):
            while 1:
                print(f"\nself.curr_train_index: {self.cur_train_index}")
                if self.cur_train_index >= self.num_samples:
                    self.cur_train_index = 0  # initialize the index for the next epoch

                # fill one batch
                traces_a = np.array(range(self.cur_train_index,
                                          self.cur_train_index + self.batch_size))
                traces_p = np.array(range(self.cur_train_index,
                                          self.cur_train_index + self.batch_size))
                
                self.cur_train_index += self.batch_size
                
                traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)

                gc.collect()

                yield ([self.Xa[traces_a],
                        self.Xp[traces_p],
                        self.Xp_all[traces_n]],
                       np.zeros(shape=(traces_a.shape[0]))
                       )


    # At first epoch we don't generate hard triplets
    all_traces_train_idx = np.array(range(len(train_windows1)))
    gen_hard = SemiHardTripletGenerator(train_windows1, train_windows2, batch_size, all_traces_train_idx,
                                        train_windows1, train_windows2, None, None)
    nb_epochs = 10000
    description = 'coffeh2'

    best_loss = sys.float_info.max


    def saveModel(epoch, logs):
        global best_loss

        loss = logs['loss']

        if loss < best_loss:
            print("\nloss is improved from {} to {}. save the model".format(str(best_loss),
                                                                          str(loss)))

            best_loss = loss
            shared_model1.save_weights(
                args.model + str(num_windows) + "_interval"+str(interval)+ '_addn'+str(addn)+"_model1_w_superpkt.h5")
            shared_model2.save_weights(
                args.model + str(num_windows) + "_interval"+str(interval)+'_addn'+str(addn)+"_model2_w_superpkt.h5")
        else:
            print("\nloss is not improved from {}.".format(str(best_loss)))


    for epoch in range(nb_epochs):
        print("built new hard generator for epoch " + str(epoch))

        if epoch % 2 == 0:
            if epoch == 0:
                model_triplet.fit_generator(generator=gen_hard.next_train(),
                                            steps_per_epoch=train_windows1.shape[0] // batch_size - 1,
                                            epochs=1, verbose=1)
                del gen_hard.Xa
                del gen_hard.Xp
                del gen_hard.Xa_all
                del gen_hard.Xp_all
            
            else:
                model_triplet.fit_generator(generator=gen_hard.next_train(),
                                            steps_per_epoch=(train_windows1.shape[0] // 2) // batch_size - 1,
                                            epochs=1, verbose=1, callbacks=[LambdaCallback(on_epoch_end=saveModel)])
                del gen_hard.similarities
                del gen_hard.Xa
                del gen_hard.Xp
                del gen_hard.Xa_all
                del gen_hard.Xp_all
        else:
            model_triplet.fit_generator(generator=gen_hard.next_train(),
                                        steps_per_epoch=(train_windows1.shape[0] // 2) // batch_size - 1,
                                        epochs=1, verbose=1, callbacks=[LambdaCallback(on_epoch_end=saveModel)])
            del gen_hard.similarities
            del gen_hard.Xa
            del gen_hard.Xp
            del gen_hard.Xa_all
            del gen_hard.Xp_all
        gc.collect()
        mid = int(len(train_windows1) / 2)
        random_ind = np.array(range(len(train_windows1)))
        np.random.shuffle(random_ind)
        X1 = np.array(random_ind[:mid])
        X2 = np.array(random_ind[mid:])
        if (epoch + 1) % 2 == 1:
            gen_hard = SemiHardTripletGenerator(train_windows1[X1], train_windows2[X1], batch_size, X2, train_windows1,
                                                    train_windows2,
                                                    shared_model1, shared_model2) # prev gen_hard_odd
        else:
            gen_hard = SemiHardTripletGenerator(train_windows1[X2], train_windows2[X2], batch_size,
                                                     X1, train_windows1, train_windows2,
                                                     shared_model1, shared_model2) # prev gen_hard_even
