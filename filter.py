import numpy as np
import pickle
import os
import random
def find_key(input_dict, value):
    return {k for k, v in input_dict.items() if v == value}

def parse_csv(csv_path, interval,final_names, threshold):#option: 'sonly', 'tonly', 'both'
    #fw=open('/data/seoh/greaterthan50.txt','w+')
    HERE_PATH = csv_path+'inflow'
    THERE_PATH = csv_path+'outflow'
    print(HERE_PATH,THERE_PATH,interval)
    #here
    here=[]
    there=[]
    here_len=[]
    there_len=[]
    h_cnt = 0
    t_cnt = 0
    flow_cnt = 0
    file_names = []
    for txt_file in os.listdir(HERE_PATH):
        file_names.append(txt_file)

    #for txt_file in open('/data/seoh/greaterthan50.txt','r').readlines():
    #    file_names.append(txt_file.strip())

    for i in range(len(file_names)):
        here_seq = []
        there_seq = []
        num_here_big_pkt_cnt = []
        num_there_big_pkt_cnt = []

        with open(HERE_PATH+'/'+file_names[i]) as f:
            #print(HERE_PATH+'/'+file_names[i])
            h_lines=[]
            full_lines=f.readlines()
            for line in full_lines:
                time=float(line.split('\t')[0])
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                h_lines.append(line)



        with open (THERE_PATH + '/' + file_names[i]) as f:

            t_lines = []
            full_lines = f.readlines ()
            for line in full_lines:
                time = float (line.split ('\t')[0])
                if float (time) > interval[1]:
                    break
                if float (time) < interval[0]:
                    continue
                t_lines.append (line)
        if(len(h_lines)>threshold) and (len(t_lines)>threshold):
            if file_names[i] in final_names.keys():
                final_names[file_names[i]] += 1
            else:
                final_names[file_names[i]] = 1


    for x  in final_names:
        print(x,final_names[x])
def create_overlap_window_csv(csv_path, out_path, threshold, interval, num_windows, addnum):
    global final_names
    final_names={}
    fw = open (out_path, 'w+')
    for win in range(num_windows):
        parse_csv(csv_path, [win*addnum,win*addnum+interval],final_names, threshold)
        #np.savez_compressed('/project/hoppernj/research/seoh/new_dcf_data/new_overlap_interval' + str(interval) + '_win' + str(win) + '_addn' + str(addnum) + '.npz',
        #         tor=here, exit=there)
    for name in list(find_key(final_names, num_windows)):

        fw.write(name)
        fw.write ('\n')
    fw.close()

data_path = '/data/website-fingerprinting/datasets/CrawlE_Proc_100/'
out_file_path = '/data/seoh/CrawlE_Proc_100_files.txt'
threshold=10 # min number of packets per window in both ends, used  30 for 500
# That is, we drop the flow pairs if either of them has pkt count < threshold.
create_overlap_window_csv(data_path, out_file_path, threshold, 5, 11, 2)
