import numpy as np
import pickle

def parse_csv(csv_path, interval,file_names):#option: 'sonly', 'tonly', 'both'
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
    final_names = []
    #for txt_file in os.listdir(HERE_PATH):
    #    file_names.append(txt_file)


    for i in range(len(file_names)):
        here_seq = []
        there_seq = []
        num_here_big_pkt_cnt = []
        num_there_big_pkt_cnt = []
        with open(HERE_PATH+'/'+file_names[i]) as f:
            #print(HERE_PATH+'/'+txt_file)
            pre_h_time = 0.0
            lines=f.readlines()
            if len(lines) == 0:
                continue
            #print('befoer filter',len(lines))
            big_pkt = []
            num_here_big_pkt = 0
            for line in lines:
                time_size = []
                time=float(line.split('\t')[0])
                size =int(line.split ('\t')[1])
                if size > 0:
                    ipd = time - pre_h_time
                else:
                    ipd = -(time - pre_h_time)
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                if abs(size) > 512:# ignore ack packets
                    if (pre_h_time != 0) and (ipd == 0):
                        big_pkt.append(size)
                        continue
                    if len(big_pkt)!=0:
                        last_pkt=here_seq.pop()
                        here_seq.append({"ipd": last_pkt["ipd"], "size": sum(big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_here_big_pkt += 1
                    time_size.append (ipd)
                    time_size.append(size)
                    time_size = np.array (time_size)
                    here_seq.append ({"ipd": time_size[0], "size": time_size[1]})
                    pre_h_time = time
        #print('after filter',len (here_seq))

        with open (THERE_PATH + '/' + file_names[i]) as f:
            pre_h_time = 0.0
            lines = f.readlines ()
            if len (lines) == 0:
                continue
            big_pkt = []
            num_there_big_pkt = 0
            for line in lines:
                time_size = []
                time = float (line.split ('\t')[0])
                size = int (line.split ('\t')[1])
                if size > 0:
                    ipd = time - pre_h_time
                else:
                    ipd = -(time - pre_h_time)
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                if abs (size) > 66:  # ignore ack packets
                    if (pre_h_time != 0) and (ipd == 0):
                        big_pkt.append (size)
                        continue
                    if len (big_pkt) != 0:
                        last_pkt = there_seq.pop ()
                        there_seq.append ({"ipd": last_pkt["ipd"], "size": sum (big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_there_big_pkt += 1

                    time_size.append (ipd)
                    time_size.append (size)
                    time_size = np.array (time_size)
                    there_seq.append ({"ipd": time_size[0], "size": time_size[1]})
                    pre_h_time = time
        #print (len (there_seq))
        if (len(here_seq)!=0) and (len(there_seq)!=0):
            here_len.append (len (here_seq))
            num_here_big_pkt_cnt.append (num_here_big_pkt)
            there_len.append (len (there_seq))
            num_there_big_pkt_cnt.append (num_there_big_pkt)
            here.append (here_seq)
            there.append (there_seq)
            final_names.append(file_names[i])
            flow_cnt += 1
        #h_cnt += len (here_seq)
        #t_cnt += len (there_seq)

    print(interval,'mean',np.mean(np.array(here_len)), np.mean(np.array(there_len)),np.mean(num_here_big_pkt_cnt),np.mean(num_there_big_pkt_cnt),flow_cnt)
    print (interval,'median', np.median (np.array (here_len)), np.median (np.array (there_len)),np.median(num_here_big_pkt_cnt),np.median(num_there_big_pkt_cnt),flow_cnt)
    return np.array(here), np.array(there), np.array(final_names)

def create_overlap_window_csv(csv_path, file_list, prefix_pickle_output, interval, num_windows, addnum):
    windows_seq = []
    file_names = []
    for txt_file in open(file_list,'r').readlines():
        file_names.append(txt_file.strip())
    for win in range(num_windows):
        here, there, labels = parse_csv(csv_path, [win*addnum,win*addnum+interval],file_names)
        windows_seq.append({"tor": here, "exit": there, "label": labels})


        with open (prefix_pickle_output + str(interval) + '_win' + str(win) + '_addn' + str(addnum) + '_w_superpkt.pickle', 'wb') as handle:
            pickle.dump ({'tor':here, 'exit':there, "label": labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    windows_seq = np.array(windows_seq)

    return windows_seq

data_path = '/data/website-fingerprinting/datasets/CrawlE_Proc/'
file_list_path = '/data/seoh/greaterthan50_final_burst.txt'
prefix_pickle_output = '/data/website-fingerprinting/datasets/new_dcf_data/crawle_new_overlap_interval'
create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 5, 11, 2)
#create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 4, 11, 2)
#create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 3, 11, 2)
#create_overlap_window_csv(npz_path, 5, 11, 1)
#create_overlap_window_csv(npz_path, 3, 11, 2)