# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
label_list=['星座','时政','科技','时尚','家居','游戏','娱乐','教育','股票','财经','社会','彩票','体育','房产']
label_id = {i:label for i, label in enumerate(label_list)}
file1=open('1.txt','w',encoding='utf-8')
if __name__ == '__main__':
    dataset = 'THUCNews' 
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    print("Loading data...")
    test_data = build_dataset(config)
    
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    for i, (trains, labels) in enumerate(test_iter):
            #print(trains)
            outputs = model(trains)
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            #print(predic,len(predic))
            for j in predic:
               label=label_id[j]
               file1.write(label+'\n')
file1.close()
