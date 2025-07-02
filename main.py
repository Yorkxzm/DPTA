import logging
import sys
from make_dataset.incre_dataset import IncrementDataset

import logging
import numpy as np

from torch.utils.data import DataLoader
import time
import torch
import json
from dl_models.DPTA import Learner
import os
#nohup python main.py > inap0output.out &
#nohup python main.py > output.out &
#nohup python main.py > outputinrin1k.out &
##nohup python main.py > outputvtabonlyce.out &
def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def _set_random(seed=1): #fix rand seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse

exp_dataset = 'vtab'

if exp_dataset == 'inr' or exp_dataset == 'ina':
    default_task= exp_dataset +'_20_20'
elif exp_dataset == 'cars':
    default_task= exp_dataset +'_16_20'
else:
    default_task= exp_dataset +'_10_10'
default_json = default_task+ '_vit-im21_adapter.json'



parser = argparse.ArgumentParser(description='Exp-DPTA')
parser.add_argument(
    '--dataset_name',
    type=str,
    default=exp_dataset)
parser.add_argument(
    '--task_name',
    type=str,
    default=default_task)
parser.add_argument(
    '--json_name',
    type=str,
    default=default_json)

main_args = parser.parse_args()

log_path = f"logs/{main_args.dataset_name}/{main_args.task_name}/"
logfilename = log_path + "loggers"
if not os.path.exists(log_path):
    os.makedirs(log_path)

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

logging.info(f"Dataset: {main_args.dataset_name}")
logging.info(f"Task name: {main_args.task_name}")

def main():

    args = load_json(f'exp_params/{main_args.json_name}')
    rd_seed = args["seed"][0]
    logging.info(f"Random seed: {rd_seed}")
    _set_random(rd_seed)
    custom_tasks = None
    init_cls = args["init_cls"]
    increments = args["increment"]
    incdata = IncrementDataset(dataset_name=main_args.dataset_name, init_cls=args["init_cls"],increment=args["increment"],seed=rd_seed,shuffle=args["shuffle"])

    if custom_tasks == None:
        print(f'there are {incdata.nb_tasks} tasks with a total of {incdata.get_total_classnum} classes')
        print(f'Init_cls = {init_cls} ')
        print(f'Each remain tasks have {increments} classes')
    else:
        print(f'there are {incdata.nb_tasks} tasks with a total of {sum(custom_tasks)} classes')
        for i in range(incdata.nb_tasks):
            print(f'task{i} contains {custom_tasks[i]}classes')

    #generate models
    model = Learner(args,load_model=False)

    incdata_test = IncrementDataset(dataset_name=main_args.dataset_name, init_cls=args["init_cls"],increment=args["increment"],seed=rd_seed,shuffle=args["shuffle"])
    acc_list=[]
    for tasks in range(incdata.nb_tasks):
        logging.info(f"Training on the task {tasks+1}: ")
        model.incremental_train(incdata)
        dataset= incdata.get_dataset(np.arange(0, model._known_classes), source="test", mode="test")
        loaders = DataLoader(dataset,batch_size=32,shuffle=True)
        start_time = time.time()
        acc = model.eval_task_custom(loaders)
        end_time = time.time()
        acc_list.append(acc)
        logging.info(f"ACC in all {model._known_classes} classes after the task {tasks+1}: {acc}")
        logging.info(f"inf time in task {tasks+1}: {end_time-start_time}s")


    logging.info(f"Final ACC in all tasks: {acc_list[-1]}") #the last Ab is Af
    logging.info(f"Avg ACC in all tasks: {sum(acc_list)/len(acc_list)}") #Avg Ab
    #model._network.ptm.adapter_unit_save('vtabadapter',5)
        
if __name__ == "__main__":
    main()