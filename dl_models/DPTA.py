import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dl_models.deep_model import AdaptDualProtoVit,AdaptDualProtoVitupd
from dl_models.base_learner import BaseLearner
from model_utils.toolkit import tensor2numpy
import os
from radam.radam import RAdam
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
num_workers = 0

from dl_models.deep_losses import CenterLoss

class Learner(BaseLearner):
    def __init__(self, args, load_model=True, load_dir = 'adapters/vtab/vtab0-10'):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        self.topk = args["top_k"]

        self._network = AdaptDualProtoVitupd(args, True)
        self. batch_size= args["batch_size"]
        self. init_lr=args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args

        self.res = args['use_res']
        self._network.res = self.res


        self.load_model = load_model

        self.load_dir = load_dir

        self._device = args["device"][0]

        if load_model == True:
            self._network.ptm.all_adapter_load(self.load_dir,sum_tasks=args["task_num"])

        self.criterion_xent = nn.CrossEntropyLoss()

        self.weight_cent = args["cent_weight"]
        self.cent_lr = args["center_lr"]
        self.first_adapt = args["first_adapt"]

        self.use_kl_div = False#KL div is useless in DPTA.
        
    def after_task(self):
        self._known_classes = self._total_classes
        
        self._network.freeze()
        self._network.ptm.add_adapter_to_list()

        
   
    def replace_fc(self, trainloader, args=None):
        # replace fc.weight with the embedding average of train data
        model = self._network.to(self._device)
        model = model.eval()

        embedding_list1 = [] #save the embeddings for the first prototype
        embedding_list2 = [] #second prototype layer embeddings
        label_list = []
        # data_list=[]aw
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)

                embedding1 = model.ptm.forward_proto_withoutft(data, first_adapt=self.first_adapt)
                embedding2 = model.ptm.forward_proto(data,adapt_index=self._cur_task)

                if self.res == True:
                    embedding2 += embedding1

                embedding_list1.append(embedding1.cpu())
                embedding_list2.append(embedding2.cpu())
                label_list.append(label.cpu())

        embedding_list1 = torch.cat(embedding_list1, dim=0)
        embedding_list2 = torch.cat(embedding_list2, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)

        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding1=embedding_list1[data_index]
            embedding2=embedding_list2[data_index]
            proto1=embedding1.mean(0)
            self._network.fc.weight.data[class_index]=proto1
            proto2=embedding2.mean(0)
            self._network.fc2.weight.data[class_index]=proto2
        return model

    

    def incremental_train(self, data_manager):
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.task_contain_class.append(torch.arange(self._known_classes, self._total_classes))
        if self._cur_task == 0:
            self.criterion_cent = CenterLoss(num_classes=data_manager.get_task_size(self._cur_task), feat_dim=self._network.feature_dim,device=self._device)
        else:
            self.criterion_cent.update_center_param(num_classes=data_manager.get_task_size(self._cur_task), feat_dim=self._network.feature_dim,device=self._device)
            
        self._network.update_fc(self._total_classes)
        self._network.generate_ft_fc(data_manager.get_task_size(self._cur_task))

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.load_model == False:
            self._train_ft(self.train_loader, self.test_loader)

        self.replace_fc(self.train_loader_for_protonet,self._network)

        self.after_task()

    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'radam':
            optimizer = RAdam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _train_ft(self, train_loader, test_loader):

        optimizer = self.get_optimizer(lr=self.args["init_lr"])
        scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])       

        # show total parameters and trainable parameters
        if self._cur_task == 0:
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

        self._init_train(train_loader, test_loader, optimizer, scheduler)
          
        
            

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        optimizer_ce = RAdam(self.criterion_cent.parameters(),self.cent_lr)
        self._network.to(self._device)

        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = targets.to(torch.long)
                #Shift the sample label to start at 0, otherwise CUDA error: device-side assert triggered will be reported
                targets = targets-self._known_classes
                
                outs = self._network(inputs)
                features = outs["features"]
                logits = outs["logits"]
                loss = self.weight_cent * self.criterion_cent(features,targets) + self.criterion_xent(logits,targets)

                optimizer_ce.zero_grad()
                optimizer.zero_grad()
 
                loss.backward()

                optimizer.step()
                for param in self.criterion_cent.parameters():
                    param.grad.data *= (1. / self.weight_cent)
                optimizer_ce.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
           
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            '''
            test_acc = self._compute_accuracy(self._network, test_loader)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            '''
            prog_bar.set_description(info)

        logging.info(info)

    def eval_task(self):
       
        accy = self._eval_acc(self.test_loader)
        
        return accy
    
    def eval_task_custom(self,test_loader):

        accy = self._eval_acc(test_loader)
        
        return accy

    def _eval_acc(self, loader ):


        self._network.eval()
        print('start evals')
        acc_list = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                predicts = self._network(inputs,test=True, res=self.res, use_kl_div = self.use_kl_div, first_adapt=self.first_adapt)
                acc = sum(predicts==targets)/targets.shape[0]
                acc_list.append(acc)
        acc = sum(acc_list)/len(acc_list)
        acc = acc.cpu().detach().numpy()
        return acc
    
    def eval_task_with_first_layer(self,loader = None):

        if loader == None:
            y_pred, y_true = self._eval_cnn1(self.test_loader)
        else:
            y_pred, y_true = self._eval_cnn1(loader)
        cnn_accy = self._evaluate(y_pred, y_true)


        return cnn_accy
    
    #overload
    def _eval_cnn1(self, loader ):


        self._network.eval()
        print('start evals')
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward_first(inputs,first_adapt=self.first_adapt)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

def accuracy(y_pred, y_true, nb_old, increment=10):
    #assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc