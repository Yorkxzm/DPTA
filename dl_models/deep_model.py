import copy
import logging
import torch
from torch import nn
import timm
from dl_models.linear_model import CosineLinear,SecondCosineLinear,TopKCosineLinear


#Intranet can not access to host= 'huggingface.co' and requires mirrors
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def get_ptms(args, pretrained=True): 
    #get the specified pre-trained models
    name = args["convnet_type"].lower()

    if name=="pretrained_vit_b16_224" or name=="vit_base_patch16_224":
        model=timm.create_model("vit_base_patch16_224",pretrained=pretrained, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_b16_224_in21k" or name=="vit_base_patch16_224_in21k":
        model=timm.create_model("vit_base_patch16_224_in21k",pretrained=pretrained, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_base_b16_224_adapter" or name=="pretrained_vit_base_b16_224_in21k_adapter":
        ffn_num = args["ffn_num"]
        from finetunes import vision_transformer_adapter
        from easydict import EasyDict
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
            _device = args["device"][0]
        )
        if name=="pretrained_vit_base_b16_224_adapter":
            model = vision_transformer_adapter.vit_base_patch16_224(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim=768
        elif name=="pretrained_vit_base_b16_224_in21k_adapter":
            model = vision_transformer_adapter.vit_base_patch16_224_in21k(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim=768
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))      
    
class BaseNet(nn.Module):
    # BaseNet exists as the parent class of all other Nets
    def __init__(self, args, pretrained=True):
        super(BaseNet, self).__init__()
        print('Start to build BaseNet')

        self.ptm = get_ptms(args, pretrained)

        print('BaseNet build done!')

        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.ptm.out_dim
    
    def extract_vector(self, x):
        return self.ptm(x)["features"]
    
    def forward(self, x):
        x = self.ptm(x)
        """
        here suppose there are three outputs in a dict
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out = self.fc(x["features"])
        out.update(x)

        return out
    
    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        #freeze specified parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
class SimpleVitNet(BaseNet):
    '''prototype model load ViT
    The network computes in FC is actually the cosine similarity between the sample representation and the prototype. 
    Selecting the class with the greatest cosine similarity is the classification result
    '''
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained=True)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        #update fc layers (add new classes to classification)
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.ptm(x)

    def forward(self, x):
        x = self.ptm(x)
        out = self.fc(x)
        # out.update(x)
        return out



class AdapterVitNet(BaseNet):

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained=True)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.ptm.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        #update fc layers (add new classes to classification)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.ptm(x)

    def forward(self, x, test=False, target_tasks=0, res =True):
        if test == True:
            out = self.ptm.forward_proto(x,adapt_index=target_tasks)
            
            if res == True:
                out = out + self.ptm.forward_proto_withoutft(x)
              
                out = self.fc(out)
                return out
        else:
            x = self.ptm(x)
        out = self.fc(x)

        return out
        
class AdaptDualProtoVit(BaseNet):

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained=True)
        self.args = args
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        self._cur_task = -1
        self.out_dim =  self.ptm.out_dim
        self.fc = None
        self.fc2 = None

        self.task_contain_class = []

        self.finetune_fc = None
        self.topk = args["top_k"]

        self.classtask_query = querytask_Bi_INCj(b=self.init_cls,inc=self.inc)


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        #update fc layers (add new classes to classification)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        
        fc2 = self.generate_second_fc(self.feature_dim,nb_classes).to(self._device)
        if self.fc2 is not None:
            nb_output = self.fc2.out_features
            weight = copy.deepcopy(self.fc2.weight.data)
            fc2.sigma.data = self.fc2.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc2.weight = nn.Parameter(weight)
        del self.fc2
        self.fc2 = fc2        

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def generate_second_fc(self, in_dim, out_dim):
        fc = SecondCosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.ptm(x)

    def forward(self, x, test=False, res =False, use_kl_div = False, first_adapt=False):
        #please note that if x,y is integers, then [x,y] can be a index.
        if test == True:
            # first layer calculates 
            emb1s = self.ptm.forward_proto_withoutft(x,first_adapt = first_adapt)
            label_probs = self.fc(emb1s)["logits"]
            topk_pred = torch.topk(label_probs, k=self.topk, dim=1, largest=True, sorted=True)[1] 
            out_labels = []

            for i in range(topk_pred.shape[0]):
                prob_list=[]
                label_list=[]
                for j in range(self.topk):
                    label_list.append(topk_pred[i][j])
                    task_index = self.classtask_query(topk_pred[i][j])
                    x_i_emb2 = self.ptm.forward_proto(x[[i]],adapt_index=task_index)
                    if res:
                        x_i_emb2+=emb1s[[i]]
                    x_i_prob = self.fc2(x_i_emb2,class_index = [int(topk_pred[i][j])], use_kl_div = use_kl_div)['logits']
                    prob_list.append(x_i_prob)
                prob_list = torch.cat(prob_list, dim=0)
                if use_kl_div:
                    maxprob_index = prob_list.argmin(0)
                else:
                    maxprob_index = prob_list.argmax(0)
                    
                label = label_list[maxprob_index]
                out_labels.append(label)

            out_labels = torch.stack(out_labels,dim=0)
            return out_labels
        else:
            out = self.ptm(x)
            outs = {"features": out,
                   "logits": self.finetune_fc(out)}
            return outs
   
    def save_fc(self,path='cosfc'):
        torch.save(self.fc,f'{path}/fc1.pth')
        torch.save(self.fc2,f'{path}/fc2.pth')
        torch.save({'tcs':self.task_contain_class},f'{path}/tcs.pth')
    def load_fc(self,path='cosfc'):
        self.fc=torch.load(f'{path}/fc1.pth')
        self.fc2=torch.load(f'{path}/fc2.pth')
        tcs = torch.load(f'{path}/tcs.pth')
        self.task_contain_class=tcs["tcs"]


    def forward_first(self, x, first_adapt = False):
        emb1s = self.ptm.forward_proto_withoutft(x,first_adapt= first_adapt)
        
        out = self.fc(emb1s)
        return out
               
    def generate_ft_fc(self,out_dim):
        in_dim =self.feature_dim
        if self.finetune_fc != None:
            self.del_ft_fc()
        self.finetune_fc = torch.nn.Linear(in_dim,out_dim)

    def del_ft_fc(self):
        self.finetune_fc = self.finetune_fc.to('cpu')
        del self.finetune_fc



class AdaptDualProtoVitupd(BaseNet):

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained=True)
        self.args = args
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        self._cur_task = -1
        self.out_dim =  self.ptm.out_dim
        self.fc = None
        self.fc2 = None

        self.task_contain_class = []

        self.finetune_fc = None
        self.topk = args["top_k"]
        self.res = False


        self.classtask_query = querytask_Bi_INCj(b=self.init_cls,inc=self.inc)



    def update_fc(self, nb_classes, nextperiod_initialization=None):
        #update fc layers (add new classes to classification)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        
        fc2 = self.generate_second_fc(self.feature_dim,nb_classes).to(self._device)
        if self.fc2 is not None:
            nb_output = self.fc2.out_features
            weight = copy.deepcopy(self.fc2.weight.data)
            fc2.sigma.data = self.fc2.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc2.weight = nn.Parameter(weight)
        del self.fc2
        self.fc2 = fc2        

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def generate_second_fc(self, in_dim, out_dim):
        fc = TopKCosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.ptm(x)

    def forward(self, x, test=False, res =False, use_kl_div = False, first_adapt=False):
        if test == True:
            emb1s = self.ptm.forward_proto_withoutft(x, first_adapt=first_adapt)
            label_probs = self.fc(emb1s)["logits"]
            topk_pred = torch.topk(label_probs, k=self.topk, dim=1)[1]  # [B, K]

            all_emb2 = []
            all_emb2 = self.ptm.forward_all_adapter(x)  # [T, B, D]

            if self.res:
                for i in range(len(self.ptm.adapter_list)):
                    all_emb2[i] += emb1s
            labels = self.fc2(all_emb2, topk_pred,self.classtask_query)
            #print(labels.shape) 
            return labels
            
        else:
            out = self.ptm(x)
            outs = {"features": out,
                   "logits": self.finetune_fc(out)}
            return outs

    def save_fc(self,path='cosfc'):
        torch.save(self.fc,f'{path}/fc1.pth')
        torch.save(self.fc2,f'{path}/fc2.pth')
        torch.save({'tcs':self.task_contain_class},f'{path}/tcs.pth')
    def load_fc(self,path='cosfc'):
        self.fc=torch.load(f'{path}/fc1.pth')
        self.fc2=torch.load(f'{path}/fc2.pth')
        tcs = torch.load(f'{path}/tcs.pth')
        self.task_contain_class=tcs["tcs"]


    def forward_first(self, x, first_adapt = False):
        emb1s = self.ptm.forward_proto_withoutft(x,first_adapt= first_adapt)
        
        out = self.fc(emb1s)
        return out
               
    def generate_ft_fc(self,out_dim):
        in_dim =self.feature_dim
        if self.finetune_fc != None:
            self.del_ft_fc()
        self.finetune_fc = torch.nn.Linear(in_dim,out_dim)

    def del_ft_fc(self):
        self.finetune_fc = self.finetune_fc.to('cpu')
        del self.finetune_fc

class querytask_Bi_INCj(nn.Module):
    def __init__(self, b, inc ):
        super(querytask_Bi_INCj, self).__init__()
        self.b=b
        self.inc = inc
    def forward(self,c):
        base_mask = c < self.b
        subsequent_tasks = ((c - self.b) // self.inc) + 1
        return torch.where(base_mask, 
                        torch.tensor(0, device=c.device), 
                        subsequent_tasks).long()
