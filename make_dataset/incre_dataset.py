import logging #please note that logging can't working on jupyter notebooks 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from make_dataset.get_data import vtab,iImageNetA,iImageNetR,iCIFAR100,CUB,CARS

### basic dataset for images
def pil_loader(path):
    """
    open path file to generate img arrays
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
### find imgdata
def get_imgdata(dataset_name):
    name = dataset_name.lower()
    
    if name=="vtab":
        return vtab()
    
    elif name=="ina":
        return iImageNetA()
    elif name=="inr":
        return iImageNetR()
    
    elif name=="cifar":
        return iCIFAR100()
    
    elif name=='cub':
        return CUB()
    elif name=='cars':
        return CARS()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
    
# basic_dataset for create data_loader
class BasicDataset(Dataset):
    def __init__(self, images, labels, trans, use_path=True): 
        #immges,labels come from classes in get_data.py,self.train_data and self.test_data
        assert len(images) == len(labels), "Data size error!"

        self.images = images

        self.labels = labels

        self.trans = trans #torch transform

        self.use_path = use_path

    def __len__(self): #return len
        return len(self.images)

    def __getitem__(self, idx): #return image and labels in iter
        if self.use_path:
            image = self.trans(pil_loader(self.images[idx]))
        else:
            image = self.trans(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label
    
# if shuffled, Change the order of x to the order after shuffle
def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

#incremental dataset
class IncrementDataset(object):
    '''
        build a dataset in incremental form

        __init__() Args:
        dataset_name(str): dataset names for find img datas, choice: vtab

        init_cls(int): number of classes do the first task contains

        increment(int): number of classes do the remain task contains

        custom_tasks(list): Customize number of classes there are on each task in list forms

    '''
    def __init__(self, dataset_name='vtab', shuffle=False, seed=0, init_cls=10, increment=10, custom_tasks=None):
        self.custom_tasks = custom_tasks

        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)

        assert init_cls <= len(self._class_order), "Error: No enough classes."

        self.custom_tasks = custom_tasks

        self._increments = []# number of data stored on each task

        if init_cls > 0:
            self._increments.append(init_cls) #add first tasks in list

        while sum(self._increments) + increment < len(self._class_order): #add remain tasks in list
            self._increments.append(increment)
        
        #If the number of remaining categories is less than increment, an automatic complement is performed
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0: #have several redundant classes
            self._increments.append(offset) 

        # have the custom_tasks setting
        if custom_tasks != None:
            assert sum(custom_tasks) <= len(self._class_order), "Error: The total number of task classes is greater than the maximum classes of the dataset"
            self._increments = custom_tasks

    def _setup_data(self, dataset_name, shuffle, seed):
        #get data by calling get_data.py and map the indexs
        idata = get_imgdata(dataset_name)

        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trans = idata.train_trans
        self._test_trans = idata.test_trans
        self._common_trans = idata.common_trans #Other transformer components,  generally set to empty lits

        #  build orders (The marshalling sequence in which each class appears）
        order = [i for i in range(len(np.unique(self._train_targets)))]

        if shuffle: 
        #shuffle the marshalling sequence of classes, eg: in vlabs, the order of trainset and testset classes is 10,11……，change it randomly like 32,1,50……
        #then could build several tasks with random classes.
        #this could be done by generate random indexes
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        
        #save the class order and total classes in logger 
        logging.info(f'class_order:{self._class_order} in {self.dataset_name} dataset')
        logging.info(f'total_class_num:{len(self._class_order)} in {self.dataset_name} dataset')

        # change the orders of the classes by the indexes
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        '''
        build a special task dataset through label indices

        *indices*:  a list contains the selected label range, range (known_classes,known_classes + i-th task classes numbers)
        for example, if indices=[0,1,2,3,4],then model will choose the data and target which labels equal 0,1,2,3 and 4 in order
        labels are externally predefined class-ids

        *source* set train/test to choose train or test data, *mode* set train/test to choose torch transform
        '''
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train": #build transform lists and compose them
            trans = transforms.Compose([*self._train_trans, *self._common_trans])
        elif mode == "flip":
            trans = transforms.Compose(
                [
                    *self._test_trans,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trans,
                ]
            )
        elif mode == "test":
            trans = transforms.Compose([*self._test_trans, *self._common_trans])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0: #add the appendent data for them
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data: #Returns all selected data together
            return data, targets, BasicDataset(data, targets, trans, self.use_path)
        else:
            return BasicDataset(data, targets, trans, self.use_path)
        
    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        '''
        select training and vaildation samples for hyperparameters tuning.
        '''
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trans = transforms.Compose([*self._train_trans, *self._common_trans])
        elif mode == "test":
            trans = transforms.Compose([*self._test_trans, *self._common_trans])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return BasicDataset(
            train_data, train_targets, trans, self.use_path
        ), BasicDataset(val_data, val_targets, trans, self.use_path)
       
    def _select(self, x, y, low_range, high_range):
        # select data with selected label ranges
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0] #  label indexs in given ranges
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        # select a certain percentage of data with selected labels 
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def special_tasks_setting(self):
        #for some strange task settings, like overlap
        pass

    def get_task_size(self, task): 
        #given task idx, return class numbers in this task
        return self._increments[task]
    
    def getlen(self, index):
        #get the number of samples in one classes
        y = self._train_targets
        return np.sum(np.where(y == index))
    
    @property
    def nb_tasks(self): #get the total number of tasks
        return len(self._increments)
    
    @property
    def get_total_classnum(self): #get the total number of classes
        return len(self._class_order)

    
def select_data_with_labels(x, y, label_list):
     # select data with pre-defined label lists
    x1 = []
    y1 = []
    for label in label_list:
        idxes = np.where(y == label)[0]
        x1.append(x[idxes])
        y1.append(y[idxes])

    return np.concatenate(x1),np.concatenate(y1) #build the list to np.arrays

#About modifying task-specific labels to start at 0
# suppose y is the label set in a task, use a mapping function to rearrage them or - known_classes。