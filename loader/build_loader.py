import os
import sys
import torch
import logging
import random
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
from .base_data_loader import BaseDataLoader
from torch.utils.data.dataloader import default_collate
import pickle

from .base_data_loader import BaseDataLoader
from .COCODataset import COCORelDataset, COCOLayoutDataset
from .VGmsdnDataset import VGmsdnRelDataset, VGmsdnLayoutDataset
# from .VGDataset import BboxDataset, RelDataset, Rel2Layout_Dataset


class DataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers = 0):
        collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)
        
def build_loader(cfg, eval_only):
    logger = logging.getLogger('dataloader')
    
    data_dir = cfg['DATASETS']['DATA_DIR_PATH']
    batch_size = cfg['SOLVER']['BATCH_SIZE']
    shuffle = cfg['DATALOADER']['SHUFFLE']
    validation_split =cfg['DATALOADER']['VAL_SPLIT']
    num_workers = cfg['DATALOADER']['NUM_WORKER']
    smart_sampling = cfg['DATALOADER']['SMART_SAMPLING']
    is_mask = True
    
    if cfg['DATASETS']['NAME'] == 'coco':
        ins_data_path = os.path.join(data_dir, 'instances_train2017.json')
        sta_data_path = os.path.join(data_dir,'stuff_train2017.json')
        obj_id_v2 = cfg['DATALOADER']['OBJ_ID_MODULE_V2']
        if eval_only:
            ins_data_path = os.path.join(data_dir, 'instances_val2017.json')
            sta_data_path = os.path.join(data_dir,'stuff_val2017.json')
            is_mask = cfg['TEST']['TEST_IS_MASK']
        dataset = COCORelDataset(ins_data_path, sta_data_path, is_mask=is_mask, 
                                 obj_id_v2=obj_id_v2)
        dataset_r = COCORelDataset(ins_data_path, sta_data_path, reverse=True,
                                   is_mask=is_mask)
            
            
    elif cfg['DATASETS']['NAME'] == 'vg_msdn':
        coco_addon = cfg['DATASETS']['COCO_ADDON']
        ins_data_path = os.path.join(data_dir, 'train.json')
        cat_path = os.path.join(data_dir, 'categories.json')
        dict_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
        if eval_only:
            ins_data_path = os.path.join(data_dir, 'test.json')
            is_mask = cfg['TEST']['TEST_IS_MASK']
        dataset = VGmsdnRelDataset(instances_json_path = ins_data_path,
                                       category_json_path = cat_path, 
                                       dict_save_path = dict_path,
                                       add_coco_rel=coco_addon,
                                       sentence_size=128,
                                       is_mask=is_mask)
        dataset_r = VGmsdnRelDataset(instances_json_path = ins_data_path,
                                       category_json_path = cat_path, 
                                       dict_save_path = dict_path,
                                       sentence_size=128, 
                                       add_coco_rel=coco_addon,
                                       reverse=True)

    elif cfg['DATASETS']['NAME'] == 'vg_co':
        ins_data_path = os.path.join(data_dir, 'train.json')
        cat_path = os.path.join(data_dir, 'categories.json')
        dict_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
        if eval_only:
            ins_data_path = os.path.join(data_dir, 'test.json')
        dataset = VGmsdnRelDataset(instances_json_path = ins_data_path,
                                       category_json_path = cat_path, 
                                       dict_save_path = dict_path,
                                       sentence_size=128)
            
    elif cfg['DATASETS']['NAME'] == 'vg':
        if eval_only:
            logger.warning('vg dataset donot have test split. Eval on training set.')
        dataset = RelDataset(smart_sampling = smart_sampling, 
                                 data_dir = data_dir, 
                                 anns_file_name = cfg['DATASETS']['ANNS_FILENAME'])
            
    else:
        raise Exception("Sorry, we only support vg, vg_msdn, vg_co and coco datasets.")
    
    mode = 'Pretrain' if cfg['MODEL']['PRETRAIN'] else 'Seq2Seq'
    logger.info('Setup [{}] dataset in [{}] mode.'.format(cfg['DATASETS']['NAME'], mode))
    logger.info('[{}] dataset in [{}] mode => Test dataset {}.'.format(cfg['DATASETS']['NAME'], mode, eval_only))
    return DataLoader(dataset, batch_size, shuffle, validation_split, num_workers), DataLoader(dataset_r, batch_size, shuffle, validation_split, num_workers)
