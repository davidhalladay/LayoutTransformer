import sys
sys.path.append("../")
sys.path.append("./")
from model import Rel2Layout, RelEncoder, Rel2Bbox, Rel2RegLayout
from trainer import Trainer, PretrainTrainer, RegTrainer
from utils import ensure_dir
from loader.COCODataset import COCORelDataset
import torch.backends.cudnn as cudnn
import argparse
import logging
import json
import cv2
from bounding_box import bounding_box as bb
import os
import numpy as np
from utils import ensure_dir
import pickle
import torch
import random
from random import randrange
import yaml
from model import build_model
from tqdm import tqdm
import time

logger = logging.getLogger('inference')

class Inference_COCO():
    def __init__(self, save_dir, vocab_dict):
        self.save_dir = save_dir
        self.device = self._prepare_gpu()
        self.vocab_dict = vocab_dict
            
    def check_GT(self, idx, dataset):
        """
        input_token = COCO[0][0]
        input_obj_id = COCO[0][1]
        input_box_label = COCO[0][2]
        output_label = COCO[0][3]
        segment_label = COCO[0][4]
        token_type = COCO[0][5]
        """
        image_id = dataset.image_ids[idx]
        single_data = dataset[idx]
        log_file_name = os.path.join(self.save_dir,str(image_id)+'.txt')
        image_wh = dataset.image_id_to_size[image_id]
        image_size = [image_wh[1], image_wh[0], 3]
        
        box_mask = np.array(single_data[2]) != 2.
        boxes = np.array(single_data[2])[box_mask].reshape(-1,4)
        boxes = self.xywh2xyxy(boxes, image_wh)
        
        id_mask = np.array(single_data[1]) != 0.
        ids = np.array(single_data[1])[id_mask]
        
        class_mask = (single_data[0][1::2] != 0)
        clss = single_data[0][1::2][class_mask]
        
        __image__mask = clss != 4
        clss = clss[__image__mask]
        clss = self.idx2vocab(clss, 'text')

        boxes = boxes[__image__mask]


        assert len(clss) == len(boxes)

        self.draw_img(image_size = image_size, boxes=boxes, labels=clss,
                      save_dir = self.save_dir,label_ids= torch.ones(len(clss)), name=str(image_id)+'_gt')
        return boxes
        
    def check_from_model(self, dataset_idx, dataset, model, random=False, layout_save=None):
        model.to(self.device)
        model.eval()
#         bb_gt = self.check_GT(idx, dataset)
        image_id = dataset.image_ids[dataset_idx]
        single_data = dataset[dataset_idx]
        image_wh = dataset.image_id_to_size[image_id]
        if layout_save is not None: 
            ensure_dir(layout_save)
            image_wh = [640, 640]
        image_size = [image_wh[1], image_wh[0], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        json_save_dir = os.path.join(self.save_dir, 'sg2im_json')
        ensure_dir(json_save_dir)
        json_file_name = os.path.join(json_save_dir, str(image_id)+'.json')

        input_token = single_data[0].unsqueeze(0).to(self.device)
        input_obj_id = single_data[1].unsqueeze(0).to(self.device)
        segment_label = single_data[5].unsqueeze(0).to(self.device)
        token_type = single_data[6].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)
        global_mask = input_token >= 2
        
        vocab_logits, obj_id_logits, token_type_logits, output_box, _, refine_box, _ = \
            model(input_token, input_obj_id, segment_label, token_type, src_mask, inference=True, epoch=0, global_mask=global_mask)
        
        pred_vocab = vocab_logits.argmax(2)
        pred_id = obj_id_logits.argmax(2)
        id_mask = (input_token == 3) * (input_token > 4) * (input_token < 160)
        input_obj_id[id_mask] = pred_id[id_mask]
        
        # get relation prior
        rel_mask = (pred_vocab >= 178) * (pred_vocab < 184)
        rel_prior = output_box[rel_mask].detach()
        rel_vocab = pred_vocab[rel_mask].detach()
        rel_classes = self.idx2vocab(rel_vocab, 'text')
        rel_prior_xy = rel_prior[:, :2]
#         print(rel_prior_xy)
        self.save_relation_prior(rel_classes, rel_prior_xy, self.save_dir)
        
        # construct mask
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        mask = torch.zeros(1,len(input_obj_id_list))
        mask_obj_avg = []
        for i in range(1, int(max(input_obj_id_list))+1):
#             idx = len(input_obj_id_list) - 1 - input_obj_id_list[::-1].index(i)
            mask_obj_avg.append((torch.LongTensor(input_obj_id_list) == i).reshape(1, -1)) 
            idx = input_obj_id_list.index(i)
            mask[0, idx] = 1
            
#         input_obj_id_list = list(input_obj_id[0].cpu().numpy())
#         mask = torch.zeros(1,len(input_obj_id_list))
#         exist_list = [0 for i in range(int(max(pred_vocab.tolist()[0]))+1)]
#         for i in range(len(input_obj_id_list)):
#             if input_obj_id_list[i] != 0:
#                 if exist_list[pred_vocab[0][i]] != input_obj_id_list[i]:
#                     exist_list[pred_vocab[0][i]] = input_obj_id_list[i]
# #                     idx = input_obj_id_list.index(i)
#                     mask[0, i] = 1
#         mask[0, input_obj_id[0] > 0] = 1
        mask = mask.bool()
        
#         pred_mask = (pred_vocab >= 4) * (pred_vocab < 176)
        pred_vocab = pred_vocab[mask].detach()
#         pred_id = pred_id[mask].detach()
        # use relation
#         print(output_box[0, 2::4, :2])
#         print(output_box[0,:20,:2])
        output_box[0, 3::4, :2] = output_box[0, 1::4, :2] - output_box[0, 2::4, :2]
#         print(output_box[0,:20,:2])
        
        
        output_boxes = output_box[mask].detach()
        
        output_class_ids = input_obj_id[mask].detach()
#         print(image_wh)
        output_boxes = self.xcycwh2xyxy(output_boxes, image_wh)

#         print(output_boxes)
        pred_classes = self.idx2vocab(pred_vocab, 'text')
        
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')
        input_sentence = self.idx2vocab(input_token.squeeze(0), 'text')
#         print(self.get_iou(output_boxes.squeeze(0), bb_gt))
        
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx)
        if refine_box is not None:
            refine_boxes = refine_box[mask].detach()
            refine_boxes = self.xcycwh2xyxy(refine_boxes, image_wh)
            self.draw_img(image_size = image_size, boxes=refine_boxes.squeeze(0),
                          labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                          save_dir = self.save_dir, name=image_id, idx=dataset_idx, mode='r')
        self.write_log(output_sentence, input_obj_id_list, log_file_name, 
                        name=image_id, idx=dataset_idx)
        self.write_json(output_sentence, input_obj_id_list, json_file_name, 
                        name=image_id, idx=dataset_idx)
        if layout_save is not None: 
            self.save_layout(boxes=refine_boxes.squeeze(0), objs=pred_classes, 
                         save_path=layout_save, label_ids=output_class_ids.squeeze(0),
                             name=image_id, image_wh=image_wh)

    def check_from_sg(self, input_dict, model, layout_save=None):
        model.to(self.device)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("PARAMETERS:", pytorch_total_params)
        model.eval()
        print("self input")
        image_id = input_dict['image_id']
        dataset_idx = input_dict['dataset_idx']
        image_wh = [640, 640]
        
        if layout_save is not None: 
            ensure_dir(layout_save)
            image_wh = [640, 640]
        image_size = [image_wh[1], image_wh[0], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        json_save_dir = os.path.join(self.save_dir, 'sg2im_json')
        ensure_dir(json_save_dir)
        json_file_name = os.path.join(json_save_dir, str(image_id)+'.json')

        input_token = input_dict['tensor_list'][0].unsqueeze(0).to(self.device)
        input_obj_id = input_dict['tensor_list'][1].unsqueeze(0).to(self.device)
        segment_label = input_dict['tensor_list'][2].unsqueeze(0).to(self.device)
        token_type = input_dict['tensor_list'][3].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)
        global_mask = input_token >= 2
        
        ####
        input_token = input_token.repeat(64, 1)
        input_obj_id = input_obj_id.repeat(64, 1)
        segment_label = segment_label.repeat(64, 1)
        token_type = token_type.repeat(64, 1)
        src_mask = src_mask.repeat(64, 1, 1)
        global_mask = global_mask.repeat(64, 1)

        ##
        with torch.no_grad():
            start = time.time()
            vocab_logits, obj_id_logits, token_type_logits, output_box, _, refine_box, _ = \
                model(input_token, input_obj_id, segment_label, token_type, src_mask, inference=True, epoch=0, global_mask=global_mask)
            end = time.time()

        print("Elapsed time,", end-start)
        print("Batch Size", vocab_logits.size(0))
        exit()
        pred_vocab = vocab_logits.argmax(2)
        pred_id = obj_id_logits.argmax(2)
        id_mask = (input_token == 3) * (pred_vocab > 4) * (pred_vocab < 177)
        input_obj_id[id_mask] = pred_id[id_mask]
        # construct mask
        
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        mask = torch.zeros(1,len(input_obj_id_list))
        mask_obj_avg = []
        for i in range(1, int(max(input_obj_id_list))+1):
#             idx = len(input_obj_id_list) - 1 - input_obj_id_list[::-1].index(i)
            mask_obj_avg.append((torch.LongTensor(input_obj_id_list) == i).reshape(1, -1)) 
            idx = input_obj_id_list.index(i)
            mask[0, idx] = 1
            
#         mask[0, input_obj_id[0] > 0] = 1
        mask = mask.bool()
#         pred_mask = (pred_vocab >= 4) * (pred_vocab < 176)
        pred_vocab = pred_vocab[mask].detach()
#         pred_id = pred_id[mask].detach()
        # use relation
#         print(output_box[0, 2::4, :2])
        output_box[0, 3::4, :2] = output_box[0, 1::4, :2] - output_box[0, 2::4, :2]
    
        output_boxes = output_box[mask].detach()
        refine_boxes = refine_box[mask].detach()
        output_class_ids = input_obj_id[mask].detach()
        
        output_boxes = self.xcycwh2xyxy(output_boxes, image_wh)
        refine_boxes = self.xcycwh2xyxy(refine_boxes, image_wh)
        pred_classes = self.idx2vocab(pred_vocab, 'text')
#         print(pred_classes)
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')
#         print(output_sentence)
        input_sentence = self.idx2vocab(input_token.squeeze(0), 'text')
#         print(self.get_iou(output_boxes.squeeze(0), bb_gt))
#         print(input_sentence)
#         print(output_sentence)
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx)
        self.draw_img(image_size = image_size, boxes=refine_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx, mode='r')
        self.write_log(output_sentence, input_obj_id_list, log_file_name, 
                        name=image_id, idx=dataset_idx)
        self.write_json(output_sentence, input_obj_id_list, json_file_name, 
                        name=image_id, idx=dataset_idx)
        if layout_save is not None: 
            self.save_layout(boxes=refine_boxes.squeeze(0), objs=pred_classes, 
                         save_path=layout_save, label_ids=output_class_ids.squeeze(0),
                             name=image_id, image_wh=image_wh)
            
    def save_relation_prior(self, rel_classes, rel_prior_xy, save_dir):
        rel_prior_xy = rel_prior_xy.tolist()
        try:
            with open(os.path.join(save_dir, 'rel_prior.json'), 'r') as fp:
                rel_dict = json.load(fp)
        except:
            rel_dict = dict()
        for i, rel in enumerate(rel_classes):
            if rel in rel_dict.keys():
                rel_dict[rel].append(rel_prior_xy[i])
            else:
                rel_dict[rel] = [rel_prior_xy[i]]
        with open(os.path.join(save_dir, 'rel_prior.json'), 'w') as fp:
            json.dump(rel_dict, fp)
        
        
#     def predict_val(self, )
    def draw_img(self, image_size, boxes, labels, label_ids, save_dir, name, idx, mode='c'):
        # setting color
        color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
                'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
        image = np.full(image_size, 200.)
        
        boxes[boxes < 0] = 0
        boxes[boxes > image_size[0]] = image_size[0]
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)
        for i in range(len(boxes)):
            bb.add(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                   str(labels[i]+'[{}]'.format(label_ids[i])), 
                   color=color[ord(labels[i][0])-ord('a')])
        self.show_and_save(image,
                   os.path.join(save_dir,str(name) + '_{}_{}'.format(idx, mode) +'.png'))
#         logger.info("Save image in {}".format(os.path.join(save_dir,str(name)+'_{}_{}'.format(idx, mode)+'.png')))

    def write_log(self, sentence, class_ids, log_file_name, name=None, idx=None):
        f = open(log_file_name, 'w')
        for i in range(1, len(sentence), 4):
            if sentence[i+1] == '__in_image__':
                break
            single_pair = ''
            single_pair += sentence[i] + '[{}]'.format(class_ids[i]) + ' '
            single_pair += sentence[i+1] + ' '
            single_pair += sentence[i+2] + '[{}]'.format(class_ids[i+2]) + ' '
            single_pair += '\n'
            f.write(single_pair)
#         logger.info("Save log file in {}".format(log_file_name))
    
    def write_json(self, sentence, class_ids, log_file_name, name=None, idx=None):
        out_dict = dict()
        out_dict['image_id'] = name
        out_dict['dataset_idx'] = idx
        out_dict['objects'] = ["None" for i in range(max(class_ids))]
        out_dict['relationships'] = []
        for i in range(1, len(sentence), 4):
            if sentence[i+1] == '__in_image__':
                break
            out_dict['objects'][int(class_ids[i]-1)] = sentence[i]
            out_dict['objects'][int(class_ids[i+2]-1)] = sentence[i+2]
            single_rel = [int(class_ids[i]-1), sentence[i+1], int(class_ids[i+2]-1)]
            out_dict['relationships'].append(single_rel)
        with open(log_file_name, 'w') as outfile:
            json.dump(out_dict, outfile)
#         logger.info("Save json file in {}".format(log_file_name))
        
    def save_layout(self, boxes, objs, save_path, label_ids, name, image_wh):
        output_dict = dict()
        output_dict['image_id'] = name
        output_dict['boxes'] = (boxes/image_wh[0]).tolist()
        output_dict['classes'] = objs
        output_dict['class_ids'] = label_ids.tolist()
        output_file_name = os.path.join(save_path,str(name)+'.json')
        with open(output_file_name, 'w') as fp:
            json.dump(output_dict, fp)
#         logger.info("Save json file in {}".format(output_file_name))
        return 0
        
    def xywh2xyxy(self, boxes, image_wh):
        center = boxes[:,:2].copy()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        return boxes
    
    def xcycwh2xyxy(self, boxes, image_wh):
        center = boxes[:,:2].clone()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        return boxes
    
    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device
    
    def idx2vocab(self, idx, modality):
        sent = []
        for i in range(len(idx)):
            if modality == 'text':
                sent.append(self.vocab_dict[int(idx[i])])
            else:
                sent.append(self.cls_dict[idx[i]])
        return sent
    
    def show_and_save(self, image, path):
        cv2.imwrite(path, image)
        
    def run(self, cfg, model, dataset):
        print("Dataset: ",len(dataset))
        layout_path = cfg['TEST']['LAYOUT_MODE'] if cfg['TEST']['LAYOUT_MODE'] != '' else None

        if cfg['TEST']['MODE'] == 'gt':
            if cfg['TEST']['RANDOM']: logger.warning('Test gt mode do not support random.')
            self.check_GT(cfg['TEST']['SAMPLE_IDX'], dataset)
        elif cfg['TEST']['MODE'] == 'model':
            if cfg['TEST']['RANDOM']: 
                if cfg['TEST']['SAMPLE_IDX'] == -1:
                    for idx in tqdm(range(5000)):
                        self.check_from_model(idx, dataset = dataset, model=model, 
                                              random=True, layout_save=layout_path)
                else:
                    self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset, model, 
                                       random=True, layout_save=layout_path)
            else: 
                if cfg['TEST']['SAMPLE_IDX'] == -1:
                    for idx in tqdm(range(5000)):
                        self.check_from_model(idx, dataset = dataset, model=model,
                                             layout_save=layout_path)
                else:
                    self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset=dataset,
                                         model=model, layout_save=layout_path)
        else:
            logger.error('We only support gt and model test mode.')
            
    def get_iou(self, bb, bb_gt):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """
        print(bb)
        print(bb_gt)

        # determine the coordinates of the intersection rectangle
        iou_list = []
        for i in range(len(bb)):
            x_left = max(bb[i][0], bb_gt[i][0])
            y_top = max(bb[i][1], bb_gt[i][1])
            x_right = min(bb[i][2], bb_gt[i][2])
            y_bottom = min(bb[i][3], bb_gt[i][3])

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb[i][2] - bb[i][0]) * (bb[i][3] - bb[i][1])
            bb2_area = (bb_gt[i][2] - bb_gt[i][0]) * (bb_gt[i][3] - bb_gt[i][1])

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            assert iou >= 0.0
            assert iou <= 1.0
            iou_list.append(iou)
        return iou_list

if __name__ == '__main__':
    data_dir = 'data/coco/'
    cfg_path = './configs/coco/coco_seq2seq_v7.yaml'
    test_output = 'saved/coco_F_seq2seq_v7/test'

#     int_json_path ='data/vg_msdn/stuff_train2017.json' 
    model_path = 'saved/coco_F_seq2seq_v7/checkpoint_12_0.44482552774563705.pth'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    vocab_dic_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
    with open(vocab_dic_path, 'rb') as file:
        vocab_dict = pickle.load(file)
        
    infer = Inference_COCO(save_dir = test_output, vocab_dict = vocab_dict)
    
    ## dataset
    dict_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
    ins_data_path = os.path.join(data_dir, 'instances_val2017.json')
    sta_data_path = os.path.join(data_dir,'stuff_val2017.json')
    dataset = COCORelDataset(ins_data_path, sta_data_path)
    
    # build model
    model = build_model(cfg)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    da = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood', 'other', '__in_image__', 'left of', 'right of', 'above', 'below', 'inside', 'surrounding']
    
    # my own input
    sentence = ['[CLS]', 'zebra', 'above', 'grass', '[SEP]',
                         'grass', 'below', 'mountain', '[SEP]',
                         'mountain', 'above', 'grass', '[SEP]',
                         'grass', 'right of', 'tree', '[SEP]',
                         'zebra', '__in_image__', '__image__', '[SEP]',
                         'grass', '__in_image__', '__image__', '[SEP]',
                         'mountain', '__in_image__', '__image__', '[SEP]',
                         'tree', '__in_image__', '__image__', '[SEP]']

#                 'man', 'wear_a', 'glass', '[SEP]']  
    for i in range(len(sentence)):
        sentence[i] = dataset.vocab['object_pred_name_to_idx'][sentence[i]]
    
    obj_id = [0, 1, 0, 2, 0,
                 2, 0, 3, 0,
                 3, 0, 2, 0,
                 2, 0, 4, 0,
                 1, 0, 0, 0,
                 2, 0, 0, 0,
                 3, 0, 0, 0,
                 4, 0, 0, 0]

    
    segment_label = [1]
    token_label = [0]
    for i in range(int((len(obj_id)-1)/4)):
        for j in range(4):
            segment_label.append(i+1)
            token_label.append(j+1)

    for i in range(128 - len(sentence)): 
        sentence.append(0)  
        obj_id.append(0)
        segment_label.append(0)
        token_label.append(0)
    print(len(sentence))
    print(len(obj_id))
    print(len(segment_label))
    print(len(token_label))
    self_input_dict = {'input_token':torch.LongTensor(sentence),
                       'input_obj_id':torch.LongTensor(obj_id),
                       'segment_label':torch.LongTensor(segment_label),
                      'token_type':torch.LongTensor(token_label),
                      'image_id':10001,
                      'image_wh':(800,600)}
    
    
    infer.check_from_model(100, dataset, model, self_input_dict = self_input_dict,
                           layout_save='../output_my/')