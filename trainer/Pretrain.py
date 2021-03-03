import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import random
import torch.nn as nn
import math
from .loss import RegLoss, FocalLoss, Log_Pdf, Rel_Loss

import matplotlib.pyplot as plt
from PIL import Image
from .scheduler import build_scheduler
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import pickle
from .iou import IOU_calculator


class PretrainTrainer:    
    def __init__(self, model, dataloader, dataloader_r, opt, cfg):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.cfg = cfg
        self.n_epochs = self.cfg['SOLVER']['EPOCHS']
        self.save_dir = self.cfg['OUTPUT']['OUTPUT_DIR']
        self.two_path = self.cfg['MODEL']['DECODER']['TWO_PATH']
        self.dataloader = dataloader
        self.dataloader_r = dataloader_r
        self.batch_size = dataloader.batch_size
        self.total_steps = len(dataloader) * self.n_epochs
        self.device = self._prepare_gpu()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tensorboard'))
        self.model = model
        self.pad_index = 0
        self.bos_index = 1
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), 
                                          betas=(0.9, 0.999), weight_decay=0.01)
        self.bbox_head_optimizer = torch.optim.Adam(self.model.bbox_head.parameters(), 
                                          betas=(0.9, 0.999), weight_decay=0.01)
#         self.scheduler = MultiStepLR(self.optimizer, milestones=[4, 30,40,60], 
# gamma=0.6667)
#         self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.6667, patience=5,
#                                            threshold = 0.02, threshold_mode='rel')
        self.encoder_scheduler = build_scheduler(cfg, self.encoder_optimizer, self.total_steps, "ENC")
        self.bbox_head_scheduler = build_scheduler(cfg, self.bbox_head_optimizer, self.total_steps, "BOX")
        self.loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        # self.focal_loss = FocalLoss(gamma=2, alpha=None, ignore_index=self.pad_index, reduction='sum')
        self.IOU_c = IOU_calculator(reduction = 'mean', cfg=cfg)
        self.val_box_loss = RegLoss(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., refine=True)
        self.refine, self.box_loss, self.rel_loss, self.refine_box_loss = self.build_loss()
        
        if self.cfg['MODEL']['ENCODER']['ENABLE_NOISE']:
            self.noise_size = self.cfg['MODEL']['ENCODER']['NOISE_SIZE']
            
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)
        self.pretrain_encoder = cfg['MODEL']['PRETRAIN']
        if not self.pretrain_encoder:
            assert self.cfg['SOLVER']['PRETRAIN_WEIGHT'] is not '', 'Please input the pretrain checkpoint.'
            self._load_encoder_weight(self.cfg['SOLVER']['PRETRAIN_WEIGHT'])
            self.pretrain_encoder = False

        # TODO:
        # This will be fix in the future
        if self.cfg['DATASETS']['NAME'] == 'coco':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                   'object_pred_idx_to_name.pkl'), 'rb') as file:
                self.vocab_dict = pickle.load(file)
        elif self.cfg['DATASETS']['NAME'] == 'vg_msdn':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                   'object_pred_idx_to_name.pkl'), 'rb') as file:
                self.vocab_dict = pickle.load(file)
        elif self.cfg['DATASETS']['NAME'] == 'vg_co':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                   'object_pred_idx_to_name.pkl'), 'rb') as file:
                self.vocab_dict = pickle.load(file)
        elif self.cfg['DATASETS']['NAME'] == 'vg':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                self.cfg['DATASETS']['REL_DICT_FILENAME']), 'rb') as file:
                self.vocab_dict = pickle.load(file)
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                self.cfg['DATASETS']['CLS_DICT_FILENAME']), 'rb') as file:
                self.cls_dict = pickle.load(file)

                
    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.model.to(self.device)
        best_val_mIOU = 0.
        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            if self.two_path and i%2 == 0:
                mode = 'w'
            elif self.two_path and i%2 == 1:
                mode = 'r'
            else: mode = 'w'
            log = self._run_epoch(i, 'train', mode)
            val_log = self._run_epoch(i, 'valid', 'w')
            merged_log = {**log, **val_log}
            all_log.append(merged_log)
            if (i + 1)%10 == 0 or val_log['valid_coarse_miou'] > best_val_mIOU:
                if val_log['valid_coarse_miou'] > best_val_mIOU:
                    best_val_mIOU = val_log['valid_coarse_miou']
                checkpoint = {
                    'log': all_log,
                    'state_dict': self.model.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'bbox_head_optimizer': self.bbox_head_optimizer.state_dict(),
                    'n_steps': self.encoder_scheduler.n_current_steps,
                }

                check_path = os.path.join(self.save_dir, 'checkpoint_' + str(i+1) + '_{}'.format(val_log['valid_coarse_miou']) + '.pth')
                torch.save(checkpoint, check_path)
                self.logger.info("SAVING CHECKPOINT: {}".format(check_path))

    def test(self):
        self.model.to(self.device)
        self._run_epoch(self.begin_epoch, 'test')

    def _run_epoch(self, epoch, phase, mode):
        self.logger.info('[Phase: {}, Epoch: {}]'.format(phase, epoch))
        if phase == 'train':
            self.model.train()
            if mode == 'w':
                dataloader = self.dataloader
            else:
                dataloader = self.dataloader_r
        else:
            self.model.eval()
            if mode == 'w':
                dataloader = self.dataloader.split_validation()
            else:
                dataloader = self.dataloader_r.split_validation()
        total_loss = 0
        total_vocab_loss = 0
        total_token_type_loss = 0
        total_obj_id_loss = 0
        total_rel_box_loss = 0
        total_coar_box_loss = 0
        total_refi_box_loss = 0

        total_correct = 0
        total_label = 0
        total_correct_id = 0
        total_label_id = 0
        total_correct_type = 0
        total_label_type = 0

        coarse_miou = 0
        refine_miou = 0
        for batch_idx, (input_token, input_obj_id, output_obj_id, coarse_box_label, 
                        output_label, segment_label, token_type)  in enumerate(dataloader):
            input_token = input_token.to(self.device)
            input_obj_id = input_obj_id.to(self.device)
            output_obj_id = output_obj_id.to(self.device)
            coarse_box_label = coarse_box_label.to(self.device)
            output_label = output_label.to(self.device)
            segment_label = segment_label.to(self.device)
            token_type = token_type.to(self.device)
            src_mask = (input_token != 0).unsqueeze(1).to(self.device)
            if self.cfg['MODEL']['ENCODER']['ENABLE_NOISE']:
                noise = torch.randn(input_token.shape[0], input_token.shape[1], 
                                    self.noise_size).to(self.device)

            trg_tmp = input_token[:,:-1]
            trg_input_box = coarse_box_label[:, :-1]
            trg_mask = (trg_tmp != self.pad_index).unsqueeze(1).to(self.device)
            global_mask = input_token >= 2

            if phase == 'train':
                vocab_logits, obj_id_logits, token_type_logits, coarse_box, coarse_gmm, refine_box, refine_gmm = self.model(input_token, input_obj_id, segment_label, token_type, src_mask, trg_input_box, trg_mask, epoch=epoch, global_mask=global_mask)
            else:
                vocab_logits, obj_id_logits, token_type_logits, coarse_box, coarse_gmm, refine_box, refine_gmm = self.model(input_token, input_obj_id, segment_label, token_type, src_mask, inference = True, epoch=epoch, global_mask=global_mask)
            
            # compute log probs
            log_probs_vocab = F.log_softmax(vocab_logits, dim=-1)
            log_probs_obj_id = F.log_softmax(obj_id_logits, dim=-1)
            # log_probs_vocab = vocab_logits
            log_probs_type = F.log_softmax(token_type_logits, dim=-1)
            # print("log_probs_cats shape:", log_probs_cats.size())

            # NLLLoss: Src-> N*C (C for classes), Trg-> N 
            log_probs_vocab = log_probs_vocab.reshape(log_probs_vocab.size(0) * log_probs_vocab.size(1), log_probs_vocab.size(2))
            log_probs_obj_id = log_probs_obj_id.reshape(log_probs_obj_id.size(0) * log_probs_obj_id.size(1), log_probs_obj_id.size(2))
            log_probs_type = log_probs_type.reshape(log_probs_type.size(0) * log_probs_type.size(1), log_probs_type.size(2))
            
            if not self.pretrain_encoder:
                coarse_box = coarse_box.reshape(coarse_box.size(0) * coarse_box.size(1), coarse_box.size(2))
                if self.cfg['MODEL']['DECODER']['BOX_LOSS'] == 'PDF':
                    coarse_gmm = coarse_gmm.reshape(coarse_gmm.size(0) * coarse_gmm.size(1), coarse_gmm.size(2))
                
                if self.refine:
                    refine_box = refine_box.reshape(refine_box.size(0) * refine_box.size(1), refine_box.size(2))
                    if self.cfg['MODEL']['REFINE']['BOX_LOSS'] == 'PDF':
                        refine_gmm = refine_gmm.reshape(refine_gmm.size(0) * refine_gmm.size(1), refine_gmm.size(2))
                        
            trg_vocab = output_label.reshape(output_label.size(0) * output_label.size(1))
            trg_obj_id=output_obj_id.reshape(output_obj_id.size(0) * output_obj_id.size(1))
            trg_type = token_type.reshape(token_type.size(0) * token_type.size(1))
            coarse_box_label = coarse_box_label.reshape(coarse_box_label.size(0) * coarse_box_label.size(1), coarse_box_label.size(2))

            # compute batch loss
            vocab_loss = self.loss(log_probs_vocab, trg_vocab) / input_token.size(0)
            obj_id_loss = self.loss(log_probs_obj_id, trg_obj_id) / input_token.size(0)
#            vocab_loss = self.focal_loss(log_probs_vocab, trg_vocab) / input_token.size(0)
            type_loss = self.loss(log_probs_type, trg_type) / input_token.size(0)
    
            if not self.pretrain_encoder:
                if self.cfg['MODEL']['DECODER']['BOX_LOSS'] == 'PDF':
                    box_loss, kl_loss = self.box_loss(coarse_gmm, coarse_box_label, False) 
                    rel_loss, rel2_loss = self.rel_loss(coarse_gmm, coarse_box_label)
#                     rel_loss = rel2_loss * 0.
                    box_loss /= input_token.size(0)
                    kl_loss /= input_token.size(0)
                    rel_loss /= input_token.size(0)
                    rel2_loss /= input_token.size(0)
                    
                else:
                    box_loss, kl_loss = self.box_loss(coarse_box, coarse_box_label)
                    box_loss /= input_token.size(0)
                    kl_loss /= input_token.size(0)
                    rel_loss = 0
                    rel2_loss = 0
                if self.refine:
                    if self.cfg['MODEL']['REFINE']['BOX_LOSS'] == 'PDF':
                        refine_box_loss, refine_kl_loss = self.refine_box_loss(refine_gmm, coarse_box_label, False) 
                        refine_box_loss /= input_token.size(0)
                        refine_kl_loss /= input_token.size(0)
                    else:
                        refine_box_loss, refine_kl_loss = self.refine_box_loss(refine_box, coarse_box_label) 
                        refine_box_loss /= input_token.size(0)
                        refine_kl_loss /= input_token.size(0)

            # print("Loss shape:", log_probs_cats.size(), trg_cats_id.size())
            tot_vocab_loss = vocab_loss * self.cfg['MODEL']['LOSS']['WEIGHT_VOCAB_LOSS']
            tot_obj_id_loss = obj_id_loss * self.cfg['MODEL']['LOSS']['WEIGHT_VOCAB_LOSS']
            tot_type_loss = type_loss * self.cfg['MODEL']['LOSS']['WEIGHT_TYPE_LOSS']
            
            tot_rel_box_loss = torch.tensor(0.).cuda()
            tot_rel2_box_loss = torch.tensor(0.).cuda()
            tot_coar_box_loss = torch.tensor(0.).cuda()
            tot_refi_box_loss = torch.tensor(0.).cuda()
            tot_coar_kl_loss = torch.tensor(0.).cuda()
            tot_refi_kl_loss = torch.tensor(0.).cuda()
            if not self.pretrain_encoder:
                tot_coar_box_loss += box_loss * self.cfg['MODEL']['LOSS']['WEIGHT_COARSE_BOX_LOSS']
                tot_coar_kl_loss += kl_loss * self.cfg['MODEL']['LOSS']['WEIGHT_COARSE_BOX_LOSS'] * 0.1
                tot_rel_box_loss += rel_loss 
                tot_rel2_box_loss += rel2_loss * self.cfg['MODEL']['LOSS']['WEIGHT_COARSE_BOX_LOSS']

                if self.refine:
                    tot_refi_box_loss += refine_box_loss * self.cfg['MODEL']['LOSS']['WEIGHT_REFINE_BOX_LOSS']
                    tot_refi_kl_loss += refine_kl_loss * self.cfg['MODEL']['LOSS']['WEIGHT_REFINE_BOX_LOSS'] * 0.1

            loss = tot_vocab_loss + tot_type_loss + tot_coar_box_loss + tot_refi_box_loss + tot_coar_kl_loss + tot_refi_kl_loss + tot_obj_id_loss + tot_rel_box_loss + tot_rel2_box_loss
            
            if phase == 'train':
                self.encoder_optimizer.zero_grad()
                self.bbox_head_optimizer.zero_grad()
                loss.backward()
                self.encoder_scheduler.step_and_update_lr()
                if not self.pretrain_encoder:
                    self.bbox_head_scheduler.step_and_update_lr()
            
            if not self.pretrain_encoder:
                coarse_miou += self.IOU_c.val_iou(coarse_box, coarse_box_label, is_std=False)
                if self.refine:
                    refine_miou += self.IOU_c.val_iou(refine_box, coarse_box_label, is_std=False)
                else: refine_miou += 0
            
            correct, total = self._calc_acc(log_probs_vocab, trg_vocab)
            correct_id, total_id = self._calc_acc(log_probs_obj_id, trg_obj_id)
            correct_type, total_type = self._calc_acc(log_probs_type, trg_type)
            total_correct += correct
            total_label += total
            total_correct_id += correct_id
            total_label_id += total_id
            total_correct_type += correct_type
            total_label_type += total_type

            total_loss += loss.item()
            total_vocab_loss += tot_vocab_loss.item()
            total_obj_id_loss += tot_obj_id_loss.item()
            total_token_type_loss += tot_type_loss.item()
            if not self.pretrain_encoder:
                total_rel_box_loss += tot_rel_box_loss.item()
                total_coar_box_loss += (tot_coar_box_loss.item() + tot_coar_kl_loss.item())
                total_refi_box_loss += (tot_refi_box_loss.item() + tot_refi_kl_loss.item())
            if phase == 'train':
                if batch_idx % self.cfg['OUTPUT']['NUM_STEPS_SHOW_LOSS'] == 0:
                    self.logger.info('[%d/%d] Loss: %.4f Loss_vocab: %.4f Loss_obj_id: %.4f Loss_token_type: %.4f Loss_box: [%.4f,%.4f] Loss_kl: [%.4f,%.4f] Loss_rel: [%.4f, %.4f] Co IOU: %.4f Re IOU: %.4f'% (batch_idx + 1, len(dataloader), loss.item(), tot_vocab_loss.item(), tot_obj_id_loss.item(), tot_type_loss.item(), tot_coar_box_loss.item(), tot_refi_box_loss.item(), tot_coar_kl_loss.item(), tot_refi_kl_loss.item(),tot_rel_box_loss.item(), tot_rel2_box_loss.item(), coarse_miou/(batch_idx+1), refine_miou/(batch_idx+1)))
                
            elif phase == 'valid' and batch_idx == 0:
                print("INPUT: {}".format(\
                     self.idx2vocab(input_token[0, :16].detach().cpu().numpy(),0)))
                print("GT: {}".format(\
                     self.idx2vocab(output_label[0, :16].detach().cpu().numpy(),0)))
                print("PRED {}".format(\
                     self.idx2vocab(torch.max(vocab_logits[0, :16], \
                               dim=1)[1].detach().cpu().numpy(),0)))
                
            elif phase == 'test':
                print("INPUT:",
                     self.idx2vocab(input_token[0, :16].detach().cpu().numpy(),0))
                print("GT:", 
                     self.idx2vocab(output_label[0, :16].detach().cpu().numpy(),0))
                print("PRED:",
                     self.idx2vocab(torch.max(vocab_logits[0, :16],\
                            dim=1)[1].detach().cpu().numpy(),0))
            
        acc = (total_correct.float() / total_label.float()).item()
        acc_id = (total_correct_id.float() / total_label_id.float()).item()
        acc_type = (total_correct_type.float() / total_label_type.float()).item()
        log = self._log_epoch(epoch, total_loss/len(dataloader), 
                   total_vocab_loss/len(dataloader), 
                   total_obj_id_loss/len(dataloader), 
                   total_token_type_loss/len(dataloader), 
                   total_rel_box_loss/len(dataloader), 
                   total_coar_box_loss/len(dataloader),
                   total_refi_box_loss/len(dataloader),
                   coarse_miou/len(dataloader),
                   refine_miou/len(dataloader),
                   acc, acc_id, acc_type, phase, self.encoder_optimizer, 
                   self.bbox_head_optimizer)
        return log

    def _calc_acc(self, logits, gt):
        """
        Param
            logits: Tensor, (B * max_length, C)
            gt:   Tensor, (B * max_length)
        """
        pred = torch.max(logits, dim=1)[1]
        correct = torch.sum((pred==gt) & (gt != 0))
        total = torch.sum((gt != 0))
        return correct, total
    
    def _log_epoch(self, epoch, total_loss, total_vocab_loss, total_obj_id_loss,
                   total_token_type_loss, total_rel_box_loss,
                   total_coar_box_loss, total_refi_box_loss, coarse_miou, refine_miou,
                   acc, acc_id, acc_type, phase, encoder_optimizer, bbox_head_optimizer):
        
        log = {
            'epoch': epoch,
            phase + '_loss': total_loss,
            phase + '_vocab_loss': total_vocab_loss,
            phase + '_obj_id_loss': total_obj_id_loss,
            phase + '_token_type_loss': total_token_type_loss,
            phase + '_rel_box_loss': total_rel_box_loss,
            phase + '_coar_box_loss': total_coar_box_loss,
            phase + '_refi_box_loss': total_refi_box_loss,
            phase + '_coarse_miou': coarse_miou,
            phase + '_refine_miou': refine_miou
        }
        self.tb_writer.add_scalar( phase + "/Loss", total_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_vocab", total_vocab_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_obj_id", total_obj_id_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_token_type", total_token_type_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_rel_box", total_rel_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_coar_box", total_coar_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_refi_box", total_refi_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Coarse_miou", coarse_miou, epoch)
        self.tb_writer.add_scalar( phase + "/Refine_miou", refine_miou, epoch)

        self.tb_writer.add_scalar( phase + "/mask_acc", acc, epoch)
        self.tb_writer.add_scalar( phase + "/obj_id_acc", acc_id, epoch)
        self.tb_writer.add_scalar( phase + "/type_acc", acc_type, epoch)
        self.tb_writer.add_scalar( phase + "/enc_lr", encoder_optimizer.param_groups[0]['lr'], epoch)
        self.tb_writer.add_scalar( phase + "/box_lr", bbox_head_optimizer.param_groups[0]['lr'], epoch)
        self.logger.info('[TOTAL] Loss: %.4f Loss_vocab: %.4f Loss_token_type: %.4f Loss_coar_box: %.4f Loss_refi_box: %.4f Loss_rel_box: %.4f'%(total_loss, total_vocab_loss, total_token_type_loss,total_coar_box_loss, total_refi_box_loss, total_rel_box_loss))
        self.logger.info('[TOTAL] Coarse_mIOU: %.4f Refine_mIOU: %.4f'%(coarse_miou, refine_miou))
        self.logger.info("[TOTAL] Mask word acc: %.4f"%(acc))
        self.logger.info("[TOTAL] Mask obj_id acc: %.4f"%(acc_id))
        self.logger.info("[TOTAL] Mask type acc: %.4f"%(acc_type))
        self.logger.debug("="*30)

        return log

    def idx2vocab(self, idx, modality):
        sent = ""
        for i in range(len(idx)):
            if modality == 'text' or modality == 0:
                sent += self.vocab_dict[idx[i]]
            elif modality == 'image' or modality == 1:
                sent += self.cls_dict[idx[i]]
            sent += " "
        return sent

    def build_loss(self):
        rel_gt = self.cfg['MODEL']['ENCODER']['REL_GT']
        raw_batch_size = self.cfg['SOLVER']['BATCH_SIZE']
        KD_ON = self.cfg['MODEL']['LOSS']['KD_LOSS']
        Topk = self.cfg['MODEL']['LOSS']['TOPK']
        if self.cfg['MODEL']['DECODER']['BOX_LOSS'] == 'PDF':
            box_loss = Log_Pdf(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., rel_gt = rel_gt, raw_batch_size=raw_batch_size, KD_ON=KD_ON, Topk=Topk)
            rel_loss = Rel_Loss(reduction = 'sum', raw_batch_size=raw_batch_size)
        else:
            box_loss = RegLoss(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1.)
            rel_loss = None
        refine = self.cfg['MODEL']['REFINE']['REFINE']
        if refine:
            if self.cfg['MODEL']['REFINE']['BOX_LOSS'] == 'PDF':
                refine_box_loss = Log_Pdf(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., rel_gt = rel_gt, raw_batch_size=raw_batch_size, KD_ON=KD_ON, Topk=Topk)

            else:
                refine_box_loss = RegLoss(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., refine=True)


            return refine, box_loss, rel_loss, refine_box_loss
        else:
            return refine, box_loss, rel_loss, None


    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        torch.backends.cudnn.benchmark = False
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            try:
                self.model.load_state_dict(checkpoint['state_dict']).to(self.device)
            except:
                self.logger.info('[Resume] Only load some ckpt from checkpoint')
                pretrain_ckpt = {k: v for k, v in checkpoint['state_dict'].items()
                                 if 'bbox_head' not in k}
                self.model.load_state_dict(pretrain_ckpt, strict=False).to(self.device)

            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            self.bbox_head_optimizer.load_state_dict(checkpoint['bbox_head_optimizer'])
            self.encoder_scheduler.load_state_dict(checkpoint['n_steps'])
            self.bbox_head_scheduler.load_state_dict(checkpoint['n_steps'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')

    def _load_encoder_weight(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        except Exception as e:
            print(e)
            self.logger.error('[Resume] Cannot load from checkpoint')
