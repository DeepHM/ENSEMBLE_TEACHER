from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader_aug import get_train_loader
from network import Network, SingleNetwork
from dataloader_aug import CityScape
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False
    
    
def update_ema_variables(model, ema_model, alpha, global_step) :
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        layer_name = ema_param[0]
        layer_name = layer_name.replace('module.','')
        ema_model.module.state_dict()[layer_name] = ema_param[1].data.mul_(alpha).add_(1 - alpha, param[1].data)            


def save_ckpt(model,optimizer_l,optimizer_r,epoch,path) :
    torch.save({'model' : model.module.state_dict(),
                'optimizer_l' : optimizer_l.state_dict(),
                'optimizer_r' : optimizer_r.state_dict(),
                'epoch' : epoch,
               }, path)
    
    
print('Labeled Ratio : ',config.labeled_ratio)
'''
For CutMix.
'''
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)

add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
    mask_generator
)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)



with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
#     train_loader, train_sampler = get_train_loader(engine, CityScape, train_source=config.train_source, \
#                                                    unsupervised=False, collate_fn=collate_fn)
#     unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, CityScape, \
#                 train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
#     unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, CityScape, \
#                 train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)
    train_loader, train_sampler = get_train_loader(engine, CityScape, train_source=config.train_source, \
                                               unsupervised=False, collate_fn=collate_fn, sampler_seed=0)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, CityScape, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn, sampler_seed=engine.local_rank*10+1)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, CityScape, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn, sampler_seed=engine.local_rank*10+2)

    
    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime())) + '_' + osp.basename(config.output_path)
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    pixel_num = 50000 * config.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                       min_kept=pixel_num, use_weight=False)
    criterion_cps = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    
    ema_branch1 = SingleNetwork(config.num_classes, criterion=criterion, norm_layer=BatchNorm2d, pretrained_model=config.pretrained_model)
    ema_branch2 = SingleNetwork(config.num_classes, criterion=criterion, norm_layer=BatchNorm2d, pretrained_model=config.pretrained_model)
    ema_branch1.load_state_dict(model.branch1.state_dict())
    ema_branch2.load_state_dict(model.branch2.state_dict())    

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                     base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                     base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
            ema_branch1.cuda()
            ema_branch2.cuda()
            ema_branch1 = DistributedDataParallel(ema_branch1)
            ema_branch2 = DistributedDataParallel(ema_branch2)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)
        
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)
    if engine.continue_state_object:
        engine.restore_checkpoint()     # it will change the state dict of optimizer also

    model.train()
    print('begin train')
    global_step = 0 
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
            unsupervised_train_sampler_0.set_epoch(epoch)
            unsupervised_train_sampler_1.set_epoch(epoch)
            
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0

        ''' supervised part '''
        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()
            
            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']
            unsup_imgs_strong_0 = unsup_minibatch_0['data_strong']
            unsup_imgs_strong_1 = unsup_minibatch_1['data_strong']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
#             mask_params = mask_params.cuda(non_blocking=True)

            # unsupervised loss on model/branch#1
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_strong_0 * (1 - batch_mix_masks) + unsup_imgs_strong_1 * batch_mix_masks
            unsup_imgs_mixed = unsup_imgs_mixed.cuda(non_blocking=True)
            batch_mix_masks = batch_mix_masks.cuda(non_blocking=True)            
     
            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                _, logits_u0_tea_1 = ema_branch1(unsup_imgs_0)
                _, logits_u1_tea_1 = ema_branch1(unsup_imgs_1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                _, logits_u0_tea_2 = ema_branch2(unsup_imgs_0)
                _, logits_u1_tea_2 = ema_branch2(unsup_imgs_1)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()

            # Mix teacher predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            # Get student#1 prediction for mixed image
            _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
            # Get student#2 prediction for mixed image
            _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)

            cps_loss = criterion_cps(logits_cons_stu_1, ps_label_2) + criterion_cps(logits_cons_stu_2, ps_label_1)
            dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            cps_loss = cps_loss / engine.world_size
            cps_loss = cps_loss * config.cps_weight

            # supervised loss on both models
            _, sup_pred_l = model(imgs, step=1)
            _, sup_pred_r = model(imgs, step=2)

            loss_sup = criterion(sup_pred_l, gts)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
            dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            loss_sup_r = loss_sup_r / engine.world_size
            
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            loss = loss_sup + loss_sup_r + cps_loss
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

            print_str = 'Epoch{}/{}'.format(epoch+1, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()
            
            global_step += 1
            update_ema_variables(model.module.branch1, ema_branch1, config.ema_decay, global_step)
            update_ema_variables(model.module.branch2, ema_branch2, config.ema_decay, global_step)
            

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
            logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))
            
        if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if (engine.distributed) and (engine.local_rank == 0):         
                print('Model save \n  > ',osp.join(config.snapshot_dir,'epoch-'+str(epoch+1)+'.pth'))
                print('EMA Model save \n  > ',osp.join(config.ema_snapshot_dir,'ema_branch1_epoch'+str(epoch+1)+'.pth'))
                print('EMA Model save \n  > ',osp.join(config.ema_snapshot_dir,'ema_branch2_epoch'+str(epoch+1)+'.pth'))
                save_ckpt(model,optimizer_l,optimizer_r,epoch,osp.join(config.snapshot_dir,'model_epoch'+str(epoch+1)+'.pth'))
                save_ckpt(ema_branch1,optimizer_l,optimizer_r,epoch,osp.join(config.ema_snapshot_dir,'ema_branch1_epoch'+str(epoch+1)+'.pth'))
                save_ckpt(ema_branch2,optimizer_l,optimizer_r,epoch,osp.join(config.ema_snapshot_dir,'ema_branch2_epoch'+str(epoch+1)+'.pth'))



                          
