import os
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import tqdm as tq_src
from models import *
import torch.nn.functional as F
from datasets import PairLoader
import torch.distributed as dist
from torch.nn.modules.loss import _Loss
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter, CosineScheduler, pad_img
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

# function to reduce tensor across all processes
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# validation function
def valid(val_loader, network):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()
    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        with torch.no_grad():
            H, W = source_img.shape[2:]
            source_img = pad_img(source_img, network.module.patch_size if hasattr(network.module, 'patch_size') else 16)
            output = network(source_img)[0].clamp_(-1, 1)
            output = output[:, :, :H, :W]
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))
    return PSNR.avg

def main():
    # define model
    print("==> Building model: " + args.model)
    network = eval(args.model)()
    network.cuda()
    if args.use_ddp:
        network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
        if b_setup['batch_size'] // world_size < 16:
            if local_rank == 0: print('==> Using SyncBN because of too small norm-batch-size.')
            nn.SyncBatchNorm.convert_sync_batchnorm(network)
    else:
        network = DataParallel(network)
        if b_setup['batch_size'] // torch.cuda.device_count() < 16:
            print('==> Using SyncBN because of too small norm-batch-size.')
            convert_model(network)
    # restore loss, edge loss, classification loss
    criterion = []
    criterion.append(nn.MSELoss(size_average=True).cuda())
    criterion.append(nn.MSELoss(size_average=True).cuda())
    criterion.append(nn.CrossEntropyLoss().cuda())
    # define optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=b_setup['lr'], weight_decay=b_setup['weight_decay'])
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=b_setup['lr'] * 1e-2, 
                                   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=b_setup['epochs'])
    scaler = GradScaler()
    # load saved model
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        best_psnr = 0
        cur_epoch = 0
    else:
        if not args.use_ddp or local_rank == 0: print('==> Loaded existing trained model.')
        model_info = torch.load(os.path.join(save_dir, args.model+'.pth'), map_location='cpu')
        network.load_state_dict(model_info['state_dict'])
        optimizer.load_state_dict(model_info['optimizer'])
        lr_scheduler.load_state_dict(model_info['lr_scheduler'])
        wd_scheduler.load_state_dict(model_info['wd_scheduler'])
        scaler.load_state_dict(model_info['scaler'])
        cur_epoch = model_info['cur_epoch']
        best_psnr = model_info['best_psnr']
    # define train dataset
    train_dataset = PairLoader(args.train_set,os.path.join(args.data_dir, args.train_set), 'train', 
                               b_setup['t_patch_size'], 
                               b_setup['edge_decay'], 
                               b_setup['data_augment'], 
                               b_setup['cache_memory'])
    train_loader = DataLoader(train_dataset,
                              batch_size=b_setup['batch_size'] // world_size,
                              sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter'] // world_size),
                              num_workers=args.num_workers // world_size,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)
    # define val dataset
    val_dataset = PairLoader(args.val_set,os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'], 
                             b_setup['v_patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=max(int(b_setup['batch_size'] * b_setup['v_batch_ratio'] // world_size), 1),
                            num_workers=args.num_workers // world_size,
                            pin_memory=True)
    # start training
    if not args.use_ddp or local_rank == 0:
        print('==> Start training, current model name: ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
    with tq_src.trange(0, b_setup['epochs'], desc='epochs', dynamic_ncols=True, leave=True) as tbar:
        display_info_epoch={}
        for epoch in tbar:
            if local_rank == 0 :
                tbar.set_postfix(display_info_epoch)
                tbar.refresh()
            pbar = tqdm(total=len(train_loader), leave=True, desc='iter', dynamic_ncols=True, position=1)
            frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs'])
            losses = AverageMeter()
            torch.cuda.empty_cache()
            network.eval() if frozen_bn else network.train()
            display_info_iter = {}
            for batch in train_loader:
                source_img = batch['source'].cuda()
                target_img = batch['target'].cuda()
                edge_img   = batch['edge'].cuda()
                class_img  = batch['class_label'].cuda()
                class_label = torch.flatten(class_img, 1)
                class_label = torch.mean(class_label, dim=1)
                class_label = torch.round(class_label).long()
                with autocast(args.use_mp):
                    output, pre_edge, pre_class = network(source_img)
                    loss_restore = criterion[0](output, target_img)
                    loss_edge = criterion[1](pre_edge, edge_img)
                    loss_classify = criterion[2](pre_class, class_label)
                    loss = loss_restore + loss_edge + loss_classify
                if local_rank == 0: 
                    display_info_iter.update({"lossa: ": format(loss.item(), '.3f'),
                                              "lossr: ": format(loss_restore.item(), '.3f'),
                                              "losse: ": format(loss_edge.item(), '.3f'),
                                              "lossc: ": format(loss_classify.item(), '.3f')
                                              })
                    pbar.set_postfix(display_info_iter)
                    pbar.update()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.use_ddp: loss = reduce_mean(loss, dist.get_world_size())
                losses.update(loss.item())
            loss = losses.avg
            lr_scheduler.step(epoch + 1)
            wd_scheduler.step(epoch + 1)
            if not args.use_ddp or local_rank == 0:
                writer.add_scalar('train_loss', loss, epoch)
            # validation
            if epoch % b_setup['period'] == 0:
                avg_psnr = valid(val_loader, network)
                if not args.use_ddp or local_rank == 0:
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        torch.save({'cur_epoch': epoch + 1,
                                    'best_psnr': best_psnr,
                                    'state_dict': network.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                    'lr_scheduler' : lr_scheduler.state_dict(),
                                    'wd_scheduler' : wd_scheduler.state_dict(),
                                    'scaler' : scaler.state_dict()},
                                    os.path.join(save_dir, args.model + '.pth'))
                    writer.add_scalar('valid_psnr', avg_psnr, epoch)
                    writer.add_scalar('best_psnr', best_psnr, epoch)
                if args.use_ddp: dist.barrier()  
            if local_rank == 0 :
                display_info_epoch.update({"train_avg_loss: ": format(loss, ".4f"), "val_avg_psnr: ": format(avg_psnr, ".4f"), "val_best_psnr: ": format(best_psnr, ".4f")})


        

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MUL_UNET', type=str, help='model name')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use Mixed Precision')
    parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/dehaze_datasets/', type=str, help='path to dataset')
    parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
    parser.add_argument('--train_set', default='allweather', type=str, help='train dataset name')
    #TODO
    parser.add_argument('--val_set', default='SOTS-IN', type=str, help='valid dataset name')
    parser.add_argument('--exp', default='all_weather', type=str, help='experiment setting')
    args = parser.parse_args()
    # initialize DDP
    if args.use_ddp:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        if local_rank == 0: print('==> Using DDP.')
    else:
        world_size = 1
    # load experiment setting
    with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
        b_setup = json.load(f)
    # start training
    main()
