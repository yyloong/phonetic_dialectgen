import os
import os.path as osp
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
warnings.simplefilter('ignore')

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mymodels import *  # 使用简化的模型
from mymeldata import build_dataloader  # 使用简化的数据加载器
from utils import *
from losses import *
from optimizers import build_optimizer
import time

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")
    
    batch_size = config.get('batch_size', 10)
    device = accelerator.device
    
    epochs = config.get('epochs_1st', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    
    max_len = config.get('max_len', 200)
    
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    
    model_params = recursive_munch(config['model_params'])
    
    # 只构建需要的模型组件 - 移除decoder和所有判别器
    model = Munch(
        text_encoder=TextEncoder(channels=model_params.hidden_dim, kernel_size=5, depth=model_params.n_layer, n_symbols=model_params.n_token),
        text_aligner=text_aligner,
        pitch_extractor=pitch_extractor,
    )

    best_loss = float('inf')  # best test loss

    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    
    for k in model:
        model[k] = accelerator.prepare(model[k])
    
    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )
    
    _ = [model[key].to(device) for key in model]

    # 只为需要训练的模型初始化优化器
    trainable_models = ['text_encoder']  # 只训练文本编码器
    if TMA_epoch > 0:  # 如果有TMA阶段，也训练text_aligner和pitch_extractor
        trainable_models.extend(['text_aligner', 'pitch_extractor'])
    
    optimizer = build_optimizer({key: model[key].parameters() for key in trainable_models},
                                  scheduler_params_dict= {key: scheduler_params.copy() for key in trainable_models},
                               lr=float(config['optimizer_params'].get('lr', 1e-4)))
    
    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])
    
    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                        load_only_params=config.get('load_only_params', True))
        else:
            start_epoch = 0
            iters = 0
    
    # in case not distributed
    try:
        n_down = model.text_aligner.module.n_down
    except:
        n_down = model.text_aligner.n_down
    
    # 只需要mel-spectrogram损失
    mel_loss_fn = nn.L1Loss()

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].train() for key in trainable_models]

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            # 统一使用简化的batch格式
            texts, input_lengths, mels, mel_input_length = batch
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)

            s2s_attn.masked_fill_(attn_mask, 0.0)
                        
            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)
    
            # get clips
            mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
            mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
        
            en = []
            gt = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

            en = torch.stack(en)
            gt = torch.stack(gt).detach()

            # 直接使用编码后的特征作为预测的mel-spectrogram
            # 可以添加一个简单的线性层将编码维度映射到mel维度
            if en.shape[1] != gt.shape[1]:
                # 如果维度不匹配，使用线性变换
                if not hasattr(model, 'mel_projection'):
                    model.mel_projection = nn.Linear(en.shape[1], gt.shape[1]).to(device)
                mel_pred = model.mel_projection(en.transpose(1, 2)).transpose(1, 2)
            else:
                mel_pred = en

            # 只计算mel-spectrogram重建损失
            optimizer.zero_grad()
            loss_mel = mel_loss_fn(mel_pred, gt)
            
            if epoch >= TMA_epoch: # start TMA training
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= texts.size(0)

                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                
                g_loss = loss_params.lambda_mel * loss_mel + \
                         loss_params.lambda_mono * loss_mono + \
                         loss_params.lambda_s2s * loss_s2s

            else:
                loss_s2s = 0
                loss_mono = 0
                g_loss = loss_mel
            
            running_loss += accelerator.gather(loss_mel).mean().item()

            accelerator.backward(g_loss)
            
            optimizer.step('text_encoder')
            
            if epoch >= TMA_epoch: 
                optimizer.step('text_aligner')
                optimizer.step('pitch_extractor')
            
            iters = iters + 1
            
            if (i+1)%log_interval == 0 and accelerator.is_main_process:
                print('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_mono, loss_s2s))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/mono_loss', loss_mono, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)

                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)
                                
        # 验证阶段
        loss_test = 0
        _ = [model[key].eval() for key in trainable_models]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                # waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                # 修正：使用统一的简化batch格式
                texts, input_lengths, mels, mel_input_length = batch

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn)

                # get clips
                mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
                mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])
                
                en = []
                gt = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start:random_start+mel_len])
                    gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                en = torch.stack(en)
                gt = torch.stack(gt).detach()

                # 预测mel-spectrogram
                if hasattr(model, 'mel_projection'):
                    mel_pred = model.mel_projection(en.transpose(1, 2)).transpose(1, 2)
                else:
                    mel_pred = en

                loss_mel = mel_loss_fn(mel_pred, gt)
                loss_test += accelerator.gather(loss_mel).mean().item()
                iters_test += 1

        if accelerator.is_main_process:
            print('Epochs:', epoch + 1)
            print('Validation loss: %.3f' % (loss_test / iters_test) + '\n\n\n\n')
            print('\n\n\n')
            writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure('eval/attn', attn_image, epoch)
            
            # 保存预测的mel-spectrogram用于可视化
            with torch.no_grad():
                for bib in range(min(len(asr), 6)):
                    mel_length = int(mel_input_length[bib].item())
                    gt_full = mels[bib, :, :mel_length].unsqueeze(0)
                    en_full = asr[bib, :, :mel_length // 2].unsqueeze(0)
                    
                    if hasattr(model, 'mel_projection'):
                        mel_pred_full = model.mel_projection(en_full.transpose(1, 2)).transpose(1, 2)
                    else:
                        mel_pred_full = en_full
                                        
                    # 可视化预测和真实的mel-spectrogram
                    pred_image = get_image(mel_pred_full[0].cpu().numpy())
                    gt_image = get_image(gt_full[0].cpu().numpy())
                    
                    writer.add_figure(f'eval/mel_pred_{bib}', pred_image, epoch)
                    if epoch == 0:
                        writer.add_figure(f'eval/mel_gt_{bib}', gt_image, epoch)

            if epoch % saving_epoch == 0:
                if (loss_test / iters_test) < best_loss:
                    best_loss = loss_test / iters_test
                print('Saving..')
                state = {
                    'net':  {key: model[key].state_dict() for key in model}, 
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': loss_test / iters_test,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, 'epoch_1st_%05d.pth' % epoch)
                torch.save(state, save_path)
                                
    if accelerator.is_main_process:
        print('Saving..')
        state = {
            'net':  {key: model[key].state_dict() for key in model}, 
            'optimizer': optimizer.state_dict(),
            'iters': iters,
            'val_loss': loss_test / iters_test,
            'epoch': epoch,
        }
        save_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
        torch.save(state, save_path)

if __name__=="__main__":
    main()
