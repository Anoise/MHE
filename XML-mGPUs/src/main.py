import argparse
from xmlrpc.client import Boolean
import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset_v3 import Dataset, createDataCSV
import utils
from model_amp_v3 import XMLModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=False, default=1)
    parser.add_argument('--update_count', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=0.0001)
    parser.add_argument('--seed', type=int, required=False, default=6088)
    parser.add_argument('--epoch', type=int, required=False, default=15)
    parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')  # amazon670k  
    parser.add_argument('--data_path', type=str, required=False, default='/home/user/Data/data')  # amazon670k  
    parser.add_argument('--model', type=str, required=False, default='bert-base')
    parser.add_argument('--model_path', type=str, required=False, default='../NLP-Model')
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--num_group', type=int, required=False, default=86)
    parser.add_argument('--hidden_dim', type=int, required=False, default=300)
    
    parser.add_argument('--candidates_topk', type=int, required=False, default=10)
    parser.add_argument('--max_len', type=int, required=False, default=512)
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--dropout', type=float, required=False, default=0.1)
    parser.add_argument('--feature_layers', type=int, required=False, default=5)
    parser.add_argument('--use_swa', action='store_true', default=True) # ,default=True
    parser.add_argument('--swa_warmup_epoch', type=int, required=False, default=1)
    parser.add_argument('--swa_step', type=int, required=False, default=100)
    
    parser.add_argument('--valid', action='store_true')  # default=True
    parser.add_argument('--eval_step', type=int, required=False, default=2000)
    parser.add_argument('--eval_model', action='store_true')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=False)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')

    return parser.parse_args()


def eval_one_epoch(epoch, eval_loader):
    bar = tqdm.tqdm(total=len(eval_loader))
    bar.set_description(f'eval-{epoch}')
    acc_meter = utils.AverageMeter()
    g_acc_meter = utils.AverageMeter()
    model.eval()
    if args.use_swa: model_cpu.swa_swap_params() #;print('swa_swap_params 1...')
    with torch.no_grad():
        for data in eval_loader:
            bar.update(1)
            inputs = tuple(_.cuda() for _ in data[:3])
            outputs, candidates, group_outputs = model(*inputs)
            
            labels, group_labels = data[3], data[4]
            p1, p3, p5 = acc_meter.get_accuracy(outputs, labels, candidates)
            g_p1, g_p3, g_p5 = g_acc_meter.get_accuracy(group_outputs, group_labels)
    if args.use_swa: model_cpu.swa_swap_params() #;print('swa_swap_params 2...')
    bar.close()
    return p1, p3, p5, g_p1, g_p3, g_p5

   
def train_one_epoch(epoch, dataloader, optimizer, loss_scaler):
    bar = tqdm.tqdm(total=len(dataloader))
    bar.set_description(f'train-{epoch}')
    train_loss = 0
    model.train()
    if args.use_swa and epoch == args.swa_warmup_epoch: model_cpu.swa_init() #;print('swa_init ...')
    
    for step, data in enumerate(dataloader):
        bar.update(1)
        inputs = tuple(_.cuda() for _ in data)
        with torch.cuda.amp.autocast():
            outputs, loss = model(*inputs)
        
        optimizer.zero_grad()
        train_loss += loss.item()
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())

        if args.use_swa and step % args.swa_step == 0: model_cpu.swa_step() #;print('swa_step ...')
        bar.set_postfix(loss=loss.item())
    bar.close()
    return train_loss / step


def main():

    train_d = Dataset(df, 'train', tokenizer, label_map, args.max_len, num_classes)
    sampler_train = torch.utils.data.DistributedSampler(train_d, num_replicas=num_tasks, 
                                                        rank=global_rank, shuffle=True)
    trainloader = DataLoader(train_d, sampler=sampler_train, batch_size=args.batch, num_workers=8,
                             pin_memory=args.pin_mem, drop_last=True, persistent_workers=True)

    test_d = Dataset(df, 'test', tokenizer, label_map, args.max_len, num_classes)
    sampler_test = torch.utils.data.DistributedSampler(test_d, num_replicas=num_tasks, 
                                                       rank=global_rank, shuffle=False)
    testloader = DataLoader(test_d, sampler=sampler_test, batch_size=args.batch, num_workers=8,
                            pin_memory=args.pin_mem, drop_last=False,persistent_workers=True)
    
    if args.valid:
        valid_d = Dataset(df, 'valid', tokenizer, label_map, args.max_len, num_classes)
        sampler_val = torch.utils.data.DistributedSampler(valid_d, num_replicas=num_tasks, 
                                                          rank=global_rank, shuffle=False)
        validloader = DataLoader(valid_d,sampler=sampler_val, batch_size=args.batch, num_workers=8,
                            pin_memory=args.pin_mem, drop_last=False, persistent_workers=True)

    start_epoch = 0
    max_p = [0,0,0,0]
    if args.resume:
        print('resume models ...')
        checkpoints = torch.load(f'models_v6/model-{args.dataset}.bin',map_location='cpu')
        model_cpu.load_state_dict(checkpoints['model'])
        start_epoch = checkpoints['epoch']
        max_p = checkpoints['max_accuracy']
        optimizer.load_state_dict(checkpoints['optimizer'])
        print('resume start epoch p1,p3,p5',start_epoch, max_p)
    
    loss_scaler = utils.get_loss_scaler(optimizer)
    for epoch in range(start_epoch, args.epoch+5):
        sampler_train.set_epoch(epoch)
        
        
        train_loss = train_one_epoch(epoch, trainloader, optimizer, loss_scaler)

        if args.valid:
            p1, p3, p5, g_p1, g_p3, g_p5  = eval_one_epoch(epoch, validloader)
        else:
            p1, p3, p5, g_p1, g_p3, g_p5 = eval_one_epoch(epoch, testloader)

        logger(f'eval epoch={epoch:>2}: p1={p1:.4f}, p3={p3:.4f}, p5={p5:.4f}, \
                    g_p1={g_p1:.4f}, g_p3={g_p3:.4f}, g_p5={g_p5:.4f}, max_p5={max_p[-1]:.4f}, train_loss:{train_loss:.6f}')

        if max_p[-1] < p5:
            max_p = [epoch, p1, p3, p5]
            if args.use_swa: model_cpu.swa_swap_params()
            utils.save_on_master({
                    'model': model_cpu.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': 0,
                    'args': args,
                    'max_accuracy': [p1,p3,p5],
                }, 'models_v6/model-{}.bin'.format(experiment_name))


        # if args.valid:
        #     p1, p3, p5, g_p1, g_p3, g_p5 = eval_one_epoch(epoch, testloader)
        #     logger(f'test epoch={epoch:>2}: p1={p1:.4f}, p3={p3:.4f}, p5={p5:.4f}, \
        #                         g_p1={g_p1:.4f}, g_p3={g_p3:.4f}, g_p5={g_p5:.4f}, \
        #                          max_p5={max_p[-1]:.4f}, train_loss:{train_loss:.6f}')
        
        if epoch >= args.epoch + 5 and max_p[0] - epoch > 5: break
    logger(f'best epoch: {max_p[0]}, p1={max_p[1]:.4f}, p3={max_p[2]:.4f}, p5={max_p[-1]:.4f}')

if __name__ == '__main__':
    
    args = get_args()
    experiment_name = utils.get_exp_name(args)
    utils.init_distributed_mode(args)
    utils.init_seed(args.seed ) # + utils.get_rank()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    device = torch.device(args.device)
    
    logger = utils.Logger('log_'+experiment_name)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    df, label_map = createDataCSV(args.dataset,args.data_path)
    print(f'load {args.dataset} dataset...',
          'df shape: ',df.shape,'num labels', len(label_map))
    

    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    logger(f'load {args.dataset} dataset with ' f'{len(df[df.dataType =="train"])} \
          train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')
    
    num_classes =  utils.check_gorup(len(label_map),args.num_group) 
    model = XMLModel(n_labels=len(label_map),num_classes=num_classes, model_name=args.model, 
                     model_path=args.model_path, candidates_topk=args.candidates_topk, 
                     hidden_dim=args.hidden_dim, feature_layers=args.feature_layers, 
                     dropout=args.dropout,device=device)
    if args.dataset in ['wiki500k', 'amazon670k']:
        tokenizer = utils.get_fast_tokenizer(args.model,args.model_path)    
    else: 
        tokenizer = utils.get_tokenizer(args.model,args.model_path)
 
    optimizer = torch.optim.AdamW(utils.get_optimizer_params(model), lr=args.lr)#, eps=1e-8)
    
    model_cpu = model
    model.to(device)
    if torch.cuda.device_count() > 1:
        logger("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=True)
        model_cpu = model.module
 
    main()
