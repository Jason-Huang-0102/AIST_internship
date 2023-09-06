import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
#from models.wide_resnet import Wide_ResNet
# from pytorch_cifar100.models.wideresidual import wideresnet ### NOTE: You need to change the model name here using the up-to-date model in torchvision.model
import torchvision
import horovod.torch as hvd
import distributed_optimizer as dis_hvd
# from distributed_optimizer import DistributedOptimizer
import os
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_MPI_THREADS_DISABLE'] = '1'
import zipfile
import os.path
import math
import time
from tqdm import tqdm
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import reducescatter_async
from horovod.torch.mpi_ops import synchronize
# from torchsummary import summary
from torch.optim.lr_scheduler import LambdaLR
# handles = {}


# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default='./', type=str)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/pdf/1605.07146v4.pdf
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.001,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--model', type=str, default="resnet50",
                    help='model')
parser.add_argument('--group-size', type=int, default=10,
                    help='combine parameter into group')

def train(epoch, log_dir):
    torch.cuda.synchronize()
    start_epoch = time.time()
    
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    rank = hvd.rank()
    if rank == 0:
        init_time_file = open(os.path.join(log_dir, "init_time.log"), "a", buffering=1)
        io_time_file = open(os.path.join(log_dir, "io_time.log"), "a", buffering=1)
        stagging_file = open(os.path.join(log_dir, "stagging.log"), "a", buffering=1)
        forward_time_file = open(os.path.join(log_dir, "forward_time.log"), "a", buffering=1)
        backward_time_file = open(os.path.join(log_dir, "backward_time.log"), "a", buffering=1)
        weightupdate_time_file = open(os.path.join(log_dir, "weightupdate_time.log"), "a", buffering=1)
        weightupdatesync_time_file = open(os.path.join(log_dir, "weightupdatesync_time.log"), "a", buffering=1)
        accuracy_file = open(os.path.join(log_dir, "accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "loss_per_epoch.log"), "a", buffering=1)
        accuracy_comp_file = open(os.path.join(log_dir, "accuracy_comp_iter.log"), "a", buffering=1)
        #accuracy_iter_file = open(os.path.join(log_dir, "accuracy_per_iter.log"), "a", buffering=1)
        #loss_iter_file = open(os.path.join(log_dir, "loss_per_iter.log"), "a", buffering=1)
        epoch_time_file = open(os.path.join(log_dir, "epoch_time.log"), "a", buffering=1)
        allgather_time_file = open(os.path.join(log_dir, "allgather_time.log"), "a", buffering=1)
        

    stop = time.time()
    print("{:.10f}".format(stop - start_epoch), file=init_time_file) if rank == 0 else None
    # print(hvd.rank())
    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        torch.cuda.synchronize()
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx == 2:
            #     break
            torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), "\t", hvd.rank(), file=io_time_file) if rank == 0 else None
            print("{:.10f}".format(stop - start), file=io_time_file) if rank == 0 else None
            # optimizer.synchronize()
            torch.cuda.synchronize()
            start = time.time()
            # adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            stop = time.time()
            print("{:.10f}".format(stop - start), file=stagging_file) if rank == 0 else None
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                # torch.cuda.synchronize()
                # print(1)
                start = time.time()
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                # forward
                # print(i)
                output = model(data_batch)
                loss = F.cross_entropy(output, target_batch)
                # torch.cuda.synchronize()
                stop = time.time()
                print("{:.10f}".format(stop - start), file=forward_time_file) if rank == 0 else None
                
                torch.cuda.synchronize()
                start = time.time()
                accuracy_iter = accuracy(output, target_batch)
                train_accuracy.update(accuracy_iter)
                train_loss.update(loss)
                # if rank == 0 and (batch_idx % number_iter_track == 0):
                    # print("{:.10f}".format(train_accuracy.avg), file=accuracy_iter_file) 
                    # print("{:.10f}".format(train_loss.avg), file=loss_iter_file) 
                # else:
                    # None
                torch.cuda.synchronize()
                # print(3.4)
                stop = time.time()
                print("{:.10f}".format(stop - start), file=accuracy_comp_file) if rank == 0 else None
                # print(4)
                
                # Average gradients among sub-batches
                # torch.cuda.synchronize()
                start = time.time()
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
                # torch.cuda.synchronize()
                stop = time.time()
                print("{:.10f}".format(stop - start), file=backward_time_file) if rank == 0 else None
                
            # print("-------for loop end-----------")
            # Gradient is applied across all ranks
            torch.cuda.synchronize()
            # weight update
            start = time.time()
            # print("------------main step() start----------------------")
            optimizer.step()
            torch.cuda.synchronize()
            # print("------------main step() end----------------------")
            # sync_start = time.time()
            # torch.cuda.synchronize()
            # _weight_exchange(model, index)
            # mg_we_forward._weight_exchange()
            stop = time.time()
            print("{:.10f}".format(stop - start), file=weightupdate_time_file) if rank == 0 else None
            # print("{:.10f}".format(stop - start), file=weightupdatesync_time_file) if rank == 0 else None
            
            start = time.time()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)
            torch.cuda.synchronize()
            stop = time.time()
            print("SYNC\t{:.10f}".format(stop - start), file=accuracy_comp_file) if rank == 0 else None
            
            start = time.time()

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        
    if rank == 0 :
        print("{:.10f}".format(train_accuracy.avg), file=accuracy_file) 
        print("{:.10f}".format(train_loss.avg), file=loss_file) 
    else:
        None
    
    torch.cuda.synchronize()
    stop_epoch = time.time()
    print("{:.10f}".format(stop_epoch - start_epoch), file=epoch_time_file) if rank == 0 else None
    
    if rank == 0:
        init_time_file.close()
        io_time_file.close()
        stagging_file.close()
        forward_time_file.close()
        backward_time_file.close()
        weightupdate_time_file.close()
        weightupdatesync_time_file.close()
        accuracy_file.close()
        loss_file.close()
        accuracy_comp_file.close()
        #accuracy_iter_file.close()
        #loss_iter_file.close()
        epoch_time_file.close()
   
def validate(epoch, log_dir):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    rank = hvd.rank()
    
    if rank ==0:
        accuracy_file = open(os.path.join(log_dir, "val_accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "val_loss_per_epoch.log"), "a", buffering=1)

    
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
    if rank == 0 :
        print("{:.10f}".format(val_accuracy.avg), file=accuracy_file) 
        print("{:.10f}".format(val_loss.avg), file=loss_file) 
    else:
        None
    if rank ==0:
        accuracy_file.close()
        loss_file.close()

# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
#  On CIFAR learning rate dropped by 0.2 at 60, 120 and 160 epochsand we train for total 200 epoch
def adjust_learning_rate(epoch):
    
    if epoch < 60:
        lr_adj = 1.
    elif epoch < 120:
        lr_adj = 0.2
    elif epoch < 160:
        lr_adj = 0.04
    else:
        lr_adj = 0.008
    return lr_adj
def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        filepath = os.path.join(args.log_dir, filepath)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        # print("update 1")
        # print(val.detach().cpu().size())
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        # print("update 2")
        self.n += 1
        # print("update 3")

    @property
    def avg(self):
        return self.sum / self.n

        
class TimeEstimation(object):
    def __init__(self, name):
        self.name = name
        self.sum_ = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum_ += val
        self.n += 1

    @property
    def avg(self):
        return self.sum_ / self.n

    @property
    def sum(self):
        return self.sum_
class Merge_GE_FORWARD():
    def __init__(self, model, seq_layernames, named_parameters, group_size) :
        self._seq_layernames = seq_layernames
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []
        self._named_parameters = {k: v for k, v
                                in named_parameters}
        if self._seq_layernames is not None:
            self._sequential_keys = self._seq_layernames
        else:
            self._sequential_keys = [k for k, v in named_parameters]
        
        self._handles = {}

        self.model = model
        self.group_size = group_size
        # print("name_params \n")
        # print(self._named_parameters.keys())
        self._generate_merged_parameters()
        


    def _generate_groups_with_layer(self):
        # sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        # self._sizes = sizes

        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0

        # considered a layer as a group 
        for name, layer in self.model.named_children():
            for k, v in layer.named_parameters():
                key_groupidx_maps[name+'.'+k] = idx
                group.append(name+'.'+k)
            if len(group)!=0:
                groups.append(group)
                group = []
                idx += 1

        return groups, key_groupidx_maps

    def _generate_groups_with_number(self) :
        num = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for name, layer in self.model.named_children():
            for k, v in layer.named_parameters():
                num += 1
                key_groupidx_maps[name+'.'+k] = idx
                if num < self.group_size:
                    group.append(name+'.'+k)
                else:
                    idx += 1
                    group.append(name+'.'+k)
                    groups.append(group)
                    group = []
                    num = 0
            #     key_groupidx_maps[name+'.'+k] = idx
            #     group.append(name+'.'+k)
            # if len(group)!=0:
            #     groups.append(group)
            #     group = []
            #     idx += 1
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_merged_parameters(self):
        self._merged_parameters = {}
        self._merged_parameter_names = {}
        # groups, key_groupidx_maps = self._generate_groups_with_layer()
        groups, key_groupidx_maps = self._generate_groups_with_number()

        new_keys = []
        self._merged_parameter_offsets = {}
        self._layerwise_compressors = None
        self._layerwise_compressors = {}
        for g in groups:
            sub_size = 0
            offsets = []
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
            new_key = ':'.join(g)
            new_keys.append(new_key)
            t = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False)
            self._merged_parameters[new_key] = t
            self._merged_parameter_names[t] = new_key
            self._merged_parameter_offsets[new_key] = offsets
        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps

        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)

    # copy data to merge_parameter
    def _push_to_buffer(self, name, tensor):
        with torch.no_grad():
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                return name, new_tensor 
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()
            self._merged_parameters[new_key].data[offset:offset+numel].copy_(tensor.view(numel))
            self._groups_flags[group_idx][layer_idx] = 1
            # record if all para in groups is done or not.
            for idx in self._groups_flags[group_idx]:
                if idx == 0:
                    return name, None
            return new_key, self._merged_parameters[new_key]

    def _allgather_weight_async(self, params, name):
        index = hvd.rank()
        tensor = params.data.view(-1)
        allgather_name = name
        if len(name) > 100:
            allgather_name = name[0:50]+'...'+name[50:100]
        # https://horovod.readthedocs.io/en/stable/_modules/horovod/torch/mpi_ops.html#allreduce_async_
        s = tensor.size().numel()//hvd.size()
        q = tensor.size().numel()%hvd.size()


        if index < q:
            handle = allgather_async(tensor[ index*s + index : (index+1)*s + index + 1])
        elif index == q:
            handle = allgather_async(tensor[ index*s + index : (index+1)*s + index])
        else:
            handle = allgather_async(tensor[index*s + q : (index+1)*s + q])
        return handle, None

    def _weight_exchange(self):
        # global handles

        # push data into group tensor, if not all data ready , new_tensor==None
        # if not data ready do allgather
        for layer, module in self.model.named_children():
            for k, v in module.named_parameters():
            # name = self._parameter_names.get(p)
                new_name, new_tensor = self._push_to_buffer(layer+'.'+k, v.data)
                if new_tensor is not None:
                    # print("new_tensor is not None")
                    handle, ctx = self._allgather_weight_async(new_tensor, new_name)
                    self._handles[new_tensor] = (handle, 1)


    # # #   k:name, v:parameter
    #     for k,v in model.named_parameters():
    #         if v.data is not None:
    #             # tensor = v.data.view(-1)
    #             # allreduce_name = name
    #             # if len(name) > 100:
    #             #     allreduce_name = name[0:50]+'...'+name[50:100]
    #             # https://horovod.readthedocs.io/en/stable/_modules/horovod/torch/mpi_ops.html#allreduce_async_
    #             # allreduce by parameter name 
    #             # handle = allreduce_async_(tensor, average=True, name=allreduce_name)
    #             # return handle, None

    #             tensor = v.data.view(-1)
                
    #             s = tensor.size().numel()//hvd.size()
    #             q = tensor.size().numel()%hvd.size()

    #             if index < q:
    #                 handle = allgather_async(tensor[ index*s + index : (index+1)*s + index + 1])
    #             elif index == q:
    #                 handle = allgather_async(tensor[ index*s + index : (index+1)*s + index])
    #             else:
    #                 handle = allgather_async(tensor[index*s + q : (index+1)*s + q])

    #             handles[k] = (handle, 1)

    # get data from merge_parameter
    def _pull_from_buffer(self, name, merged_tensor):
        if len(self._groups) == len(self._sequential_keys):
            shape = self._named_parameters[name].data.shape
            return {name: merged_tensor.view(shape)} 
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx])
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensors[k] = merged_tensor.data[offset:offset+numel].view(original_tensor.shape)
        return tensors

    def _make_hook(self):
        def hook(*ignore):
        # global handles
        # # k : name, v: params
        # for k, v in module.named_parameters():
        #     if k in handles:
        #         handle = handles[k][0]
        #         output = synchronize(handle)

        #         del handles[k] 
        #         tensor = v.data.view(-1)
        #         tensor.data = output.data.clone()

            for p, value in self._handles.items():
                # name = self._merged_parameter_names.get(p)
                handle, ctx  = value
                # print("synchronize\n")
                output = synchronize(handle)
                # print(output.size())
                p.set_(output.data)
                # if self._profiling:
                #     utils.force_insert_item(self._update_times, name, time.time()-stime)
            if len(self._groups) != len(self._sequential_keys):
                for merged_p, v in self._handles.items():
                    new_name = self._merged_parameter_names.get(merged_p)
                    tensors = self._pull_from_buffer(new_name, merged_p)
                    # print(tensors.keys())
                    for n in tensors:
                        p = self._named_parameters.get(n)
                        # if settings.FP16:
                        #     p.grad.set_(tensors[n].data.type(p.grad.type()))
                        # else:
                            # p.grad.set_(tensors[n].data)
                        # with torch.no_grad():
                        p.data.copy_(tensors[n])
                    # torch.cuda.empty_cache()
            # print("len(self._handles) : {}".format(len(self._handles)))
            self._handles.clear()
            # print("len(self._handles) : {}".format(len(self._handles)))
        return hook
    def _register_pre_forward_hooks(self):
        # for i, layer in enumerate(model.modules()):
        #     # print(num)
        #     # TODO: check transfomer is ok?
        #     layer.register_forward_pre_hook(self._make_hook())
        for k, v in self.model.named_modules():
            v.register_forward_pre_hook(self._make_hook())


    # def _index(self):
    #     tensor = torch.linspace(0, hvd.size()*2-2, hvd.size()*2-1).cuda()
    #     handle = reducescatter_async(tensor)
    #     output = synchronize(handle)
    #     index = int(output.detach().cpu().tolist()[0]//2)
    #     return index

if __name__ == '__main__':
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)


    print ("Run with arguments:")
    for key, value in args._get_kwargs():
        if value is not None:
            print(value,key) if hvd.rank() == 0 else None
    
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    num_classes = 100 if args.dataset == 'CIFAR100' else 10 ## Only support CIFAR
        
    # From the WRN paper, "In general, we observed that CIFAR mean/std preprocessing allows training wider anddeeper networks with better accuracy"
    # Preprocessing with mean/std. Value pick up from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)    
        
    if args.dataset == 'CIFAR100':    
        train_dataset = \
            datasets.CIFAR100(args.dataroot,
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
                                 
        val_dataset = \
            datasets.CIFAR100(args.dataroot,
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
    else:
        train_dataset = \
            datasets.CIFAR10(args.dataroot,
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
                                 
        val_dataset = \
            datasets.CIFAR10(args.dataroot,
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    ## Import MPI here because it should call after the multiprocessing.set_start_method() call
    ## due to using of 'fork server': https://github.com/chainer/chainermn/issues/204
    from mpi4py import MPI
        
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


    # Set up Wide_RESNET-28-10 model.
    # model = LeNet()
    if args.model=='resnet50':
        model = torchvision.models.resnet50()
    elif args.model=='resnet101':
        model = torchvision.models.resnet101()
    if args.model=='vgg16':
        model = torchvision.models.vgg16()
    if args.model=='efficientnet':
        model = torchvision.models.efficientnet_v2_l()
    # if hvd.rank()==0:
    #     print(summary(model.cuda(), input_size=(3, 224, 224)))


    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)
    
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # index = _index()
    # Horovod: wrap optimizer with DistributedOptimizer.
    # print("optimizer = dis_hvd.DistributedOptimizer")
    seq_layernames = []
    for name, p in model.named_parameters():
        seq_layernames.append(name)
    optimizer = dis_hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        seq_layernames=seq_layernames,
        group_size = args.group_size,
        op = "allreduce"
        # index = index,
        # op=hvd.Adasum if args.use_adasum else hvd.Average,
        # gradient_predivide_factor=args.gradient_predivide_factor
        )
    scheduler = LambdaLR(optimizer, lr_lambda=adjust_learning_rate)
    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # print("hvd.broadcast_parameters(model.state_dict(), root_rank=0)")
    # Horovod: broadcast parameters & optimizer state.
    dis_hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # mg_we_forward = Merge_GE_FORWARD(model = model, seq_layernames = seq_layernames, named_parameters = model.named_parameters(), group_size = args.group_size)
    # mg_we_forward._register_pre_forward_hooks()

    
    # why
    # MPI.COMM_WORLD.Barrier()
    for epoch in range(resume_from_epoch, args.epochs):
        # print('---------------------train start---------------------------')
        train(epoch,args.log_dir)
        # print('---------------------train end---------------------------')
        validate(epoch, args.log_dir)
        scheduler.step()
        #save_checkpoint(epoch)
        # if epoch % 10 == 0:
           # save_checkpoint(epoch)