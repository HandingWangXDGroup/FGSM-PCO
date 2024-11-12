import argparse
import copy
import logging
import os
import time
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
import matplotlib.pyplot as plt
#from Cifar100_models import *
import random
from models import *
from utils import *
logger = logging.getLogger(__name__)
device_ids = [0]


def get_args():
    parser = argparse.ArgumentParser(description='Adversarial Training on CIFAR-10')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--data-dir', default='../data', type=str, help='Directory for dataset')
    parser.add_argument('--epochs', default=110, type=int, help='Number of epochs to train')
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'], help='Learning rate schedule')
    parser.add_argument('--lr-min', default=0., type=float, help='Minimum learning rate')
    parser.add_argument('--lr-max', default=0.1, type=float, help='Maximum learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer')
    parser.add_argument('--model', default='ResNet18', type=str, help='Model name')
    parser.add_argument('--epsilon', default=8, type=int, help='Perturbation budget')
    parser.add_argument('--alpha', default=8, type=float, help='Step size for adversarial attack')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--factor', default=0.6, type=float, help='Label smoothing factor')
    return parser.parse_args()

def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.to(device_ids[0]).data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


def atta_aug(input_tensor, rst):
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size

    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 8)
        y_t = random.randint(0, 8)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, f'epochs_{args.epochs}', f'lr_max_{args.lr_max}',
                               f'model_{args.model}', f'lr_schedule_{args.lr_schedule}',
                               f'alpha_{args.alpha}', f'epsilon_{args.epsilon}', f'factor_{args.factor}')

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, 'output_cifar10_eps.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile
    )
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load dataset
    train_loader, test_loader = get_all_loaders(args.data_dir, args.batch_size)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    # Build model
    logger.info('==> Building model..')
    model = {
        "VGG": VGG('VGG19'),
        "ResNet18": ResNet18(),
        "PreActResNet18": PreActResNet18(),
        "WideResNet": WideResNet()
    }.get(args.model, ResNet18())

    model = model.to(device_ids[0])
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    num_of_example = 50000
    batch_size = args.batch_size

    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 100/110, lr_steps * 105 / 110],
                                                         gamma=0.1)


        
    # Training process
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    
    for i, (X, y) in enumerate(train_loader):
        cifar_x, cifar_y = X.to(device_ids[0]), y.to(device_ids[0])

    for epoch in range(args.epochs):
        Confidence = 0
        delta_norm = 0.0

        batch_size = args.batch_size
        cur_order = np.random.permutation(num_of_example)
        #print(cur_order)
        iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
        batch_idx = -batch_size
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        print("epoch:,",epoch)
        print(iter_num)
        if epoch == 0:
            temp=torch.rand(50000,3,32,32)
            if args.delta_init != 'previous':
                all_delta = torch.zeros_like(temp).to(device_ids[0])
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    all_delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                all_delta.data = clamp(alpha * torch.sign(all_delta), -epsilon, epsilon)
        idx = torch.randperm(cifar_x.shape[0])
        cifar_x =cifar_x[idx, :,:,:].view(cifar_x.size())
        cifar_y = cifar_y[idx].view(cifar_y.size())
        all_delta=all_delta[idx, :, :, :].view(all_delta.size())
        for i in range(iter_num):
            batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
            X=cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            y= cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            delta =all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            X=X.to(device_ids[0])
            y=y.to(device_ids[0])
            batch_size = X.shape[0]
            rst = torch.zeros(batch_size, 3, 32, 32).to(device_ids[0])
            X, transform_info = atta_aug(X, rst)
            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).to(device_ids[0])).float()
            delta.requires_grad = True
            ori_output = model(X + delta[:X.size(0)])
            ori_loss = LabelSmoothLoss(ori_output, label_smoothing.float())
            onehot = torch.eye(10)[y].to(device_ids[0])
            confidence = (onehot*torch.softmax(ori_output,dim=1))
            confidence = torch.max(confidence, dim=1, keepdim=True).values
            confidence = confidence
            confidence = (1-confidence).unsqueeze(dim=2).unsqueeze(dim=3)
            confidence = torch.clamp(confidence,0,1)
            Confidence+=confidence.sum().detach().cpu()
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()
            delta_FGSM = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta_FGSM = clamp(delta_FGSM, lower_limit - X, upper_limit - X)
            X_FGSM = X+1.5*delta_FGSM
            X_new = confidence*(X + delta[:X.size(0)])+(1-confidence)*X_FGSM
            Y_new = Variable(torch.tensor(_label_smoothing(y, 0.6)).to(device_ids[0])).float()
            next_delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            next_delta.data[:X.size(0)] = clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X_new)
            fgsm_output = model(X + delta[:X.size(0)])
            loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            loss = LabelSmoothLoss(output, Y_new)+10*(torch.abs(loss_fn(fgsm_output.float(), ori_output.float())-loss_fn(fgsm_output.float(), output.float())))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
            all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]=next_delta


        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)

        
        logger.info('==> Building model..')
        
        model_test = {
        "VGG": VGG('VGG19'),
        "ResNet18": ResNet18(),
        "PreActResNet18": PreActResNet18(),
        "WideResNet": WideResNet()
    }.get(args.model, ResNet18())
        model_test = model_test.to(device_ids[0])
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1,epsilon)

        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        with open("./Cifar10_PCO_Res.txt","a+") as f:
            f.write(str(pgd_acc)+'\n')
            f.write(str(test_acc)+'\n')
        print('PGD',pgd_acc)
        print('clean',test_acc)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model.state_dict(), os.path.join(output_path, 'Cifar10_PCO_Res.pth'))

    torch.save(model.state_dict(), os.path.join(output_path, 'Cifar10_PCO_Res.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    print(epoch_clean_list)
    print(epoch_pgd_list)


if __name__ == "__main__":
    main()
