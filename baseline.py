import time
from easydict import EasyDict
import yaml
from utils import *
import torch.nn as nn
import torchvision
import argparse
from models.resnet_ import *
import os

par_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(par_dir)

parser = argparse.ArgumentParser(description='Baseline')
# parser.add_argument('--work_path', type=str, default='./experiments/resnet18')
parser.add_argument('--net_arch', type=str, default='ResNet50', help='network architecture',
                    choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'])
parser.add_argument('--sample_ratio', type=float, default=0.96657)
# parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

CONTINUE = False
global last_epoch, best_prec, config


# 训练
def train(train_loader, net, criterion, optimizer, epoch, device):
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print("===  Epoch: [{}/{}]  === ".format(epoch + 1, config.epochs))
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_index + 1) % 100 == 0:
            print("===  step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}  ===".format(
                batch_index + 1, len(train_loader), train_loss / (batch_index + 1), 100.0 * correct / total,
                get_current_lr(optimizer)))
    print("===  step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}  ===".format(
        batch_index + 1, len(train_loader), train_loss / (batch_index + 1), 100.0 * correct / total,
        get_current_lr(optimizer)))

    end = time.time()
    print("===  cost time: {:.4f}s  ===".format(end - start))


# 测试
def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    print("===  Validate [{}/{}] ===".format(epoch + 1, config.epochs))
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("===  test loss: {:.3f} | test acc: {:6.3f}%  ===".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))

    # 保存检查点
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, config.ckpt_name)
    if is_best:
        best_prec = acc


if __name__ == "__main__":
    print("==============model：" + args.net_arch + ' sample_ratio:' + str(args.sample_ratio))
    experiment_model = args.net_arch

    # 打开配置文件
    with open("_config/" + args.net_arch + ".yaml") as f:
        config = yaml.load(f)
    config = EasyDict(config)

    # 选择ResNet18
    if experiment_model == 'ResNet18':
        net = resnet18(pretrained=config.pretrained)
        print(config.pretrained)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(channel_in, config.num_classes))

    # 选择ResNet34
    elif experiment_model == 'ResNet34':
        net = resnet34(pretrained=config.pretrained)
        print(config.pretrained)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(channel_in, config.num_classes))

    # 选择ResNet50
    elif experiment_model == 'ResNet50':
        net = resnet50(pretrained=config.pretrained)
        print(config.pretrained)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(channel_in, config.num_classes))

    # 选择ResNet101
    elif experiment_model == 'ResNet101':
        net = resnet101(pretrained=config.pretrained)
        print(config.pretrained)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(channel_in, config.num_classes))

    # 训练设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), config.lr_scheduler.base_lr)
    # 继续上一次训练
    if CONTINUE:
        ckpt_file_name = experiment_model + '/' + config.ckpt_name + '.pth.tar'
        best_prec, last_epoch, optimizer = load_checkpoint(ckpt_file_name, net, optimizer=optimizer)
    else:
        last_epoch = -1
        best_prec = 0

    # 加载训练数据
    transform1 = transforms.Compose([transforms.Resize((config.input_size, config.input_size))])
    transform2 = transforms.Compose([transforms.ToTensor()])
    train_loader, test_loader = get_data_loader(transform1, transform2, config, args.sample_ratio, args.net_arch)

    # 训练
    print(("=======  Training  ======="))
    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    print(("=======  Training Finished.Best_test_acc: {:.3f}% ========".format(best_prec)))

    # 预测
    net_dict = net.state_dict()
    state_dict = torch.load("Net_best.pth.tar")['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net_dict.update(new_state_dict)
    net.load_state_dict(net_dict)
    net.to(device)
    net.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for batch_index, (inputs_batch, targets_batch) in enumerate(test_loader):
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)

            outputs = net(inputs_batch)

            preds.extend(outputs.max(1)[1].cpu().numpy())
            gts.extend(targets_batch.cpu().numpy())

    # 计算F1和Balanced accuracy
    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    bal_acc = (tpr + tnr) / 2
    print("####################")
    print(experiment_model + " sample_ratio " + str(args.sample_ratio) + " : f1={:.3f},bal_acc={:.3f}".format(f1, bal_acc))
