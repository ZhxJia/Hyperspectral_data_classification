import numpy as np
from utils.hsidataset import HsiDataset
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import HsiNet
from absl import logging, flags, app
import os.path
from tqdm import tqdm

flags.DEFINE_boolean('enable_pretrained', True, 'Enable pretrained model')
flags.DEFINE_string('ckpt_dir', './ckpt/best_model.pt', 'pretrained model dir')
flags.DEFINE_integer('epoch', 300, 'number of epoch to train')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_integer('batch', 16, 'batch size')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.xavier_normal_(m.bias.data)


def check_val(model, loss_fn, loader, device):
    val_correct = 0
    val_samples = 0
    loss = 0

    model.eval()
    for t, (x, y) in enumerate(loader):
        x_val = x.to(device)
        y_val = y.to(device)

        scores = model(x_val)
        loss += loss_fn(scores, y_val.long()).item()
        t = t + 1
        _, preds = scores.data.cpu().max(dim=1)
        val_correct += (preds == y).sum()
        val_samples += preds.size(0)
    val_acc = float(val_correct) / val_samples
    val_loss = loss / t
    print('val_loss:%.4f, Got %d / %d correct (%.4f%%)' % (val_loss, val_correct, val_samples, 100 * val_acc))
    print("-------------------------------------")
    return val_acc, val_loss


def train(model, loss_fn, optimizer, lr_schedule, train_loader, test_loader, writer, device,
          num_epochs=1):
    nb = len(train_loader)
    best_loss = 1
    best_test_acc = 0
    best_test_loss = 10
    n_iter = 0
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=nb)
        num_correct = 0
        num_samples = 0

        for i, (patch, label) in pbar:
            n_iter = n_iter + 1
            patch = patch.to(device)
            label = label.to(device)

            scores = model(patch)
            loss = loss_fn(scores, label.long())

            _, preds = scores.data.max(dim=1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_samples = 0
            num_correct = 0
            writer.add_scalar('Loss/train', loss.item(), n_iter)
            writer.add_scalar('Accuracy/train', acc, n_iter)
            # ------------------end batch-------------------------------------------

        print(('\n' + '%10s' * 4) % ('Epoch', 'loss', 'acc', 'lr'))
        s = ('%10s' + '%10.3g' * 3) % (
            f'{epoch}/{num_epochs}', loss, acc, optimizer.state_dict()['param_groups'][0]['lr'])
        print(s)

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_loss': loss.item(),
                    'optimizer': optimizer.state_dict()},
                   f'./ckpt/model.pt')

        test_acc, test_loss = check_val(model, loss_fn, test_loader, device)
        # update scheduler
        lr_schedule.step(test_acc)
        writer.add_scalars('Loss', {'test': test_loss, 'train': loss.item()}, n_iter)
        writer.add_scalars('Accuracy', {'test': test_acc, 'train': acc}, n_iter)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            print(f"best test accuracy:{best_test_acc}")
            print("--------------------")
            print(f"save model")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_loss': test_loss,
                        'optimizer': optimizer.state_dict()},
                       f'./ckpt/best_model.pt')
            print("--------------------")


FLAGS = flags.FLAGS


def main(unused_argv):
    start_epoch = 0
    writer = SummaryWriter()
    hsi_traindatas = HsiDataset("./data", type='train', oversampling=True)
    hsi_testdatas = HsiDataset("./data", type='test', oversampling=True)
    train_dataloader = DataLoader(hsi_traindatas, batch_size=FLAGS.batch, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(hsi_testdatas, batch_size=FLAGS.batch, shuffle=True, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    hsinet = HsiNet(num_class=16).to(device)
    optimizer = optim.Adam(params=hsinet.parameters(), lr=FLAGS.lr, weight_decay=1e-10)
    if os.path.exists(FLAGS.ckpt_dir):
        model_dict = torch.load(FLAGS.ckpt_dir, map_location=device)
        if FLAGS.enable_pretrained:
            print(f"Loading pretrained model ...")
            hsinet.load_state_dict(model_dict['model_state_dict'])
            if model_dict['optimizer'] is not None:
                optimizer.load_state_dict(model_dict['optimizer'])

    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, verbose=True, patience=10,
                                                       min_lr=0.000001)
    loss_fn = nn.CrossEntropyLoss()

    train(hsinet, loss_fn, optimizer, lr_schedule, train_dataloader, test_dataloader, writer, device,
          num_epochs=FLAGS.epoch)


if __name__ == "__main__":
    app.run(main)
