import io
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional import precision_recall
import typer
from bats.data import CLASSES, SpectrogramDataset
from bats.train import IntHandler


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], nc=1):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet18EncClassifier(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], nc=1):
        super().__init__()
        self._resnet_18_enc = ResNet18Enc(num_Blocks=num_Blocks, nc=nc)
        self.classification_head = nn.Sequential(
            nn.Linear(27648, 1000),
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(100),
            nn.GELU(),
            nn.Linear(100, len(CLASSES)),
            nn.BatchNorm1d(len(CLASSES)),
            nn.Softmax()
        )

    def forward(self, x):
        enc = self._resnet_18_enc(x)
        flat = enc.view(enc.shape[0], -1)
        return self.classification_head(flat)


def collate_fn(batch):
    return {'spect': [item['spect'][np.newaxis, np.newaxis, ...] for item in batch],
            'target': torch.stack([torch.tensor(item['target']) for item in batch])}


class BatsModelVanillaClassifier(pl.LightningModule):
    def __init__(self):
        super(BatsModelVanillaClassifier, self).__init__()

        self.net = ResNet18EncClassifier()

        self.forward = self.net.forward

        self.train_acc = torchmetrics.Accuracy(num_classes=len(CLASSES),
                                               multiclass=True)
        self.val_acc = torchmetrics.Accuracy(num_classes=len(CLASSES), multiclass=True)
        cfg = OmegaConf.load('/mnt/drive1/home/ohaddoron1/Projects/bats/bats/configs/wav2vec2-pretraining.yaml')

        dataset = SpectrogramDataset(cfg=cfg)
        self.train_dataset = deepcopy(dataset)
        self.test_dataset = deepcopy(dataset)
        self.train_dataset.data, self.test_dataset.data = train_test_split(dataset.data,
                                                                           stratify=[item['species'] for item in
                                                                                     dataset.data],
                                                                           random_state=42,

                                                                           )

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True,
                                       drop_last=True)
        self.val_loader = DataLoader(self.test_dataset, batch_size=32, drop_last=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, purpose='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, purpose='val')

    def step(self, batch, batch_idx, purpose=None):

        x = batch['spect']
        target = batch['target']

        out = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(out, target)
        precision, recall = precision_recall(out, target, num_classes=len(CLASSES),
                                             threshold=0.1)
        self.log(name=f'{purpose}/loss', value=loss)
        self.log(name=f'{purpose}/precision', value=precision)
        self.log(name=f'{purpose}/recall', value=recall)
        if purpose == 'train':
            self.train_acc.update(out, target)
            self.log(name=f'train/accuracy', value=self.train_acc, on_epoch=True)
        elif purpose == 'val':
            self.val_acc.update(out, target)
            self.log(name=f'val/accuracy', value=self.val_acc)

        return dict(loss=loss, outputs=out, labels=target)

    def training_epoch_end(self, outs) -> None:
        tb = self.logger.experiment
        outputs = torch.cat([tmp['outputs'] for tmp in outs])
        labels = torch.cat([tmp['labels'] for tmp in outs])
        confusion = torchmetrics.ConfusionMatrix(num_classes=len(CLASSES)).to(
            outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=CLASSES,
            columns=CLASSES,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            CLASSES,
            CLASSES,
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("train/confusion_matrix", im, global_step=self.current_epoch)
        plt.close()

        # Precision recall curve
        pr_curve = BinaryPrecisionRecallCurve(thresholds=None)
        precision = dict()
        recall = dict()
        for i in range(outputs.shape[1]):
            precision[i], recall[i], _ = pr_curve(outputs[:, i],
                                                  torch.tensor(labels == i).long())
            plt.plot(recall[i].detach().cpu(), precision[i].detach().cpu(), lw=2, label='{}'.format(CLASSES[i]))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc='best')
        plt.title("precision vs. recall curve")
        tb.add_figure(tag='train/precision_recall', figure=plt.gcf(), global_step=self.current_epoch)

        plt.close()

    def validation_epoch_end(self, outs) -> None:
        tb = self.logger.experiment
        outputs = torch.cat([tmp['outputs'] for tmp in outs])
        labels = torch.cat([tmp['labels'] for tmp in outs])
        confusion = torchmetrics.ConfusionMatrix(num_classes=len(CLASSES)).to(
            outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=CLASSES,
            columns=CLASSES,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            CLASSES,
            CLASSES,
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val/confusion_matrix", im, global_step=self.current_epoch)
        plt.close()

        # Precision recall curve
        pr_curve = BinaryPrecisionRecallCurve(thresholds=None)
        precision = dict()
        recall = dict()
        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(outputs.shape[1]):
            precision[i], recall[i], _ = pr_curve(outputs[:, i],
                                                  torch.tensor(labels == i).long())
            ax.plot(recall[i].detach().cpu(), precision[i].detach().cpu(), lw=2, label='{}'.format(CLASSES[i]))
        plt.xlabel("recall")
        plt.ylabel("precision")
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #           ncol=3, fancybox=True, shadow=True)

        plt.title("precision vs. recall curve")
        tb.add_figure(tag='val/precision_recall', figure=plt.gcf(), global_step=self.current_epoch)

        plt.close()

    def configure_optimizers(self):
        return Adam(lr=1e-4, params=self.parameters())


@typer.run
def train_classifier():
    trainer = pl.Trainer(max_epochs=100, gpus=[3],
                         # callbacks=EarlyStopping(monitor='val/accuracy')
                         )
    model = BatsModelVanillaClassifier()
    trainer.fit(model)
