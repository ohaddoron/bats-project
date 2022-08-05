import io
from copy import deepcopy
from enum import Enum
import seaborn as sn
import fairseq.checkpoint_utils
import pandas as pd
import pytorch_lightning as pl
import fairseq
import torch.nn
import torchmetrics
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import PrecisionRecallCurve
from typer import Typer
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional import precision_recall
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config

from bats.data import BatsDataset, DataCollatorForWav2Vec2Pretraining, CLASSES
from torchmetrics.classification import BinaryPrecisionRecallCurve

app = Typer()


class BatsModelAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            ['/mnt/drive1/home/ohaddoron1/Projects/bats/models/wav2vec_small.pt.1'])
        self.model = model[0]
        cfg = OmegaConf.load('/mnt/drive1/home/ohaddoron1/Projects/bats/bats/configs/wav2vec2-pretraining.yaml')
        feature_extractor = Wav2Vec2FeatureExtractor()
        w2v_config = Wav2Vec2Config()
        dataset = BatsDataset(cfg)
        collate_fn = DataCollatorForWav2Vec2Pretraining(self.model, feature_extractor, padding='longest',
                                                        config=w2v_config)
        self.train_dataset = deepcopy(dataset)
        self.test_dataset = deepcopy(dataset)

        self.train_dataset.data, self.test_dataset.data = train_test_split(dataset.data,
                                                                           stratify=[item['species'] for item in
                                                                                     dataset.data],
                                                                           random_state=42,

                                                                           )

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True,
                                       drop_last=True)
        self.val_loader = DataLoader(self.test_dataset, batch_size=32, collate_fn=collate_fn, drop_last=True)

        self.forward = self.model.forward

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, purpose='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, purpose='val')

    def step(self, batch, batch_idx, purpose=None):
        x, mask, _ = batch

        out = self.forward(x, padding_mask=mask, mask=False)
        loss = out['features_pen'] + out['prob_perplexity']
        self.log(name=f'{purpose}/prob_perplexity', value=out['prob_perplexity'])
        self.log(name=f'{purpose}/code_perplexity', value=out['code_perplexity'])
        self.log(name=f'{purpose}/features_pen', value=out['features_pen'])
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        return Adam(lr=1e-4, params=self.model.parameters())


class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


class BatsModelClassifier(pl.LightningModule):
    def __init__(self):
        super(BatsModelClassifier, self).__init__()
        self.bats_model_ae: BatsModelAE = BatsModelAE.load_from_checkpoint(
            '/mnt/drive1/home/ohaddoron1/Projects/bats/bats/lightning_logs/ae/checkpoints/epoch=2-step=159.ckpt',
            map_location='cpu')
        self.bats_model_ae.freeze()
        self.classifier_model = nn.Sequential(
            nn.Linear(23040, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, len(self.bats_model_ae.train_dataset.species)),
            nn.BatchNorm1d(len(self.bats_model_ae.train_dataset.species)),
            nn.Softmax()
        )
        self.train_acc = torchmetrics.Accuracy(num_classes=len(self.bats_model_ae.train_dataset.species),
                                               multiclass=True)
        self.val_acc = torchmetrics.Accuracy(num_classes=len(self.bats_model_ae.train_dataset.species), multiclass=True)

    def train_dataloader(self):
        return self.bats_model_ae.train_loader

    def val_dataloader(self):
        return self.bats_model_ae.val_loader

    def forward(self, x, padding_mask):
        features = nn.functional.interpolate(
            self.bats_model_ae.forward(x, padding_mask=padding_mask, mask=False, features_only=True)[
                'features'].unsqueeze(1), size=(45, 512)).reshape(32, -1).detach()
        predictions = self.classifier_model(features)

        return predictions

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, purpose='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, purpose='val')

    def step(self, batch, batch_idx, purpose=None):
        x, padding_mask, target = batch

        out = self.forward(x, padding_mask=padding_mask)
        loss = torch.nn.CrossEntropyLoss()(out, target)
        precision, recall = precision_recall(out, target, num_classes=len(self.bats_model_ae.train_dataset.species),
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
        confusion = torchmetrics.ConfusionMatrix(num_classes=len(self.bats_model_ae.train_dataset.species)).to(
            outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=self.bats_model_ae.train_dataset.species,
            columns=self.bats_model_ae.train_dataset.species,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            self.bats_model_ae.train_dataset.species,
            self.bats_model_ae.train_dataset.species,
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
            plt.plot(recall[i].detach().cpu(), precision[i].detach().cpu(), lw=2, label='class {}'.format(CLASSES[i]))
        plt.xlabel("recall")
        plt.ylabel("precision")
        ax.legend(loc='best')
        plt.title("precision vs. recall curve")
        tb.add_figure(tag='train/precision_recall', figure=plt.gcf(), global_step=self.current_epoch)

        plt.close()

    def validation_epoch_end(self, outs) -> None:
        tb = self.logger.experiment
        outputs = torch.cat([tmp['outputs'] for tmp in outs])
        labels = torch.cat([tmp['labels'] for tmp in outs])
        confusion = torchmetrics.ConfusionMatrix(num_classes=len(self.bats_model_ae.train_dataset.species)).to(
            outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=self.bats_model_ae.train_dataset.species,
            columns=self.bats_model_ae.train_dataset.species,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            self.bats_model_ae.train_dataset.species,
            self.bats_model_ae.train_dataset.species,
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
        ax.legend(loc='best')
        plt.title("precision vs. recall curve")
        tb.add_figure(tag='val/precision_recall', figure=plt.gcf(), global_step=self.current_epoch)

        plt.close()

    def configure_optimizers(self):
        return Adam(lr=1e-4, params=self.classifier_model.parameters())


@app.command()
def train_ae():
    trainer = pl.Trainer(max_epochs=100, gpus=[4])
    model = BatsModelAE()
    trainer.fit(model)


@app.command()
def train_classifier():
    trainer = pl.Trainer(max_epochs=100, gpus=[4],
                         # callbacks=EarlyStopping(monitor='val/accuracy')
                         )
    model = BatsModelClassifier()
    trainer.fit(model)


if __name__ == '__main__':
    app()
