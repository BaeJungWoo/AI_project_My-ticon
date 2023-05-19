import pandas as pd
import torch
import re
import emoji

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from soynlp.normalizer import repeat_normalize


from pytorch_lightning.utilities.types import EVAL_DATALOADERS

class Model(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.training_outputs = []
        self.validation_outputs = []
        self.training_loss = []
        self.validation_loss = []

        self.clsfier = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=4)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, **kwargs):
        return self.clsfier(**kwargs)
    
    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def training_step(self, batch, batch_idx):
        
        ret = self.step(batch, batch_idx)
        self.training_step_outputs.append(ret)
        return ret
    
    def validation_step(self, batch, batch_idx):
        
        ret = self.step(batch, batch_idx)
        self.validation_step_outputs.append(ret)
        return ret

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        if state=='val': print(outputs)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        if state =='train':
          self.training_outputs.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
          self.training_loss.append(loss)
        elif state == 'val':
          self.validation_outputs.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
          self.validation_loss.append(loss)

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')
        return {'loss': loss}     

    def on_train_epoch_end(self): # fix
        self.epoch_end(self.training_step_outputs, state='train')
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self): # fix
        self.epoch_end(self.validation_step_outputs, state='val')   
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    
    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')
        
    def clean(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def encode(self, x, **kwargs):
        return self.tokenizer.encode(
            self.clean(str(x)),
            padding='max_length',
            max_length = self.hparams.max_length,
            truncation=True,
            **kwargs,
        )
    
    def preprocess_dataframe(self, df):
        df['sentence'] = df['sentence'].map(self.encode)
        return df
    
    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['sentence'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long)
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 1 if not self.hparams.tpu_cores else self.hparams.tpu_cores,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers
        )
    
    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)
    
    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=True)

    def test_dataloader(self):
        return self.dataloader(self.hparams.test_data_path, shuffle=True)