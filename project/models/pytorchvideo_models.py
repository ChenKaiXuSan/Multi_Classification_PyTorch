# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torchmetrics import classification

from models.make_model import MakeVideoModule, early_fusion, late_fusion, single_frame

from pytorch_lightning import LightningModule

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt 
import seaborn as sns

# %%

class WalkVideoClassificationLightningModule(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model
        self.img_size = hparams.img_size

        self.lr=hparams.lr
        self.num_class = hparams.model_class_num
        self.uniform_temporal_subsample_num = hparams.uniform_temporal_subsample_num

        self.fusion_method = hparams.fusion_method

        if self.fusion_method == 'slow_fusion':
            self.model = MakeVideoModule(hparams)

            # select the network structure 
            if self.model_type == 'resnet':
                self.model=self.model.make_walk_resnet()

            elif self.model_type == 'r2plus1d':
                self.model = self.model.make_walk_r2plus1d()

            elif self.model_type == 'csn':
                self.model=self.model.make_walk_csn()

            elif self.model_type == 'x3d':
                self.model = self.model.make_walk_x3d()

            elif self.model_type == 'slowfast':
                self.model = self.model.make_walk_slow_fast()

            elif self.model_type == 'i3d':
                self.model = self.model.make_walk_i3d()

            elif self.model_type == 'c2d':
                self.model = self.model.make_walk_c2d()


        elif self.fusion_method == 'single_frame':
            self.model = single_frame(hparams)
        elif self.fusion_method == 'early_fusion':
            self.model = early_fusion(hparams)
        elif self.fusion_method == 'late_fusion':
            self.model = late_fusion(hparams)
        else:
            raise ValueError('no choiced model selected, get {self.fusion_method}')

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # set the metrics
        self._accuracy = classification.MulticlassAccuracy(num_classes=self.num_class,)
        self._precision = classification.MulticlassPrecision(num_classes=self.num_class, )
        self._f1_score = classification.MulticlassF1Score(num_classes=self.num_class, )
        self._auroc = classification.MulticlassAUROC(num_classes=self.num_class,)

        self._confusion_matrix = classification.MulticlassConfusionMatrix(num_classes=self.num_class)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        '''
        
        # input and label
        video = batch['video'].detach()

        if self.fusion_method == 'single_frame': 
            # for single frame
            label = batch['label'].detach()

            # when batch > 1, for multi label, to repeat label in (bxt)
            label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        else:
            label = batch['label'].detach() # b, class_num

        # model predicts
        preds = self.model(video)

        # when torch.size([1]), not squeeze.
        preds_softmax = torch.softmax(preds, dim=1)

        loss = F.cross_entropy(preds_softmax, label)

        accuracy = self._accuracy(preds_softmax, label)
        precision = self._precision(preds_softmax, label)
        f1_score = self._f1_score(preds_softmax, label)
        auroc = self._auroc(preds_softmax, label)
        confusion_matrix = self._confusion_matrix(preds_softmax, label)

        self.log_dict({'train_loss': loss, 'train_acc': accuracy, 'train_f1_score': f1_score})

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     '''
    #     after validattion_step end.

    #     Args:
    #         outputs (list): a list of the train_step return value.
    #     '''
        
    #     # log epoch metric
    #     # self.log('train_acc_epoch', self.accuracy)
    #     pass

    def validation_step(self, batch, batch_idx):
        '''
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss 
            accuract: selected accuracy result.
        '''

        # input and label
        video = batch['video'].detach() # b, c, t, h, w

        if self.fusion_method == 'single_frame': 
            label = batch['label'].detach()

            # when batch > 1, for multi label, to repeat label in (bxt)
            label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        else:
            label = batch['label'].detach() # b, class_num

        # pred the video frames
        with torch.no_grad():
            preds = self.model(video)

        preds_softmax = torch.softmax(preds, dim=1)

        val_loss = F.cross_entropy(preds_softmax, label)

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_softmax, label)
        precision = self._precision(preds_softmax, label)
        f1_score = self._f1_score(preds_softmax, label)
        auroc = self._auroc(preds_softmax, label)

        confusion_matrix = self._confusion_matrix(preds_softmax, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val_loss': val_loss, 'val_acc': accuracy, 'val_f1_score': f1_score}, on_step=False, on_epoch=True)
        
        return accuracy

    def validation_epoch_end(self, outputs):
        pass
        
        # val_metric = torch.stack(outputs, dim=0)

        # final_acc = (torch.sum(val_metric) / len(val_metric)).item()

        # print('Epoch: %s, avgAcc: %s' % (self.current_epoch, final_acc))

        # self.ACC[self.current_epoch] = final_acc

    def on_validation_end(self) -> None:
        pass
            
    def test_step(self, batch, batch_idx):
        pass
        
    def test_epoch_end(self, outputs):
       pass

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type