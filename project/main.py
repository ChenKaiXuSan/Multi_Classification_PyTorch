'''
this project were based the pytorch, pytorch lightning and pytorch video library, 
for rapid development.
'''

# %%
import os, time, subprocess, warnings, sys, logging
warnings.filterwarnings("ignore")

# logger = logging.getLogger("lightning.pytorch")
# logger.setLevel(logging.WARNING)

import torch 

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, RichProgressBar, ModelCheckpoint, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor

from dataloader.data_loader import WalkDataModule
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from argparse import ArgumentParser

import pytorch_lightning
# %%

def get_parameters():
    '''
    The parameters for the model training, can be called out via the --h menu
    '''
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'csn', 'r2plus1d', 'x3d', 'slowfast', 'c2d', 'i3d'])
    parser.add_argument('--img_size', type=int, default=224)
    # parser.add_argument('--version', type=str, default='test', help='the version of logger, such data')
    parser.add_argument('--model_class_num', type=int, default=4, help='the class num of model')
    # parser.add_argument('--model_depth', type=int, default=50, choices=[50, 101, 152], help='the depth of used model')

    # Training setting
    parser.add_argument('--max_epochs', type=int, default=50, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader for load video')
    parser.add_argument('--clip_duration', type=float, default=1, help='clip duration for the video')
    parser.add_argument('--uniform_temporal_subsample_num', type=int,
                        default=8, help='num frame from the clip duration')
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')

    # ablation experment 
    # different fusion method 
    parser.add_argument('--fusion_method', type=str, default='slow_fusion', choices=['single_frame', 'early_fusion', 'late_fusion', 'slow_fusion'], help="select the different fusion method from ['single_frame', 'early_fusion', 'late_fusion']")
    # pre process flag
    # parser.add_argument('--pre_process_flag', action='store_true', help='if use the pre process video, which detection.')
    # Transfor_learning
    # parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')

    # TTUR
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/dataset/", help='meta dataset path')
    parser.add_argument('--four_data_path', type=str, default="/workspace/data/multi_class_segmentation_dataset_512",
                        help="4 type class dataset (ASD, DHS, LCS, HipOA) with segmentation, after 5 fold cross validation. The img size is 512 pixel.")

    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    # add the parser to ther Trainer
    # parser = Trainer.add_argparse_args(parser)

    return parser.parse_known_args()

# %%

def train(hparams):

    # set seed
    seed_everything(42, workers=True)

    classification_module = WalkVideoClassificationLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(hparams.log_path, hparams.model), name=hparams.version, version=hparams.fold)

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar(refresh_rate=hparams.batch_size)

    # define the checkpoint behavier.
    model_check_point = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}-{val_f1_score:.4f}',
        auto_insert_metric_name=True,
        monitor="val_acc",
        mode="max",
        save_last=True,
        save_top_k=3,

    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max',
    )

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=50)

    trainer = Trainer(
                      devices=[hparams.gpu_num,],
                      accelerator="gpu",
                      max_epochs=hparams.max_epochs,
                      logger=tb_logger,
                      #   log_every_n_steps=100,
                      check_val_every_n_epoch=1,
                      callbacks=[progress_bar, rich_model_summary, table_metrics_callback, monitor, model_check_point, early_stopping],
                    #   deterministic=True
                      )
    
    # from the params
    # trainer = Trainer.from_argparse_args(hparams)

    # training and val
    trainer.fit(classification_module, data_module)

    Acc_list = trainer.validate(classification_module, data_module, ckpt_path='best')

    # return the best acc score.
    return model_check_point.best_model_score.item()
 
# %%
if __name__ == '__main__':

    # for test in jupyter
    config, unkonwn = get_parameters()

    #############
    # K Fold CV
    #############

    DATE = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
    DATA_PATH = config.four_data_path

    # set the version
    uniform_temporal_subsample_num = config.uniform_temporal_subsample_num
    clip_duration = config.clip_duration
    config.version = '_'.join([DATE, str(clip_duration), str(uniform_temporal_subsample_num)])

    # output log to file
    log_path = '/workspace/Multi_Classification_PyTorch/logs/' + '_'.join([config.version, config.model]) + '.log'
    # fh = logging.FileHandler(log_path, mode="a")
    # fh.setLevel(logging.WARNING)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # sys.stderr = open(log_path, 'w')
    sys.stdout = open(log_path, 'w')
    
    # get the fold number
    fold_num = os.listdir(DATA_PATH)
    fold_num.sort()

    store_Acc_Dict = {}
    sum_list = []

    for fold in fold_num:
        #################
        # start k Fold CV
        #################

        print('#' * 50)
        print('Strat %s' % fold)
        print('#' * 50)

        config.train_path = os.path.join(DATA_PATH, fold)
        config.fold = fold

        Acc_score = train(config)

        store_Acc_Dict[fold] = Acc_score
        sum_list.append(Acc_score)

    print('#' * 50)
    print('different fold Acc:')
    print(store_Acc_Dict)
    print('Final avg Acc is: %s' % (sum(sum_list) / len(sum_list)))
    

