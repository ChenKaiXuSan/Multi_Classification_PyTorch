# %%
from pytorchvideo.models import resnet, csn, r2plus1d, x3d, slowfast

import torch
import torch.nn as nn
import copy


# %%

class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num


    def set_parameter_requires_grad(self, model: torch.nn.Module, flag:bool = True):

        for param in model.parameters():
            param.requires_grad = flag

    def make_walk_csn(self):

        if self.transfor_learning:
            CSN = torch.hub.load("facebookresearch/pytorchvideo:main", model='csn_r101', pretrained=True)
            CSN.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
        
        else:
            CSN = csn.create_csn(
            input_channel=3,
            model_depth=self.model_depth,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            )

        return CSN

    def make_walk_r2plus1d(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='r2plus1d_r50', pretrained=True)

            # change the head layer.
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
            model.blocks[-1].activation = None
        
        else:
            model = r2plus1d.create_r2plus1d(
                input_channel=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                head_activation=None,
            )

        return model

    def make_walk_c2d(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='c2d_r50', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num, bias=True)

            return model
        else:
            print('no orignal model supported!')

    def make_walk_i3d(self) -> nn.Module:
        
        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='i3d_r50', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

            return model
        else:
            print('no orignal model supported!')

    def make_walk_x3d(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_m', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
            model.blocks[-1].activation = None

        else:
            model = x3d.create_x3d(
                input_channel=3, 
                input_clip_length=16,
                input_crop_size=224,
                model_num_class=1,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                head_activation=None,
            )

        return model

    def make_walk_slow_fast(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='slowfast_r50', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:

            model = slowfast.create_slowfast(
                input_channels=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return model


    def make_walk_resnet(self):
        
        # make model
        slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        
        # change the knetics-400 output 400 to model class num
        slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        model_stem = slow.blocks[0]
        model_head = slow.blocks[-1]
        model_stage = slow.blocks[1:-1]

        return slow

# %%
class single_frame(nn.Module):

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num

        self.transfor_learning = hparams.transfor_learning

        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfor_learning)

        self.resnet_model.fc = torch.nn.Linear(2048, self.model_class_num, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        b, c, t, h, w = img.size()
        # make frame to batch image, wich (b*t, c, h, w)
        batch_imgs = img.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

        output = self.resnet_model(batch_imgs)

        return output

class early_fusion(nn.Module):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model_class_num

        self.transfor_learning = hparams.transfor_learning
        self.uniform_temporal_subsample_num = hparams.uniform_temporal_subsample_num

        # change the resnet work structure
        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfor_learning)

        self.resnet_model.conv1 = torch.nn.Conv2d(3 * self.uniform_temporal_subsample_num, out_channels=64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_model.fc = torch.nn.Linear(2048, self.model_class_num, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        b, c, t, h, w = img.size()

        batch_imgs = img.permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)

        output = self.resnet_model(batch_imgs)

        return output

class late_fusion(nn.Module):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model_class_num

        self.transfor_learning = hparams.transfor_learning

        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfor_learning)

        # late fusion model
        self.first_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-2])
        self.last_model = copy.deepcopy(self.first_model)

        self.last_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.last_layer = torch.nn.Linear(4096, self.model_class_num, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        b, c, t, h, w = img.size()
        batch_imgs = img.permute(0, 2, 1, 3, 4) # b, t, c, h, w

        first_single_frame =  batch_imgs[:, 10, :] # b, c, h, w
        last_single_frame = batch_imgs[:, -10, :] # b, c, h, w

        first_feat = self.first_model(first_single_frame)
        last_feat = self.last_model(last_single_frame)

        cat_feat = torch.cat((first_feat, last_feat), dim = 1) # b, c

        output = self.last_layer(self.last_pool(cat_feat).squeeze())

        return output


# # %%
# class opt: 

#     model_class_num = 1
#     model_depth = 50
#     transfor_learning = True
#     fix_layer = 'stage_head'

# make_video_module = MakeVideoModule(opt)

# model = make_video_module.make_walk_i3d()

# from torchinfo import summary

# summary(model, input_size=(4, 3, 16, 224, 224))
# # %%

# single_frame_model = single_frame(opt)

# batch_video = torch.randn(size=[4, 3, 16, 224, 224])

# output = single_frame_model(batch_video)

# %%
# list the model in repo.
# torch.hub.list('facebookresearch/pytorchvideo', force_reload=True)
# # %%