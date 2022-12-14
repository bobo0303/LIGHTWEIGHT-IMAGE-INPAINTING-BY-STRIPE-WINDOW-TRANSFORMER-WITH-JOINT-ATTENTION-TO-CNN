import torch.nn as nn
import logging, torch, yaml
import numpy as np
from .transformer_layer import BlockAxial, my_Block_2
from .cswin_transformer_attfromself import cswin_Transformer
from .RDB_layer import RDB
from einops import rearrange


logger = logging.getLogger(__name__)

class inpaint_model(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, args):
        super().__init__()

        #first three layers 256>128>64>32
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)


        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        # self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        # self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        # self.conv4_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.n_head = args['n_head']
        self.value = nn.Linear(args['n_embd'], args['n_embd'])

        self.cswin_Transformer_layers = cswin_Transformer(args)

        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(128)
        self.ln_xLxG = nn.LayerNorm(32)

        # layer local (RDB)
        self.RDB1 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        self.RDB2 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        self.RDB3 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        self.RDB4 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        # self.RDB3 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(args['nFeat'] * 4, args['nFeat'], kernel_size=1, padding=0, bias=True)  # ????????? concat??????RDB???????????? ??????????????? *2
        self.GFF_3x3 = nn.Conv2d(args['nFeat'], args['nFeat'], kernel_size=3, padding=1, bias=True)


        self.convt_LG = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm = nn.BatchNorm2d(256)

        #last three layers 32>64>128>256
        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        # self.convt1_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.convt2_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.convt3_3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)

        self.act_last = nn.Sigmoid()    # ZitS first stage
        # self.act_last = nn.Tanh()     # Lama ????????????????????? ??????????????? >0


        self.block_size = 32
        #self.config = config

        self.apply(self._init_weights)

        # calculate parameters
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):    #https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, args, new_lr):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb') #????????????transformer???????????????????????????????????????????????????????????????
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": float(args['weight_decay'])},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=float(new_lr), betas=(0.9, 0.95))
        return optimizer

    def forward(self, img_idx, masks=None):
        img_idx = img_idx * (1 - masks)
        # edge_idx = edge_idx * (1 - masks)
        x = torch.cat((img_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        # xG = x.clone()
        # xL = x.clone()
        # x0 = x.clone()
        xG, xL = torch.split(x, [128, 128], dim=1)

        # mid layer-G
        [b, c, h, w] = xG.shape

        xG, att2, att4 = self.cswin_Transformer_layers(xG)

        xL1 = self.RDB1(xL)   # 4 cnn layer
        xL2 = self.RDB2(xL1)  # 4 cnn layer
        xL2_mid = rearrange(xL2, 'b c h w -> b (h w) c')
        B, T, C = xL2_mid.size()
        xL2_mid = self.value(xL2_mid).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        xmid2 = xL2_mid @ att2
        xmid2 = rearrange(xmid2, 'b c h w -> b (h w) c')
        xmid2 = xmid2.permute(0, 2, 1).reshape(b, c, h, w)  # reshape??????
        xG = xG + xmid2

        xL3 = self.RDB3(xL2)  # 4 cnn layer
        xL4 = self.RDB4(xL3)  # 4 cnn layer

        xL4_mid = rearrange(xL4, 'b c h w -> b (h w) c')
        B, T, C = xL4_mid.size()
        xL4_mid = self.value(xL4_mid).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        xmid4 = xL4_mid @ att4
        xmid4 = rearrange(xmid4, 'b c h w -> b (h w) c')
        xmid4 = xmid4.permute(0, 2, 1).reshape(b, c, h, w)  # reshape??????
        xL4 = xL4 + xmid2 + xmid4

        # xG = xG.permute(0, 2, 3, 1)
        # xG = self.ln_f(xG).permute(0, 3, 1, 2).contiguous()
        XLn = torch.cat((xL1, xL2, xL3, xL4), 1)  # concat
        XLn = self.GFF_1x1(XLn)
        XLn = self.GFF_3x3(XLn)
        xL = XLn + xL  # Residual

        # x = xG + xL     # x + L + G (x?????????L???)
        xL = self.act(self.ln_xLxG(xL))
        xG = self.act(self.ln_xLxG(xG))
        x = torch.cat((xG, xL), 1)  # concat

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        x = self.act_last(x)    # Sigmoid

        # pred_img, edge = torch.split(x, [3, 1], dim=1)
        # return pred_img, edge
        return x

'''if __name__ == '__main__':
    from thop import profile
    import torch
    import time

    # open config file
    with open('C:/Users/Lab722-2080/subject20220725/config/model_config.yml', 'r') as config:
        args = yaml.safe_load(config)

    input_1 = torch.ones(1, 3, 256, 256, dtype=torch.float, requires_grad=False)
    input_2 = torch.ones(1, 1, 256, 256, dtype=torch.float, requires_grad=False)

    Inpaint_model = inpaint_model(args)
    pred_img = Inpaint_model(input_1, input_2)
    # for i in range(10):
    #     s = time.time()
    #     torch.cuda.synchronize()
    #     out = rod(input)  # cone(input)
    #     torch.cuda.synchronize()
    #     e = time.time()
    #     total_time += (e - s)
    # print('time:', total_time / 10)
    flops, params = profile(Inpaint_model, inputs=(input_1, input_2))

    print('input shape:', input.shape)
    print('parameters:', params)
    print('flops', flops)
    # print('output shape', out.shape)'''

