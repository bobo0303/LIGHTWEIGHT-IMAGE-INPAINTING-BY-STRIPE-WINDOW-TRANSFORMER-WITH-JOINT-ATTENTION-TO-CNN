import torch.nn as nn
import logging, torch, yaml
from .transformer_layer import BlockAxial, my_Block_2
from .Global_transformer_layer import Mid_Golobal_layer
from .RDB_layer import RDB

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

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 128))  #1b 32*32hw 256C
        self.drop = nn.Dropout(args['embd_pdrop'])

        #兩條路線Global:Transformer Local:CNN
        # layer global (TRANSFORMER)
        self.blocks = []
        for _ in range(args['n_layer']):
            # self.blocks.append(BlockAxial(args))
            # self.blocks.append(my_Block_2(args))
            self.blocks.append(Mid_Golobal_layer(args))
        self.blocks = nn.Sequential(*self.blocks)

        self.blocks_2 = []
        for _ in range(args['n_layer']):
            self.blocks_2.append(Mid_Golobal_layer(args))
        self.blocks_2 = nn.Sequential(*self.blocks_2)

        self.blocks_3 = []
        for _ in range(args['n_layer']):
            self.blocks_3.append(Mid_Golobal_layer(args))
        self.blocks_3 = nn.Sequential(*self.blocks_3)

        self.blocks_4 = []
        for _ in range(args['n_layer']):
            self.blocks_4.append(Mid_Golobal_layer(args))
        self.blocks_4 = nn.Sequential(*self.blocks_4)

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
        self.GFF_1x1 = nn.Conv2d(args['nFeat'] * 4, args['nFeat'], kernel_size=1, padding=0, bias=True)  # 因為是 concat兩個RDB而以所以 輸入通道是 *2
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
        # self.act_last = nn.Tanh()     # Lama 會造成顏色錯誤 因為結果非 >0


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
        no_decay.add('pos_emb') #這個應是transformer的東西現在我沒加所以會抱錯之後有家要放回去
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
        xG = xG.view(b, c, h * w).transpose(1, 2).contiguous()
        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        xG = self.drop(xG + position_embeddings)  # [b,hw,c] 原來CNN跑完後的權重加上 隨機給予的可學習資料 並設定0.1部分資料不可更新
        xG = xG.permute(0, 2, 1).reshape(b, c, h, w)  # reshape回去

        xG_1 = self.blocks(xG)
        xG_2 = self.blocks_2(xG_1)
        xL1 = self.RDB1(xL)   # 4 cnn layer
        xL2 = self.RDB2(xL1)  # 4 cnn layer

        xG_1 = self.act(self.ln_xLxG(xG_1))
        xL1 = self.act(self.ln_xLxG(xL1))
        x_mid = xG_1 + xL1
        xL2 = self.act(self.ln_xLxG(xL2))
        xG_2 = self.act(self.ln_xLxG(xG_2))
        xL2 = x_mid + xL2
        xG_2 = x_mid + xG_2

        xG_3 = self.blocks_3(xG_2)
        xG_4 = self.blocks_4(xG_3)
        xL3 = self.RDB3(xL2)  # 4 cnn layer
        xL4 = self.RDB4(xL3)  # 4 cnn layer

        xG_3 = self.act(self.ln_xLxG(xG_3))
        xL3 = self.act(self.ln_xLxG(xL3))
        x_mid2 = xG_3 + xL3
        xL4 = self.act(self.ln_xLxG(xL4))
        xG_4 = self.act(self.ln_xLxG(xG_4))
        xL4 = x_mid2 + xL4
        xG_4 = x_mid2 + xG_4

        xG = xG_4.permute(0, 2, 3, 1)
        xG = self.ln_f(xG).permute(0, 3, 1, 2).contiguous()
        XLn = torch.cat((xL1, xL2, xL3, xL4), 1)  # concat
        XLn = self.GFF_1x1(XLn)
        XLn = self.GFF_3x3(XLn)
        xL = XLn + xL  # Residual

        # x = xG + xL     # x + L + G (x包含在L裡)
        xL = self.act(self.ln_xLxG(xL))
        xG = self.act(self.ln_xLxG(xG))
        x = torch.cat((xG, xL), 1)  # concat
        # x = self.convt_LG(x)    #目前是3_1_1的反卷機 之後應該也可以改成1*1 C:512>256 試試看
        # x = self.act(x)
        # x = self.GFF_1x1(x)
        # x = self.GFF_3x3(x)

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

