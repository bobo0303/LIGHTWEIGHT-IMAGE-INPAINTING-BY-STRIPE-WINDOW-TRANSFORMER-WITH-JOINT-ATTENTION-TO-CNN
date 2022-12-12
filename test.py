import os, torch, yaml, time
import colorful as c
from torch.utils.data.dataloader import DataLoader
from my_utils import set_seed, SSIM, PSNR, LPIPS, LPIPS_SET, save_img
from models.model import inpaint_model
from dataset import get_dataset


# set seed
set_seed(1234)
gpu = torch.device("cuda")

# open config file
with open('./config/model_config.yml', 'r') as config:
    args = yaml.safe_load(config)

# Define the model
Inpaint_model = inpaint_model(args)

# Define the dataset
test_dataset = get_dataset(args['test_path'], test_mask_path=args['test_mask_1~60_path'],is_train=False, image_size=args['image_size'])

# set initial
iterations = 0
Total_time = []

# loaded_ckpt
if os.path.exists(args['test_ckpt']):
    data = torch.load(args['test_ckpt'])
    Inpaint_model.load_state_dict(data['state_dict'],strict=False)
    Inpaint_model = Inpaint_model.to(args['gpu'])
    print('Finished reloading the Epoch '+c.yellow(str(data['epoch']))+' model')
    # Optimizer
    raw_model = Inpaint_model.module if hasattr(Inpaint_model, "module") else Inpaint_model  # 要問一下
    iterations = data['iterations']
    previous_epoch = data['epoch']
    print(c.blue('------------------------------------------------------------------'))
    print('resume training with iterations: '+c.yellow(str(iterations))+ ', previous_epoch: '+c.yellow(str(previous_epoch)))
    print(c.blue('------------------------------------------------------------------'))
else:
    raise Exception('Warnning: There is no test model found.')

# DataLoaders
test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                         batch_size=args['batch_size'] // args['world_size'],
                         num_workers=args['num_workers'])

print('Try load alex mdoel for LPIPS')
alex = LPIPS_SET()
alex = alex.to(args['gpu'])

if alex is None:
    print(c.red('Warnning: There is no pre_trained alex model found. Try again or Check the Internet.'))
else:
    print(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))

epoch_start = time.time()
# start EVAL
print(c.blue('-----------------------------')+c.cyan(' test ! ')+c.blue('-------------------------------------'))
# Evaluation (Validation)
raw_model.eval()    # eval MODE
loader = test_loader
PSNR_center = []
SSIM_center = []
LPIPS_center = []

for val_it, val_items in enumerate(loader):
    for k in val_items:
        if type(val_items[k]) is torch.Tensor:
            val_items[k] = val_items[k].to(args['gpu'])

    # in to the model (no_grad)
    with torch.no_grad():
        # pred_img, pred_edge = raw_model(val_items['img'], val_items['edge'], val_items['mask'])
        pred_img = raw_model(val_items['img'], val_items['mask'])

        PSNR_center.append(PSNR(val_items['img'], pred_img))
        SSIM_center.append(SSIM(val_items['img'], pred_img))
        LPIPS_center.append(LPIPS(val_items['img'], pred_img, alex))
        # FID_center.append(FID(val_items['img'], pred_img))
        save_img(pred_img, args['save_img_path'], val_items['name'])

PSNR_center = torch.stack(PSNR_center).mean().item()
SSIM_center = torch.stack(SSIM_center).mean().item()
LPIPS_center = torch.stack(LPIPS_center).mean().item()
# FID_center = torch.stack(FID_center).mean().item()
print(c.green('---------------------------------------------------------------------------------'))
print(c.blue('------------------------------')+c.cyan(' PSNR: ')+c.magenta('%.2f'%PSNR_center)+c.blue(' --------------------------------------'))
print(c.blue('------------------------------')+c.cyan(' SSIM: ')+c.magenta('%.3f'%SSIM_center)+c.blue(' --------------------------------------'))
print(c.blue('-----------------------------')+c.cyan(' LPIPS: ')+c.magenta('%.3f'%LPIPS_center)+c.blue(' --------------------------------------'))
print(c.green('---------------------------------------------------------------------------------'))

epoch_time = (time.time() - epoch_start)  # teain one epoch time
print(c.blue('------------------------')+f"This epoch cost {epoch_time:.5f} seconds!"+c.blue('-----------------------'))
Total_time.append(epoch_time)
print(c.blue('----------------------------------')+c.cyan(' End EVAL! ')+c.blue('-----------------------------------'))

Total_time = sum(Total_time)
print(f" ---------------------- Totale time cost {Total_time: .5f} Sec! ------------------------"
            f"\n --------------------------- ALL TRAINING IS DONE!! -----------------------------")
