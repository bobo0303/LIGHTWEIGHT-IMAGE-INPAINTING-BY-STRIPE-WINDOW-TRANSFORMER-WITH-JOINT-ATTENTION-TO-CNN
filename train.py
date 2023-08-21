import os, sys, torch, logging, yaml, time
import colorful as c
from torchsummary import summary
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from my_utils import set_seed, Visualization_of_training_results, SSIM, PSNR, LPIPS, LPIPS_SET
from models.model import inpaint_model
from models.Discriminator import NLayerDiscriminator
from dataset import get_dataset
from tqdm import tqdm
from losses.L1 import l1_loss, masked_l1_loss
from losses.Perceptual import PerceptualLoss
from losses.Adversarial import NonSaturatingWithR1
from losses.HSV import HSV
from losses.Style import StyleLoss
from losses.Edge import Edge_MSE_loss
from tensorboardX import SummaryWriter

# set seed
set_seed(1234)
gpu = torch.device("cuda")

# open config file
with open('./config/model_config.yml', 'r') as config:
    args = yaml.safe_load(config)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# set log
log_path = args['ckpt_path'] + args['name']
if not os.path.isdir(log_path):
    os.makedirs(log_path)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(sh)
logger.propagate = False
fh = logging.FileHandler(os.path.join(log_path, args['name'] + '.txt'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# set tensorboardx log
log_dir = os.path.join(args['ckpt_path'], args['name'], 'log')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_'+args['name'])

# Define the model
Inpaint_model = inpaint_model(args)
Discriminator = NLayerDiscriminator(input_nc=3)

# Define the dataset
train_dataset = get_dataset(args['data_path'], mask_path=args['mask_path'], is_train=True,image_size=args['image_size'])

test_dataset = get_dataset(args['val_path'], test_mask_path=args['val_mask_path'],is_train=False, image_size=args['image_size'])

# set initial
previous_epoch = -1
bestAverageF1 = 0
iterations = 0
best_psnr = 0
best_ssim = 0
best_lpips = 1
best_epoch_psnr = 0
best_epoch_ssim = 0
best_epoch_lpips = 0
Total_time = []

# loaded_ckpt
if os.path.exists(args['resume_ckpt']):
    data = torch.load(args['resume_ckpt'])
    D_data = torch.load(args['resume_D_ckpt'])
    Inpaint_model.load_state_dict(data['state_dict'],strict=False)
    Discriminator.load_state_dict(D_data['discriminator'])
    Inpaint_model = Inpaint_model.to(args['gpu'])
    Discriminator = Discriminator.to(args['gpu'])
    logger.info('Finished reloading the Epoch '+c.yellow(str(data['epoch']))+' model')
    # Optimizer
    raw_model = Inpaint_model.module if hasattr(Inpaint_model, "module") else Inpaint_model  # 要問一下
    optimizer = raw_model.configure_optimizers(args, new_lr=args['learning_rate'])
    # update optimizer info
    optimizer.load_state_dict(data['optimizer'])
    D_optimizer = optim.Adam(Discriminator.parameters(), lr=args['D_learning_rate'], betas=(0.9, 0.95))
    D_optimizer.load_state_dict(D_data['D_optimizer'])
    iterations = data['iterations']
    #bestAverageF1 = data['best_validation']
    previous_epoch = data['epoch']
    logger.info('Finished reloading the Epoch '+c.yellow(str(data['epoch']))+' optimizer')
    logger.info(c.blue('------------------------------------------------------------------'))
    logger.info('resume training with iterations: '+c.yellow(str(iterations))+ ', previous_epoch: '+c.yellow(str(previous_epoch)))
    logger.info(c.blue('------------------------------------------------------------------'))
else:
    # start form head
    Inpaint_model = Inpaint_model.to(args['gpu'])
    Discriminator = Discriminator.to(args['gpu'])
    logger.info(c.blue('------------------------------------------------------------------'))
    logger.info(c.red('Warnning: There is no trained model found. An initialized model will be used.'))
    logger.info(c.red('Warnning: There is no previous optimizer found. An initialized optimizer will be used.'))
    logger.info(c.blue('------------------------------------------------------------------'))
    # Optimizer
    raw_model = Inpaint_model.module if hasattr(Inpaint_model, "module") else Inpaint_model
    optimizer = raw_model.configure_optimizers(args, new_lr=args['learning_rate'] )
    D_optimizer = optim.Adam(Discriminator.parameters(), lr=args['D_learning_rate'], betas=(0.9, 0.95))

if args['lr_decay']:
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args['train_epoch'] - args['warmup_epoch'],
                                                            eta_min=float(args['lr_min']))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args['warmup_epoch'],
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    D_scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(D_optimizer, args['train_epoch'] - args['D_warmup_epoch'],
                                                            eta_min=float(args['D_lr_min']))
    D_scheduler = GradualWarmupScheduler(D_optimizer, multiplier=1, total_epoch=args['D_warmup_epoch'],
                                       after_scheduler=D_scheduler_cosine)
    D_scheduler.step()

    for i in range(1, previous_epoch):
        scheduler.step()
        D_scheduler.step()

# DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                                  batch_size=args['batch_size'] // args['world_size'],  # BS of each GPU
                                  num_workers=args['num_workers'])

test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                         batch_size=args['batch_size'] // args['world_size'],
                         num_workers=args['num_workers'])

# Load pre_trained VGG model (for Perceptual_loss)
if args['Lambda_Perceptual'] is not 0 or not None:
    logger.info('Try load VGG mdoel for Perceptual_loss')
    VGG = PerceptualLoss()
if VGG is None:
    logger.info(c.red('Warnning: There is no pre_trained VGG model found. Try again or Check the Internet.'))
else:
    logger.info(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))
if args['Lambda_Style'] is not 0 or not None:
    logger.info('Try load VGG mdoel for Style_loss')
    style = StyleLoss()
if VGG is None:
    logger.info(c.red('Warnning: There is no pre_trained VGG model found. Try again or Check the Internet.'))
else:
    logger.info(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))

adv = NonSaturatingWithR1()

logger.info('Try load alex mdoel for LPIPS')
alex = LPIPS_SET()
alex = alex.to(args['gpu'])
if alex is None:
    logger.info(c.red('Warnning: There is no pre_trained alex model found. Try again or Check the Internet.'))
else:
    logger.info(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))

# print('==> Training start: ')
summary(raw_model, [(3, 256, 256), (1, 256, 256)])  # won't write in log (model show)

for epoch in range(args['train_epoch']):
    if previous_epoch != -1 and epoch < previous_epoch:
        continue
    if epoch == previous_epoch + 1:
        logger.info("Resume from Epoch %d" % epoch)

    epoch_start = time.time()
    raw_model.train()   # Train MODE
    Discriminator.train()
    loader = train_loader

    for it, items in enumerate(tqdm(loader, disable=False)):
        for k in items:
            if type(items[k]) is torch.Tensor:
                items[k] = items[k].to(args['gpu']).requires_grad_()    # img mask edge name

        Discriminator.zero_grad()
        D_img, _ = Discriminator(items['img'])
        LD1 = adv.discriminator_real_loss(items['img'], D_img)    # LD1+LGP (real loss)

        pred_img = raw_model(items['img'], items['mask'])
        pred_img_ = pred_img.detach()
        D_pred_img, _ = Discriminator(pred_img_)
        LD2 = adv.discriminator_fake_loss(D_pred_img, mask=items['mask'])  #LD23 (fake loss)
        D_loss = args['Lambda_LD1']*LD1 + args['Lambda_LD2']*LD2

        D_loss.backward()
        D_optimizer.step()

        raw_model.zero_grad()

        img = items['img']
        pred_img = pred_img
        mask = items['mask']
        Edge = items['edge']

        L1_loss = l1_loss(img, pred_img)
        Edge_loss = Edge_MSE_loss(img, pred_img, Edge)
        Perceptual_loss = VGG.forward(img, pred_img, mask=mask)
        style_loss = style(pred_img * mask, img * mask)
        HSV_loss_H, HSV_loss_S, HSV_loss_V, HSV_loss = HSV(img, pred_img, Edge)
        # HSV_edge_loss_H, HSV_edge_loss_S, HSV_edge_loss_V, HSV_edge_loss = HSV_edge(img, pred_img, Edge)
        # LHSV_H = args['Lambda_HSV'] * HSV_loss_H + args['Lambda_HSV_edge'] * HSV_edge_loss_H
        # LHSV_S = args['Lambda_HSV'] * HSV_loss_S + args['Lambda_HSV_edge'] * HSV_edge_loss_S
        # LHSV_V = args['Lambda_HSV']*HSV_loss_V + args['Lambda_HSV_edge']*HSV_edge_loss_V
        # LHSV = args['Lambda_HSV']*HSV_loss + args['Lambda_HSV_edge']*HSV_edge_loss
        LHSV = args['Lambda_HSV'] * HSV_loss_H + args['Lambda_HSV'] * HSV_loss_S

        D_pred_img, _ = Discriminator(pred_img)
        LG = adv.generator_loss(D_pred_img, mask=None)  # LG
        # G_loss = args['Lambda_L1']*L1_loss + args['Lambda_Perceptual']*Perceptual_loss + args['Lambda_LG']*LG
        G_loss = args['Lambda_L1'] * L1_loss + args['Lambda_Perceptual'] * Perceptual_loss + args['Lambda_LG'] * LG + \
                 args['Lambda_LHSV'] * LHSV + args['Lambda_Style'] * style_loss + args['Lambda_Edge'] * Edge_loss

        G_loss.backward()
        optimizer.step()

        iterations += 1  # number of iterations processed this step

        '''if iterations % args['print_freq'] == 0:
            logger.info(f"epoch {epoch + 1} iter {it}/{args['iterations_per_epoch']}: D1_loss {LD1.item():.5f}. | D2_loss {LD2.item():.5f}. | "
                        f"L1_loss {L1_loss.item():.5f} | Perceptual_loss {Perceptual_loss.item():.5f}. | HSV_Total_loss {LHSV.item():.5f}. | "
                        f"G_loss {LG.item():.5f}. | D_Total_loss {D_loss.item():.5f}. | G_Total_loss {G_loss.item():.5f}. | lr {scheduler.get_lr()[0]:e}")'''

        if iterations % 500 == 0:
            # Visualization of training results (Edge and GT)
            # Visualization_of_training_results(pred_edge, pred_img, items['img'], items['edge'], items['mask'], log_path, iterations)
            Visualization_of_training_results(pred_img, items['img'], items['mask'], log_path, iterations)
            writer.add_scalar('loss/LD1', LD1.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/LD2', LD2.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('Total_loss/D_total_loss', D_loss.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/L1_loss', L1_loss.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/Perceptual_loss', Perceptual_loss.item(), iterations)
            writer.add_scalar('loss/style_loss', style_loss.item(), iterations)
            writer.add_scalar('loss/Edge_loss', Edge_loss.item(), iterations)
            writer.add_scalar('loss/LG', LG.item(), iterations)
            writer.add_scalar('HSV/H_loss', HSV_loss_H.item(), iterations)
            writer.add_scalar('HSV/S_loss', HSV_loss_S.item(), iterations)
            # writer.add_scalar('HSV/V_loss', HSV_loss_V.item(), iterations)
            # writer.add_scalar('HSV/H_edge_loss', HSV_edge_loss_H.item(), iterations)
            # writer.add_scalar('HSV/S_edge_loss', HSV_edge_loss_S.item(), iterations)
            # writer.add_scalar('HSV/V_edge_loss', HSV_edge_loss_V.item(), iterations)
            writer.add_scalar('loss/HSV_loss', HSV_loss.item(), iterations)
            # writer.add_scalar('loss/HSV_edge_loss', HSV_edge_loss.item(), iterations)
            writer.add_scalar('Total_loss/HSV_total_loss', LHSV.item(), iterations)
            writer.add_scalar('Total_loss/G_total_loss', G_loss.item(), iterations)

    logger.info(
        f"epoch {epoch + 1} iter {iterations}: D1_loss {LD1.item():.5f}. | D2_loss {LD2.item():.5f}. | {c.magenta('D_Total_loss')} {D_loss.item():.5f}. | "
        f"L1_loss {L1_loss.item():.5f} | Perceptual_loss {Perceptual_loss.item():.5f}. | style_loss {style_loss.item():.5f}. | Edge_loss {Edge_loss.item():.5f} | HSV_Total_loss {LHSV.item():.5f} | "
        f"G_loss {LG.item():.5f}. | {c.magenta('G_Total_loss')} {G_loss.item():.5f}. | lr {scheduler.get_lr()[0]:e}")

    # start EVAL
    logger.info(c.blue('-----------------------------')+c.cyan(' Start EVAL! ')+c.blue('-------------------------------------'))
    # Evaluation (Validation)
    raw_model.eval()    # eval MODE
    Discriminator.eval()
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

    PSNR_center = torch.stack(PSNR_center).mean().item()
    SSIM_center = torch.stack(SSIM_center).mean().item()
    LPIPS_center = torch.stack(LPIPS_center).mean().item()
    # FID_center = torch.stack(FID_center).mean().item()

    writer.add_scalar('Evaluation_Metrics/PSNR_center', PSNR_center, epoch + 1)     # tensorboard for log (visualize)
    writer.add_scalar('Evaluation_Metrics/SSIM_center', SSIM_center, epoch + 1)
    writer.add_scalar('Evaluation_Metrics/LPIPS_center', LPIPS_center, epoch + 1)
    # writer.add_scalar('Evaluation_Metrics/FID_center', FID_center, epoch + 1)

    # Save ckpt (best PSNR)
    if PSNR_center > best_psnr:
        best_psnr = PSNR_center
        best_epoch_psnr = epoch
        torch.save({'epoch': epoch + 1,
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iterations': iterations
                    }, os.path.join(log_path, "model_bestPSNR.pth"))

    # Save ckpt (best SSIM)
    if SSIM_center > best_ssim:
        best_ssim = SSIM_center
        best_epoch_ssim = epoch
        torch.save({'epoch': epoch + 1,
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iterations': iterations
                    }, os.path.join(log_path, "model_bestSSIM.pth"))

    # Save ckpt (best LPIPS)
    if LPIPS_center < best_lpips:
        best_lpips = LPIPS_center
        best_epoch_lpips = epoch
        torch.save({'epoch': epoch + 1,
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iterations': iterations
                    }, os.path.join(log_path, "model_bestLPIPS.pth"))

    # Save ckpt (each EPOCH)
    torch.save({'epoch': epoch + 1,
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iterations': iterations
                }, os.path.join(log_path, "model_" + str(epoch + 1) + ".pth"))

    logger.info(f"current_epoch {epoch + 1} : current_PSNR {PSNR_center:.5f}. | "
                f"best_epoch {best_epoch_psnr+1} : best_PSNR {best_psnr:.5f}."
                f"\ncurrent_epoch {epoch + 1} : current_SSIM {SSIM_center:.5f}.  | "
                f"best_epoch {best_epoch_ssim+1} : best_SSIM {best_ssim:.5f}."
                f"\ncurrent_epoch {epoch + 1} : current_LPIPS {LPIPS_center:.5f}. | "
                f"best_epoch {best_epoch_lpips+1} : best_LPIPS {best_lpips:.5f}.")

    # Save D_ckpt (each EPOCH)
    torch.save({'discriminator': Discriminator.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                }, os.path.join(log_path, "Discriminator_" + str(epoch + 1) + ".pth"))

    # Save the last model
    torch.save({'epoch': epoch + 1,
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iterations': iterations
                }, os.path.join(log_path, "model_last.pth"))

    # Save D_ckpt (last EPOCH)
    torch.save({'discriminator': Discriminator.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                }, os.path.join(log_path, "Discriminator_last.pth"))

    if args['lr_decay']:
        scheduler.step()
        D_scheduler.step()
        writer.add_scalar('lr/G_lr', scheduler.get_lr()[0], epoch + 1)
        writer.add_scalar('lr/D_lr', D_scheduler.get_lr()[0], epoch + 1)

    epoch_time = (time.time() - epoch_start)  # teain one epoch time
    logger.info(f"This epoch cost {epoch_time:.5f} seconds!")
    Total_time.append(epoch_time)
    logger.info(c.blue('------------------------------')+c.cyan(' End EVAL! ')+c.blue('--------------------------------------'))

writer.close()
Total_time = sum(Total_time)
logger.info(f"\n --------------------- Total time cost {Total_time/60/60:.5f} HR! ----------------------"
            f"\n -------------------------- ALL TRAINING IS DONE!! --------------------------")
