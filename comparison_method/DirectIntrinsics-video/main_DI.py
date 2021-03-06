import torch
import torchvision
from torch.utils.data import DataLoader
# import my_data_RD as my_data
import my_data as my_data
from configs.intrinsic_DI import opt

from utils import save_eval_images, check_dir, visualize_inspector
import numpy as np
import argparse
import pdb

"""
v3: use vgg19; use L1 loss rather than MSE loss, discard gradient and ssim,
    enlarge idt loss weight, change fea_div_dict
v4: use gradient loss, average across 3 channels (rgb), add loss weight
v6: set grad threshold for gt_grad, only large grad values are considered
v7: use mask for crop window selection when feed data
v8: perceptual loss use L2 + cosine, more strict trian patch selection
v9: Grad use 3 channels and x y splited, preserve_info_loss, patch select 0.35,
    fea_distance_loss: cos 0.85, L2 0.15, lr use wave pattern
v10: cropsize=256, L1 loss weight = 30.0
v11: cropsize=288, L1 loss weight = 25.0, lambda_r = lambda_b = 1.0, use
    div_dict for perceptual, use cosine loss for preserve_info
RD: refined data, gray scale shading, I=A.*S; modified from v11
    ms+fd+pers
RD_v8: align_corners=False, conv3 has pad 1, feat_dict=[low,mid,mid2,deep,out], dr = [1,1,1]

DI: direct Intrinsics
[2021.03.30] for video extension, using Temporal Consistency Loss
"""

#>>>>>>> 2021.03.24 >>>>>>> 
######## argparse for flexible change variables #####
#### need to define:
# if 'MPI' in opt.data.name:
#     opt.output_root = '../ckpoints-'+opt.train.trainer_mode+'-'+opt.train.mode+'-'+opt.data.name+'-'+opt.data.split
# else:
#     opt.output_root = '../ckpoints-' + opt.train.trainer_mode + '-' + opt.train.mode + '-' + opt.data.name

# opt.logger.log_dir = opt.output_root+'/log/'      # The log dir for saving train log and image information
# opt.logger.root_dir = opt.output_root       # The root dir for logging
######## dataloader, load from refined_final
parser = argparse.ArgumentParser(description="ablation study of loss terms for MPI_video dataset")
# video ablation extension settings:
parser.add_argument('--w_flow', type=float, default=50.0)
parser.add_argument('--lambda_r', type=float, default=10.0, help="lambda_r for flow")
parser.add_argument('--lambda_s', type=float, default=5.0, help="lambda_s for flow")

parser.add_argument('--phase', type=str, default='train', help="train or test")
parser.add_argument('--best_ep', type=int, default=None)

parser.add_argument('--cuda_id', type=str, default='0')

parser.add_argument('--is_continue', action='store_true')
parser.add_argument('--from_ep', type=int, default=None)

parser.add_argument('--save_train_img', action='store_true')
parser.add_argument('--save_per_n_ep', type=int, default=5)
parser.add_argument('--total_ep', type=int, default=250)

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_id

opt.optim.flow_idt_w = FLAGS.w_flow
opt.optim.lambda_flow_r = FLAGS.lambda_r
opt.optim.lambda_flow_s = FLAGS.lambda_s
opt.output_root = '../video-ckpoints-DI-scratch-'+opt.train.mode+'-'+opt.data.name+'-'+opt.data.split \
                  +'-w_flow-'+str(FLAGS.w_flow)+ '-lambda_r-' + str(FLAGS.lambda_r) + '-lambda_s-' + str(FLAGS.lambda_s)
opt.logger.log_dir = opt.output_root + '/log/'

opt.train.save_train_img = FLAGS.save_train_img
opt.train.save_per_n_ep = FLAGS.save_per_n_ep
opt.train.total_ep = FLAGS.total_ep  # 250, MIT-50

if FLAGS.is_continue and FLAGS.from_ep is not None:
    opt.continue_train = True
    opt.which_epoch = FLAGS.from_ep
    print('continue from ', opt.which_epoch)

#<<<<<<< 2021.03.24 <<<<<<< 
import trainer_DI as Trainer

check_dir(opt.output_root)
check_dir(opt.logger.log_dir)


# loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'iter_step': []}
def train_one_epoch(trainer, dataset, epoch, loss_inspector):
    opt.is_train = True
    dataloader = DataLoader(dataset, batch_size=opt.data.batch_size, shuffle=True)
    batch_num = len(dataset)//opt.data.batch_size

    log_list = []

    for batch_idx, samples in enumerate(dataloader):
        trainer.set_input(samples)
        trainer.optimize_parameters()

        losses = trainer.get_current_errors()

        if np.isnan(losses['loss_total'].cpu().item()):
            print('Warning: nan loss!')
            pdb.set_trace()
        loss_inspector['loss_total'].append(losses['loss_total'].cpu().item())
        loss_inspector['loss_idt'].append(losses['idt_S'].cpu().item() + losses['idt_R'].cpu().item())
        loss_inspector['loss_grad'].append(losses['grad_S'].cpu().item() + losses['grad_R'].cpu().item())
        # video flow loss
        loss_inspector['loss_flow'].append(losses['flow'].cpu().item())
        loss_inspector['loss_flow_r'].append(losses['flow_r'].cpu().item())
        loss_inspector['loss_flow_s'].append(losses['flow_s'].cpu().item())

        loss_inspector['lr'].append(trainer.get_lr())
        loss_inspector['step'].append((epoch-1) * batch_num + batch_idx + 1)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_num,
                100.0*batch_idx/batch_num, losses['loss_total'].cpu().item()))
            log_detail = trainer.loss_log(losses)
            print(log_detail)
            log_list.append(log_detail)
        if opt.train.save_train_img and (epoch % opt.train.save_per_n_ep) == 0:
            visuals = trainer.get_current_visuals()
            img_dir = opt.logger.log_dir + 'train-imgs_ep' + str(epoch)  # '../checkpoints/log/'
            check_dir(img_dir)
            save_eval_images(visuals, img_dir, batch_idx, opt)
    if epoch % opt.train.save_per_n_ep == 0:
        with open(opt.logger.log_dir + '/train_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.writelines(["%s\n" % item for item in log_list])


def evaluate_one_epoch(trainer, dataset, epoch, loss_inspector, save_pred_results=False):
    opt.is_train = False
    # Todo:batch_size should be 1, or change save_images function
    batch_size = opt.data.batch_size_test if not save_pred_results else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_num = len(dataset) // batch_size

    log_list = []
    loss_total = []
    loss_idt = []
    loss_grad = []
    # video flow loss
    loss_flow = []
    loss_flow_r = []
    loss_flow_s = []

    for batch_idx, samples in enumerate(dataloader):
        trainer.set_input(samples)
        trainer.inference()
        losses = trainer.get_current_errors()

        loss_total.append(losses['loss_total'].cpu().item())
        loss_idt.append(losses['idt_S'].cpu().item() + losses['idt_R'].cpu().item())
        loss_grad.append(losses['grad_S'].cpu().item() + losses['grad_R'].cpu().item())
        # video flow loss
        loss_flow.append(losses['flow'].cpu().item())
        loss_flow_r.append(losses['flow_r'].cpu().item())
        loss_flow_s.append(losses['flow_s'].cpu().item())

        # print eval losses
        log_str = 'Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx, batch_num,
            100.0 * batch_idx / batch_num, losses['loss_total'].cpu().item())
        if batch_idx % 100 == 0:
            print(log_str)
        log_detail = trainer.loss_log(losses)
        # print(log_detail)
        log_list.append(log_str)
        log_list.append(log_detail)

        if save_pred_results:
            # save eval imgs into epoch dir
            # if batch_idx % 10 == 0:
            #     print(log_str)
            visuals = trainer.get_current_visuals()
            img_dir = opt.logger.log_dir + 'test-imgs_ep' + str(epoch)  # '../checkpoints/log/'
            check_dir(img_dir)
            save_eval_images(visuals, img_dir, batch_idx, opt)

    loss_inspector['loss_total'].append(np.mean(loss_total))
    loss_inspector['loss_idt'].append(np.mean(loss_idt))
    loss_inspector['loss_grad'].append(np.mean(loss_grad))
    
    # video flow loss
    loss_inspector['loss_flow'].append(np.mean(loss_flow))
    loss_inspector['loss_flow_r'].append(np.mean(loss_flow_r))
    loss_inspector['loss_flow_s'].append(np.mean(loss_flow_s))

    cur_lr = opt.optim.lr_g if trainer.get_lr() is None else trainer.get_lr()
    loss_inspector['lr'].append(cur_lr)
    loss_inspector['step'].append(epoch)

    # save log info into file
    eval_log = 'Evaluation_loss_total-Ep{}: {:.4f}, learning rate: {}\nloss_idt: {:.4f}, \tloss_grad: {:.4f} \
    \tloss_flow: {:.4f}, \tloss_flow_r: {:4f}, \tloss_flow_s: {:4f}\n'.format(
               epoch, np.mean(loss_total), trainer.get_lr(), np.mean(loss_idt), np.mean(loss_grad),
               np.mean(loss_flow), np.mean(loss_flow_r), np.mean(loss_flow_s))

    print(eval_log)
    if save_pred_results:
        with open(opt.logger.log_dir + '/eval_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.write(eval_log)
            f.writelines(["%s\n" % item for item in log_list])


def basic_settings():
    run_settings = {}
    """setup datasets"""
    TRAIN_setting = EasyDict({
        'data_root': r'/home/wzj/ws/intrinsic/data',
        'image_height': 320,
        'image_width':  320,
        'fast_check': FLAGS.fast_check,
    })
    # if TRAIN_SET.mode in ['train', 'valid']:
    IMAGE_NAMES_train = [ 'input_1', 'input_2', 
                    'albedo_1', 'albedo_2', 
                    'shading_1', 'shading_2',
                    'occ', 'flow']
    # elif TRAIN_SET.mode == 'test':
    IMAGE_NAMES_test = ['input_1', 'albedo_1', 'shading_1']
    
    if 'MPI' in opt.data.name:
        train_dataset = my_data.IntrinsicImageList(params=TRAIN_setting, image_names=IMAGE_NAMES_train, mode='train')
        valid_dataset = my_data.IntrinsicImageList(params=TRAIN_setting, image_names=IMAGE_NAMES_train, mode='valid')
        test_dataset =  my_data.IntrinsicImageList(params=TRAIN_setting, image_names=IMAGE_NAMES_test, mode='test')
    else:
        raise NotImplementedError

    """setup model trainer"""
    model_trainer = Trainer.Trainer_Basic(opt)
    opt_test = opt
    opt_test.is_train = False
    test_model = Trainer.Trainer_Basic(opt_test)

    # evaluate(trainer, test_dataset, 0)
    # pdb.set_trace()
    """training process"""
    """training process video flow

    Returns:
        _type_: _description_
    """
    train_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_grad': [],
                            'lr': [],
                            'loss_flow': [], 'loss_flow_r': [], 'loss_flow_s': [],
                            'step': []}
    test_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_grad': [],
                           'lr': [],
                           'loss_flow': [], 'loss_flow_r': [], 'loss_flow_s': [],
                           'step': []}

    run_settings['train_dataset'] = train_dataset
    run_settings['valid_dataset'] = valid_dataset
    run_settings['test_dataset'] = test_dataset
    run_settings['model_trainer'] = model_trainer
    run_settings['test_model'] = test_model
    run_settings['train_loss_inspector'] = train_loss_inspector
    run_settings['test_loss_inspector'] = test_loss_inspector

    return run_settings


def train():
    s = basic_settings()

    start_epoch = opt.train.epoch_count  # 1

    if opt.continue_train:
        evaluate_one_epoch(s['model_trainer'], s['valid_dataset'], start_epoch-1, s['test_loss_inspector'])
    for ep in range(opt.train.total_ep):  # total_ep=300
        epoch = ep + start_epoch
        train_one_epoch(s['model_trainer'], s['train_dataset'], epoch, s['train_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            evaluate_one_epoch(s['model_trainer'], s['valid_dataset'], epoch, s['test_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            s['model_trainer'].save(epoch)

            """save to log"""

            with open(opt.logger.log_dir + '/all_eval_log-' + str(start_epoch - 1 + opt.train.total_ep) + '.txt', 'w') as f:
                f.write(
                    'epoch: total_loss: idt_loss:  grad_loss: \
                    loss_flow: loss_flow_r: loss_flow_s: \
                    learning_rate:\n')
                for i in range(len(s['test_loss_inspector']['step'])):
                    log_string = '{:04d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                        s['test_loss_inspector']['step'][i],
                        s['test_loss_inspector']['loss_total'][i],
                        s['test_loss_inspector']['loss_idt'][i],
                        s['test_loss_inspector']['loss_grad'][i],
                        s['test_loss_inspector']['loss_flow'][i],
                        s['test_loss_inspector']['loss_flow_r'][i],
                        s['test_loss_inspector']['loss_flow_s'][i],
                        s['test_loss_inspector']['lr'][i]
                    )
                    f.write(log_string)

            """visualize inspectors"""
            train_loss_dir = opt.logger.log_dir + 'train_losses-' + \
                             str(start_epoch) + '-' + str(epoch) + '_' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
            test_loss_dir = opt.logger.log_dir + 'test_losses-' + \
                            str(start_epoch) + '-' + str(epoch) + '_' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
            visualize_inspector(s['train_loss_inspector'], train_loss_dir, step_num=None, mode=opt.train.trainer_mode)
            visualize_inspector(s['test_loss_inspector'], test_loss_dir, step_num=None, mode=opt.train.trainer_mode)

        s['model_trainer'].update_learning_rate()


def test(epoch=None):
    settings = basic_settings()
    test_dataset = settings['test_dataset']
    model_trainer = settings['test_model']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_split, 'G_decompose', epoch_name=epoch)

    evaluate_one_epoch(model_trainer, test_dataset, epoch=epoch,
                       loss_inspector=test_loss_inspector, save_pred_results=True, phase='test')


if __name__ == '__main__':
    if FLAGS.phase == "train":
        train()
    if FLAGS.phase == "test":
        if FLAGS.best_ep is not None:
            test(epoch=FLAGS.best_ep)
        else:
            test(epoch=100)
    #test(epoch=75)
    #test(epoch=150)
