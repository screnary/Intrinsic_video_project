"""
Script for loading paired data for low-level vision
"""

import os
import torch
import random
import os.path
from os.path import exists, join, dirname, abspath
import numpy as np
import h5py
from skimage import io
import torch.utils.data as data
from skimage.transform import resize, rotate
from torchvision.datasets.folder import is_image_file

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
import pdb


class Warper2d(nn.Module):
    def __init__(self, img_size):
        super(Warper2d, self).__init__()
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        img_src: [B, 1, H1, W1] (source image used for prediction, size 32)
        img_smp: [B, 1, H2, W2] (image for sampling, size 44)
        flow: [B, 2, H1, W1] flow predicted from source image pair
        """
        self.img_size = img_size  # H and W, tuple
        H, W = img_size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W)
        yy = yy.view(1,H,W)
        self.grid = torch.cat((xx,yy),0).float() # [2, H, W]
            
    def forward(self, flow, img):
        grid = self.grid.repeat(flow.shape[0],1,1,1) #[bs, 2, H, W]
        if img.is_cuda:
            grid = grid.cuda()
        if flow.shape[2:]!=img.shape[2:]:
            pad = int((img.shape[2] - flow.shape[2]) / 2)
            flow = F.pad(flow, [pad]*4, 'replicate')#max_disp=6, 32->44
        vgrid = Variable(grid, requires_grad = False) + flow
 
        # scale grid to [-1,1] 
        H, W = self.img_size  # (436, 1024)
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/(W-1)-1.0 #max(W-1,1)
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/(H-1)-1.0 #max(H-1,1)
        # pdb.set_trace()
        # >>> if img_size is scalar, is square shape
        # vgrid = 2.0*vgrid/(self.img_size-1)-1.0 #max(W-1,1)
        # <<< if img_size is scalar, is square shape
 
        vgrid = vgrid.permute(0,2,3,1)
        output = F.grid_sample(img, vgrid, align_corners=False)
        # mask = Variable(torch.ones(img.size())).cuda()
        # mask = F.grid_sample(mask, vgrid)
       
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        return output  #*mask


def h5_reader(path):
    f = h5py.File(path, 'r')
    flo = f['data'][:]
    order = list(range(np.ndim(flo)))[::-1]
    return np.transpose(flo, order)


def default_loader(path):
    file_ext = path.split('.')[-1]
    if file_ext in ['png', 'jpg']:  # is image
        return io.imread(path)
    elif file_ext == 'h5':
        return h5_reader(path)


def make_dataset(root, file_name):
    images = []
    pwd_file = os.path.join(root, file_name)

    if os.path.isfile(pwd_file):
        # the file must record each image's full path
        with open(pwd_file, 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip()
                images.append(os.path.join(root, line))
    else:
        assert os.path.isdir(pwd_file), f'{pwd_file} must be a folder or file.'
        names = os.listdir(pwd_file)
        images = [os.path.join(pwd_file, x) for x in names if is_image_file(x)]
    
    sorted(images)
    # WARNING: use the top 1000 images
    images = images[:1000]
    return images


class LowLevelVisionFolder(data.Dataset):
    def __init__(self, params, 
                loader=default_loader, image_names=None, is_train=True) -> None:
        super().__init__()
        self.image_names    = image_names
        self.root           = params.data_root
        self.loader         = loader
        self.height         = params.image_height
        self.width          = params.image_width
        self.rotation_range = 15.0 # deg
        self.is_train       = is_train

        img_dict = {}
        for i in self.image_names:
            img_dict[i] = make_dataset(self.root, i)

        # paired data, all images' length must be same
        assert all([len(x) == len(img_dict[image_names[0]]) for x in img_dict.values()])
        
        self.image_dict = img_dict

        # print(self.image_dict)
    
    def __len__(self) -> int:
        return len(self.image_dict[self.image_names[0]])
    
    def data_arguments(self, img, mode, random_pos, random_angle, random_flip):
        if random_flip > 0.5:
            img = np.fliplr(img)

        img = rotate(img, random_angle, order = mode)
        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        img = resize(img, (self.height, self.width), order=mode)

        return img
    
    def load_images(self, index, use_da=True):
        images = {}
        for k in self.image_names:
            images[k] = self.loader(self.image_dict[k][index]) /255.

        file_name = os.path.basename(self.image_dict[self.image_names[0]][index])
        ori_h, ori_w = images[self.image_names[0]].shape[:2]

        if use_da:
            random_flip    = random.random()
            random_angle   = (random.random() - 0.5) * self.rotation_range
            random_start_x = random.randint(0, 9)
            random_start_y = random.randint(0, 9)

            random_pos = [random_start_y, random_start_y + ori_h - 10, random_start_x,
                          random_start_x + ori_w - 10]
            
            for k in images.keys():
                images[k] = self.data_arguments(images[k],
                    mode=1, random_pos=random_pos, random_angle=random_angle,
                    random_flip=random_flip)
        else:
            for k in images.keys():
                images[k] = resize(images[k], (self.height, self.width), order=1)

        images['filename'] = file_name

        return images
    
    def __getitem__(self, index: int):
        images = self.load_images(index, use_da=self.is_train)

        for k in images.keys():
            if type(images[k]) is str:
                continue
            images[k][images[k] < 1e-4] = 1e-4
            images[k] = torch.from_numpy(np.transpose(images[k], (2, 0, 1))).contiguous().float()   # * 255 - 255 / 2.
        
        return images


class IntrinsicImageList(data.Dataset):
    """
    A loader for loading intrinsic image from split files (train.txt, test.txt)
    """
    def __init__(self, params, 
                loader=default_loader, image_names=None, mode='train') -> None:
        super().__init__()
        self.image_names    = image_names  # folders to read images, [clean, albedo, shading, occlusions, flow]
        self.root           = params.data_root
        self.loader         = loader
        self.height         = params.image_height  # 320
        self.width          = params.image_width  # 320
        self.rotation_range = 15.0 # deg
        self.is_train       = True if mode=='train' else False
        self.mode = mode

        if self.mode == 'train':
            split_fn = 'MPI_video-320-train.txt'
        elif self.mode == 'valid':
            split_fn = 'MPI_video-320-valid.txt'
        elif self.mode == 'test':
            split_fn = 'MPI_video-fullsize-test.txt'
        else:
            raise NotImplementedError("mode should be in ['train', 'valid', 'test']")
        
        self.name_list = []
        with open(os.path.join(self.root, 'MPI_Video', split_fn), 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                img_name = line.strip().split('.')[0]
                self.name_list.append(img_name)
        
        if params.fast_check:
            # WARNING: use the top 30 images
            self.name_list = self.name_list[:30]
            print('FAST CHECK WARNING: use the top 30 images')

        if self.mode == 'train' or self.mode == 'valid':
            dataset_dir = join(self.root, 'MPI_Video')

            img_dict = {}
            img_dict['input_1'] = []
            img_dict['input_2'] = []
            img_dict['albedo_1'] = []
            img_dict['albedo_2'] = []
            img_dict['shading_1'] = []
            img_dict['shading_2'] = []
            img_dict['occ'] = []
            img_dict['flow'] = []
            for img_n in self.name_list:
                img_dict['input_1'].append(join(dataset_dir, 'MPI-main-clean', 'from', img_n+'.png'))
                img_dict['input_2'].append(join(dataset_dir, 'MPI-main-clean', 'to', img_n+'.png'))
                img_dict['albedo_1'].append(join(dataset_dir, 'MPI-main-albedo', 'from', img_n+'.png'))
                img_dict['albedo_2'].append(join(dataset_dir, 'MPI-main-albedo', 'to', img_n+'.png'))
                img_dict['shading_1'].append(join(dataset_dir, 'MPI-main-shading', 'from', img_n+'.png'))
                img_dict['shading_2'].append(join(dataset_dir, 'MPI-main-shading', 'to', img_n+'.png'))
                img_dict['occ'].append(join(dataset_dir, 'occlusions', img_n+'.png'))
                img_dict['flow'].append(join(dataset_dir, 'flow', img_n+'.h5'))
        elif self.mode == 'test':
            dataset_dir = join(self.root, 'MPI', 'refined_final')

            img_dict = {}
            img_dict['input_1'] = []
            img_dict['albedo_1'] = []
            img_dict['shading_1'] = []
            for img_n in self.name_list:
                img_dict['input_1'].append(join(dataset_dir, 'MPI-main-clean', img_n+'.png'))
                img_dict['albedo_1'].append(join(dataset_dir, 'MPI-main-albedo', img_n+'.png'))
                img_dict['shading_1'].append(join(dataset_dir, 'MPI-main-shading', img_n+'.png'))

        self.image_dict = img_dict

    def data_arguments(self, img, random_flip):
        # do not rotate or risize
        if random_flip > 0.5:
            img = np.fliplr(img)

        # img = rotate(img, random_angle, order = mode)
        # img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        # img = resize(img, (self.height, self.width), order=mode)  # donot resize

        return img

    def __len__(self):
        return len(self.name_list)

    def load_images(self, index, use_da=False):
        images = {}  # ['input', 'albedo', 'shading', 'occ', 'flow'], flow is not an image

        for k in self.image_names:
            if k not in ['flow']:
                images[k] = self.loader(self.image_dict[k][index]) /255.
            elif k =='flow':
                images[k] = self.loader(self.image_dict[k][index])


        file_name = os.path.basename(self.image_dict[self.image_names[0]][index])
        ori_h, ori_w = images[self.image_names[0]].shape[:2]
        is_flip = 'wo_flip'
        if use_da:
            random_flip = random.random()
            if random_flip > 0.5:
                is_flip = 'fliped'
            for k in images.keys():
                images[k] = self.data_arguments(images[k], random_flip=random_flip)
                if k == 'flow' and random_flip>0.5:
                    images[k][:,:,0] = -1*images[k][:,:,0]  # reverse direction of delta_x

        images['filename'] = file_name+'-'+is_flip
        return images

    def __getitem__(self, index: int):
        images = self.load_images(index, use_da=self.is_train)

        for k in images.keys():
            if type(images[k]) is str:
                continue
            if k not in ['flow', 'occ']:
                images[k][images[k] < 1e-4] = 1e-4
            # images[k] = torch.from_numpy(np.transpose(images[k], (2, 0, 1))).contiguous().float()   # * 255 - 255 / 2.
            images[k] = torch.from_numpy(images[k].astype("float")).permute(2,0,1)
        return images


def gen_train_valid_list():
    # split: from origin list
    train_file_path = '/home/wzj/intrinsic/intrinsic_image_project/datasets/MPI/MPI_main_sceneSplit-fullsize-NoDefect-train.txt'
    valid_file_path = '/home/wzj/intrinsic/intrinsic_image_project/datasets/MPI/MPI_main_sceneSplit-fullsize-NoDefect-test.txt'

    save_root = '/home/wzj/intrinsic/data/MPI_Video'
    train_f = os.path.join(save_root, 'MPI_video-320-train.txt')
    valid_f = os.path.join(save_root, 'MPI_video-320-valid.txt')
    test_f = os.path.join(save_root, 'MPI_video-fullsize-test.txt')

    train_list = []
    with open(train_file_path, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            img_name = line.split('.')[0]
            image_names = sorted([x for x in os.listdir(os.path.join(save_root, 'occlusions')) if x.startswith(img_name)])
            train_list.extend(image_names)

    valid_list = []
    with open(valid_file_path, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            img_name = line.split('.')[0]
            image_names = sorted([x for x in os.listdir(os.path.join(save_root, 'occlusions')) if x.startswith(img_name)])
            valid_list.extend(image_names)
    
    with open(train_f, 'w+') as fid:
        fid.writelines(['{}\n'.format(img_n) for img_n in train_list])
    
    with open(valid_f, 'w+') as fid:
        fid.writelines(['{}\n'.format(img_n) for img_n in valid_list])
    
    shutil.copy(valid_file_path, test_f)

##################################################################################
# Test code
##################################################################################

from easydict import EasyDict
# from  matplotlib import pyplot as plt


def test_dataloder():
    TRAIN_SET = EasyDict({
        'data_root': r'/home/wzj/intrinsic/data',
        'image_height': 320,
        'image_width':  320,
        'mode': 'train'
    })

    if TRAIN_SET.mode in ['train', 'valid']:
        IMAGE_NAMES = [ 'input_1', 'input_2', 
                        'albedo_1', 'albedo_2', 
                        'shading_1', 'shading_2',
                        'occ', 'flow']
    elif TRAIN_SET.mode == 'test':
        IMAGE_NAMES = ['input_1', 'albedo_1', 'shading_1']

    train_set = IntrinsicImageList(params=TRAIN_SET, image_names=IMAGE_NAMES, mode=TRAIN_SET.mode)
    data = train_set[0]
    tr_i1 = np.uint8(np.transpose(data['input_1'].detach().cpu().numpy(), axes=[1, 2, 0]) * 255)
    tr_a1  = np.uint8(np.transpose(data['albedo_1'].detach().cpu().numpy(), axes=[1, 2, 0]) * 255)
    tr_a2  = np.uint8(np.transpose(data['albedo_2'].detach().cpu().numpy(), axes=[1, 2, 0]) * 255)
    tr_s1 = np.uint8(np.transpose(data['shading_1'].detach().cpu().numpy(), axes=[1, 2, 0]) * 255)
    tr_occ  = np.float32(np.transpose(data['occ'].detach().cpu().numpy(), axes=[1, 2, 0]))


    print(tr_i1.shape)
    print(tr_s1.shape)
    print(data['filename'])
    warp_op = Warper2d((TRAIN_SET.image_height, TRAIN_SET.image_width))
    input_flow = data['flow'].unsqueeze(0)
    input_img2 = data['albedo_2'].unsqueeze(0)
    output_img = warp_op(input_flow, input_img2)
    warped_img = output_img.squeeze().permute(1,2,0).numpy()

    # has black edges on image boarders, need to ignore this
    mask = warped_img[:,:,0] < 1e-4
    tr_occ = tr_occ + np.expand_dims(np.float32(mask),-1)

    # plt.subplot(121); plt.imshow(tr_img); plt.title('Train Input')
    # plt.subplot(122); plt.imshow(tr_gt);  plt.title('Train GT')
    # plt.show()


if __name__ == "__main__":
    test_dataloder()