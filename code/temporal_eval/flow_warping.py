""" demo of warping using optical flow """
import os
from os.path import abspath, dirname, join
from glob import glob
import cv2
import numpy as np
import h5py

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb


def h5_reader(path):
    f = h5py.File(path, 'r')
    flo = f['data'][:]
    order = list(range(np.ndim(flo)))[::-1]
    return np.transpose(flo, order)


def load_img(path):
    img = cv2.imread(path)  # [H,W,C]
    img = img.astype("float") / 255.0
    cur_img = np.zeros(img.shape)
    # (bgr)-->(rgb)
    cur_img[:,:,0] = img[:,:,2]
    cur_img[:,:,1] = img[:,:,1]
    cur_img[:,:,2] = img[:,:,0]
    return cur_img


def load_img_and_resize(path, tar_size=((436, 1024))):
    img = cv2.imread(path)  # [H,W,C]
    dim = (tar_size[1], tar_size[0])
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    assert img.shape[:2] == tar_size, 'resized shape should equal to target'
    img = img.astype("float") / 255.0
    cur_img = np.zeros(img.shape)
    # (bgr)-->(rgb)
    cur_img[:,:,0] = img[:,:,2]
    cur_img[:,:,1] = img[:,:,1]
    cur_img[:,:,2] = img[:,:,0]
    return cur_img


def load_img_and_extend_resize(path, tar_size=((436, 1024))):
    img_small = cv2.imread(path)  # [h,w,c]
    [h,w,c] = img_small.shape
    [H,W,C] = [327, 768, 3]
    img = np.zeros((H,W,C), dtype=img_small.dtype)
    img[:h, :w, :] = img_small

    dim = (tar_size[1], tar_size[0])
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    assert img.shape[:2] == tar_size, 'resized shape should equal to target'
    img = img.astype("float") / 255.0
    cur_img = np.zeros(img.shape)
    # (bgr)-->(rgb)
    cur_img[:,:,0] = img[:,:,2]
    cur_img[:,:,1] = img[:,:,1]
    cur_img[:,:,2] = img[:,:,0]
    return cur_img


def img2tensor(img):
    img_t = torch.from_numpy(img.astype("float"))
    img_t = img_t.permute(2,0,1).unsqueeze(0)
    return img_t


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1,2,0))
    return img


def vis_demo(flow_files):
    flow_path = flow_files[0]
    flo = h5_reader(flow_path)
    u_flo = np.array(np.tile(flo[:,:,0:1], [1,1,3]), dtype=np.uint8)
    v_flo = np.array(np.tile(flo[:,:,1:2], [1,1,3]), dtype=np.uint8)

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Image(z=u_flo), 1,1)
    fig.add_trace(go.Image(z=v_flo), 2,1)
    fig.show()


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
        return output #*mask


def run_demo():
    root_dir = dirname(dirname(dirname(abspath(__file__))))
    flow_dir = join(root_dir, 'data', 'MPI', 'flow_hdf5')

    flow_files = sorted(glob(join(flow_dir, '*.h5')))
    # pdb.set_trace()
    image_size = (436, 1024)
    warp_op = Warper2d(image_size)
    # vis_demo(flow_files)
    # for i in range(len(flow_files)-1):
    i = 441  # market_5 0001
    flo_path = flow_files[i]

    name = flo_path.split('/')[-1].split('.')[0]
    synname = '_'.join(name.split('_')[0:2])
    img_path = join(root_dir, 'data', 'MPI', 'origin', 'MPI-main-albedo', name+'.png')
    frame_num = name.split('_')[-1]
    print(name)
    # read flow
    flow_data = h5_reader(flo_path)
    input_flow = torch.from_numpy(flow_data.astype("float"))
    input_flow = input_flow.permute(2,0,1).unsqueeze(0)  # (1,C,H,W)
    # read image
    cur_img = load_img(img_path)  # [H,W,C]
    input_img = torch.from_numpy(cur_img)
    input_img = input_img.permute(2,0,1).unsqueeze(0)  # (1,C,H,W)
    img_size = input_img.shape[2:4]

    next_name = flow_files[i+1].split('/')[-1].split('.')[0]
    next_img_path = join(root_dir, 'data', 'MPI', 'origin', 'MPI-main-albedo', next_name+'.png')
    next_img = load_img(next_img_path)
    input_img2 = torch.from_numpy(next_img).permute(2,0,1).unsqueeze(0)

    # warping
    assert img_size == image_size, 'img size not fit'
    output_img = warp_op(input_flow, input_img2) # sample from img2 to reconstruct img1
    warped_img = output_img.squeeze().permute(1,2,0).numpy()  # sample from coordinates+flow from next image

    # need occlusions images to perfectly reconstruct the valid region
    mask_path = join(root_dir, 'data', 'MPI', 'occlusions', synname, 'frame_{:s}.png'.format(frame_num))  # the occlusion is from i to i+1 th frame.
    mask = load_img(mask_path)

    # fig = make_subplots(rows=3, cols=1,
    #                     subplot_titles=("current", "warped from next", "next"))
    # fig.add_trace(go.Image(z=(cur_img*255).astype("uint8")), 1,1)
    # fig.add_trace(go.Image(z=((warped_img*(1-mask)*255).astype("uint8"))), 2,1)
    # fig.add_trace(go.Image(z=(next_img*255).astype("uint8")), 3,1)
    # fig.show()
    # pdb.set_trace()
    return cur_img, warped_img*(1-mask), next_img


if __name__ == '__main__':
    run_demo()
