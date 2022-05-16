""" modified from https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/evaluate_WarpError.py """

### python lib
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, argparse, glob, re, math, pickle, cv2
from os.path import exists, dirname, abspath, join
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
import flow_warping as Flow
from easydict import EasyDict
from tqdm import tqdm

import argparse
import datetime
import json
import pdb


def get_configs():
    opts = EasyDict()
    opts.phase =  'refined_final'  # 'refined_final'
    opts.channel = 'shading'  # 'reflect', need change to 'albedo' while search gt
    opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5/log/test-imgs_ep200_renamed'
    opts.data_dir = '/home/wzj/intrinsic/data/MPI'
    # opts.ref_dir = '/home/wzj/intrinsic/data/MPI-Sintel-complete'
    opts.ref_dir = '/home/wzj/intrinsic/data/MPI/origin'
    opts.synsets = ['market_5']
    opts.cuda_id = '0'
    opts.is_DVP = False  # is Deep Video Prior setting
    return opts


def run_demo(opts):
    # compute TCM (occlusion-aware video temporal consistency)
    opts.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda_id
    print(" ")
    print(opts)

    output_dir = os.path.join(opts.data_dir, opts.phase, 'TCM')

    ## flow warping layer
    image_size = (436, 1024)
    device = torch.device("cuda" if opts.cuda else "cpu")
    warp_op = Flow.Warper2d(image_size).to(device)

    ### load video list

    ### start evaluation
    TCM_all = []
    err_all = []
    err_ref_all = []

    for v in range(len(opts.synsets)):
        syn = opts.synsets[v]
        frame_dir = os.path.join(opts.data_dir, opts.phase, 'MPI-main-'+opts.channel)
        occ_dir = os.path.join(opts.data_dir, "occlusions", syn)
        flow_dir = os.path.join(opts.data_dir, "flow_hdf5")

        frame_list = sorted(glob.glob(os.path.join(frame_dir, syn+"*.png")))

        TCM = []
        errs = []
        errs_ref = []
        err = 0.0
        err_ref = 0.0
        for t in tqdm(range(1, len(frame_list))):
            ### frame path
            frame_path = frame_list[t-1]
            img_name = frame_path.split('/')[-1].split('.')[0]
            ### flow path
            flow_path = os.path.join(flow_dir, img_name+".h5")
            ### occlution path
            occ_name = 'frame_'+img_name.split('_')[-1]
            occ_path = os.path.join(occ_dir, occ_name+'.png')

            ### reference frame path
            ref1_path = os.path.join(opts.ref_dir, 'MPI-main-'+opts.channel, frame_list[t-1].split('/')[-1])
            ref2_path = os.path.join(opts.ref_dir, 'MPI-main-'+opts.channel, frame_list[t].split('/')[-1])


            ### load input images and flow
            img1 = Flow.load_img(frame_list[t-1])
            img2 = Flow.load_img(frame_list[t])
            flow_data = Flow.h5_reader(flow_path)

            img1_ref = Flow.load_img(ref1_path)
            img2_ref = Flow.load_img(ref2_path)

            # print("Evaluate TCM on %s-%s: video %d / %d, %s" %(opts.phase, opts.channel, v + 1, len(opts.synsets), img_name))

            ### load occlusion mask
            occ_mask = Flow.load_img(occ_path)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = Flow.img2tensor(img2).to(device)
                flow = Flow.img2tensor(flow_data).to(device)
                img2_ref = Flow.img2tensor(img2_ref).to(device)

                ## warp img2
                warp_img2 = warp_op(flow, img2)
                warp_ref2 = warp_op(flow, img2_ref)

                ## convert to numpy array
                warp_img2 = Flow.tensor2img(warp_img2)
                warp_ref2 = Flow.tensor2img(warp_ref2)

            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            diff_ref = np.multiply(warp_ref2 - img1_ref, noc_mask)
            tcm_cur = np.exp(-np.abs( np.sum(np.power(diff,2)) / (np.sum(np.power(diff_ref,2))+1e-9) - 1 ))
            TCM.append(tcm_cur)  # the larger, the better temporal consistency
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err = np.sum(np.square(diff)) / N
            err_ref = np.sum(np.square(diff_ref)) / N

            errs.append(err)
            errs_ref.append(err_ref)
        # err_all[v] = err / (len(frame_list) - 1)
        TCM_all.append(np.mean(TCM))
        err_all.append(np.mean(errs))
        err_ref_all.append(np.mean(errs_ref))

        metric_filename = os.path.join(output_dir, syn+'.txt')
        # pdb.set_trace()
        np.savetxt(metric_filename, np.concatenate([np.reshape(TCM,(-1,1)), np.reshape(errs,(-1,1)), np.reshape(errs_ref,(-1,1))], axis=1), fmt="%f")
    metric_filename2 = os.path.join(output_dir, 'total.txt')
    np.savetxt(metric_filename2, np.concatenate([np.reshape(TCM_all,(-1,1)), np.reshape(err_all,(-1,1)), np.reshape(err_ref_all,(-1,1))], axis=1), fmt="%f")
    
    return TCM, errs, errs_ref


def mu_std_consistency(opts):
    # compute temporal consistency based on lightness statistics (mu, std)

    print(opts)

    output_dir = os.path.join(opts.data_dir, opts.phase, 'TCM')

    ### start evaluation
    TCM_all = []
    err_all = []
    err_ref_all = []

    for v in range(len(opts.synsets)):
        syn = opts.synsets[v]
        frame_dir = os.path.join(opts.data_dir, opts.phase, 'MPI-main-'+opts.channel)
        occ_dir = os.path.join(opts.data_dir, "occlusions", syn)
        flow_dir = os.path.join(opts.data_dir, "flow_hdf5")

        frame_list = sorted(glob.glob(os.path.join(frame_dir, syn+"*.png")))

        TCM_mu = []
        TCM_std = []
        errs_mu = []
        errs_mu_ref = []
        errs_std = []
        errs_std_ref = []
        err_mu = 0.0
        err_mu_ref = 0.0
        err_std = 0.0
        err_std_ref = 0.0
        for t in tqdm(range(1, len(frame_list))):
            ### frame path
            frame_path = frame_list[t-1]
            img_name = frame_path.split('/')[-1].split('.')[0]

            ### reference frame path
            ref1_path = os.path.join(opts.ref_dir, 'MPI-main-'+opts.channel, frame_list[t-1].split('/')[-1])
            ref2_path = os.path.join(opts.ref_dir, 'MPI-main-'+opts.channel, frame_list[t].split('/')[-1])


            ### load input images and flow
            img1 = cv2.imread(frame_list[t-1])
            img2 = cv2.imread(frame_list[t])

            img1_ref = cv2.imread(ref1_path)
            img2_ref = cv2.imread(ref2_path)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

            img1_ref = cv2.cvtColor(img1_ref, cv2.COLOR_BGR2HSV)
            img2_ref = cv2.cvtColor(img2_ref, cv2.COLOR_BGR2HSV)

            ### get v-channel of hsv
            l1 = img1[..., 2]
            l2 = img2[..., 2]
            l1_ref = img1_ref[..., 2]
            l2_ref = img2_ref[..., 2]

            ## compute statistical error
            diff_mu = np.mean(l1) - np.mean(l2)
            diff_mu_ref = np.mean(l1_ref) - np.mean(l2_ref)
            diff_std = np.std(l1) - np.std(l2)
            diff_std_ref = np.std(l1_ref) - np.std(l2_ref)

            diff, diff_ref = diff_mu, diff_mu_ref
            tcm_mu = np.exp(-np.abs( np.sum(np.power(diff,2)) / (np.sum(np.power(diff_ref,2))+1e-9) - 1 ))
            TCM_mu.append(tcm_mu)  # the larger, the better temporal consistency
            
            err = np.sum(np.square(diff))
            err_ref = np.sum(np.square(diff_ref))

            errs_mu.append(err)
            errs_mu_ref.append(err_ref)

            diff, diff_ref = diff_std, diff_std_ref
            tcm_std = np.exp(-np.abs( np.sum(np.power(diff,2)) / (np.sum(np.power(diff_ref,2))+1e-9) - 1 ))
            TCM_std.append(tcm_std)  # the larger, the better temporal consistency
            
            err = np.sum(np.square(diff))
            err_ref = np.sum(np.square(diff_ref))

            errs_std.append(err)
            errs_std_ref.append(err_ref)
        # err_all[v] = err / (len(frame_list) - 1)
        TCM_all.append([np.mean(TCM_mu), np.mean(TCM_std)])
        err_all.append([np.mean(errs_mu), np.mean(errs_std)])
        err_ref_all.append([np.mean(errs_mu_ref), np.mean(errs_std_ref)])

        metric_filename = os.path.join(output_dir, syn+'_mu.txt')
        np.savetxt(metric_filename, np.concatenate([np.reshape(TCM_mu,(-1,1)), np.reshape(errs_mu,(-1,1)), np.reshape(errs_mu_ref,(-1,1))], axis=1), fmt="%f")
        metric_filename = os.path.join(output_dir, syn+'_std.txt')
        np.savetxt(metric_filename, np.concatenate([np.reshape(TCM_std,(-1,1)), np.reshape(errs_std,(-1,1)), np.reshape(errs_std_ref,(-1,1))], axis=1), fmt="%f")
    metric_filename2 = os.path.join(output_dir, '_mu_std_total.txt')
    np.savetxt(metric_filename2, np.concatenate([np.reshape(TCM_all,(-1,2)), np.reshape(err_all,(-1,2)), np.reshape(err_ref_all,(-1,2))], axis=1), fmt="%f")
    
    return [TCM_mu, TCM_std], [errs_mu, errs_std], [errs_mu_ref, errs_std_ref]


def eval_TCM(opts):
    # compute TCM (occlusion-aware video temporal consistency)
    # check the reconstructed results, frame-by-frame methods: [1)ours, 2)Direct Intrinsic, 3)Revisiting]
    # the results intrinsic images have different size, so we have to resize them after read
    # res_dir: 1) /home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints/log/test-imgs_epxxx_renamed
    opts.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda_id
    print(" ")
    print(opts)

    output_dir = os.path.join(opts.res_dir, 'TCM', "_".join(opts.synsets), opts.channel)
    if not exists(output_dir):
        print("create output_dir!... ", output_dir)
        os.makedirs(output_dir)

    ## flow warping layer
    image_size = (436, 1024)
    device = torch.device("cuda" if opts.cuda else "cpu")
    warp_op = Flow.Warper2d(image_size).to(device)

    ### load video list

    ### start evaluation
    TCM_all = []
    err_all = []
    err_ref_all = []

    for v in range(len(opts.synsets)):
        syn = opts.synsets[v]
        
        if not opts.is_DVP:
            frame_dir = os.path.join(opts.res_dir)
            frame_list = sorted(glob.glob(os.path.join(frame_dir, syn+"*"+opts.channel+"-pred.png")))
        else:
            frame_dir = os.path.join(opts.res_dir)
            frame_list = sorted(glob.glob(os.path.join(frame_dir, syn+"*")))

        occ_dir = os.path.join(opts.data_dir, "occlusions", syn)
        flow_dir = os.path.join(opts.data_dir, "flow_hdf5")

        TCM = []
        errs = []
        errs_ref = []
        err = 0.0
        err_ref = 0.0
        for t in tqdm(range(1, len(frame_list))):
            ### frame path
            img1_name = "_".join(frame_list[t-1].split('/')[-1].split('.')[0].split('_')[:4])
            img2_name = "_".join(frame_list[t].split('/')[-1].split('.')[0].split('_')[:4])
            ### flow path
            flow_path = os.path.join(flow_dir, img1_name+".h5")
            ### occlution path
            occ_name = 'frame_'+img1_name.split('_')[-1]
            occ_path = os.path.join(occ_dir, occ_name+'.png')

            ### reference frame path
            if opts.channel == 'reflect':
                channel_name = 'albedo'
            elif opts.channel == 'shading':
                channel_name = 'shading'
            ref1_path = os.path.join(opts.ref_dir, 'MPI-main-'+channel_name, img1_name+'.png')
            ref2_path = os.path.join(opts.ref_dir, 'MPI-main-'+channel_name, img2_name+'.png')

            ### load input images and flow
            # pdb.set_trace()
            # if not opts.is_DVP:
            #     img1 = Flow.load_img_and_resize(frame_list[t-1], image_size)
            #     img2 = Flow.load_img_and_resize(frame_list[t], image_size)
            # else:
            #     img1 = Flow.load_img_and_extend_resize(frame_list[t-1], image_size)
            #     img2 = Flow.load_img_and_extend_resize(frame_list[t], image_size)
 
            img1 = Flow.load_img_and_resize(frame_list[t-1], image_size)
            img2 = Flow.load_img_and_resize(frame_list[t], image_size)

            flow_data = Flow.h5_reader(flow_path)

            img1_ref = Flow.load_img(ref1_path)
            img2_ref = Flow.load_img(ref2_path)

            ### load occlusion mask
            occ_mask = Flow.load_img(occ_path)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = Flow.img2tensor(img2).to(device)
                flow = Flow.img2tensor(flow_data).to(device)
                img2_ref = Flow.img2tensor(img2_ref).to(device)

                ## warp img2
                warp_img2 = warp_op(flow, img2)
                warp_ref2 = warp_op(flow, img2_ref)

                ## convert to numpy array
                warp_img2 = Flow.tensor2img(warp_img2)
                warp_ref2 = Flow.tensor2img(warp_ref2)

            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            diff_ref = np.multiply(warp_ref2 - img1_ref, noc_mask)
            tcm_cur = np.exp(-np.abs( np.sum(np.power(diff,2)) / (np.sum(np.power(diff_ref,2))+1e-9) - 1 ))
            TCM.append(tcm_cur)  # the larger, the better temporal consistency
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err = np.sum(np.square(diff)) / N
            err_ref = np.sum(np.square(diff_ref)) / N

            errs.append(err)
            errs_ref.append(err_ref)
    return TCM, errs, errs_ref


def compare_framewise_methods(synname='market_5', channel='shading'):
    opts = get_configs()
    opts.synsets = [synname]
    opts.channel = channel

    opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5/log/test-imgs_ep195_renamed'
    tcms_1, errs_1, errs_ref = eval_TCM(opts)

    # opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-direct_intrinsics-MPI-main-RD-sceneSplit/log/test-imgs_ep225_renamed'
    # tcms_2, errs_2, _ = eval_TCM(opts)

    # opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/IntrinsicImage-master/results/test/RD_MPI-main-clean-video-renamed'
    # tcms_3, errs_3, _ = eval_TCM(opts)

    # #########
    opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5/log/test-imgs_ep200_renamed'
    tcms_2, errs_2, _ = eval_TCM(opts)
    tcms_3, errs_3 = tcms_2, errs_2
    # #########

    k1 = errs_ref[0] / errs_1[0]
    errs_1_rescale = np.array(errs_1) * k1
    tcm_1_rescale = np.exp(-np.abs(np.array(errs_1_rescale)/np.array(errs_ref) - 1))

    k2 = errs_ref[0] / errs_2[0]
    errs_2_rescale = np.array(errs_2) * k2
    tcm_2_rescale = np.exp(-np.abs(np.array(errs_2_rescale)/np.array(errs_ref) - 1))

    k3 = errs_ref[0] / errs_3[0]
    errs_3_rescale = np.array(errs_3) * k3
    tcm_3_rescale = np.exp(-np.abs(np.array(errs_3_rescale)/np.array(errs_ref) - 1))

    print('TCM comparison:')
    print('TCM 1 rescale: ', np.mean(tcm_1_rescale))
    print('TCM 2 rescale: ', np.mean(tcm_2_rescale))
    print('TCM 3 rescale: ', np.mean(tcm_3_rescale))

    outfig_dir = os.path.join('./frame-by-frame-ep200_based-on-195', "_".join(opts.synsets), opts.channel)
    if not exists(outfig_dir):
        os.makedirs(outfig_dir)
        print('created outfig_dir: ', outfig_dir)

    fig = plt.figure()
    plt.hold(True)
    plt.plot(errs_1_rescale, 'c-')
    plt.plot(errs_2_rescale, 'g-')
    plt.plot(errs_3_rescale, 'b-')
    plt.plot(errs_ref, 'r-')
    fig.canvas.draw()
    plt.savefig(join(outfig_dir, 'errs.png'))
    plt.close()

    fig = plt.figure()
    plt.hold(True)
    plt.plot(tcm_1_rescale, 'c-')
    plt.plot(tcm_2_rescale, 'g-')
    plt.plot(tcm_3_rescale, 'b-')
    fig.canvas.draw()
    plt.savefig(join(outfig_dir, 'tcms.png'))
    plt.close()

    with open(join(outfig_dir, 'TCM_rescale.txt'), 'w+') as f:
        f.write('TCM 1 rescale: {:.6f}\nTCM 2 rescale: {:.6f}\nTCM 3 rescale: {:.6f}\n\nTCM 1: {:.6f}\nTCM 2: {:.6f}\nTCM 3: {:.6f}'.format(
            np.mean(tcm_1_rescale), np.mean(tcm_2_rescale), np.mean(tcm_3_rescale), np.mean(tcms_1), np.mean(tcms_2), np.mean(tcms_3)
        ))


def compare_DVP_and_ours_fbf(synname='market_5', channel='shading'):
    opts = get_configs()
    opts.synsets = [synname]
    opts.channel = channel

    opts.is_DVP = False
    opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5/log/test-imgs_ep195_renamed'
    tcms_1, errs_1, errs_ref = eval_TCM(opts)

    opts.is_DVP = True
    # TODO: change this opts.res_dir to compare Intrinsic_IRT0_initial0_ep200
    # opts.res_dir = '/home/wzj/intrinsic/3rdparty/deep-video-prior/result/Intrinsic_IRT0_initial0/'+opts.channel+'/0025'
    opts.res_dir = '/home/wzj/intrinsic/3rdparty/deep-video-prior/result/Intrinsic_IRT0_initial0_ep200/'+opts.channel+'/0025'
    tcms_2, errs_2, _ = eval_TCM(opts)

    k1 = errs_ref[0] / errs_1[0]
    errs_1_rescale = np.array(errs_1) * k1
    tcm_1_rescale = np.exp(-np.abs(np.array(errs_1_rescale)/np.array(errs_ref) - 1))

    k2 = errs_ref[0] / errs_2[0]
    errs_2_rescale = np.array(errs_2) * k2
    tcm_2_rescale = np.exp(-np.abs(np.array(errs_2_rescale)/np.array(errs_ref) - 1))

    print('TCM comparison:')
    print('TCM 1: ', np.mean(tcm_1_rescale))
    print('TCM 2: ', np.mean(tcm_2_rescale))

    print('TCM origin comparison:')
    print('TCM 1 origin: ', np.mean(tcms_1))
    print('TCM 2 origin: ', np.mean(tcms_2))

    outfig_dir = os.path.join('./DVP-ep200_and_ours_fbf-ep195', "_".join(opts.synsets), opts.channel)
    if not exists(outfig_dir):
        os.makedirs(outfig_dir)
        print('created outfig_dir: ', outfig_dir)

    fig = plt.figure()
    plt.hold(True)
    plt.plot(errs_1_rescale, 'c-')
    plt.plot(errs_2_rescale, 'b-')
    plt.plot(errs_ref, 'r-')
    fig.canvas.draw()
    plt.savefig(join(outfig_dir, 'errs.png'))
    plt.close()

    fig = plt.figure()
    plt.hold(True)
    plt.plot(tcm_1_rescale, 'c-')
    plt.plot(tcm_2_rescale, 'b-')
    fig.canvas.draw()
    plt.savefig(join(outfig_dir, 'tcms.png'))
    plt.close()

    with open(join(outfig_dir, 'TCM_rescale.txt'), 'w+') as f:
        f.write('TCM 1 rescale: {:.6f}\nTCM 2 rescale: {:.6f}\n\nTCM 1: {:.6f}\nTCM 2: {:.6f}'.format(
            np.mean(tcm_1_rescale), np.mean(tcm_2_rescale), np.mean(tcms_1), np.mean(tcms_2)
        ))


def compare_our_video_methods(synname='market_5', channel='shading'):
    opts = get_configs()
    opts.synsets = [synname]
    opts.channel = channel

    opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5/log/test-imgs_ep195_renamed'
    tcms_1, errs_1, errs_ref = eval_TCM(opts)

    # opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/framewise-ckpoints-direct_intrinsics-MPI-main-RD-sceneSplit/log/test-imgs_ep225_renamed'
    # tcms_2, errs_2, _ = eval_TCM(opts)

    # opts.res_dir = '/home/wzj/intrinsic/intrinsic_image_project/IntrinsicImage-master/results/test/RD_MPI-main-clean-video-renamed'
    # tcms_3, errs_3, _ = eval_TCM(opts)

    # #########
    # code/video-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-1.0-lambda_s-1.0
    opts.res_dir = '/home/wzj/intrinsic/code/video-ckpoints-fromV8-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-10.0-lambda_s-5.0/log/test-imgs_ep5-renamed'
    tcms_2, errs_2, _ = eval_TCM(opts)
    # #########

    k1 = errs_ref[0] / errs_1[0]
    errs_1_rescale = np.array(errs_1) * k1
    tcm_1_rescale = np.exp(-np.abs(np.array(errs_1_rescale)/np.array(errs_ref) - 1))

    k2 = errs_ref[0] / errs_2[0]
    errs_2_rescale = np.array(errs_2) * k2
    tcm_2_rescale = np.exp(-np.abs(np.array(errs_2_rescale)/np.array(errs_ref) - 1))

    print('TCM comparison:')
    print('ref TCM 1 rescale: ', np.mean(tcm_1_rescale))
    print('out TCM 2 rescale: ', np.mean(tcm_2_rescale))

    outfig_dir = os.path.join('./ours_video-ep5_r10s5', "_".join(opts.synsets), opts.channel)
    if not exists(outfig_dir):
        os.makedirs(outfig_dir)
        print('created outfig_dir: ', outfig_dir)

    fig = plt.figure()
    plt.hold(True)
    plt.plot(errs_1_rescale, 'c-')
    plt.plot(errs_2_rescale, 'b-')
    plt.plot(errs_ref, 'r-')
    fig.canvas.draw()
    plt.savefig(join(outfig_dir, 'errs.png'))
    plt.close()

    fig = plt.figure()
    plt.hold(True)
    plt.plot(tcm_1_rescale, 'c-')
    plt.plot(tcm_2_rescale, 'b-')
    fig.canvas.draw()
    plt.savefig(join(outfig_dir, 'tcms.png'))
    plt.close()

    with open(join(outfig_dir, 'TCM_rescale.txt'), 'w+') as f:
        f.write('ref TCM 1 rescale: {:.6f}\nout TCM 2 rescale: {:.6f}\n\nref TCM 1: {:.6f}\nout TCM 2: {:.6f}'.format(
            np.mean(tcm_1_rescale), np.mean(tcm_2_rescale), np.mean(tcms_1), np.mean(tcms_2)
        ))


def run_eval_dataset(opts):
    ## flow warping layer
    image_size = (436, 1024)
    device = torch.device("cuda" if opts.cuda else "cpu")
    warp_op = Flow.Warper2d(image_size).to(device)

    ### start evaluation
    TCM_all = []
    err_all = []
    err_ref_all = []

    for v in range(len(opts.synsets)):
        syn = opts.synsets[v]
        ### reference frame path
        if opts.channel == 'reflect':
            channel_name = 'albedo'
        elif opts.channel == 'shading':
            channel_name = 'shading'

        frame_dir = os.path.join(opts.data_dir, opts.phase, 'MPI-main-'+channel_name)
        occ_dir = os.path.join(opts.data_dir, "occlusions", syn)
        flow_dir = os.path.join(opts.data_dir, "flow_hdf5")

        frame_list = sorted(glob.glob(os.path.join(frame_dir, syn+"*.png")))

        TCM = []
        errs = []
        errs_ref = []
        err = 0.0
        err_ref = 0.0
        for t in tqdm(range(1, len(frame_list))):
            ### frame path
            frame_path = frame_list[t-1]
            img_name = frame_path.split('/')[-1].split('.')[0]
            ### flow path
            flow_path = os.path.join(flow_dir, img_name+".h5")
            ### occlution path
            occ_name = 'frame_'+img_name.split('_')[-1]
            occ_path = os.path.join(occ_dir, occ_name+'.png')

            ### reference frame path
            ref1_path = os.path.join(opts.ref_dir, 'MPI-main-'+channel_name, frame_list[t-1].split('/')[-1])
            ref2_path = os.path.join(opts.ref_dir, 'MPI-main-'+channel_name, frame_list[t].split('/')[-1])

            ### load input images and flow
            img1 = Flow.load_img(frame_list[t-1])
            img2 = Flow.load_img(frame_list[t])
            flow_data = Flow.h5_reader(flow_path)

            img1_ref = Flow.load_img(ref1_path)
            img2_ref = Flow.load_img(ref2_path)

            # print("Evaluate TCM on %s-%s: video %d / %d, %s" %(opts.phase, opts.channel, v + 1, len(opts.synsets), img_name))

            ### load occlusion mask
            occ_mask = Flow.load_img(occ_path)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = Flow.img2tensor(img2).to(device)
                flow = Flow.img2tensor(flow_data).to(device)
                img2_ref = Flow.img2tensor(img2_ref).to(device)

                ## warp img2
                warp_img2 = warp_op(flow, img2)
                warp_ref2 = warp_op(flow, img2_ref)

                ## convert to numpy array
                warp_img2 = Flow.tensor2img(warp_img2)
                warp_ref2 = Flow.tensor2img(warp_ref2)

            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            diff_ref = np.multiply(warp_ref2 - img1_ref, noc_mask)
            tcm_cur = np.exp(-np.abs( np.sum(np.power(diff,2)) / (np.sum(np.power(diff_ref,2))+1e-9) - 1 ))
            TCM.append(tcm_cur)  # the larger, the better temporal consistency
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err = np.sum(np.square(diff)) / N
            err_ref = np.sum(np.square(diff_ref)) / N

            errs.append(err)
            errs_ref.append(err_ref)
    return TCM, errs, errs_ref


def eval_dataset_refine_effect(synname='market_5', channel='shading'):
    opts = get_configs()
    opts.synsets = [synname]
    opts.channel = channel
    opts.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda_id

    # >>> before temporal consistency refine
    opts.phase = 'refined_gs'
    tcm_gs, errs_gs, errs_ref = run_eval_dataset(opts)
    # rescale errs
    k_gs = errs_ref[0] / errs_gs[0]
    errs_gs_rescale = np.array(errs_gs) * k_gs
    tcm_gs_new = np.exp(-np.abs(np.array(errs_gs_rescale) / np.array(errs_ref) -1))

    # >>> after temporal consistency refine
    opts.phase = 'refined_final'
    tcm_gf, errs_gf, _ = run_eval_dataset(opts)
    # rescale errs
    k_gf = errs_ref[0] / errs_gf[0]
    errs_gf_rescale = np.array(errs_gf) * k_gf
    tcm_gf_new = np.exp(-np.abs(np.array(errs_gf_rescale) / np.array(errs_ref) -1))

    
    outfig_dir = os.path.join('./eval_dataset_refine_effect', "_".join(opts.synsets), opts.channel)
    if not exists(outfig_dir):
        os.makedirs(outfig_dir)
        print('created outfig_dir: ', outfig_dir)

    
    with open(join(outfig_dir, 'TCM_eval.txt'), 'w+') as f:
        f.write('TCM-rescale before: {:.6f}\nTCM-rescale after : {:.6f}\n\nTCM before: {:.6f}\nTCM after : {:.6f}'.format(
            np.mean(tcm_gs_new), np.mean(tcm_gf_new), np.mean(tcm_gs), np.mean(tcm_gf)
        ))


if __name__ == "__main__":
    # compare_framewise_methods()
    parser = argparse.ArgumentParser(description='Temporal Consistency Metric')
    parser.add_argument('--synname',           type=str,     default="market_5")
    parser.add_argument('--channel',           type=str,     default="shading",  choices=["shading", "reflect"])
    ARGS = parser.parse_args()

    # compare_DVP_and_ours_fbf(synname=ARGS.synname, channel=ARGS.channel)
    # compare_framewise_methods(synname=ARGS.synname, channel=ARGS.channel)
    # compare_our_video_methods(synname=ARGS.synname, channel=ARGS.channel)

    # eval dataset
    eval_dataset_refine_effect(synname=ARGS.synname, channel=ARGS.channel)
