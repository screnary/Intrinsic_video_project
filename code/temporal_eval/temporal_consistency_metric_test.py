""" modified from https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/evaluate_WarpError.py """

### python lib
import os, sys, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
import flow_warping as Flow
from easydict import EasyDict
from tqdm import tqdm

import pdb


def run_demo():
    opts = EasyDict()
    opts.phase =  'refined_final'  # 'refined_final'
    opts.channel = 'shading'
    opts.data_dir = '/home/wzj/intrinsic/data/MPI'
    # opts.ref_dir = '/home/wzj/intrinsic/data/MPI-Sintel-complete'
    opts.ref_dir = '/home/wzj/intrinsic/data/MPI/origin'
    opts.synsets = ['market_5']
    opts.cuda_id = '0'
    opts.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda_id

    print(opts)

    output_dir = os.path.join(opts.data_dir, opts.phase, 'TCM')

    ### print average if result already exists
    # metric_filename = os.path.join(output_dir, "TCM_evalscore"+opts.channel+"_"+opts.task+"_"+opts.synsets+".txt")
    # if os.path.exists(metric_filename) and not opts.redo:
    #     print("Output %s exists, skip..." %metric_filename)

    #     # cmd = 'tail -n1 %s' %metric_filename
    #     # utils.run_dcmd(cmd)
    #     sys.exit()


    ## flow warping layer
    image_size = (436, 1024)
    device = torch.device("cuda" if opts.cuda else "cpu")
    warp_op = Flow.Warper2d(image_size).to(device)

    ### load video list
    # list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    # with open(list_filename) as f:
    #     video_list = [line.rstrip() for line in f.readlines()]

    ### start evaluation
    # err_all = np.zeros(len(video_list))
    TCM_all = []

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
            # ref1_name = 'frame_'+frame_list[t-1].split('/')[-1].split('.')[0].split('_')[-1]
            # ref2_name = 'frame_'+frame_list[t].split('/')[-1].split('.')[0].split('_')[-1]
            # ref1_path = os.path.join(opts.ref_dir, 'training', 'albedo', syn, ref1_name+'.png')
            # ref2_path = os.path.join(opts.ref_dir, 'training', 'albedo', syn, ref2_name+'.png')
            # pdb.set_trace()
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
            # print('*** error :{}'.format(err))
            # print('*** referr:{}'.format(err_ref))
            errs.append(err)
            errs_ref.append(err_ref)
        # err_all[v] = err / (len(frame_list) - 1)


    # print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    # err_all = np.append(err_all, err_all.mean())
    # print("Save %s" %metric_filename)
    # np.savetxt(metric_filename, err_all, fmt="%f")
    return TCM, errs, errs_ref


if __name__ == "__main__":
    tcms, errs, errs_ref = run_demo()
    # total image diff
