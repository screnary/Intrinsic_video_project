"""
The pre-processing script for MPI Sintel dataset. Randomly crop images into 300x300 with 
sequence preserving, for the same scene, the cropped patch is named as 
`<scene>-<cropped batch id>-frame_<frame id>`
- sample based on the masks, valid pixels should occupy major region
"""
import argparse
import os
import tqdm
import cv2
import numpy as np
import h5py
import pdb


args = argparse.ArgumentParser(description='Pre-processor for MPI sequence')
args.add_argument('--data_root', type=str, 
                  default=r'/home/wzj/intrinsic/data/MPI',
                  help='the pwd of MPI Sintel dataset')
args.add_argument('--out_dir', type=str, 
                  default=r'/home/wzj/intrinsic/data/MPI_Video',
                  help='the pwd of generated dataset')
args.add_argument('--crop_w', type=int, default=320, help='the width of cropped image')
args.add_argument('--crop_h', type=int, default=320, help='the width of cropped image')
args.add_argument('--count', type=int, default=10, help='the number of random cropped images from one image')

args = args.parse_args()


def random_crop_and_flip(img, random_flip=0, random_pos=None):
    # img = rotate(img,random_angle, order = mode)
    if random_pos is not None:
        if len(img.shape) > 2:
            img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        else:
            img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3]]

    if random_flip > 0.5:
        img = np.fliplr(img)

    return img.copy()


def check_dir(s_dir):
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)


def flowh5_reader(path):
    f = h5py.File(path, 'r')
    flo = f['data'][:]  # (2,1024,436)
    order = list(range(np.ndim(flo)))[::-1]

    return np.transpose(flo, order)


def flowh5_writer(path, flow_data):
    h5_fout = h5py.File(path)

    order = list(range(np.ndim(flow_data)))[::-1]
    flo = np.transpose(flow_data, order)

    h5_fout.create_dataset('data', data=flo,
                            compression='gzip', compression_opts=4,
                            dtype='float32')
    h5_fout.close()


def main():
    name_in = 'MPI-main-clean'
    name_ab = 'MPI-main-albedo'
    name_sh = 'MPI-main-shading'

    out_dir_in_1 = os.path.join(args.out_dir, name_in, 'from')
    out_dir_ab_1 = os.path.join(args.out_dir, name_ab, 'from')
    out_dir_sh_1 = os.path.join(args.out_dir, name_sh, 'from')
    out_dir_in_2 = os.path.join(args.out_dir, name_in, 'to')
    out_dir_ab_2 = os.path.join(args.out_dir, name_ab, 'to')
    out_dir_sh_2 = os.path.join(args.out_dir, name_sh, 'to')
    out_dir_flow = os.path.join(args.out_dir, 'flow')
    out_dir_occ = os.path.join(args.out_dir, 'occlusions')

    check_dir(out_dir_in_1)
    check_dir(out_dir_ab_1)
    check_dir(out_dir_sh_1)
    check_dir(out_dir_in_2)
    check_dir(out_dir_ab_2)
    check_dir(out_dir_sh_2)
    check_dir(out_dir_flow)
    check_dir(out_dir_occ)

    scenes = list(np.unique(['_'.join(x.split('_')[:2]) for x in os.listdir(os.path.join(args.data_root, 'refined_final', name_in))]))

    t_bar = tqdm.tqdm(scenes)
    for scene_name in t_bar:
        image_names = sorted([x for x in os.listdir(os.path.join(args.data_root, 'refined_final', name_in)) if x.startswith(scene_name)])
        h, w = cv2.imread(os.path.join(args.data_root, 'refined_final', name_in, image_names[0])).shape[:2]
        for i in range(len(image_names)-1):
            img_name_1 = image_names[i]
            img_name_2 = image_names[i+1]
            base_name = img_name_1.split('.')[0]
            flow_path = os.path.join(args.data_root, 'flow_hdf5', img_name_1.split('.')[0]+'.h5')
            flow_data = flowh5_reader(flow_path)  # (436,1024,2)
            occ_path = os.path.join(args.data_root, 'occlusions', scene_name, '_'.join(img_name_1.split('_')[-2:]))
            occ_data = cv2.imread(occ_path)

            mask_img = cv2.imread(os.path.join(args.data_root, 'refined_final', 'MPI-main-mask', img_name_1))
            mask_data = (mask_img * 255)[:,:,0]  # (436,1024)

            for cnt in range(args.count):  # random crop args.count patches per frame
                resample_flag = True
                loop_n = 0
                while resample_flag and loop_n <= 50:
                    if loop_n >= 10 and loop_n % 10 == 0:
                        print('resample valid patch...loop times:', loop_n)
                    else:
                        pass
                    loop_n += 1

                    y = np.random.randint(0, h - args.crop_h)  # h [0~436], w [0~1024]
                    x = np.random.randint(0, w - args.crop_w)
                    # random_flip = np.random.random()
                    random_flip = 0  # do not flip
                    random_pos = [y, y + args.crop_h, x, x + args.crop_w]

                    patch_mask = random_crop_and_flip(mask_img, random_flip=random_flip, random_pos=random_pos)
                    invalid_idx = np.array(patch_mask, dtype=np.float32) == 0
                    invalid_num = invalid_idx.sum().astype(np.float32)
                    resample_flag =  invalid_num > 0.05 * invalid_idx.size

                img_in1 = cv2.imread(os.path.join(args.data_root, 'refined_final', name_in, img_name_1))  # (436,1024,3)
                img_ab1 = cv2.imread(os.path.join(args.data_root, 'refined_final', name_ab, img_name_1))
                img_sh1 = cv2.imread(os.path.join(args.data_root, 'refined_final', name_sh, img_name_1))

                img_in2 = cv2.imread(os.path.join(args.data_root, 'refined_final', name_in, img_name_2))
                img_ab2 = cv2.imread(os.path.join(args.data_root, 'refined_final', name_ab, img_name_2))
                img_sh2 = cv2.imread(os.path.join(args.data_root, 'refined_final', name_sh, img_name_2))

                try:
                    out_img_in1 = random_crop_and_flip(img_in1, random_flip=random_flip, random_pos=random_pos)
                    out_img_ab1 = random_crop_and_flip(img_ab1, random_flip=random_flip, random_pos=random_pos)
                    out_img_sh1 = random_crop_and_flip(img_sh1, random_flip=random_flip, random_pos=random_pos)

                    out_img_in2 = random_crop_and_flip(img_in2, random_flip=random_flip, random_pos=random_pos)
                    out_img_ab2 = random_crop_and_flip(img_ab2, random_flip=random_flip, random_pos=random_pos)
                    out_img_sh2 = random_crop_and_flip(img_sh2, random_flip=random_flip, random_pos=random_pos)

                    out_flow = random_crop_and_flip(flow_data, random_flip=random_flip, random_pos=random_pos)
                    out_occ  = random_crop_and_flip(occ_data, random_flip=random_flip, random_pos=random_pos)
                except Exception as e:
                    print('Exception! ', e)
                    pdb.set_trace()
                    continue

                out_file_name = '{}-{:0>2d}.png'.format(base_name, cnt)

                cv2.imwrite(os.path.join(out_dir_in_1, out_file_name), out_img_in1)
                cv2.imwrite(os.path.join(out_dir_ab_1, out_file_name), out_img_ab1)
                cv2.imwrite(os.path.join(out_dir_sh_1, out_file_name), out_img_sh1)

                cv2.imwrite(os.path.join(out_dir_in_2, out_file_name), out_img_in2)
                cv2.imwrite(os.path.join(out_dir_ab_2, out_file_name), out_img_ab2)
                cv2.imwrite(os.path.join(out_dir_sh_2, out_file_name), out_img_sh2)

                cv2.imwrite(os.path.join(out_dir_occ, out_file_name), out_occ)
                flowh5_writer(os.path.join(out_dir_flow, out_file_name.split('.')[0]+'.h5'), out_flow)


if __name__ == '__main__':
    main()
