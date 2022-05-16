"""
Created on 1/2/2021
@Author: DreamTale
"""
import os
import cv2
from numpy.core.defchararray import not_equal
from numpy.lib.npyio import save
import tqdm
import skimage
import argparse
import numpy as np
from numpy.lib.utils import source
from matplotlib import pyplot as plt, use
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from glob import glob
from os.path import abspath, dirname, join
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def image_measurement(img1, img2, methods=['ssim', 'psnr', 'mse']):
    """Measure the similarity or difference between `img1` and `img2`

    Args:
        img1 (np.array, unit8):   The input image
        img2 (np.array, unit8):   another input image
        methods (list, optional): Measurement methods. 
                                  Support ['mse', 'ssim', 'psnr', 'si-ssim', 'si-psnr', 'si-mse']
                                  Defaults to ['ssim', 'psnr].
    """
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))

    img1 = skimage.img_as_float32(img1)
    img2 = skimage.img_as_float32(img2)
    
    result_dict = {}
    for method in methods:
        pf = ''
        if 'si-' in method.lower():
            img1 = (img1 - img1.min()) / (img1.max() - img1.min())
            img2 = (img2 - img2.min()) / (img2.max() - img2.min())
            pf   = 'SI-'
        if 'ssim' in method.lower():
            result_dict.update({f'{pf}SSIM': structural_similarity(img1, img2, multichannel=True)})
        elif 'psnr' in method.lower():
            result_dict.update({f'{pf}PSNR': peak_signal_noise_ratio(img1, img2)})
        elif 'mse' in method.lower():
            result_dict.update({f'{pf}MSE':  mean_squared_error(img1, img2)})
    
    return result_dict


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def align_image_histogram(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def generate_matched_details(img, img_ref, use_hsv=False, save_pwd=None, is_calc=False, img_gt=None):

    h, w = img_ref.shape[:2]
    img  = cv2.resize(img, (w, h))

    if img_gt is None:
        img_gt = img_ref
    else:
        img_gt = cv2.resize(img_gt, (w, h))

    measure_before = image_measurement(img, img_gt) if is_calc else None

    if use_hsv:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
        img     = cv2.cvtColor(img,     cv2.COLOR_BGR2HSV)
        
        matched = np.zeros_like(img)
        matched[..., 0] = img[..., 0]
        matched[..., 1] = img[..., 1]
        matched[..., 2] = align_image_histogram(img[..., 2], img_ref[..., 2])

    else:
        matched = np.zeros_like(img)
        matched[..., 0] = align_image_histogram(img[..., 0], img_ref[..., 0])
        matched[..., 1] = align_image_histogram(img[..., 1], img_ref[..., 1])
        matched[..., 2] = align_image_histogram(img[..., 2], img_ref[..., 2])

    if use_hsv:
        source       = img[..., 2]
        template     = img_ref[..., 2]
        matched_gray = matched[..., 2]

        img     = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_HSV2BGR)
        matched = cv2.cvtColor(matched, cv2.COLOR_HSV2BGR)
    else:
        source       = cv2.cvtColor(img,     cv2.COLOR_BGR2GRAY)
        template     = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        matched_gray = cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY)
        
    measure_after = image_measurement(matched, img_gt) if is_calc else None

    def ecdf(x):
        """convenience function for computing the empirical CDF"""
        vals, counts = np.unique(x, return_counts=True)
        ecdf = np.cumsum(counts).astype(np.float64)
        ecdf /= ecdf[-1]
        return vals, ecdf

    if save_pwd is not None:

        x1, y1 = ecdf(source.ravel())
        x2, y2 = ecdf(template.ravel())
        x3, y3 = ecdf(matched_gray.ravel())

        fig = plt.figure()
        gs = plt.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()

        ax1.imshow(img[..., ::-1], cmap=plt.cm.gray)
        ax1.set_title('Source')
        ax2.imshow(matched[..., ::-1], cmap=plt.cm.gray)
        ax2.set_title('Matched')
        ax3.imshow(img_ref[..., ::-1], cmap=plt.cm.gray)
        ax3.set_title('Reference')

        ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
        ax4.plot(x2, y2 * 100, '-k', lw=3, label='Reference')
        ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
        ax4.set_xlim(x1[0], x1[-1])
        if use_hsv:
            ax4.set_xlabel('Pixel value (V-channel)')
        else:
            ax4.set_xlabel('Pixel value (gray-scale)')
        ax4.set_ylabel('Cumulative %')
        ax4.legend(loc=5)

        plt.savefig(save_pwd)

    return matched, measure_before, measure_after


# def test_match():
#     img     = cv2.imread(r"E:\ws\RR\RR_TIP_images\real_input_1.jpg")
#     img_ref = cv2.imread(r"E:\ws\RR\RR_TIP_images\real_input_1.jpg")
#     img_gt  = cv2.imread(r"E:\ws\RR\RR_TIP_images\real_gt_1.png")

#     # _, m_before, m_after = generate_matched_details(img, img_ref, use_hsv=False, save_pwd='result_histogram_match-rgb-input.png', is_calc=True, img_gt=img_gt)
#     _, m_before, m_after = generate_matched_details(img, img_ref, use_hsv=True, save_pwd='result_histogram_match-hsv-input.png', is_calc=True, img_gt=img_gt)
#     # _, m_before, m_after = generate_matched_details(img, img_gt, use_hsv=True, save_pwd='result_histogram_match-hsv-gt.png', is_calc=True)
#     # _, m_before, m_after = generate_matched_details(img, img_gt, use_hsv=False, save_pwd='result_histogram_match-rgb-gt.png', is_calc=True)

#     print('==> Before histogram alignment:')
#     print(m_before)
#     print('==> After histogram alignment:')
#     print(m_after)



def main(args):

    print(' ########################## Arguments: ##############################')
    print(' ┌───────────────────────────────────────────────────────────────────')
    for arg in vars(args):
        print(' │ {:<20} : {}'.format(arg, getattr(args, arg)))
    print(' └───────────────────────────────────────────────────────────────────')

    image_names     = [x for x in os.listdir(args.input_dir) if is_image_file(x)]
    exp_image_names = [x for x in image_names if x.startswith(args.exp_name)]

    image_input  = sorted([x for x in exp_image_names if 'input' in x])
    image_gt     = sorted([x for x in exp_image_names if 'gt'    in x])
    method_names = set([x.split('.')[0].split('_')[1] for x in exp_image_names])
    method_names = [x for x in method_names if 'input' not in x and 'gt' not in x]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    t_bar = tqdm.tqdm(image_input)
    t_bar.set_description('Processing')
    results = []

    for idx, name_in in enumerate(t_bar):
        name_gt = '.'.join([name_in.replace('input', 'gt').split('.')[0], image_gt[idx].split('.')[-1]])
        img_in = cv2.imread(os.path.join(args.input_dir, name_in))
        img_gt = cv2.imread(os.path.join(args.input_dir, name_gt))
        cv2.imwrite(os.path.join(args.output_path, name_in), img_in)
        cv2.imwrite(os.path.join(args.output_path, name_gt), img_gt)
        
        rlt_img, p_max, p_min = generate_matched_details(img_in, img_in, use_hsv=args.use_hsv, is_calc=True, img_gt=img_gt)
        cv2.imwrite(os.path.join(args.output_path, name_in), rlt_img)
        results.append(('Input', p_max, p_min))

        for pred_name in method_names:
            proc_image_name = sorted([x for x in exp_image_names if pred_name in x])[idx]
            img_pred = cv2.imread(os.path.join(args.input_dir, proc_image_name))
            rlt_img, p_max, p_min = generate_matched_details(img_pred, img_in, use_hsv=args.use_hsv, is_calc=True, img_gt=img_gt)
            cv2.imwrite(os.path.join(args.output_path, proc_image_name), rlt_img)
            results.append((pred_name, p_max, p_min))

    [print(x) for x in results]
    print('\nDone.')


def get_ref_name(image_names, step=5):
    mvalue_list = []
    for img_n in image_names:
        img_in = cv2.imread(img_n)
        img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        img_v = img_hsv[..., 2]
        mvalue_list.append(np.mean(img_v.ravel()))
    ref_list = []
    for i in range(0,len(image_names),step):
        idx = np.argmax(mvalue_list[i:i+step])
        ref_list.append(image_names[i:i+step][idx])
    return ref_list


def main_sequence(args):
    """ process image sequence to fit the 1st frame histogram """
    print(' ########################## Arguments: ##############################')
    print(' ┌───────────────────────────────────────────────────────────────────')
    for arg in vars(args):
        print(' │ {:<20} : {}'.format(arg, getattr(args, arg)))
    print(' └───────────────────────────────────────────────────────────────────')
    data_root = join(dirname(dirname(abspath(__file__))), 'datasets', 'MPI')
    syn_name = args.syn_name
    image_names = sorted(glob(join(data_root, 'refined_final', 'MPI-main-albedo', syn_name+'*.png')))

    path_ref_list = get_ref_name(image_names, step=args.step)  # find the frame with the largest average lightness
    print('ref image is: ', path_ref_list)

    output_path = join(data_root, 'refined_final', 'MPI-main-albedo-aligned')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    t_bar = tqdm.tqdm(image_names)
    t_bar.set_description('Processing')
    results = []

    for idx, path_in in enumerate(t_bar):
        path_ref = path_ref_list[idx // args.step]
        img_name = path_in.split('/')[-1]
        img_in = cv2.imread(path_in)
        img_ref = cv2.imread(path_ref)
        # cv2.imwrite(os.path.join(args.output_path, img_name), img_in)
        # cv2.imwrite(os.path.join(args.output_path, ref_name), img_ref)
        
        rlt_img, p_max, p_min = generate_matched_details(img_in, img_ref, use_hsv=args.use_hsv, is_calc=True, img_gt=None)
        cv2.imwrite(os.path.join(output_path, img_name), rlt_img)
        results.append(('Input', p_max, p_min))

    [print(x) for x in results]
    print('\nDone.')


def test_match():
    data_root = join(dirname(dirname(abspath(__file__))), 'datasets', 'MPI')
    syn_name = 'market_5'
    image_names = sorted(glob(join(data_root, 'refined_gs_v3', 'MPI-main-shading', syn_name+'*.png')))

    in_path = image_names[47]
    ref_path = image_names[44]
    img     = cv2.imread(in_path) # 45,46,47 |48
    img_ref = cv2.imread(ref_path)

    in_name = in_path.split('/')[-1]
    # pdb.set_trace()

    # _, m_before, m_after = generate_matched_details(img, img_ref, use_hsv=False, save_pwd='result_histogram_match-rgb-input.png', is_calc=True, img_gt=img_gt)
    rlt_img, m_before, m_after = generate_matched_details(img, img_ref, use_hsv=True, save_pwd='result_histogram_match-hsv-input.png', is_calc=True, img_gt=None)
    # _, m_before, m_after = generate_matched_details(img, img_gt, use_hsv=True, save_pwd='result_histogram_match-hsv-gt.png', is_calc=True)
    # _, m_before, m_after = generate_matched_details(img, img_gt, use_hsv=False, save_pwd='result_histogram_match-rgb-gt.png', is_calc=True)

    output_path = join(data_root, 'refined_gs_v3_aligned', 'MPI-main-shading')
    cv2.imwrite(os.path.join(output_path, in_name), rlt_img)
    print('==> Before histogram alignment:')
    print(m_before)
    print('==> After histogram alignment:')
    print(m_after)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Align image to the reference')
    parser.add_argument('-n', '--syn_name', type=str, default='temple_3')
    parser.add_argument('-s', '--step', type=int, default=5)
    # parser.add_argument('-i', '--input_dir',   type=str, 
    #                     default=r'E:\ws\RR\RR_TIP_images', help="root of data path")
    # parser.add_argument('-o', '--output_path', type=str, 
    #                     default='results',     help="outputs path")
    # parser.add_argument('-e', '--exp_name',    type=str, 
    #                     default='real',        help="experiment name")
    parser.add_argument('-u', '--use_hsv',     action='store_false',
                        help='use HSV mode or not')

    args = parser.parse_args()

    # main(args)

    # test_match()
    main_sequence(args)


# python .\evalutate_ssim_psnr.py -f ssim,psnr