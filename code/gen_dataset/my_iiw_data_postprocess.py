import os
import shutil as sh
import pdb

"""
rename files according to data list file
"""
if __name__ == '__main__':
    data_root = '../ckpoints--iiw_v7-oneway+pixsupv2+vgg19-IIW-decoder_Residual/log/'
    data_dir = os.path.join(data_root, 'test-imgs_ep12')
    out_dir = os.path.join(data_root, 'test-imgs_ep12')  # -renamed')
    file_dir = '../datasets/IIW/test_list/img_batch.p'
    out_file_dir = '../datasets/IIW/test_list.txt'

    with open(file_dir, 'r+') as f:
        lines = f.readlines()

    file_names = []
    for l in lines:
        if '.png' in l:
            img_n = l.strip('\n').split('/')[-1][:-4]
            file_names.append(img_n)

    # file_list = [fn + '\n' for fn in file_names]
    # with open(out_file_dir, 'w+') as f:
    #     f.writelines(file_list)

    suf_i = '_input.png'
    suf_r = '_reflect-pred.png'
    suf_sp = '_shading-pred.png'  # shading predict
    suf_sr = '_shading-rec.png'  # shading reconstruct

    suf_i_out = '.png'
    suf_r_out = '_r.png'
    suf_sp_out = '_sp.png'  # shading predict
    suf_sr_out = '_sr.png'  # shading reconstruct
    for i in range(len(file_names)):
        fn = file_names[i][:-4]
        i_name = os.path.join(data_dir, str(i) + suf_i)
        r_name = os.path.join(data_dir, str(i) + suf_r)
        sp_name = os.path.join(data_dir, str(i) + suf_sp)
        sr_name = os.path.join(data_dir, str(i) + suf_sr)

        i_name_out = os.path.join(out_dir, fn + suf_i_out)
        r_name_out = os.path.join(out_dir, fn + suf_r_out)
        sp_name_out = os.path.join(out_dir, fn + suf_sp_out)
        sr_name_out = os.path.join(out_dir, fn + suf_sr_out)

        os.rename(i_name, i_name_out)
        os.rename(r_name, r_name_out)
        os.rename(sp_name, sp_name_out)
        os.rename(sr_name, sr_name_out)
