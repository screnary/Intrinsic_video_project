import os
import shutil as sh
import pdb


"""
rename files according to data list file
"""


def rename_my_mpi_pred():
    data_root = '../ckpoints-Basic-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-decoder_Residual/log/'
    data_dir = os.path.join(data_root, 'test-imgs_ep200')
    out_dir = os.path.join(data_root, 'test-imgs_ep200-renamed')
    file_dir = '../datasets/MPI/MPI_main_sceneSplit-fullsize-NoDefect-test.txt'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(file_dir, 'r+') as f:
        lines = f.readlines()

    file_names = []
    for l in lines:
        if '.png' in l:
            img_n = l.strip('\n')[:-4]
            file_names.append(img_n)

    # file_list = [fn + '\n' for fn in file_names]
    # with open(out_file_dir, 'w+') as f:
    #     f.writelines(file_list)

    suf_i = '_diffuse.png'
    suf_r = '_reflect-pred.png'
    suf_s = '_shading-pred.png'  # shading predict
    suf_rgt = '_reflect-real.png'  # shading reconstruct
    suf_sgt = '_shading-real.png'  # shading reconstruct

    for i in range(len(file_names)):
        fn = file_names[i]
        i_name = os.path.join(data_dir, str(i) + suf_i)
        r_name = os.path.join(data_dir, str(i) + suf_r)
        s_name = os.path.join(data_dir, str(i) + suf_s)
        rgt_name = os.path.join(data_dir, str(i) + suf_rgt)
        sgt_name = os.path.join(data_dir, str(i) + suf_sgt)

        i_name_out = os.path.join(out_dir, fn + suf_i)
        r_name_out = os.path.join(out_dir, fn + suf_r)
        s_name_out = os.path.join(out_dir, fn + suf_s)
        rgt_name_out = os.path.join(out_dir, fn + suf_rgt)
        sgt_name_out = os.path.join(out_dir, fn + suf_sgt)

        sh.copyfile(i_name, i_name_out)
        sh.copyfile(r_name, r_name_out)
        sh.copyfile(s_name, s_name_out)
        sh.copyfile(rgt_name, rgt_name_out)
        sh.copyfile(sgt_name, sgt_name_out)


def rename_revisiting_mpi_pred():
    data_root = '../IntrinsicImage-master/results/test/bkup/'
    # data_dir = os.path.join(data_root, 'RD_MPI-main-clean')
    # out_dir = os.path.join(data_root, 'RD_MPI-main-clean-renamed')
    data_dir = os.path.join(data_root, 'MPI-main-clean-with-rescale')
    out_dir = os.path.join(data_root, 'MPI-main-clean-with-rescale-renamed')
    file_dir = '../datasets/MPI/MPI_main_sceneSplit-fullsize-NoDefect-test.txt'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(file_dir, 'r+') as f:
        lines = f.readlines()

    file_names = []
    for l in lines:
        if '.png' in l:
            img_n = l.strip('\n')[:-4]
            file_names.append(img_n)

    # file_list = [fn + '\n' for fn in file_names]
    # with open(out_file_dir, 'w+') as f:
    #     f.writelines(file_list)

    suf_i = '-input.png'
    suf_r = '_reflect-pred.png'
    suf_s = '_shading-pred.png'  # shading predict
    suf_rgt = '_reflect-real.png'  # shading reconstruct
    suf_sgt = '_shading-real.png'  # shading reconstruct

    for i in range(len(file_names)):
        fn = file_names[i]
        i_name = os.path.join(data_dir, str(i) + suf_i)
        r_name = os.path.join(data_dir, str(i) + suf_r)
        s_name = os.path.join(data_dir, str(i) + suf_s)
        rgt_name = os.path.join(data_dir, str(i) + suf_rgt)
        sgt_name = os.path.join(data_dir, str(i) + suf_sgt)

        i_name_out = os.path.join(out_dir, fn + suf_i)
        r_name_out = os.path.join(out_dir, fn + suf_r)
        s_name_out = os.path.join(out_dir, fn + suf_s)
        rgt_name_out = os.path.join(out_dir, fn + suf_rgt)
        sgt_name_out = os.path.join(out_dir, fn + suf_sgt)

        sh.copyfile(i_name, i_name_out)
        sh.copyfile(r_name, r_name_out)
        sh.copyfile(s_name, s_name_out)
        sh.copyfile(rgt_name, rgt_name_out)
        sh.copyfile(sgt_name, sgt_name_out)


if __name__ == '__main__':
    # rename_my_mpi_pred()
    rename_revisiting_mpi_pred()
