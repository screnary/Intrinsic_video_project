# argparser
import argparse
import datetime
import json
import os
from os.path import exists, join, dirname
from easydict import EasyDict



def parser():
    """ first load yaml file, then change params according to the parser """
    parser = argparse.ArgumentParser(description='Temporal Consistency Metric')

    ### testing options
    parser.add_argument('-task',            type=str,     default="train",            help='evaluated task')
    parser.add_argument('-phase',           type=str,     default="refined_final",           choices=["refined_gs", "refined_final"])
    parser.add_argument('-channel',         type=str,     default="shading")
    parser.add_argument('-data_dir',        type=str,     default='/home/wzj/intrinsic/data/MPI',           help='path to data folder')
    parser.add_argument('-ref_dir',        type=str,     default='/home/wzj/intrinsic/data/MPI/origin',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-redo',            action="store_true",                    help='redo evaluation')
    
    parser.add_argument("-synsets", nargs='+', type=str, default=['market_5'], help="category list for classification")

    parser.add_argument('-cuda_id', type=str, default='0', help="cuda id, 0 or 1")

    parser.add_argument('-res_dir',        type=str,     default='/home/wzj/intrinsic/data/MPI',           help='path to data folder')

    opts = parser.parse_args()
    opts.cuda = True

    print(opts)

    opts = parser.parse_args()
    opts = EasyDict(opts.__dict__)
    return opts


if __name__ == '__main__':
    opts = parser()
    print(opts)
