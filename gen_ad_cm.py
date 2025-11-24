# import required module
import os
from multiprocessing import Pool, set_start_method
import argparse
from tqdm import *
from random import randrange
import torch
import pandas as pd
from src.models.adversarial import AdversarialNoiseAugmentor

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--input_path', type=str, default="",required=True, help='Audio file path')
    
    parser.add_argument('--output_path', type=str, default="",required=True, help='Feature output path')

    parser.add_argument('--adv_method1', type=str, default="",required=True, help='Adversarial attack method 1')
    
    parser.add_argument('--out_format', type=str, default="flac", required=False, help='Output format. \n'
                        +'Suported: flac, ogg, mp3, wav. Default: flac. \n'
                        +'Encode by pydub + ffmpeg. Please install ffmpeg first. \n')
    parser.add_argument('--seed', type=int, default=251, help='Random seed for reproducibility')
    
    # load argument
    args = parser.parse_args()
        
    return args
   
def main():
    args = parse_argument()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    config = {
            "aug_type": "adversarial",
            "output_path": args.output_path,
            "out_format": args.out_format,
            "batch_size": args.batch_size,

# Choose Surrogate Model
            
            "model_name": "aasist",
            "model_pretrained": "/home/eoil/aasist/models/weights/AASIST.pth",
            "config_path": "./AASIST_conf.yaml",

            # "model_name": "aasistssl",
            # "model_pretrained": "./pretrained/LA_model.pth",

            # "model_name": "conformer_tcm",
            # "model_pretrained": "./best_4.pth",

            # "model_name": "rawnet2",
            # "model_pretrained": "./pretrained_rawnet2/pre_trained_DF_RawNet2.pth",
            
            "device": "cuda",

            "adv_method1": args.adv_method1,
            "input_path": args.input_path,

            "seed": args.seed
        }
    file_path = './PeerJ/protocol/ASVspoof2019.csv' 
    protocol_df = pd.read_csv(file_path, sep=" ", header=None, names=["utt1", "utt2", "label", "type"])

    filenames = protocol_df["utt2"].apply(lambda x: os.path.join(args.input_path, f"{x}.flac")).tolist()
    labels = protocol_df["type"].tolist()  # 'target', 'nontarget', 'spoof'

    ana = AdversarialNoiseAugmentor(config)
    ana.load_batch(filenames, labels)
    ana.transform_batch(protocol_df)

if __name__ == '__main__':
    main()