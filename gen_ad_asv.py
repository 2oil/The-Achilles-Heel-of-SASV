# import required module
import os
import argparse
import torch
import pandas as pd
from tqdm import *
from src.models.adversarial import AdversarialNoiseAugmentor

def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--input_path', type=str, required=True, help='Audio file path')
    parser.add_argument('--output_path', type=str, required=True, help='Feature output path')
    parser.add_argument('--adv_method1', type=str, required=True, help='Adversarial attack method 1')
    parser.add_argument('--out_format', type=str, default="flac", help='Output format (default: flac)')
    parser.add_argument('--seed', type=int, default=251, help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    args = parse_argument()

    os.makedirs(args.output_path, exist_ok=True)

    config = {
        "aug_type": "adversarial",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "batch_size": args.batch_size,

# Choose surrogate model

        "model_name": "resnetse34v2",
        "model_pretrained": "./ResNetModels/baseline_v2_ap.model",  

        # "model_name": "ecapa_tdnn",
        # "model_pretrained": "./pretrained/pretrain.model",  

        # "model_name": "next_tdnn",
        # "model_pretrained": "./pretrained/model.pt",

        "eval_list": "./PeerJ/protocol/ASVspoof2019.csv",
        "enroll_path": "./enr_audio/eval",

        "device": "cuda",
        "adv_method1": args.adv_method1,
        "input_path": args.input_path,

        "seed": args.seed
    }

    file_path = config["eval_list"]
    protocol_df = pd.read_csv(file_path, sep=" ", header=None, names=["utt1", "utt2", "label", "type"])

    filenames = protocol_df["utt2"].apply(lambda x: os.path.join(args.input_path, f"{x}.flac")).tolist()
    labels = protocol_df["type"].tolist() 

    ana = AdversarialNoiseAugmentor(config)
    ana.load_batch(filenames, labels)
    ana.transform_batch(protocol_df)


if __name__ == '__main__':
    main()