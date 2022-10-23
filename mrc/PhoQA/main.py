import torch


from visquaddataset import build_dataloader
from model import build_model
from train import train
from checkpointer import Checkpointer

import argparse
import os


def main(cfg):
    train_loader, test_loader = build_dataloader(cfg.train_file, cfg.test_file, cfg.batch_size, cfg.max_seq_length, cfg.max_query_length, cfg.doc_stride)
    model = build_model(model_path=cfg.model_path)
    model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    checkpointer = Checkpointer(model, optimizer, cfg.save_dir)
    #print(typcheckpointer)
    if cfg.resume is not None:
        checkpointer.load(cfg.resume)
    
    #train(model, train_loader, test_loader, optimizer, checkpointer, cfg.device, cfg.num_epoch)
    #device = torch.device(cfg.device)
    
    train(model, train_loader, test_loader, optimizer, checkpointer, cfg.device, cfg.num_epoch)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--lr', type=float, default=1e-5)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--num_epoch', type=int, default=20)
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--model_path', type=str, default='vinai/phobert-base')
    args.add_argument('--save_dir', type=str, default='./output')
    args.add_argument('--train_file', type=str, default='./dataset/w_seg/train.json')
    args.add_argument('--test_file', type=str, default='./dataset/w_seg/test.json')
    args.add_argument('--max_seq_length', type=int, default=256)
    args.add_argument('--max_query_length', type=int, default=64)
    args.add_argument('--doc_stride', type=int, default=128)
    args.add_argument('--resume', type=str, default=None)
    
    config = args.parse_args()
    main(config)



