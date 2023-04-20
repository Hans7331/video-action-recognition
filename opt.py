import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--test_batch_size',type=int,default=4)
    parser.add_argument('--frames_per_clip',type=int,default=16)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--num_classes',type=int,default=101)
    parser.add_argument('--annot_dir',type=str,default=r"../anet_annotations/activitynet-21-category.json")
    parser.add_argument('--patch_size',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--tt_split',type=float,default=0.8)
    parser.add_argument('--image_size',type=int,default=112)
    parser.add_argument('--dataset_dir', type=str, default=r"../UCF101")
    parser.add_argument('--pr',type=int,default=2)
    parser.add_argument('--dataset',type=str,default='UCF101')
    parser.add_argument('--checkpoint_flag',type=int,default=0)
    parser.add_argument('--temperature',type=int,default=0.1)
    
    
    args = parser.parse_args()
    
    return args

