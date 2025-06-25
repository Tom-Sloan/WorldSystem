# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for self-captured img sequence
# --------------------------------------------------------
import os.path as osp
import torch

SLAM3R_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import sys # noqa: E402
sys.path.insert(0, SLAM3R_DIR) # noqa: E402
from slam3r.utils.image import load_images, load_images_with_padding

class Seq_Data():
    def __init__(self, 
                 img_dir,     # the directory of the img sequence
                 img_size=224,  # only img_size=224 is supported now 
                 silent=False,  
                 sample_freq=1, # the frequency of the imgs to be sampled
                 num_views=-1, # only take the first num_views imgs in the img_dir
                 start_freq=1,  
                 postfix=None,   # the postfix of the img in the img_dir(.jpg, .png, ...)
                 to_tensor=False,
                 start_idx=0,
                 use_padding=False):  # NEW: option to preserve entire image
        
        # Note that only img_size=224 is supported now.
        # With use_padding=True: Imgs will be resized and padded to 224x224, preserving all content.
        # With use_padding=False: Imgs will be cropped and resized to 224x224, losing border information.
        assert img_size==224, "Sorry, only img_size=224 is supported now."

        # load imgs with sequential number.
        # Imgs in the img_dir should have number in their names to indicate the order,
        # such as frame-0031.color.png, output_414.jpg, ...
        if use_padding:
            self.imgs = load_images_with_padding(img_dir, size=img_size, 
                                    verbose=not silent, img_freq=sample_freq,
                                    postfix=postfix, start_idx=start_idx, 
                                    img_num=num_views, use_padding=True)
        else:
            self.imgs = load_images(img_dir, size=img_size, 
                                    verbose=not silent, img_freq=sample_freq,
                                    postfix=postfix, start_idx=start_idx, img_num=num_views)
        
        self.num_views = num_views if num_views > 0 else len(self.imgs)
        self.stride = start_freq
        self.img_num = len(self.imgs)
        self.use_padding = use_padding
        if to_tensor:
            for img in self.imgs:
                img['true_shape'] = torch.tensor(img['true_shape'])
        self.make_groups()
        self.length = len(self.groups)
        
        if isinstance(img_dir, str):
            if img_dir[-1] == '/':
                img_dir = img_dir[:-1]
            self.scene_names = ['_'.join(img_dir.split('/')[-2:])]
        
    def make_groups(self):
        self.groups = []
        for start in range(0,self.img_num, self.stride):
            end = start + self.num_views 
            if end > self.img_num:
                break
            self.groups.append(self.imgs[start:end])
    
    def __len__(self):
        return len(self.groups)
                        
    def __getitem__(self, idx):
        return self.groups[idx]



if __name__ == "__main__":
    from slam3r.datasets.base.base_stereo_view_dataset import view_name
    from slam3r.viz import SceneViz, auto_cam_size
    from slam3r.utils.image import rgb

    dataset = Seq_Data(img_dir="dataset/7Scenes/office-09",
                       img_size=224, silent=False, sample_freq=10, 
                       num_views=5, start_freq=2, postfix="color.png")
    for i in range(len(dataset)):
        data = dataset[i]
        print([img['idx'] for img in data])
        # break