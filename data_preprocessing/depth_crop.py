import numpy as np
import os
import glob
import SimpleITK as sitk
from datetime import datetime

def center_crop_nii(image, start_index):
    crop_start = [0,0,start_index]
    cropped_image = sitk.RegionOfInterest(image, size=[256,256,32], index=crop_start)
    return cropped_image

def cropNii(vol, start_index):
    vol = center_crop_nii(vol, start_index)
    assert vol.GetSize() == (256, 256,32)
    return vol



IMG_BNAME="./data/ABDOMENCT1K/img_resampled/*.nii.gz"
SEG_BNAME="./data/ABDOMENCT1K/mask_resampled/*.nii.gz"

imgs = glob.glob(IMG_BNAME)
segs = glob.glob(SEG_BNAME)
imgs = [ fid for fid in sorted(imgs, key = lambda x: (x.split("_")[-2])  ) ]
segs = [ fid for fid in sorted(segs, key = lambda x: (x.split("_")[-2])  ) ]

print(len(imgs), imgs)

print(segs)

lb = sitk.ReadImage(segs[4])
print(lb.GetSize())

img = sitk.ReadImage(imgs[4])
print(img.GetSize())


LABEL_NAME = ["BGD", "LIVER", "KID", "SPLEEN", "PANCREAS"]     
MIN_TP=1 # minimum number of true positive pixels in a slice
fid = f'./special_case.txt'

img_dir = "./data/ABDOMENCT1K/image_cropped"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

lbl_dir = "./data/ABDOMENCT1K/label_cropped"
if not os.path.exists(lbl_dir):
    os.makedirs(lbl_dir)

print('total # of niis', len(segs))

for pid, seg in enumerate(segs):
    img_path  = os.path.join(img_dir, seg.split("/")[-1] )
    lb_path  = os.path.join(lbl_dir, seg.split("/")[-1] )

    slice_index_list = []
    lb_vol = sitk.ReadImage(seg)

    n_slice = lb_vol.GetSize()[-1] 

    if os.path.exists(img_path):
        continue

    print("[Calculating]", datetime.now())

    for slc in range(n_slice):
        cls = 4
        if cls in lb_vol[:,:,slc]:
            if np.sum( lb_vol[:,:,slc] == cls) >= MIN_TP:
                slice_index_list.append(1)
        else:
            slice_index_list.append(0)

    assert(len(slice_index_list) == n_slice)

    window = 32
    num_of_most_pancreas = 0
    start_index_has_most_pancreas = 0
    other_index = []
    for i in range(n_slice):
        if i+window-1 > n_slice-1:
            break
        num = np.sum(slice_index_list[i:i+window])

        if num > num_of_most_pancreas:
            num_of_most_pancreas = num
            start_index_has_most_pancreas = i
            other_index = []
        elif num == num_of_most_pancreas:
            other_index.append(i)
    
    if len(other_index) != 0:  
        if ((other_index[-1] - start_index_has_most_pancreas) == (len(other_index))):  # 窗口连续
            start_index_has_most_pancreas = len(other_index) // 2 + start_index_has_most_pancreas
        else:
            with open(fid, 'w') as fopen:
                fopen.write(f'{seg} has incontinous start index, which is {start_index_has_most_pancreas} and {other_index}')
            continue
        
    lb_vol = cropNii(lb_vol, start_index_has_most_pancreas)

    pid = segs.index(seg)
    img_vol = sitk.ReadImage(imgs[pid])
    img_vol = cropNii(img_vol, start_index_has_most_pancreas)

    sitk.WriteImage( lb_vol, lb_path)
    sitk.WriteImage( img_vol, img_path)
    print(pid, "th volume, i.e.", seg, " is finished", datetime.now())