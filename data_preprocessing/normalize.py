import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk

class CTExtractor:
    def __init__(self, input_path, out_path, predict_out_path):
        super(CTExtractor, self).__init__()

        self.MIN_BOUND = -1000.0
        self.MAX_BOUND = 400.0
        self.PIXEL_MEAN = 0.25
        self.roi = 256
        self.roi2 = 32

        self.path = input_path
        self.outpath = out_path
        self.predict_out_path = predict_out_path
        self.slices = []
        self.fname = ''
    
    def normalize(self, image):
        image = (image - self.MIN_BOUND) / (self.MAX_BOUND - self.MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image*2-1.

    def save(self):
        path = os.path.join(self.outpath, self.fname.split('.')[0] + '.npy')
        np.save(path, self.vol[:,::-1,::-1])
        
        predict_path = os.path.join(self.path.replace("image_cropped", "mask_nii"), self.fname[0:10] + ".nii.gz")
        os.makedirs(path_output, exist_ok=True)
        predict_image = sitk.ReadImage(predict_path)
        predict_array = sitk.GetArrayFromImage(predict_image)
        np.save(os.path.join(self.predict_out_path, self.fname.split('.')[0] + '.npy'), predict_array[:,::-1,::-1])

    def run(self, fname):
        self.fname = fname
        nii_path = os.path.join(self.path, self.fname)

        self.vol = sitk.ReadImage(nii_path)
        self.vol = sitk.GetArrayFromImage(self.vol)
        self.vol = self.normalize(self.vol)
        
        self.save()


def worker(fname, extractor):
        extractor.run(fname)  # fname = Case_00580_0000.nii.gz



if __name__ == "__main__": 
    input_path = './data/ABDOMENCT1K/image_cropped'
    path_output = './data/ABDOMENCT1K/image'
    predict_out_path = './data/ABDOMENCT1K/mask'
    
    
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(predict_out_path, exist_ok=True)
    extractor = CTExtractor(input_path, path_output, predict_out_path)

    def worker_partial(fname):
        return worker(fname, extractor)

    fnames = os.listdir(input_path)
    fnames = [fid for fid in sorted(fnames)]
    print('total # of niis', len(fnames))

    with Pool(processes=4) as pool:
        res = list(tqdm(pool.imap(
            worker_partial, iter(fnames)), total=len(fnames)))
