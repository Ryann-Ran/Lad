import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk
import math

class CTExtractor:
    def __init__(self, input_path, out_path, mask_output_path):
        super(CTExtractor, self).__init__()
        self.PIXEL_MEAN = 0.25
        self.roi = 256
        self.roi2 = 32

        self.path = input_path
        self.outpath = out_path
        self.mask_output_path = mask_output_path
        self.slices = []
        self.fname = ''

    def resample(self, vol, outspacing=[1.0, 1.0, 1.0]):
        outsize = [0,0,0]
        inputspacing = 0
        inputsize = 0
        
        inputsize = vol.GetSize()
        inputspacing = vol.GetSpacing()

        transform = sitk.Transform()
        transform.SetIdentity()
        outsize[0] = int(inputsize[0]*inputspacing[0]/outspacing[0] + 0.5)
        outsize[1] = int(inputsize[1]*inputspacing[1]/outspacing[1] + 0.5)
        outsize[2] = int(inputsize[2]*inputspacing[2]/outspacing[2] + 0.5)

        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(vol.GetOrigin())
        resampler.SetOutputSpacing(outspacing)
        resampler.SetOutputDirection(vol.GetDirection())
        resampler.SetSize(outsize)
        newvol = resampler.Execute(vol)
        return newvol
    
    def resample_by_res(self, mov_img_obj, new_spacing, interpolator = sitk.sitkLinear, logging = True):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputDirection(mov_img_obj.GetDirection())
        resample.SetOutputOrigin(mov_img_obj.GetOrigin())
        mov_spacing = mov_img_obj.GetSpacing()

        resample.SetOutputSpacing(new_spacing)
        RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
        new_size = np.array(mov_img_obj.GetSize()) *  RES_COE 

        resample.SetSize( [int(sz) for sz in new_size] )

        return resample.Execute(mov_img_obj)
    
    def resample_lb_by_res(self, mov_lb_obj, new_spacing, interpolator = sitk.sitkLinear, ref_img = None, logging = True):
        src_mat = sitk.GetArrayFromImage(mov_lb_obj)
        lbvs = np.unique(src_mat)
        for idx, lbv in enumerate(lbvs):
            _src_curr_mat = np.float32(src_mat == lbv) 
            _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
            _src_curr_obj.CopyInformation(mov_lb_obj)
            _tar_curr_obj = self.resample_by_res( _src_curr_obj, new_spacing, interpolator, logging )
            _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
            if idx == 0:
                out_vol = _tar_curr_mat
            else:
                out_vol[_tar_curr_mat == lbv] = lbv
        out_obj = sitk.GetImageFromArray(out_vol)
        out_obj.SetSpacing( _tar_curr_obj.GetSpacing() )
        if ref_img != None:
            out_obj.CopyInformation(ref_img)
        return out_obj

    def zero_center(self, image):
        image = image - self.PIXEL_MEAN
        return image
    
    def pad_center(self, pix_resampled):
        tmp_npy = sitk.GetArrayFromImage(pix_resampled)
        pad_constant = np.min(tmp_npy)
        pad_constant = math.floor(pad_constant)
       
        pad_x = max(self.roi - pix_resampled.GetSize()[0], 0)
        pad_y = max(self.roi - pix_resampled.GetSize()[1], 0)
        pad_z = max(self.roi2 - pix_resampled.GetSize()[2], 0)

        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad_x // 2, pad_y // 2, pad_z // 2])
        pad_filter.SetPadUpperBound([pad_x - pad_x // 2, pad_y - pad_y // 2, pad_z - pad_z // 2])
        pad_filter.SetConstant(pad_constant)

        padded_image = pad_filter.Execute(pix_resampled)
        return padded_image


    def center_crop_nii(self, image, crop_size=[256, 256, 32]):
        original_size = image.GetSize()
        crop_start = [int((original_size[i] - crop_size[i]) / 2) for i in range(len(original_size))]
        cropped_image = sitk.RegionOfInterest(image, size=crop_size, index=crop_start)
        return cropped_image


    def crop_center(self, vol, cropz, cropy, cropx):
        z, y, x = vol.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        startz = z//2-(cropz//2)
        return vol[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]

    def save(self, mask):
        path = os.path.join(self.outpath, self.fname.split('.')[0] + '.nii.gz')
        mask_path = os.path.join(self.mask_output_path, self.fname.split('.')[0] + '.nii.gz')
        
        sitk.WriteImage(self.vol, path)
        sitk.WriteImage(mask, mask_path)
        print("finished ", path)

    def run(self, fname):
        self.fname = fname
        nii_path = os.path.join(self.path, self.fname)
        mask_path = os.path.join("./mask", self.fname[0:10] + ".nii.gz")

        self.vol = sitk.ReadImage(nii_path)  # [x,y,z]
        mask = sitk.ReadImage(mask_path)

        # Resample
        self.vol = self.resample_by_res(self.vol, [1.6, 1.6, 2.3], interpolator = sitk.sitkLinear, logging = True)
        mask = self.resample_lb_by_res(mask, [1.6, 1.6, 2.3], interpolator = sitk.sitkLinear, ref_img = self.vol, logging = True)
        
        assert mask.GetSize() == self.vol.GetSize()

        # Center Crop
        self.roi2 = self.vol.GetSize()[-1]
        if self.vol.GetSize()[-1] >= self.roi2 and self.vol.GetSize()[1] >= self.roi and self.vol.GetSize()[0] >= self.roi:
            self.vol = self.center_crop_nii(self.vol, crop_size=[self.roi, self.roi, self.roi2])
            mask = self.center_crop_nii(mask, crop_size=[self.roi, self.roi, self.roi2])
        else:
            print(self.fname, " needs pad_center, its size = ", self.vol.GetSize())
            self.vol = self.pad_center(self.vol)
            self.vol = self.center_crop_nii(self.vol, crop_size=[self.roi, self.roi, self.roi2])
            mask = self.pad_center(mask)
            mask = self.center_crop_nii(mask, crop_size=[self.roi, self.roi, self.roi2])
        assert self.vol.GetSize() == (self.roi, self.roi, self.roi2)
        assert mask.GetSize() == (self.roi, self.roi, self.roi2)
        
        self.save(mask)


def worker(fname, extractor):
        extractor.run(fname)  # fname = Case_00580_0000.nii.gz



if __name__ == "__main__": 
    input_path = './data/ABDOMENCT1K/nii'
    img_output_path = './data/ABDOMENCT1K/image_resampled'
    mask_output_path = './data/ABDOMENCT1K/label_resampled'
    os.makedirs(img_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    
    extractor = CTExtractor(input_path, img_output_path, mask_output_path)

    def worker_partial(fname):
        return worker(fname, extractor)

    fnames = os.listdir(input_path)
    fnames = [fid for fid in sorted(fnames)]
    print('total # of niis', len(fnames))

    with Pool(processes=4) as pool:
        res = list(tqdm(pool.imap(worker_partial, iter(fnames)), total=len(fnames)))
