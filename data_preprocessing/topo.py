import gudhi as gd
import glob
import os
import numpy as np
from multiprocessing import Pool
from datetime import datetime

def calc_topo_slice(slice):
    cc_0 = gd.CubicalComplex(dimensions=slice.shape, top_dimensional_cells=1-(slice == 0).flatten().astype(int))
    cc_1 = gd.CubicalComplex(dimensions=slice.shape, top_dimensional_cells=1-(slice == 1).flatten().astype(int))
    cc_2 = gd.CubicalComplex(dimensions=slice.shape, top_dimensional_cells=1-(slice == 2).flatten().astype(int))

    diag_0 = cc_0.persistence()
    diag_1 = cc_1.persistence()
    diag_2 = cc_2.persistence()
    
    topo_vector = []
    for _, diag in enumerate([diag_0, diag_1, diag_2]):
        count_label_0 = 0
        count_label_1 = 0
        for idx, (label, (birth, death)) in enumerate(diag):
            if label == 0:
                count_label_0 += 1
            elif label == 1:
                count_label_1 += 1

        topo_vector.append(count_label_0)
        topo_vector.append(count_label_1)
    
    if topo_vector[4] != 1 and topo_vector[5] != 0:
        print(topo_vector)
    
    return topo_vector

def calc_topo(mask):  
    d, h, w = mask.shape
    topo_vector = []
    for slice_idx in range(d):
        topo_vector_slice = calc_topo_slice(mask[slice_idx])
        topo_vector.extend(topo_vector_slice)

    return np.array(topo_vector)
     
class CTExtractor:
    def __init__(self, mask_path):
        self.save_path = mask_path.replace("mask", "topo")
        os.makedirs(self.save_path, exist_ok=True)

    def run(self, mask_path_i):
        topo_path = mask_path_i.replace("mask", "topo")
        mask_array = np.load(mask_path_i)
        topo = calc_topo(mask_array)
        print(mask_path_i, topo.shape)
        np.save(topo_path, topo)

def worker_partial(mask_path_i):
    return extractor.run(mask_path_i)

if __name__ == "__main__": 
    mask_path = "./mask"
    mask_list = glob.glob(mask_path + "/*.npy")
    mask_list.sort()
    extractor = CTExtractor(mask_path)

    print("[Calculating topo]", datetime.now())
    with Pool(processes=4) as pool:
        res = list(pool.imap(worker_partial, iter(mask_list)))
    print("[Finishing Calculating]", datetime.now())
