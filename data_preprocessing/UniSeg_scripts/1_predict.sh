export nnUNet_raw_data_base="./data/userdisk0/ywye/nnUNet_raw"
export nnUNet_preprocessed="./erwen_SSD/1T/nnUNet_preprocessed"
export RESULTS_FOLDER="./data/userdisk0/ywye/nnUNet_trained_models"


CUDA_VISIBLE_DEVICES=0 nnUNet_n_proc_DA=32 nnUNet_predict -i ./data/userdisk0/ywye/nnUNet_raw/nnUNet_raw_data/Test/img/ -o ./data/userdisk0/ywye/nnUNet_raw/nnUNet_raw_data/Test/Predict/10/ -t 97 -m 3d_fullres  -tr UniSeg_Trainer -f 0 -task_id 3 -exp_name UniSeg_Trainer -num_image 1 -modality CT -spacing 2.3,1.6,1.6