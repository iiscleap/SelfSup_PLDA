#!/bin/bash
# @author: prachi singh 
# @email: prachisingh@iisc.ac.in 

# This script calls DNN training python script xvec_ahc_train.py 
# All the parameters needed for training are passed as argument in this script
# If system supports multiple jobs then TYPE is set as parallel and nj represents number of jobs
# which_python variable is to pass the python full path where the required libraries are installed otherwise it will throw errors
# --reco2num_spk lists/$dataset/tmp/split$nj/JOB/reco2num_spk \

# tf_events/xvec_ssc_train_th_0.0_onlyahc_phi_0.5/${dataset}_scores/ - only ahc for filelength < 500

# tf_events/xvec_ssc_train_th_0.0_onlyahcfull_phi_0.5/${dataset}_scores/ - only ahc for all files
kaldi_recipe_path=/data1/prachis/Dihard_2020/Dihard_2020_track1
# $kaldi_recipe_path/run_extract_xvectors_2.sh
dataset_ftdnn=dihard_dev_2020_track1_wide
band=wide
dataset=dihard_dev_2020_track1_$band  # interchange for training callhome2 files
dataset2=dihard_dev_2020_track1_wide
# outf=tf_events/xvec_ssc_train_th_0.0_onlyahc_phi_0.5/${dataset}_scores/ # output folder path
# outf=tf_events/xvec_ssc_train_th_0.0_onlyahcfull/${dataset}_scores/
outf=tf_events/xvec_ssc_train_pic_alpha_1.0/${dataset_ftdnn}_scores/

TYPE=parallel # training parallely multiple utterances
nj=40 # number of jobs for parallelizing
which_python=/home/prachis/miniconda3/envs/mytorch/bin/python # python with all needed installation

cosinefold=cosine_pca_baseline/${dataset_ftdnn}_scores/cosine_scores/
pldafold=plda_pca_baseline/${dataset_ftdnn}_scores/plda_scores/
rm -rf $cosinefold
# mv $cosinefold cosine_pca_baseline/${dataset_ftdnn}_scores/cosine_scores_old/
# mv $pldafold plda_pca_baseline/${dataset_ftdnn}_scores/plda_scores_old

mkdir -p $cosinefold
mkdir -p $pldafold
mkdir -p $outf/

. ./cmd.sh
. ./path.sh

. ./utils/parse_options.sh

main_dir=/data1/prachis/Dihard_2020/SSC/

if [ $main_dir = "default" -o $kaldi_recipe_path = "default" ]; then
    echo "need main_directory full path as argument"
    echo " Set arguments for training in the code"
    echo "Usage : bash run_xvec_ahc.sh --TYPE <parallel/None> --nj <number of jobs> --which_python <python with all requirements> <full path of main directory>"
    exit 1
fi

# reco2num_spk: lists/$dataset/tmp/split$nj/JOB/reco2numspk_ahcinit_eend_overlap

if [ $TYPE == "parallel" ]; then 
     if [ ! -d lists/$dataset/tmp/split$nj ] || [ ! "$(ls -A lists/$dataset/tmp/split$nj/1)" ]; then
        
        utils/split_data_mine.sh $main_dir/lists/$dataset/tmp $nj || exit 1;
        
    fi
    
    JOB=2
    # for i in 10 19 21 26 40 4; do
    $train_cmd JOB=1:$nj $outf/log_plda/Deep_SSC.JOB.log \
    $which_python xvec_ssc_train.py \
    --which_python $which_python \
    --gpuid '0' \
    --N_batches 50 \
    --epochs 10 \
    --lr 1e-3 \
    --eta 0.5 \
    --alpha 1.0 \
    --band $band \
    --dataset $dataset_ftdnn \
    --outf $outf \
    --xvecpath $kaldi_recipe_path/xvectors_npy/${dataset_ftdnn}/ \
    --filetrain_list lists/$dataset/tmp/split$nj/JOB/dataset.list \
    --reco2utt_list lists/$dataset/tmp/split$nj/JOB/spk2utt \
    --segments lists/$dataset/segments_xvec \
    --kaldimodel lists/$dataset2/plda_${dataset2}.pkl \
    --rttm_ground_path $kaldi_recipe_path/data/$dataset/filewise_rttms/ \
    --kaldi_recipe_path $kaldi_recipe_path
    # done
else
    $which_python xvec_ssc_train.py \
    --which_python $which_python \
    --gpuid '0' \
    --batchSize 64 \
    --N_batches 1 \
    --epoch 10 \
    --lr 1e-3 \
    --eta 0.2 \
    --lamda 0.0 \
    --gamma 0.4 \
    --dataset $dataset \
    --outf $outf \
    --xvecpath $kaldi_recipe_path/xvectors_npy/${dataset}/ \
    --filetrain_list lists/$dataset/${dataset}.list \
    --reco2utt_list lists/$dataset/tmp/spk2utt \
    --threshold 'None' \
    --segments lists/$dataset/segments_xvec \
    --reco2num_spk swbd_diar/data/$dataset/reco2num_spk \
    --kaldimodel lists/$dataset2/plda_${dataset2}.pkl \
    --rttm_ground_path $kaldi_recipe_path/data/$dataset/filewise_rttms/

fi
