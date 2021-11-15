#!/bin/bash
# @author: Prachi Singh 
# @email: prachisingh@iisc.ac.in 

# This script calls DNN training python script xvec_ssc_plda_train.py 
# All the parameters needed for training are passed as argument in this script
# If system supports multiple jobs then TYPE is set as parallel and nj represents number of jobs
# which_python variable is to pass the python full path where the required libraries are installed otherwise it will throw errors
# dataset_etdnn : data dir on which to evaluate
# dataset_plda : path of plda pickle file
# outf :
echo "set dataset_etdnn: name of data directory to train and evaluate options : dihard_dev_2020, dihard_eval_2020 
ami_dev, ami_eval"
echo "outf: path of output directory"
echo "lr: learning rate"
echo "alpha:"
echo "eta: factor to reduce the lr after saturation for few epochs/minibatches"

band=wide

dataset_etdnn=dihard_eval_2020

dataset=$dataset_etdnn

dataset_plda=dihard_dev_2020

alpha=0.8
eta=0.5

outf=exp/selfsup_pic_alpha${alpha}_eta${eta}/${dataset_etdnn}_scores/


TYPE=parallel # training parallely multiple utterances
nj=40 # number of jobs for parallelizing
which_python=python # python with all needed installation
kaldi_recipe_path=/data1/prachis/Dihard_2020/Dihard_2020_track1
pldafold=plda_pca_baseline/${dataset_etdnn}_scores/plda_scores/
echo "path of plda basline scores: $pldafold"
mkdir -p $pldafold
mkdir -p $outf/

. ./cmd.sh
. ./path.sh

. ./utils/parse_options.sh

main_dir=/data1/prachis/Dihard_2020/SSC/

if [ $main_dir = "default" -o $kaldi_recipe_path = "default" ]; then
	echo "need main_directory full path as argument"
	echo " Set arguments for training in the code"
	echo "Usage : bash run_xvec_ssc_asru.sh --TYPE <parallel/None> --nj <number of jobs> --which_python <python with all requirements> <full path of main directory>"
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
	$which_python xvec_ssc_plda_train.py \
	--which_python $which_python \
	--gpuid '0' \
	--N_batches 50 \
	--epochs 5 \
	--lr 1e-3 \
    --eta $eta \
	--alpha $alpha \
	--band $band \
	--dataset $dataset_etdnn \
	--outf $outf \
	--xvecpath $kaldi_recipe_path/xvectors_npy/${dataset_etdnn}/ \
	--reco2num_spk lists/$dataset_etdnn/tmp/split$nj/JOB/reco2num_spk_ahc_init_th-0.7 \
	--filetrain_list lists/$dataset/tmp/split$nj/JOB/dataset.list \
	--reco2utt_list lists/$dataset/tmp/split$nj/JOB/spk2utt \
	--segments lists/$dataset/segments_xvec \
	--kaldimodel lists/$dataset_plda/plda_${dataset_plda}.pkl \
	--rttm_ground_path $kaldi_recipe_path/data/$dataset/filewise_rttms/ \
	--kaldi_recipe_path $kaldi_recipe_path
	# done
else
	$which_python xvec_ssc_plda_train.py \
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
	--kaldimodel lists/$dataset_plda/plda_${dataset_plda}.pkl \
	--rttm_ground_path $kaldi_recipe_path/data/$dataset/filewise_rttms/

fi
