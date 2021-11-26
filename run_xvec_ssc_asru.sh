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
echo "################################################INFO########################################################"
echo "set dataset_etdnn: name of data directory to train and evaluate options : dihard_dev_2020, dihard_eval_2020 
ami_dev, ami_eval (default:dihard_eval_2020)"
echo "outf: path of output directory (default:exp/selfsup_pic_eta0.5/dihard_eval_2020_scores/)"
echo "lr: learning rate (default:0.001)"
echo "eta: factor to reduce the lr after saturation for few epochs/minibatches (default:0.5)"
echo "clustering: type of clustering for initialization pic/ahc (Default:pic)"
echo "###############################################################################################################"

band=wide

dataset_etdnn=dihard_eval_2020

dataset=$dataset_etdnn

dataset_plda=dihard_dev_2020
clustering=pic # or ahc
eta=0.5

outf=exp/selfsup_${clustering}_eta${eta}/${dataset_etdnn}_scores/


TYPE=parallel # training parallely multiple utterances
nj=40 # number of jobs for parallelizing
which_python=python # python with all needed installation


pldafold=plda_pca_baseline/${dataset_etdnn}_scores/plda_scores/
echo "path of plda basline scores: $pldafold"
mkdir -p $pldafold
mkdir -p $outf/

. ./cmd.sh
. ./path.sh

if [ ! -d "utils" ];then
      ln -sf $KALDI_ROOT/egs/wsj/s5/utils .
fi

main_dir=default
. ./utils/parse_options.sh


kaldi_recipe_path=$main_dir/tools_diar

if [ $main_dir = "default" -o $kaldi_recipe_path = "default" ]; then
	echo "need main_directory full path as argument"
	echo " Set arguments for training in the code"
	echo "Usage : bash run_xvec_ssc_asru.sh --TYPE <parallel/None> --nj <number of jobs> --which_python <python with all requirements> --main_dir <full path of main directory>"
	exit 1
fi
njobs=`cat lists/$dataset_etdnn/$dataset_etdnn.list | wc -l`
nj=$((${nj}<${njobs}?${nj}:${njobs}))
# reco2num_spk: lists/$dataset/tmp/split$nj/JOB/reco2numspk_ahcinit_eend_overlap

if [ $TYPE == "parallel" ]; then 
     if [ ! -d lists/$dataset/tmp/split$nj ] || [ ! "$(ls -A lists/$dataset/tmp/split$nj/1)" ]; then
        
    	tools_diar/split_data_mine.sh $main_dir/lists/$dataset/tmp $nj || exit 1;
        
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
	--clustering $clustering \
	--band $band \
	--dataset $dataset_etdnn \
	--outf $outf \
	--xvecpath $kaldi_recipe_path/xvectors_npy/${dataset_etdnn}/ \
	--reco2num_spk lists/$dataset_etdnn/tmp/split$nj/JOB/reco2num_spk \
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
	--N_batches 50 \
	--epoch 5 \
	--lr 1e-3 \
    --eta $eta \
	--clustering $clustering \
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
