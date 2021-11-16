#!/bin/bash


# This script is modified version of https://github.com/dihardchallenge/dihard3_baseline/blob/master/recipes/track1/run.sh,
# which is part of DIHARD III baseline.
# Till stage 2, involves feature extraction and x-vector extraction using pretrained model
# Stage 2 and Stage 3 involves converting x-vectors into numpy format and creating data folders needed for SSC training

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
################################################################################
# Paths to DIHARD III releases
################################################################################
default_dev=/data/working/nryant/dihard3/delivery/builds/LDC2020E12_Third_DIHARD_Challenge_Development_Data

default_eval=/data/working/nryant/dihard3/delivery/builds/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete

DIHARD_DEV_DIR=$default_dev
DIHARD_EVAL_DIR=$default_eval

stage=0
modelpath=etdnn_fbank_xvector_models/exp/xvector_nnet_1a/
nnet_dir=exp/xvector_nnet_1a/  # path of xvector model
conf=etdnn_fbank_xvector_models/conf
njobs=40

. utils/parse_options.sh

if [ $DIHARD_DEV_DIR = $default_dev -o $DIHARD_EVAL_DIR = $default_eval ]; then
  echo " Usage
  $0 --DIHARD_DEV_DIR <path of dev set> --DIHARD_EVAL_DIR <path of eval set> --stage <0/1/2/3>
  DIHARD_DEV_DIR: path of dihard III development dataset
  DIHARD_EVAL_DIR: path of dihard III evaluation dataset.  
  stage: 0 (default) - data folder generation till end, 1: MFCC extraction till end, 2: X-vector extraction till end, 3: SelfSup data preparation
  
  "
  
  exit 1;
fi

# Prepare datasets
if [ $stage -le 0 ]; then

  if [ ! -d "utils" ];then
      ln -sf $KALDI_ROOT/egs/wsj/s5/utils .
  fi

  if [ ! -d "steps" ];then
      ln -sf $KALDI_ROOT/egs/wsj/s5/steps .
  fi

   
  
  # dev
  local/make_data_dir.py \
  --rttm-dir $DIHARD_DEV_DIR/data/rttm \
    data/dihard_dev_2020 \
    $DIHARD_DEV_DIR/data/flac \
    $DIHARD_DEV_DIR/data/sad
  utils/utt2spk_to_spk2utt.pl \
    data/dihard_dev_2020/utt2spk > data/dihard_dev_2020/spk2utt
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/dihard_dev_2020/

  # eval
  local/make_data_dir.py \
    --rttm-dir $DIHARD_EVAL_DIR/data/rttm \
    data/dihard_eval_2020 \
    $DIHARD_EVAL_DIR/data/flac \
    $DIHARD_EVAL_DIR/data/sad
  utils/utt2spk_to_spk2utt.pl \
    data/dihard_eval_2020/utt2spk > data/dihard_eval_2020/spk2utt
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/dihard_eval_2020/

fi

# Prepare features
if [ $stage -le 1 ]; then
 
  for name in dihard_dev_2020 dihard_eval_2020; do
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config $conf/fbank.conf --nj $njobs \
        --cmd "${train_cmd}" data/${name} exp/make_fbank/${name} $fbankdir
    utils/fix_data_dir.sh data/${name}
  done
 
  for name in dihard_dev_2020 dihard_eval_2020; do
    local/nnet3/xvector/prepare_feats.sh --nj $njobs --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
      cp data/$name/segments data/${name}_cmn/
      
      utils/fix_data_dir.sh data/${name}_cmn
  done
fi

  # Extract x-vectors
if [ $stage -le 2 ]; then
  # Extract x-vectors for the two partitions of callhome.
  for dataset in dihard_dev_2020 dihard_eval_2020; do
    local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
      --nj 40 --window 1.5 --period 0.25 --apply-cmn false \
      --min-segment 0.5 $modelpath \
      data/${dataset}_cmn $nnet_dir/xvectors_${dataset}
  done
fi

# convert x-vectors from ark to numpy , convert kaldi plda, transform.mat, mean.vec into pickle format
#and copy spk2utt,utt2spk, segments in lists folder
SSC_fold=`pwd`/../
if [ $stage -le 3 ]; then
# converts x-vectors from ark to numpy and convert kaldi models into pickle
for dataset in dihard_dev_2020 dihard_eval_2020; do
    srcdir=$nnet_dir/xvectors_${dataset}   # path of xvectors.scp
    awk '{print $1}' $srcdir/spk2utt > data/$dataset/${dataset}.list
    python $SSC_fold/services/read_scp_write_npy_embeddings.py vec $srcdir/xvector.scp xvectors_npy/${dataset}/ data/$dataset2/${dataset2}.list
    
done
python $SSC_fold/services/convert_kaldi_to_pkl.py --kaldi_feats_path $modelpath/xvectors_dihard --dataset dihard_dev_2020 --output_dir $SSC_fold

fi
if [ $stage -le 4 ]; then
# copy spk2utt,utt2spk, segments in lists folder required for training
for dataset in dihard_dev_2020 dihard_eval_2020; do
    srcdir=$nnet_dir/xvectors_${dataset}   # path of xvectors.scp
    mkdir -p $SSC_fold/lists/$dataset/tmp
    cp $srcdir/spk2utt $SSC_fold/lists/$dataset/tmp/spk2utt
    cp $srcdir/utt2spk $SSC_fold/lists/$dataset/tmp/utt2spk
    cp $srcdir/segments $SSC_fold/lists/$dataset/tmp/segments
    cp data/$dataset/reco2num_spk $SSC_fold/lists/$dataset/reco2num_spk
    cp data/$dataset/reco2num_spk $SSC_fold/lists/$dataset/tmp/reco2num_spk

    awk '{print $1}' $srcdir/spk2utt > $SSC_fold/lists/$dataset/${dataset}.list
    cp $SSC_fold/lists/$dataset/$dataset.list $SSC_fold/lists/$dataset/tmp/dataset.list
     
   mkdir data/$dataset/filewise_rttms
   # store segments filewise in folder segments_xvec and create filewise_rttms
    mkdir -p $SSC_fold/lists/$dataset/segments_xvec
    cat $SSC_fold/lists/$dataset/${dataset}.list | while read i; do
        grep $i $SSC_fold/lists/$dataset/tmp/segments > $SSC_fold/lists/$dataset/segments_xvec/${i}.segments
        grep $i data/$dataset/rttm > data/$dataset/filewise_rttms/${i}.rttm
    done
done

fi
