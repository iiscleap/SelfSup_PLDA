#!/bin/bash


# This script is modified version of egs/callhome_diarization/v2/run.sh,
# Till stage 2, involves feature extraction and x-vector extraction using pretrained model
# Stage 2 and Stage 3 involves converting x-vectors into numpy format and creating data folders needed for SSC training

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=/export/corpora/NIST/LDC2001S97/   # callhome dataset path
stage=0
nnet_dir=etdnn_fbank_xvector_models/exp/xvector_nnet_1a/  # path of xvector model
conf=etdnn_fbank_xvector_models/conf
njobs=40
# Prepare datasets
if [ $stage -le 0 ]; then

  if [ ! -d "utils" ];then
      ln -sf $KALDI_ROOT/egs/wsj/s5/utils .
  fi

  if [ ! -d "steps" ];then
      ln -sf $KALDI_ROOT/egs/wsj/s5/steps .
  fi

fi

# Prepare features
if [ $stage -le 1 ]; then
 
  for name in dihard_dev_2020 dihard_eval_2020; do
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config $conf/fbank_16k.conf --nj $njobs \
        --cmd "${train_cmd}" data/${name} exp/make_fbank/${name} $fbankdir
    utils/fix_data_dir.sh data/${name}
  done
  
  for name in dihard_dev_2020 dihard_eval_2020; do
    local/nnet3/xvector/prepare_feats.sh --nj $njobs --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
      utils/fix_data_dir.sh data/${name}_cmn
  done
fi

  # Extract x-vectors
if [ $stage -le 2 ]; then
  # Extract x-vectors for the two partitions of callhome.
  for dataset in dihard_dev_2020 dihard_eval_2020; do
    diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
      --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
      --min-segment 0.5 $nnet_dir \
      data/${dataset}_cmn $nnet_dir/xvectors_${dataset}
  done
fi

# convert x-vectors from ark to numpy , convert kaldi plda, transform.mat, mean.vec into pickle format
#and copy spk2utt,utt2spk, segments in lists folder

if [ $stage -le 3 ]; then
# converts x-vectors from ark to numpy and convert kaldi models into pickle
for dataset in dihard_dev_2020 dihard_eval_2020; do
    srcdir=$nnet_dir/xvectors_${dataset}   # path of xvectors.scp
    awk '{print $1}' $srcdir/spk2utt > data/$dataset/${dataset}.list
    python $SSC_fold/services/read_scp_write_npy_embeddings.py vec $srcdir/xvector.scp xvectors_npy/${dataset}/ data/$dataset2/${dataset2}.list
    python $SSC_fold/services/convert_kaldi_to_pkl.py --kaldi_feats_path $nnet_dir/xvectors_$dataset --dataset $dataset --output_dir $SSC_fold

done
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
