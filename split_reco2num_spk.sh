numsplit=40
#data=lists/dihard_dev_2020_track1_fbank_jhu/tmp/
#data=lists/dihard_eval_2020_track1_fbank_jhu_wide/tmp
data=lists/dihard_eval_resnet_2020_track1/tmp/
reco2num_spk=reco2num_spk_ahc_init_th-0.2
for n in `seq $numsplit`; do
dsn=$data/split${numsplit}${utt}/$n
#rm $dsn/$reco2num_spk
cat $dsn/dataset.list | while read i; do

grep $i $data/$reco2num_spk >> $dsn/$reco2num_spk
done
done
