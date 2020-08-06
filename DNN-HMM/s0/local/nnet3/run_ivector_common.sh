#!/bin/bash

set -e -o pipefail

# This script is called from scripts like local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more scripts).  It
# contains the common feature preparation and iVector-related parts of the
# script.  See those scripts for examples of usage.


stage=0
nj=32
train_set=train  # you might set this to e.g. train.
test_sets="test dev"
gmm=tri5                 # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.

num_threads_ubm=32

# folder configs
exp_dir=exp
lang_dir=lang
lmdir=google_lm
mfcc_dir=feats/mfcc

nj_extractor=5
# It runs a JOB with '-pe smp N', where N=$[threads*processes]
num_processes_extractor=2
num_threads_extractor=2

nnet3_affix=             # affix for exp/nnet3 directory to put iVector stuff in (e.g.
                         # in the tedlium recip it's _cleaned).

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

data_dir=$mfcc_dir
gmm_dir=$exp_dir/${gmm}
ali_dir=$exp_dir/${gmm}_ali_${train_set}_sp

# remove folder
rm -rf $data_dir/${train_set}_sp
for data in ${train_set}_sp ${test_sets}; do
  rm -rf $data_dir/${data}_hires
done
# check files
for f in $data_dir/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

# check files 2
if [ $stage -le 2 ] && [ -f $data_dir/${train_set}_sp_hires/feats.scp ]; then
  echo "$0: $data_dir/${train_set}_sp_hires/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi

# data speed edit to have more data 
if [ $stage -le 1 ]; then
  echo "$0: preparing directory for speed-perturbed data"
  utils/data/perturb_data_dir_speed_3way.sh $data_dir/${train_set} $data_dir/${train_set}_sp
fi


if [ $stage -le 2 ]; then
  echo "$0: creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  for data in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh $data_dir/$data $data_dir/${data}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh $data_dir/${train_set}_sp_hires   # $data_dir/${train_set}_sp_hires was created at line:69

  for data in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" $data_dir/${data}_hires  # $data_dir/${data}_hires/log & $data_dir/${data}_hires/data are default dir 
    steps/compute_cmvn_stats.sh $data_dir/${data}_hires
    utils/fix_data_dir.sh $data_dir/${data}_hires
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."

  mkdir -p $exp_dir/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=$exp_dir/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  num_utts_total=$(wc -l <$data_dir/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh $data_dir/${train_set}_sp_hires $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" --max-utts 10000 --subsample 2 \
      ${temp_data_root}/${train_set}_sp_hires_subset $exp_dir/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj --num-frames 700000 --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
	$exp_dir/nnet3${nnet3_affix}/pca_transform $exp_dir/nnet3${nnet3_affix}/diag_ubm

fi

if [ $stage -le 4 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj --num-threads $num_threads_extractor --num-processes $num_processes_extractor \
    $data_dir/${train_set}_sp_hires $exp_dir/nnet3${nnet3_affix}/diag_ubm $exp_dir/nnet3${nnet3_affix}/extractor || exit 1;
fi
##################
if [ $stage -le 5 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data; the utterance list is the same.
  ivectordir=$exp_dir/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  
  # We now extract iVectors on the speed-perturbed training data .  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online' (they vary within the utterance).

  # Having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (the iVector starts at zero at the beginning
  # of each pseudo-speaker).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $data_dir/${train_set}_sp_hires ${temp_data_root}/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_max2 \
    $exp_dir/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in ${test_sets}; do
    # nspk=$(wc -l <$data_dir/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      $data_dir/${data}_hires $exp_dir/nnet3${nnet3_affix}/extractor $exp_dir/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

if [ -f $data_dir/${train_set}_sp/feats.scp ] && [ $stage -le 7 ]; then
  echo "$0: $data_dir/${train_set}_sp/feats.scp already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi


if [ $stage -le 6 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    $data_dir/${train_set} $data_dir/${train_set}_sp
fi

if [ $stage -le 7 ]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data (needed for alignments)"
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $data_dir/${train_set}_sp
  steps/compute_cmvn_stats.sh $data_dir/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh $data_dir/${train_set}_sp
fi

if [ $stage -le 8 ]; then
  rm -rf $ali_dir
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $data_dir/${train_set}_sp $lang_dir/$lmdir $gmm_dir $ali_dir
fi


exit 0;