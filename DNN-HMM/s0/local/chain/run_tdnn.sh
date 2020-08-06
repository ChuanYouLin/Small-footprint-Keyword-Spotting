#!/bin/bash

# This script is based on run_tdnn_1a.sh in formosa chain recipe.

set -e -o pipefail

# folder configs
exp_dir=exp
lang_dir=lang
lmdir=tcc300_lm
mfcc_dir=feats/mfcc

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
# 13
stage=0
nj=10
train_set=train
test_sets="dev test"
gmm=tri5         # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# training chunk-options
#chunk_width=140,100,160
chunk_width=90,75,60
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=true

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

echo "--exp_dir $exp_dir --lang_dir $lang_dir --mfcc_dir $mfcc_dir"

# if we are using the speed-perturbed data we need to generate
# alignments for it.
# local/nnet3/run_ivector_common.sh --stage $stage --nj $nj --gmm $gmm --num-threads-ubm $num_threads_ubm \
#    --train_set $train_set --test_sets "$test_sets" \
#    --exp_dir $exp_dir --lang_dir $lang_dir --mfcc_dir $mfcc_dir --lmdir $lmdir || exit 1;

data_dir=$mfcc_dir
gmm_dir=$exp_dir/${gmm}
ali_dir=$exp_dir/${gmm}_ali_${train_set}_sp
lat_dir=$exp_dir/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=$exp_dir/chain${nnet3_affix}/tdnn${affix:+_$affix}_sp
train_data_dir=$data_dir/${train_set}_sp_hires
train_ivector_dir=$exp_dir/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=$data_dir/${train_set}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=$exp_dir/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang_chain=lang_chain

# check is files exitis
for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 12 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang_chain ]; then
    if [ $lang_chain/L.fst -nt $lang_dir/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r $lang_dir $lang_chain
    silphonelist=$(cat $lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang_chain/topo || exit 1;
  fi
fi

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} $lang_dir $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  rm -rf $tree_dir
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} $lang_chain $ali_dir $tree_dir || exit 1;
fi

if [ $stage -le 15 ]; then
  rm -rf $dir
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=30 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # input-dim = input * frame + ivector(30)
  # 230 * x + x
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=25
  # relu-renorm-layer name=tdnn2 dim=512 input=Append(-1,0,1)
  # relu-renorm-layer name=tdnn3 dim=512 input=Append(-1,0,1)
  # relu-renorm-layer name=tdnn4 dim=512 input=Append(-3,0,3)
  # relu-renorm-layer name=tdnn5 dim=512 input=Append(-3,0,3)
  # relu-renorm-layer name=tdnn6 dim=512 input=Append(-6,-3,0)

  ## adding the layers for chain branch
  # x * 200 + 200
  # relu-renorm-layer name=prefinal-chain dim=512 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  # x * x + x
  relu-renorm-layer name=prefinal-xent input=tdnn1 dim=512 target-rms=0.5
  # x * 200 + 200
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 16 ]; then

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=2 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0 --constrained false" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir || exit 1;
fi

if [ $stage -le 17 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh $lang_dir/$lmdir/phones.txt $lang_chain/phones.txt || exit 1;
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_dir/$lmdir $tree_dir $tree_dir/graph || exit 1;
  utils/mkgraph_lookahead.sh --self-loop-scale 1.0 $lang_dir/$lmdir $tree_dir $tree_dir/graph_lookahead || exit 1;
  # utils/mkgraph.sh --self-loop-scale 1.0 $lang_dir/$lmdir $dir $dir/graph || exit 1;
fi

if [ $stage -le 18 ]; then
  for data in $test_sets; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" \
      --online-ivector-dir $exp_dir/nnet3${nnet3_affix}/ivectors_${data}_hires \
      $tree_dir/graph $data_dir/${data}_hires $dir/decode_${data} || exit 1;
    steps/nnet3/decode_lookahead.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" \
      --online-ivector-dir $exp_dir/nnet3${nnet3_affix}/ivectors_${data}_hires \
      $tree_dir/graph_lookahead $data_dir/${data}_hires $dir/decode_${data}_lookahead || exit 1;
  done
  wait;
fi

exit 0;
