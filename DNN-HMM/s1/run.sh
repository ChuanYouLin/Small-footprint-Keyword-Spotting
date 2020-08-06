#!/bin/bash

ln -s ../../wsj/s5/steps ./
ln -s ../../wsj/s5/utils ./

stage=0
train=true   # set to false to disable the training-related scripts
             # note: you probably only want to set --train false if you
             # are using at least --stage 1.
decode=true  # set to false to disable the decoding-related scripts.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.


if [ $stage -le 0 ]; then
  # make MFCC features.
  for x in train dev test; do
    utils/copy_data_dir.sh data/$x feats/mfcc/$x
    steps/make_mfcc.sh --nj 32 --mfcc-config conf/mfcc.conf --cmd "$train_cmd" feats/mfcc/$x
    steps/compute_cmvn_stats.sh feats/mfcc/$x
  done
fi

if [ $stage -le 1 ]; then
  # monophone
  if $train; then
    steps/train_mono.sh --num_iters 40 --boost-silence 1.25 --nj 32 --cmd "$train_cmd" \
      feats/mfcc/train lang/ exp/mono0a || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh lang/google_lm/ exp/mono0a exp/mono0a/graph || exit 1;
	  for data in dev test;do
		  steps/decode.sh --nj 32 --cmd "$decode_cmd" exp/mono0a/graph \
		    feats/mfcc/$data exp/mono0a/decode_$data
	done
  fi
fi

if [ $stage -le 2 ]; then
  # tri1
  if $train; then
    steps/align_si.sh --nj 32 --cmd "$train_cmd" \
      feats/mfcc/train lang exp/mono0a exp/mono0a_ali || exit 1;

    steps/train_deltas.sh --num_iters 40 --cmd "$train_cmd" 250 350 \
      feats/mfcc/train lang exp/mono0a_ali exp/tri1 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh lang/google_lm \
      exp/tri1 exp/tri1/graph_tgpr || exit 1;

    for data in dev test; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode.sh --nj 32 --cmd "$decode_cmd" exp/tri1/graph_tgpr \
        feats/mfcc/${data} exp/tri1/decode_tgpr_${data} || exit 1;
    done
  fi
fi

if [ $stage -le 3 ]; then
  # tri2
  if $train; then
    steps/align_si.sh --boost-silence 1.25 --nj 32 --cmd "$train_cmd" \
      feats/mfcc/train lang exp/tri1 exp/tri1_ali || exit 1;

    steps/train_deltas.sh --num_iters 35 --boost-silence 1.25 --cmd "$train_cmd" 350 550 \
      feats/mfcc/train lang exp/tri1_ali exp/tri2 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh lang/google_lm \
      exp/tri2 exp/tri2/graph_tgpr || exit 1;

    for data in dev test; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode.sh --nj 32 --cmd "$decode_cmd" exp/tri2/graph_tgpr \
        feats/mfcc/${data} exp/tri2/decode_tgpr_${data} || exit 1;
    done
  fi
fi

if [ $stage -le 4 ]; then
  # tri3
  if $train; then
    steps/align_si.sh --nj 32 --cmd "$train_cmd" \
      feats/mfcc/train lang exp/tri2 exp/tri2_ali || exit 1;

    steps/train_lda_mllt.sh --num_iters 35 --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" 350 550 \
      feats/mfcc/train lang exp/tri2_ali exp/tri3 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh lang/google_lm \
      exp/tri3 exp/tri3/graph_tgpr || exit 1;
    for data in dev test; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode.sh --nj 32 --cmd "$decode_cmd" exp/tri3/graph_tgpr \
        feats/mfcc/${data} exp/tri3/decode_tgpr_${data} || exit 1;
    done
  fi
fi

if [ $stage -le 5 ]; then
  # tri4
  if $train; then
    steps/align_si.sh  --nj 32 --cmd "$train_cmd" \
      feats/mfcc/train lang exp/tri3 exp/tri3_ali  || exit 1;

    steps/train_sat.sh --num_iters 35 --cmd "$train_cmd" 350 550 \
      feats/mfcc/train lang exp/tri3_ali exp/tri4 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh lang/google_lm \
      exp/tri4 exp/tri4/graph_tgpr || exit 1;

    for data in dev test; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode_fmllr.sh --nj 32 --cmd "$decode_cmd" \
        exp/tri4/graph_tgpr feats/mfcc/${data} \
        exp/tri4/decode_tgpr_${data} || exit 1;
    done
  fi
fi

if [ $stage -le 6 ]; then
  # tri5
  if $train; then
    steps/align_fmllr.sh  --nj 32 --cmd "$train_cmd" \
      feats/mfcc/train lang exp/tri4 exp/tri4_ali  || exit 1;

    steps/train_sat.sh --num_iters 35 --cmd "$train_cmd" 500 800 \
      feats/mfcc/train lang exp/tri4_ali exp/tri5 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh lang/google_lm \
      exp/tri5 exp/tri5/graph_tgpr || exit 1;

    for data in dev test; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode_fmllr.sh --nj 32 --cmd "$decode_cmd" \
        exp/tri5/graph_tgpr feats/mfcc/${data} \
        exp/tri5/decode_tgpr_${data} || exit 1;
    done
  fi
fi

if [ $stage -le 7 ]; then
  # nnet
  local/nnet3/run_ivector_common.sh || exit 1;
  local/chain/run_tdnn.sh --nj 32 \
    --exp_dir exp --lang_dir lang --mfcc_dir feats/mfcc --lmdir google_lm \
    --train_set train --test_sets "dev test" || exit 1;
fi
