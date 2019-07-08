# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import torch.autograd as autograd

from . import data_utils, FairseqDataset

def get_adj(src_tokens, src_lengths, labels, node1, node2, labels_dict, node1_dict, node2_dict, pad_idx):

    # Not very nice but we do not have access to value comming from opt.gpuid command line parameter here.
    # use_cuda = batch.src[0].is_cuda

    batch_size = src_lengths.size(0)

    _MAX_BATCH_LEN = src_lengths[0]

    _MAX_DEGREE = 10  # If the average degree is much higher than this, it must be changed.

    sent_mask = torch.lt(torch.eq(src_tokens.transpose(0, 1), pad_idx), pad_idx)

    adj_arc_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 2), dtype='int32')
    adj_lab_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 1), dtype='int32')
    adj_arc_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 2), dtype='int32')
    adj_lab_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE, 1), dtype='int32')


    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE), dtype='float32')
    mask_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE), dtype='float32')
    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype='float32')

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(node1):  # iterates over the batch
        for a, arc in enumerate(de):

            arc_0 = labels_dict[labels[d, a]]

            if arc_0 == '<unk>' or arc_0 == '<pad>' or arc_0 == '</s>':
                pass
            else:
                arc_1 = int(node1_dict[arc])
                arc_2 = int(node2_dict[node2[d, a]])

                if arc_1 >= _MAX_BATCH_LEN - 1 or arc_2 >= _MAX_BATCH_LEN - 1:
                    continue

                if arc_1 in tmp_in:
                    tmp_in[arc_1] += 1
                else:
                    tmp_in[arc_1] = 0

                if arc_2 in tmp_out:
                    tmp_out[arc_2] += 1
                else:
                    tmp_out[arc_2] = 0

                idx_in = (d * _MAX_BATCH_LEN * _MAX_DEGREE) + arc_1 * _MAX_DEGREE + tmp_in[arc_1]

                idx_out = (d * _MAX_BATCH_LEN * _MAX_DEGREE) + arc_2 * _MAX_DEGREE + tmp_out[arc_2]

                if tmp_in[arc_1] < _MAX_DEGREE:

                    adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                    adj_lab_in[idx_in] = np.array([labels[d, a]])  # incoming arcs
                    mask_in[idx_in] = 1.

                if tmp_out[arc_2] < _MAX_DEGREE:

                    adj_arc_out[idx_out] = np.array([d, arc_1])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array([labels[d, a]])  # outgoing arcs
                    mask_out[idx_out] = 1.

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = autograd.Variable(torch.LongTensor(np.transpose(adj_arc_in).tolist()))
    adj_arc_out = autograd.Variable(torch.LongTensor(np.transpose(adj_arc_out).tolist()))

    adj_lab_in = autograd.Variable(torch.LongTensor(np.transpose(adj_lab_in).tolist()))
    adj_lab_out = autograd.Variable(torch.LongTensor(np.transpose(adj_lab_out).tolist()))

    mask_in = autograd.Variable(torch.FloatTensor(mask_in.reshape((_MAX_BATCH_LEN * node1.size()[0], _MAX_DEGREE)).tolist()), requires_grad=False)
    mask_out = autograd.Variable(torch.FloatTensor(mask_out.reshape((_MAX_BATCH_LEN * node2.size()[0], _MAX_DEGREE)).tolist()), requires_grad=False)
    mask_loop = autograd.Variable(torch.FloatTensor(mask_loop.tolist()), requires_grad=False)
    sent_mask = autograd.Variable(torch.FloatTensor(sent_mask.tolist()), requires_grad=False)
    # if use_cuda:
    #     adj_arc_in = adj_arc_in.cuda()
    #     adj_arc_out = adj_arc_out.cuda()
    #     adj_lab_in = adj_lab_in.cuda()
    #     adj_lab_out = adj_lab_out.cuda()
    #     mask_in = mask_in.cuda()
    #     mask_out = mask_out.cuda()
    #     mask_loop = mask_loop.cuda()
    #     sent_mask = sent_mask.cuda()
    return adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop, sent_mask

def collate(samples, labels_dict, node1_dict, node2_dict, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    labels = merge('labels', left_pad_source)
    labels = labels.index_select(0, sort_order)
    node1 = merge('node1', left_pad_source)
    node1 = node1.index_select(0, sort_order)
    node2 = merge('node2', left_pad_source)
    node2 = node2.index_select(0, sort_order)
    adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop, sent_mask \
        = get_adj(src_tokens, src_lengths, labels, node1, node2, labels_dict, node1_dict, node2_dict, pad_idx)

    return {
            'id': id,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'prev_output_tokens': prev_output_tokens,
                'adj_arc_in': adj_arc_in,
                'adj_arc_out': adj_arc_out,
                'adj_lab_in': adj_lab_in,
                'adj_lab_out': adj_lab_out,
                'mask_in': mask_in,
                'mask_out': mask_out,
                'mask_loop': mask_loop,
                'sent_mask': sent_mask
            },
            'target': target,
        }


class GCNDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        labels=None, labels_dict=None,
        node1=None, node1_dict=None,
        node2=None, node2_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.labels = labels
        self.node1 = node1
        self.node2 = node2
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.labels_dict = labels_dict
        self.node1_dict = node1_dict
        self.node2_dict = node2_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
            'labels': self.labels[index] if self.labels is not None else None,
            'node1': self.node1[index] if self.node1 is not None else None,
            'node2': self.node2[index] if self.node2 is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, labels_dict=self.labels_dict, node1_dict=self.node1_dict, node2_dict=self.node2_dict,
            pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'labels': self.labels[i] if self.labels is not None else None,
                'node1': self.node1[i] if self.node1 is not None else None,
                'node2': self.node2[i] if self.node2 is not None else None,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)
