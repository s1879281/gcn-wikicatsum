# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fairseq import utils
from fairseq.modules import ( GradMultiply, GCNLayer,
)

from . import (
    FairseqEncoder, FairseqGCNModel, register_model, register_model_architecture,
)

from .fconv import FConvDecoder, extend_conv_spec, Embedding, PositionalEmbedding, Linear, ConvTBC


@register_model('fconv_gcn_on_top')
class FConvGCNOnTopModel(FairseqGCNModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder.conv_encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--gcn_num_inputs', type=int, default=256,
                           help='Input size for the gcn layer')
        parser.add_argument('--gcn_num_units', type=int, default=256,
                           help='Output size for the gcn layer')
        parser.add_argument('--gcn_num_labels', type=int, default=10,
                           help='Number of labels for the edges of the gcn layer')
        parser.add_argument('--gcn_num_layers', type=int, default=1,
                           help='Number of gcn layers')
        parser.add_argument('--gcn_in_arcs', action="store_false", default=True,
                           help='Use incoming edges of the gcn layer')
        parser.add_argument('--gcn_out_arcs', action="store_false", default=True,
                           help='Use outgoing edges of the gcn layer')
        parser.add_argument('--gcn_batch_first', action="store_false", default=True,
                           help='Batchfirst for the gcn layer')
        parser.add_argument('--gcn_residual', type=str, default="",
                           choices=['residual', 'dense'],
                           help='Ddcide wich skip connection to use between GCN layers')
        parser.add_argument('--gcn_use_gates', action="store_true", default=False,
                           help='Switch to activate edgewise gates')
        parser.add_argument('--gcn_use_glus', action="store_true", default=False,
                           help='Node gates.')

        parser.add_argument('--dropout', default=0.2, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--normalization-constant', type=float, default=0.5, metavar='D',
                            help='multiplies the result of the residual block by sqrt(value)')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        conv_encoder = FConvEncoder3(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            normalization_constant=args.normalization_constant,
        )
        gcn_encoder = GCNEncoder2(
             dictionary=task.source_dictionary,
             dropout=args.dropout,
             num_inputs=args.gcn_num_inputs,
             num_units=args.gcn_num_units,
             num_labels=args.gcn_num_labels,
             num_layers=args.gcn_num_layers,
             in_arcs=args.gcn_in_arcs,
             out_arcs=args.gcn_out_arcs,
             batch_first=args.gcn_batch_first,
             residual=args.gcn_residual,
             use_gates=args.gcn_use_gates,
             use_glus=args.gcn_use_glus,
             normalization_constant=args.normalization_constant,
        )
        encoder = FConvGCNOnTopEncoder(task.source_dictionary, conv_encoder, gcn_encoder)
        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
            normalization_constant=args.normalization_constant,
        )
        return FConvGCNOnTopModel(encoder, decoder)


class FConvEncoder3(FairseqEncoder):
    """Convolutional encoder"""

    def __init__(
        self, dictionary, embed_dim=512, embed_dict=None, max_positions=1024,
        convolutions=((512, 3),) * 20, dropout=0.1, normalization_constant=0.5,
        left_pad=True,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.normalization_constant = normalization_constant
        self.left_pad = left_pad
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()

        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
            left_pad=self.left_pad
            # left_pad=False, #TODO: check LearnedPositionalEmbedding.forward() for the case of True
        )

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size,
                        dropout=dropout, padding=padding)
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        torch.set_printoptions(threshold=8000)

        x1 = self.embed_tokens(src_tokens)
        x2 = self.embed_positions(src_tokens)

        if src_tokens.lt(0).sum() > 0:
            print("negative voc idx (voc size {})".format(len(self.dictionary)))
            print(src_tokens)
            exit()

        x = x1 + x2

        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B  ###puts 1's where the pad index 0's otherwise
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(self.normalization_constant)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        if self.num_attention_layers > 0:
          x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(self.normalization_constant)

        return {
            'encoder_out': (input_embedding, y),           # modified here
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


class GCNEncoder2(FairseqEncoder):
    """ A graph convolutional encoder"""

    def __init__(self, dictionary, dropout,
                 num_inputs, num_units, num_labels, num_layers=1,
                 in_arcs=True,
                 out_arcs=True,
                 batch_first=False,
                 residual='',
                 use_gates=True,
                 use_glus=False,
                 # morph_embeddings=None,
                 left_pad=True,
                 normalization_constant=5):
        super(GCNEncoder2, self).__init__(dictionary)
        # num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.left_pad = left_pad
        self.dropout = dropout
        self.batch_first = batch_first
        self.normalization_constant = normalization_constant
        # self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        # if embed_dict:
        #     self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)
        #
        # self.embed_positions = PositionalEmbedding(
        #     max_positions,
        #     embed_dim,
        #     self.padding_idx,
        #     left_pad=self.left_pad
        #     # left_pad=False, #TODO: check LearnedPositionalEmbedding.forward() for the case of True
        # )

        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.residual = residual
        self.use_gates = use_gates
        self.use_glus = use_glus


        # if morph_embeddings is not None:
        #     self.morph_embeddings = morph_embeddings
        #     self.emb_morph_emb = nn.Linear(num_inputs+morph_embeddings.embedding_size, num_inputs)

        # self.H_1 = nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        # nn.init.xavier_normal_(self.H_1)
        # self.H_2 = nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        # nn.init.xavier_normal_(self.H_2)
        # self.H_3 = nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        # nn.init.xavier_normal_(self.H_3)
        # self.H_4 = nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        # nn.init.xavier_normal_(self.H_4)

        self.gcn_layers = []
        if residual == '' or residual == 'residual':

            for i in range(self.num_layers):
                gcn = GCNLayer(num_inputs, num_units, num_labels,
                               in_arcs=in_arcs, out_arcs=out_arcs,
                               batch_first=self.batch_first,
                               use_gates=self.use_gates,
                               use_glus=self.use_glus)
                self.gcn_layers.append(gcn)

            self.gcn_seq = nn.Sequential(*self.gcn_layers)

        elif residual == 'dense':
            for i in range(self.num_layers):
                input_size = num_inputs + (i * num_units)
                gcn = GCNLayer(input_size, num_units, num_labels,
                               in_arcs=in_arcs, out_arcs=out_arcs,
                               batch_first=self.batch_first,
                               use_gates=self.use_gates,
                               use_glus=self.use_glus)
                self.gcn_layers.append(gcn)

            self.gcn_seq = nn.Sequential(*self.gcn_layers)


    def forward(self, conv_output, lengths=None, arc_tensor_in=None, arc_tensor_out=None,
                label_tensor_in=None, label_tensor_out=None,
                mask_in=None, mask_out=None,  # batch* t, degree
                mask_loop=None, sent_mask=None):

        input_embedding, y = conv_output['encoder_out']
        encoder_padding_mask = conv_output['encoder_padding_mask']
        y = F.dropout(y, p=self.dropout, training=self.training)
        # if morph is None:

        # else:
        #     embeddings = self.embeddings(src)  # [t,b,e]
        #     morph_size = morph.data.size()  # [B,t,max_m]
        #     embeddings_m = self.morph_embeddings(morph.view(morph_size[0] * morph_size[1],
        #                                                     morph_size[2], 1))  # [B*t,max_m, m_e]
        #     embeddings_m = embeddings_m.view((morph_size[0], morph_size[1], morph_size[2],
        #                                       embeddings_m.data.size()[2]))  # [B,t,max_m, m_e]
        #     embeddings_m = embeddings_m.permute(3, 0, 1, 2).contiguous()  # [m_e ,B , max_m, t]
        #     masked_morph = embeddings_m * morph_mask  # [m_e ,B , max_m, t]*[B,t,max_m] = [m_e , B, t, max_m]
        #
        #     morph_sum = masked_morph.sum(3).permute(2, 1, 0).contiguous()  # [t,B,m_e]
        #
        #     embeddings = torch.cat([embeddings, morph_sum], dim=2)
        #
        #     embeddings = F.relu(self.emb_morph_emb(embeddings))

        if self.residual == '':

            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(y, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                else:
                    memory_bank = gcn(memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                memory_bank = F.dropout(memory_bank, p=self.dropout, training=self.training)

        elif self.residual == 'residual':

            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(y, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                elif g == 1:
                    prev_memory_bank = y + memory_bank
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                      label_tensor_in, label_tensor_out,
                                      mask_in, mask_out,
                                      mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                else:
                    prev_memory_bank = prev_memory_bank + memory_bank
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                memory_bank = F.dropout(memory_bank, p=self.dropout, training=self.training)

        elif self.residual == 'dense':
            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(y, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                elif g == 1:
                    prev_memory_bank = torch.cat([y, memory_bank], dim=2)
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                      label_tensor_in, label_tensor_out,
                                      mask_in, mask_out,
                                      mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                else:
                    prev_memory_bank = torch.cat([prev_memory_bank, memory_bank], dim=2)
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                    memory_bank = memory_bank.transpose(0, 1)  # [b, t, h]

                memory_bank = F.dropout(memory_bank, p=self.dropout, training=self.training)

        # batch_size = memory_bank.size()[1]
        # result_ = memory_bank.permute(2, 1, 0)  # [h,b,t]
        # res_sum = result_.sum(2)  # [h,b]
        # sent_mask = sent_mask.permute(1, 0).contiguous()  # [b,t]
        # mask_sum = sent_mask.sum(1)  # [b]
        # encoder_final = res_sum / mask_sum  # [h, b]
        # encoder_final = encoder_final.permute(1, 0)  # [b, h]

        # h_1 = torch.mm(encoder_final, self.H_1).view((1, batch_size, self.num_units))  # [1, b, h]
        # h_2 = torch.mm(encoder_final, self.H_2).view((1, batch_size, self.num_units))
        # h_3 = torch.mm(encoder_final, self.H_3).view((1, batch_size, self.num_units))
        # h_4 = torch.mm(encoder_final, self.H_4).view((1, batch_size, self.num_units))
        # h__1 = torch.cat([h_1, h_2], dim=0)  # [2, b, h]
        # h__2 = torch.cat([h_3, h_4], dim=0)  # [2, b, h]


        # add output to input embedding for attention
        y = (memory_bank + input_embedding) * math.sqrt(self.normalization_constant)

        return {
            'encoder_out': (memory_bank, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class FConvGCNOnTopEncoder(FairseqEncoder):
    """ A convolutional-GCN encoder."""

    def __init__(self, dictionary, conv_encoder, gcn_encoder):
        super(FConvGCNOnTopEncoder, self).__init__(dictionary)
        self.gcn_encoder = gcn_encoder
        self.conv_encoder = conv_encoder

    def forward(self, src, lengths=None, arc_tensor_in=None, arc_tensor_out=None,
                label_tensor_in=None, label_tensor_out=None,
                mask_in=None, mask_out=None,  # batch* t, degree
                mask_loop=None, sent_mask=None, morph=None, morph_mask=None):
        conv_output = self.conv_encoder(src, lengths)
        encoder_output = self.gcn_encoder(conv_output, lengths, arc_tensor_in, arc_tensor_out,
                label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)

        return encoder_output

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_out'] is not None:
            encoder_out_dict['encoder_out'] = (
                encoder_out_dict['encoder_out'][0].index_select(0, new_order),
                encoder_out_dict['encoder_out'][1].index_select(0, new_order),
            )
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.conv_encoder.max_positions()


@register_model_architecture('fconv_gcn_on_top', 'fconv_gcn_on_top')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)


@register_model_architecture('fconv_gcn_on_top', 'fconv_gcn_on_top_wikicatsum')
def fconv_gcn_wikicatsum(args):
    args.gcn_num_inputs = getattr(args, 'gcn_num_inputs', 256)
    args.gcn_num_units = getattr(args, 'gcn_num_units', 256)
    args.gcn_num_labels = getattr(args, 'gcn_num_labels', 10)
    args.gcn_num_layers = getattr(args, 'gcn_num_layers', 1)
    args.gcn_residual = getattr(args, 'gcn_residual', '')

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3)] * 4')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    base_architecture(args)
