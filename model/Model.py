from .Encoder import TransformerEncoder, RelEncoder
from .Decoder import TransformerDecoder, BboxDecoder, BboxRegDecoder
from .Embedding import Sentence_Embeddings, Concat_Embeddings, Add_Embeddings
from .bbox_head import BBox_Head
from .Inference import greedy, beam_search
from .Inference_Reg import greedy_Reg
import torch.nn as nn
from torch import Tensor
import numpy as np
from transformers import BertModel, BertTokenizer
import torch

class Layout_Transformer(nn.Module):
    """
    Base Model class
    """

    def __init__(self, input_embeddings, output_embeddings) -> None:
        super(Layout_Transformer, self).__init__()

        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings
        self.bos_index = 1
        self.pad_index = 0
        self.eos_index = 2

    # pylint: disable=arguments-differ
    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                trg_mask: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.
        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(src=src, src_mask=src_mask)
        # print("Encoder output:", encoder_output.size())
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, src: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.
        :param src:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.
        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=trg_input,
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    # def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
    #         -> Tensor:
    #     """
    #     Compute non-normalized loss and number of tokens for a batch
    #     :param batch: batch to compute loss for
    #     :param loss_function: loss function, computes for input and target
    #         a scalar loss for the complete batch
    #     :return: batch_loss: sum of losses over non-pad elements in the batch
    #     """
    #     # pylint: disable=unused-variable
    #     out, hidden, att_probs, _ = self.forward(
    #         src=batch.src, trg_input=batch.trg_input,
    #         src_mask=batch.src_mask, src_lengths=batch.src_lengths,
    #         trg_mask=batch.trg_mask)

    #     # compute log probs
    #     log_probs = F.log_softmax(out, dim=-1)

    #     # compute batch loss
    #     batch_loss = loss_function(log_probs, batch.trg)
    #     # return batch loss = sum over all elements in batch that are not pad
    #     return batch_loss

    def inference(self, src: Tensor, src_mask: Tensor, trg_embed: Add_Embeddings, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch
        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoder_output, encoder_hidden = self.encode(src, src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = 64

        # greedy decoding
        if beam_size < 2:
            output_cats, output_pos, output_shape = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output, eos_index=self.eos_index,
                    src_mask=src_mask, embed=trg_embed,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    beam_search(
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=batch.src_mask, embed=self.trg_embed,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder)

        return output_cats, output_pos, output_shape

    # def __repr__(self) -> str:
    #     """
    #     String representation: a description of encoder, decoder and embeddings
    #     :return: string representation
    #     """
    #     return "%s(\n" \
    #            "\tencoder=%s,\n" \
    #            "\tdecoder=%s,\n" \
    #            "\tsrc_embed=%s,\n" \
    #            "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
    #                self.decoder, self.src_embed, self.trg_embed)


# def build_model(cfg: dict = None,
#                 src_vocab: Vocabulary = None,
#                 trg_vocab: Vocabulary = None) -> Model:
#     """
#     Build and initialize the model according to the configuration.
#     :param cfg: dictionary configuration containing model specifications
#     :param src_vocab: source vocabulary
#     :param trg_vocab: target vocabulary
#     :return: built and initialized model
#     """
#     src_padding_idx = 0
#     trg_padding_idx = 0

#     src_embed = Embeddings(
#         **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
#         padding_idx=src_padding_idx)

#     # this ties source and target embeddings
#     # for softmax layer tying, see further below
#     if cfg.get("tied_embeddings", False):
#         if src_vocab.itos == trg_vocab.itos:
#             # share embeddings for src and trg
#             trg_embed = src_embed
#         else:
#             raise ConfigurationError(
#                 "Embedding cannot be tied since vocabularies differ.")
#     else:
#         trg_embed = Embeddings(
#             **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
#             padding_idx=trg_padding_idx)

#     # build encoder
#     enc_dropout = cfg["encoder"].get("dropout", 0.)
#     enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
#     if cfg["encoder"].get("type", "recurrent") == "transformer":
#         assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
#                cfg["encoder"]["hidden_size"], \
#                "for transformer, emb_size must be hidden_size"

class Text2Layout(nn.Module):
    """
    Base Model class
    """

    def __init__(self, input_embeddings, output_embeddings) -> None:
        super(Text2Layout, self).__init__()

        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = TransformerDecoder()
        self.bos_index = 1
        self.pad_index = 0
        self.eos_index = 2

    # pylint: disable=arguments-differ
    def forward(self, caption, trg_input, trg_mask):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.
        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoded = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=caption, max_length = 64, pad_to_max_length = True)
        input_ids = torch.tensor(encoded['input_ids']).cuda()
        attention_mask = torch.tensor(encoded['attention_mask']).bool().cuda()
        token_type_ids = torch.tensor(encoded['token_type_ids']).cuda()
        encoder_output = self.encode(input_ids, attention_mask, token_type_ids)[0]
        encoder_hidden = None
        attention_mask = attention_mask.unsqueeze(1)
        # print("Encoder output:", encoder_output.size())
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=attention_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """
        Encodes the source sentence.
        :param src:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.
        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=trg_input,
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    def inference(self, caption, max_output_length, beam_size, beam_alpha):
        """
        Get outputs and attentions scores for a given batch
        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoded = self.tokenizer.batch_encode_plus(caption, max_length = 64, pad_to_max_length = True)
        input_ids = torch.tensor(encoded['input_ids']).cuda()
        attention_mask = torch.tensor(encoded['attention_mask']).bool().cuda()
        token_type_ids = torch.tensor(encoded['token_type_ids']).cuda()
        encoder_output = self.encode(input_ids, attention_mask, token_type_ids)[0]
        encoder_hidden = None
        attention_mask = attention_mask.unsqueeze(1)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = 64

        # greedy decoding
        if beam_size < 2:
            output_cats, output_pos, output_shape, pred_cats, pred_pos, pred_shape = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output, eos_index=self.eos_index,
                    src_mask=attention_mask, embed=self.output_embeddings,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    beam_search(
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=attention_mask, embed=self.output_embeddings,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder)

        return output_cats, output_pos, output_shape, pred_cats, pred_pos, pred_shape

class Rel2Layout(nn.Module):

    def __init__(self, vocab_size=204, cls_size=154, pos_size=68, shape_size=68,\
                 hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1):
        super(Rel2Layout, self).__init__()

        self.encoder = RelEncoder(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, \
            attn_heads=attn_heads, dropout=dropout)
        self.decoder = BboxDecoder(cls_size=cls_size, pos_size=pos_size, shape_size=shape_size,\
            hidden_size=hidden_size, num_layers=num_layers, attn_heads=attn_heads, dropout=dropout)

        self.hidden_size = hidden_size
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2        

    def forward(self, input_token, input_ids, segment_label, token_type, src_mask, output_cls, output_pos, output_shape, trg_mask):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)

        return self.decoder(output_cls, output_pos, output_shape, encoder_output, src_mask, trg_mask)

    def inference(self, input_token, input_ids, segment_label, token_type, src_mask):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)

        max_output_length = 64

        # greedy decoding
        return greedy(encoder_hidden=None, encoder_output=encoder_output, eos_index=self.eos_index, \
            src_mask=src_mask, bos_index=self.bos_index, \
            decoder=self.decoder, max_output_length=max_output_length)

class Rel2RegLayout(nn.Module):

    def __init__(self, vocab_size=204, cls_size=154, box_size=4,
                 hidden_size=512, num_layers=6, max_out_len = 128, attn_heads=8, dropout=0.1):
        super(Rel2RegLayout, self).__init__()

        self.encoder = RelEncoder(vocab_size=vocab_size, hidden_size=hidden_size,
                                  num_layers=num_layers,attn_heads=attn_heads,
                                  dropout=dropout)
        self.decoder = BboxRegDecoder(cls_size=cls_size, box_size=box_size,
                                      hidden_size=hidden_size, num_layers=num_layers,
                                      attn_heads=attn_heads, dropout=dropout)

        self.hidden_size = hidden_size
        self.max_out_len = max_out_len
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2        

    def forward(self, input_token, input_ids, segment_label, token_type, src_mask,
                output_cls, output_box, trg_mask, trg_input_template):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)
        return self.decoder(output_cls, output_box, trg_input_template,
                            encoder_output, src_mask, trg_mask)

    def inference(self, input_token, input_ids, segment_label, token_type, src_mask,
                  trg_input_template):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)

        max_output_length = self.max_out_len
        # greedy decoding
        return greedy_Reg(encoder_hidden=None, encoder_output=encoder_output,
                          eos_index=self.eos_index, src_mask=src_mask, 
                          bos_index=self.bos_index, decoder=self.decoder, 
                          max_output_length=max_output_length, 
                          template = trg_input_template)
    
class Rel2Bbox(nn.Module):
    def __init__(self, vocab_size=204, obj_classes_size=154, noise_size=64,\
                 hidden_size=256, num_layers=4, attn_heads=4, dropout=0.1, cfg=None):
        super(Rel2Bbox, self).__init__()
        
        self.pretrain_encoder = cfg['MODEL']['PRETRAIN']
        
        # Encoder Module
        self.encoder = RelEncoder(vocab_size=vocab_size, obj_classes_size=obj_classes_size, 
                                  hidden_size=hidden_size, num_layers=num_layers, 
                                  attn_heads=attn_heads, dropout=dropout, cfg=cfg)

        # Decoder & Refine Module
        self.bbox_head = BBox_Head(hidden_size=hidden_size, dropout=dropout, cfg=cfg)

    def forward(self, input_token, input_obj_id, segment_label, token_type, src_mask,
               trg_input_box=None, trg_mask=None, inference=False, epoch=0, 
               global_mask=None):

        # [B, M, D] M = Sentence Length
        encoder_output, vocab_logits, obj_id_logits, token_type_logits, \
        src, class_embeds = self.encoder(input_token, input_obj_id, segment_label,
                                         token_type, src_mask)
        if self.pretrain_encoder:
            return vocab_logits, obj_id_logits, token_type_logits, None, None, None, None
        else:
            # Decoder inference
            if inference:
                coarse_box, coarse_gmm, refine_box, refine_gmm = \
                self.bbox_head.inference(encoder_output, src_mask, src, class_embeds,
                                         global_mask)
            else:
                coarse_box, coarse_gmm, refine_box, refine_gmm = \
                self.bbox_head(epoch, encoder_output, src_mask, 
                               src, class_embeds, trg_input_box, trg_mask, global_mask)

            return vocab_logits, obj_id_logits, token_type_logits, coarse_box, coarse_gmm, refine_box, refine_gmm