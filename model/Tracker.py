from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=(256,256), patch_size=(16, 16), in_chans=1, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class UnPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=(256,256), patch_size=(16, 16), out_chans=1, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.out_chans = out_chans
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1])

    def forward(self, x, **kwargs):
        B, N, D = x.shape
        p = self.patch_size[0]
        q = self.patch_size[1]
        h = self.img_size[0]//p
        w = self.img_size[1]//q
        x = self.proj(x)
        x = x.reshape(shape=(x.shape[0], h, w, p, q, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, self.img_size[0], self.img_size[1]))
        return imgs

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)



class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class ReferringTracker(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=1024,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
    ):
        super(ReferringTracker, self).__init__()

        # init transformer layers
        self.patchify = PatchEmbed(img_size=(256,256), patch_size=(16, 16), in_chans=1, embed_dim=hidden_channel)
        self.unpatchify = UnPatchEmbed(img_size=(256, 256), patch_size=(16, 16), out_chans=1, embed_dim=hidden_channel)
        self.outputlayer = nn.Sequential(
            nn.Conv2d(4,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,1,3,1,1),
            #nn.Softmax2d()
        )

        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.3,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.3,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.3,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # init heads
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        return

    def forward(self, frame_segment, resume=False, return_indices=False):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        frame_embeds = self.patchify(frame_segment)  # t, c1, h, w --> t,b,c,

        #n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []
        ms_output = []
        for j in range(self.num_layers):
            if j == 0:
                ms_output.append(frame_embeds)
                output = self.transformer_cross_attention_layers[j](
                    frame_embeds, frame_embeds, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)
            else:
                output = self.transformer_cross_attention_layers[j](
                    ms_output[-1], ms_output[-1], frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)


            self.last_outputs = ms_output
            outputs.append(ms_output[1:]) # (t, layers, q, b, c)

        ori_output = ms_output[-1]
        ori_output_pic = self.unpatchify(ori_output)

        past_one_q = torch.roll(frame_embeds, shifts=1, dims=0)
        future_one_q = torch.roll(frame_embeds, shifts=-1, dims=0)


        for j in range(self.num_layers):
            if j == 0:
                #ret_indices.append(self.match_embds(frame_embeds, frame_embeds))
                output = self.transformer_cross_attention_layers[j](
                    past_one_q, ori_output, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)
            else:
                output = self.transformer_cross_attention_layers[j](
                    ms_output[-1], ms_output[-1], frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)

        past_output = ms_output[-1]
        past_output_pic = self.unpatchify(past_output)
        past_output_pic[0] = torch.zeros_like(past_output_pic[0])


        for j in range(self.num_layers):
            if j == 0:
                #ret_indices.append(self.match_embds(frame_embeds, frame_embeds))
                output = self.transformer_cross_attention_layers[j](
                    future_one_q, ori_output, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)
            else:
                output = self.transformer_cross_attention_layers[j](
                    ms_output[-1], ms_output[-1], frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)

        future_output = ms_output[-1]
        future_output_pic = self.unpatchify(future_output)
        future_output_pic[-1] = torch.zeros_like(future_output_pic[-1])

        output_pic = torch.cat((past_output_pic,ori_output_pic,future_output_pic,frame_segment),dim=1)
        out = self.outputlayer(output_pic)
        return out

if __name__ == '__main__':

    a = torch.tensor([1,2,3,4,5,6,7])
    print(torch.roll(a, shifts=1, dims=0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patchifylayer = PatchEmbed(img_size=(256,256), patch_size=(16, 16), in_chans=1, embed_dim=1024).to(device)
    unpatchifylayer = UnPatchEmbed(img_size=(256, 256), patch_size=(16, 16), out_chans=1, embed_dim=1024).to(device)
    layer = SelfAttentionLayer(
        d_model=1024,
        nhead=8,
        dropout=0.0,
        normalize_before=False,
    ).to(device)
    TestData = torch.randn((16,1,256,256)).to(device)
    '''TestData = patchifylayer(TestData)
    TestData = layer(TestData)
    TestData = unpatchifylayer(TestData)'''
    layer = ReferringTracker().to(device)
    TestData = layer(TestData)
    print(TestData.shape)