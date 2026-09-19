
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a

        combined_dim_orig = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3

        extra_dim = self.d_l * 3 + self.d_l * 3   # 6 * d_l
        combined_dim = combined_dim_orig + extra_dim
        output_dim = 1

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2.1 Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2.2 Modality-invariant encoder
        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 3. Decoders
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # projections for cosine similarity
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # align projections for margin loss
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        self.text_type_emb = nn.Parameter(torch.randn(1, 1, self.d_l))
        self.visual_type_emb = nn.Parameter(torch.randn(1, 1, self.d_v))
        self.audio_type_emb = nn.Parameter(torch.randn(1, 1, self.d_a))

        self.intra_modal_encoder = self.get_network(self_type='intra', layers=1)
        self.inter_modal_encoder = self.get_network(self_type='inter', layers=1)
        self.intra_norm = nn.LayerNorm(self.d_l)
        self.inter_norm = nn.LayerNorm(self.d_l)
        self.fusion_weight = nn.Parameter(torch.tensor(0.05))

        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        self.gate_l = nn.Sequential(
            nn.Linear(2 * self.d_l, 2 * self.d_l),
            nn.ReLU(),
            nn.Linear(2 * self.d_l, 2 * self.d_l),
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(2 * self.d_v, 2 * self.d_v),
            nn.ReLU(),
            nn.Linear(2 * self.d_v, 2 * self.d_v),
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Linear(2 * self.d_a, 2 * self.d_a),
            nn.ReLU(),
            nn.Linear(2 * self.d_a, 2 * self.d_a),
            nn.Sigmoid()
        )
        for gate in [self.gate_l, self.gate_v, self.gate_a]:
            last_linear = gate[-2]
            if isinstance(last_linear, nn.Linear):
                nn.init.constant_(last_linear.bias, 1.0)

        # Ensemble weights
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)

        # Final projection
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.K = getattr(args, 'dict_size', 768)
        self.dict_homo = nn.Parameter(torch.empty(self.K, self.d_l))
        self.dict_hetero = nn.Parameter(torch.empty(self.K, self.d_l))
        nn.init.orthogonal_(self.dict_homo)
        nn.init.orthogonal_(self.dict_hetero)
        self.dict_scale = nn.Parameter(torch.tensor(1.0))

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'intra', 'inter']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def generate_masks(self, len_l, len_v, len_a, device):
        total_len = len_l + len_v + len_a
        l_pos = torch.arange(len_l, device=device)
        v_pos = torch.arange(len_l, len_l + len_v, device=device)
        a_pos = torch.arange(len_l + len_v, total_len, device=device)

        intra_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        intra_mask[l_pos[:, None], l_pos] = False
        intra_mask[v_pos[:, None], v_pos] = False
        intra_mask[a_pos[:, None], a_pos] = False

        inter_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        inter_mask[l_pos[:, None], v_pos] = False
        inter_mask[l_pos[:, None], a_pos] = False
        inter_mask[v_pos[:, None], l_pos] = False
        inter_mask[v_pos[:, None], a_pos] = False
        inter_mask[a_pos[:, None], l_pos] = False
        inter_mask[a_pos[:, None], v_pos] = False

        return intra_mask, inter_mask

    def forward(self, text, audio, video, is_distill=False):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)#一维卷积把维度特征都压缩到40，序列长度变为46
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        s_l = self.encoder_s_l(proj_x_l)#异质特征
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)#同质特征
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_l, c_v, c_a]

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)

        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        c_l_raw = c_l.clone()
        c_v_raw = c_v.clone()
        c_a_raw = c_a.clone()
        s_l_raw = s_l.clone()
        s_v_raw = s_v.clone()
        s_a_raw = s_a.clone()

        len_l, len_v, len_a = s_l.shape[0], s_v.shape[0], s_a.shape[0]
        intra_mask, inter_mask = self.generate_masks(len_l, len_v, len_a, s_l.device)

        s_l_typed = s_l + self.text_type_emb
        s_v_typed = s_v + self.visual_type_emb
        s_a_typed = s_a + self.audio_type_emb
        s_multimodal_seq = torch.cat([s_l_typed, s_v_typed, s_a_typed], dim=0)
        s_intra_encoded = self.intra_modal_encoder(s_multimodal_seq, intra_mask)
        if type(s_intra_encoded) == tuple:
            s_intra_encoded = s_intra_encoded[0]
        s_intra_l = s_intra_encoded[:len_l]
        s_intra_v = s_intra_encoded[len_l:len_l + len_v]
        s_intra_a = s_intra_encoded[len_l + len_v:]
        s_intra_l = self.intra_norm(s_intra_l)
        s_intra_v = self.intra_norm(s_intra_v)
        s_intra_a = self.intra_norm(s_intra_a)
        s_l_enhanced = s_l + self.fusion_weight * s_intra_l
        s_v_enhanced = s_v + self.fusion_weight * s_intra_v
        s_a_enhanced = s_a + self.fusion_weight * s_intra_a

        c_l_typed = c_l + self.text_type_emb
        c_v_typed = c_v + self.visual_type_emb
        c_a_typed = c_a + self.audio_type_emb
        c_multimodal_seq = torch.cat([c_l_typed, c_v_typed, c_a_typed], dim=0)
        c_inter_encoded = self.inter_modal_encoder(c_multimodal_seq, inter_mask)
        if type(c_inter_encoded) == tuple:
            c_inter_encoded = c_inter_encoded[0]
        c_inter_l = c_inter_encoded[:len_l]
        c_inter_v = c_inter_encoded[len_l:len_l + len_v]
        c_inter_a = c_inter_encoded[len_l + len_v:]
        c_inter_l = self.inter_norm(c_inter_l)
        c_inter_v = self.inter_norm(c_inter_v)
        c_inter_a = self.inter_norm(c_inter_a)
        c_l_enhanced = c_l + self.fusion_weight * c_inter_l
        c_v_enhanced = c_v + self.fusion_weight * c_inter_v
        c_a_enhanced = c_a + self.fusion_weight * c_inter_a

        hs_l_low = c_l_enhanced.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(F.dropout(F.relu(repr_l_low), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v_enhanced.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(F.dropout(F.relu(repr_v_low), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a_enhanced.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(F.dropout(F.relu(repr_a_low), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        proj_s_l = self.proj_cosine_l(s_l_enhanced.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_v = self.proj_cosine_v(s_v_enhanced.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_a = self.proj_cosine_a(s_a_enhanced.transpose(0, 1).contiguous().view(x_l.size(0), -1))

        c_l_att = self.self_attentions_c_l(c_l_enhanced)
        if type(c_l_att) == tuple: c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_c_v(c_v_enhanced)
        if type(c_v_att) == tuple: c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_c_a(c_a_enhanced)
        if type(c_a_att) == tuple: c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)
        c_proj = self.proj2_c(F.dropout(F.relu(self.proj1_c(c_fusion)), p=self.output_dropout, training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        h_l_with_as = self.trans_l_with_a(s_l_enhanced, s_a_enhanced, s_a_enhanced)
        h_l_with_vs = self.trans_l_with_v(s_l_enhanced, s_v_enhanced, s_v_enhanced)
        h_l_inter = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_l_pool = h_l_inter.mean(dim=0)
        gate_l_value = self.gate_l(h_l_pool)
        h_l_inter = h_l_inter * gate_l_value.unsqueeze(0)
        h_ls = self.trans_l_mem(h_l_inter)
        if type(h_ls) == tuple: h_ls = h_ls[0]
        last_h_l = h_ls[-1]
        # A
        h_a_with_ls = self.trans_a_with_l(s_a_enhanced, s_l_enhanced, s_l_enhanced)
        h_a_with_vs = self.trans_a_with_v(s_a_enhanced, s_v_enhanced, s_v_enhanced)
        h_a_inter = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_a_pool = h_a_inter.mean(dim=0)
        gate_a_value = self.gate_a(h_a_pool)
        h_a_inter = h_a_inter * gate_a_value.unsqueeze(0)
        h_as = self.trans_a_mem(h_a_inter)
        if type(h_as) == tuple: h_as = h_as[0]
        last_h_a = h_as[-1]
        # V
        h_v_with_ls = self.trans_v_with_l(s_v_enhanced, s_l_enhanced, s_l_enhanced)
        h_v_with_as = self.trans_v_with_a(s_v_enhanced, s_a_enhanced, s_a_enhanced)
        h_v_inter = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_v_pool = h_v_inter.mean(dim=0)
        gate_v_value = self.gate_v(h_v_pool)
        h_v_inter = h_v_inter * gate_v_value.unsqueeze(0)
        h_vs = self.trans_v_mem(h_v_inter)
        if type(h_vs) == tuple: h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        hs_proj_l_high = self.proj2_l_high(F.dropout(F.relu(self.proj1_l_high(last_h_l)), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(F.dropout(F.relu(self.proj1_v_high(last_h_v)), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(F.dropout(F.relu(self.proj1_a_high(last_h_a)), p=self.output_dropout, training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion = torch.sigmoid(self.weight_c(c_fusion))

        c_l_pool = c_l_raw.mean(dim=0)
        c_v_pool = c_v_raw.mean(dim=0)
        c_a_pool = c_a_raw.mean(dim=0)
        att_homo_l = c_l_pool @ self.dict_homo.t()
        att_homo_v = c_v_pool @ self.dict_homo.t()
        att_homo_a = c_a_pool @ self.dict_homo.t()
        alpha_homo_l = F.softmax(att_homo_l * self.dict_scale, dim=-1)
        alpha_homo_v = F.softmax(att_homo_v * self.dict_scale, dim=-1)
        alpha_homo_a = F.softmax(att_homo_a * self.dict_scale, dim=-1)
        rec_homo_l = alpha_homo_l @ self.dict_homo
        rec_homo_v = alpha_homo_v @ self.dict_homo
        rec_homo_a = alpha_homo_a @ self.dict_homo

        s_l_pool = s_l_raw.mean(dim=0)  # (N, d_l)
        s_v_pool = s_v_raw.mean(dim=0)
        s_a_pool = s_a_raw.mean(dim=0)
        att_hetero_l = s_l_pool @ self.dict_hetero.t()
        att_hetero_v = s_v_pool @ self.dict_hetero.t()
        att_hetero_a = s_a_pool @ self.dict_hetero.t()
        alpha_hetero_l = F.softmax(att_hetero_l * self.dict_scale, dim=-1)
        alpha_hetero_v = F.softmax(att_hetero_v * self.dict_scale, dim=-1)
        alpha_hetero_a = F.softmax(att_hetero_a * self.dict_scale, dim=-1)
        rec_hetero_l = alpha_hetero_l @ self.dict_hetero  # (N, d_l)
        rec_hetero_v = alpha_hetero_v @ self.dict_hetero
        rec_hetero_a = alpha_hetero_a @ self.dict_hetero

        c_l_enh_pool = c_l_enhanced.mean(dim=0)
        c_v_enh_pool = c_v_enhanced.mean(dim=0)
        c_a_enh_pool = c_a_enhanced.mean(dim=0)
        att_homo_l_enh = c_l_enh_pool @ self.dict_homo.t()
        att_homo_v_enh = c_v_enh_pool @ self.dict_homo.t()
        att_homo_a_enh = c_a_enh_pool @ self.dict_homo.t()
        rec_homo_l_enh = F.softmax(att_homo_l_enh * self.dict_scale, dim=-1) @ self.dict_homo
        rec_homo_v_enh = F.softmax(att_homo_v_enh * self.dict_scale, dim=-1) @ self.dict_homo
        rec_homo_a_enh = F.softmax(att_homo_a_enh * self.dict_scale, dim=-1) @ self.dict_homo

        last_hs = torch.cat([
            last_h_l, last_h_v, last_h_a, c_fusion,
            rec_homo_l, rec_homo_v, rec_homo_a,
            rec_hetero_l, rec_hetero_v, rec_hetero_a
        ], dim=1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        res = {
            'logits_l_homo': logits_l_low,
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            'repr_l_homo': repr_l_low,
            'repr_v_homo': repr_v_low,
            'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l_enhanced,
            's_v': s_v_enhanced,
            's_a': s_a_enhanced,
            'proj_s_l': proj_s_l,
            'proj_s_v': proj_s_v,
            'proj_s_a': proj_s_a,
            'c_l': c_l_enhanced,
            'c_v': c_v_enhanced,
            'c_a': c_a_enhanced,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,
            'logits_v_hetero': logits_v_high,
            'logits_a_hetero': logits_a_high,
            'repr_l_hetero': hs_proj_l_high,
            'repr_v_hetero': hs_proj_v_high,
            'repr_a_hetero': hs_proj_a_high,
            'last_h_l': h_ls[-1],
            'last_h_v': h_vs[-1],
            'last_h_a': h_as[-1],
            'logits_c': logits_c,
            'output_logit': output,
            'c_l_pool': c_l_pool, 'c_v_pool': c_v_pool, 'c_a_pool': c_a_pool,
            'rec_homo_l': rec_homo_l, 'rec_homo_v': rec_homo_v, 'rec_homo_a': rec_homo_a,
            'rec_hetero_l': rec_hetero_l, 'rec_hetero_v': rec_hetero_v, 'rec_hetero_a': rec_hetero_a,
            'alpha_homo_l': alpha_homo_l, 'alpha_homo_v': alpha_homo_v, 'alpha_homo_a': alpha_homo_a,
            'alpha_hetero_l': alpha_hetero_l, 'alpha_hetero_v': alpha_hetero_v, 'alpha_hetero_a': alpha_hetero_a,
            'rec_homo_l_enh': rec_homo_l_enh,
            'rec_homo_v_enh': rec_homo_v_enh,
            'rec_homo_a_enh': rec_homo_a_enh,
            's_intra_l': s_intra_l, 's_intra_v': s_intra_v, 's_intra_a': s_intra_a,
            'c_inter_l': c_inter_l, 'c_inter_v': c_inter_v, 'c_inter_a': c_inter_a,
        }
        return res