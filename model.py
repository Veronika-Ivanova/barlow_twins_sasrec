import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecBackBone(nn.Module):
    def __init__(self, item_num, config):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, config['hidden_units'], padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)

        for _ in range(config['num_blocks']):
            new_attn_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                config['hidden_units'],config['num_heads'],config['dropout_rate']
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_units'], config['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(config['manual_seed'])
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :, :] * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, log_feats

    def score(self, seq):
        '''
        Takes 1d sequence as input and returns prediction scores.
        '''
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

class Projector(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        proj_dim = config["proj_dim"]
        self.num_proj_layers = config["proj_layers"]
        if self.num_proj_layers is not None:
            proj_layers = []
            for i in range(self.num_proj_layers):
                in_size = hidden_dim if i == 0 else proj_dim
                out_size = proj_dim
                proj_layers.append(nn.Linear(in_size, out_size, bias=False))
                if i != self.num_proj_layers-1:
                    proj_layers.append(nn.BatchNorm1d(out_size, affine=False, eps=1e-8))
                    proj_layers.append(nn.ReLU(inplace=True))
            self.projector = nn.Sequential(*proj_layers)
        
        #print("PROJECTOR")
        #print(self.projector)
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        if self.num_proj_layers is None:
            return x
        else:
            return self.projector(x)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class SASRec(SASRecBackBone):
    def __init__(self, item_num, config, item_poularity_buckets = None):
        super().__init__(item_num + 1, config)

        self.fwd_type = config['fwd_type']

        if self.fwd_type in ['bce', 'gbce']:
            self.n_neg_samples = config['n_neg_samples']

        if self.fwd_type == 'sce':
            self.n_buckets = config['n_buckets']
            self.bucket_size_x = eval(config['bucket_size_x']) if type(config['bucket_size_x']) == str else config['bucket_size_x']
            self.bucket_size_y = eval(config['bucket_size_y']) if type(config['bucket_size_y']) == str else config['bucket_size_y']
                
            self.mix_x = config['mix_x']

        elif self.fwd_type == 'gbce':
            alpha = self.n_neg_samples / (item_num - 1.)
            self.beta = alpha * (config['gbce_t'] * (1. - 1. / alpha) + 1. / alpha)

        self.ssl_config = None
        if config.get("ssl", None) is not None:
            self.ssl_config = config["ssl"]
            if "barlow_twins" in self.ssl_config:
                self.projector = Projector(self.ssl_config["barlow_twins"], config["hidden_units"])
        
        self.item_poularity_buckets = item_poularity_buckets
        self.item_bucket_grad_norms = {}

    def forward(self, log_seqs, pos_seqs, neg_seqs, aug_seqs):
        if self.ssl_config is not None:
            loss, emb = self.forward_task(log_seqs, pos_seqs, neg_seqs)

            loss_ssl = 0
            aug_emb = self.log2feats(aug_seqs)
            if "barlow_twins" in self.ssl_config:
                loss_ssl = self.bt_loss(aug_emb, emb, self.ssl_config["barlow_twins"]["lmd_sem"])
                #print(loss_ssl.item())

            loss += self.ssl_config["lmd"] * loss_ssl

            with torch.no_grad():
                alignment, uniformity = self.decompose(aug_emb[:,-1,:], emb[:,-1,:], emb[:,-1,:], batch_size=emb.shape[0])

        else:
            loss, emb = self.forward_task(log_seqs, pos_seqs, neg_seqs)
            with torch.no_grad():
                alignment, uniformity = self.decompose(emb[:,-1,:], emb[:,-1,:], emb[:,-1,:], batch_size=emb.shape[0])

        return loss, (alignment, uniformity)

    def forward_task(self, log_seqs, pos_seqs, neg_seqs):
        if self.fwd_type == 'sce':
            return self.sce_forward(log_seqs, pos_seqs)
        elif self.fwd_type == 'bce':
            return self.bce_forward(log_seqs, pos_seqs, neg_seqs)
        
        elif self.fwd_type == 'gbce':
            return self.gbce_forward(log_seqs, pos_seqs, neg_seqs)

        elif self.fwd_type == 'ce':
            return self.ce_forward(log_seqs, pos_seqs)
        
        elif self.fwd_type == 'dross':
            return self.dross_forward(log_seqs, pos_seqs, neg_seqs)

        else:
            raise ValueError(f'Wrong fwd_type type - {self.fwd_type}')

    def bce_forward(self, log_seqs, pos_seqs, neg_seqs):
        device = log_seqs.device
        pos_logits, neg_logits, emb = super().forward(log_seqs, pos_seqs, neg_seqs)

        pos_logits = pos_logits[:, :, None]
        neg_logits = neg_logits.permute(0, 2, 1)

        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)

        logits = torch.cat([pos_logits, neg_logits], -1)

        gt = torch.cat([pos_labels, neg_labels], -1)

        mask = (pos_seqs != self.pad_token).float()

        loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1) * mask
        loss = loss_per_element.sum() / mask.sum()
        return loss, emb

    def gbce_forward(self, log_seqs, pos_seqs, neg_seqs):
        device = log_seqs.device

        pos_logits, neg_logits, emb = super().forward(log_seqs, pos_seqs, neg_seqs)

        pos_logits = pos_logits[:, :, None]
        neg_logits = neg_logits.permute(0, 2, 1)

        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)

        pos_logits = torch.log(1 / (F.sigmoid(pos_logits) ** (- self.beta) - 1.))

        logits = torch.cat([pos_logits, neg_logits], -1)

        gt = torch.cat([pos_labels, neg_labels], -1)

        mask = (pos_seqs != self.pad_token).float()

        loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1) * mask
        loss = loss_per_element.sum() / mask.sum()

        return loss, emb
    
    def dross_forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs) # bs, seq, hd
        pos_embs = self.item_emb(pos_seqs) # bs, seq, hd
        neg_embs = self.item_emb(neg_seqs) # bs, n_neg, hd

        pos_logits = (log_feats * pos_embs).sum(dim=-1)[:, :, None] # bs, seq, 1
        neg_logits = (log_feats[:, :, None, :] * neg_embs[:, None, :, :]).sum(dim=-1) # bs, seq, n_neg


        logits = torch.cat([pos_logits, neg_logits], dim=-1) # bs, seq, 1 + n_neg
        labels = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.int64, device=logits.device)

        mask = (pos_seqs != self.pad_token).float()

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='none') * mask.view(-1)
        loss = loss.sum() / mask.sum()

        return loss, log_feats

    def ce_forward(self, log_seqs, pos_seqs, return_emb = False):
        emb = self.log2feats(log_seqs)
        logits = emb @ self.item_emb.weight.T
        indices = torch.where(pos_seqs.view(-1) != self.pad_token)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos_seqs.view(-1)[indices], reduction='mean')

        if self.item_poularity_buckets is not None:
            self.compute_bucket_gradients(logits, pos_seqs)

        return loss, emb

    def sce_forward(self, log_seqs, pos_seqs):
        emb = self.log2feats(log_seqs)
        hd = emb.shape[-1]

        x = emb.view(-1, hd)
        y = pos_seqs.view(-1)
        w = self.item_emb.weight

        correct_class_logits_ = (x * torch.index_select(w, dim=0, index=y)).sum(dim=1) # (bs,)

        with torch.no_grad():
            if self.mix_x:
                omega = 1/np.sqrt(np.sqrt(hd)) * torch.randn(x.shape[0], self.n_buckets, device=x.device)
                buckets = omega.T @ x
                del omega
            else:
                buckets = 1/np.sqrt(np.sqrt(hd)) * torch.randn(self.n_buckets, hd, device=x.device) # (n_b, hd)

        with torch.no_grad():
            x_bucket = buckets @ x.T # (n_b, hd) x (hd, b) -> (n_b, b)
            x_bucket[:, log_seqs.view(-1) == self.pad_token] = float('-inf')
            _, top_x_bucket = torch.topk(x_bucket, dim=1, k=self.bucket_size_x) # (n_b, bs_x)
            del x_bucket

            y_bucket = buckets @ w.T # (n_b, hd) x (hd, n_cl) -> (n_b, n_cl)

            y_bucket[:, self.pad_token] = float('-inf')
            _, top_y_bucket = torch.topk(y_bucket, dim=1, k=self.bucket_size_y) # (n_b, bs_y)
            del y_bucket

        x_bucket = torch.gather(x, 0, top_x_bucket.view(-1, 1).expand(-1, hd)).view(self.n_buckets, self.bucket_size_x, hd) # (n_b, bs_x, hd)
        y_bucket = torch.gather(w, 0, top_y_bucket.view(-1, 1).expand(-1, hd)).view(self.n_buckets, self.bucket_size_y, hd) # (n_b, bs_y, hd)
        
        wrong_class_logits = (x_bucket @ y_bucket.transpose(-1, -2)) # (n_b, bs_x, bs_y)
        mask = torch.index_select(y, dim=0, index=top_x_bucket.view(-1)).view(self.n_buckets, self.bucket_size_x)[:, :, None] == top_y_bucket[:, None, :] # (n_b, bs_x, bs_y)
        wrong_class_logits = wrong_class_logits.masked_fill(mask, float('-inf')) # (n_b, bs_x, bs_y)
        correct_class_logits = torch.index_select(correct_class_logits_, dim=0, index=top_x_bucket.view(-1)).view(self.n_buckets, self.bucket_size_x)[:, :, None] # (n_b, bs_x, 1)
        logits = torch.cat((wrong_class_logits, correct_class_logits), dim=2) # (n_b, bs_x, bs_y + 1)

        loss_ = F.cross_entropy(logits.view(-1, logits.shape[-1]), (logits.shape[-1] - 1) * torch.ones(logits.shape[0] * logits.shape[1], dtype=torch.int64, device=logits.device), reduction='none') # (n_b * bs_x,)
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        loss.scatter_reduce_(0, top_x_bucket.view(-1), loss_, reduce='amax', include_self=False)
        loss = loss[(loss != 0) & (y != self.pad_token)]
        loss = torch.mean(loss)

        return loss, emb
    

    def bt_loss(self, aug_emb, seq_emb, lmd_sem):
        seq_emb = seq_emb[:,-1,:]
        aug_emb = aug_emb[:,-1,:]

        #L2 normalize
        seq_emb = nn.functional.normalize(seq_emb, dim=1, p=2)
        aug_emb = nn.functional.normalize(aug_emb, dim=1, p=2)

        z_i = self.projector(seq_emb)
        z_j = self.projector(aug_emb)
        batch_size = aug_emb.shape[0]

        z_i_norm = (z_i - z_i.mean(0)) / z_i.std(0)
        z_j_norm = (z_j - z_j.mean(0)) / z_j.std(0)

        feature_dim = z_i_norm.shape[1]

        C = (z_i_norm.T @ z_j_norm) / batch_size

        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum() / feature_dim
        off_diag = off_diagonal(C).pow_(2).sum() / feature_dim #* (feature_dim-1)
        
        return (on_diag + lmd_sem * off_diag)
    

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())
        
        return alignment, uniformity

    def get_embeddings(self, seq):
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        return log_feats[:, -1, :]

    def compute_bucket_gradients(self, logits, pos_seqs):
        """emb = self.log2feats(log_seqs)
        logits = emb @ self.item_emb.weight.T
        indices = torch.where(pos_seqs.view(-1) != self.pad_token)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos_seqs.view(-1)[indices], reduction='mean')
        return loss, emb"""

        bucket_items = defaultdict(list)
        [bucket_items[v].append(k) for k,v in self.item_poularity_buckets.items()]
        res = {}
        for bucket_idx, items in bucket_items.items():
            pos_seqs_i = pos_seqs[:, items]
            indices_i = torch.where(pos_seqs_i.view(-1) != self.pad_token)
            logits_i = logits[:, items]
            loss_i = F.cross_entropy(logits_i.view(-1, logits_i.shape[-1])[indices_i], pos_seqs_i.view(-1)[indices_i], reduction='sum')
            monitoring_grads = torch.autograd.grad(loss_i, self.parameters(), retain_graph=True)
            grad_norm = torch.sqrt(sum(g.norm(2) ** 2 for g in monitoring_grads if g is not None))
            self.item_bucket_grad_norms[bucket_idx] = grad_norm.item()