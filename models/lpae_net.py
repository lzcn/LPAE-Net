import math

import attr
import torch
import torch.nn.functional as F
from torch import nn

import torchutils
from torchutils.param import Param


@attr.s
class LPAENetParam(Param):
    name = attr.ib(default="LPAENet")
    num_users = attr.ib(default=630)
    backbone = attr.ib(default="alexnet")
    embd_dim = attr.ib(default=128)
    com_memory = attr.ib(default=False)
    loss_weight = attr.ib(factory=dict)
    num_points = attr.ib(default=0)
    num_proto = attr.ib(default=16)
    num_sab = attr.ib(default=2)
    num_seeds = attr.ib(default=1)
    num_heads = attr.ib(default=4)
    logdet = attr.ib(default=True)
    use_nn_feature = attr.ib(default=False)
    use_semantic = attr.ib(default=False)
    use_visual = attr.ib(default=False)
    cold_start = attr.ib(default=False)


@attr.s
class LatentFactorNetParam(Param):
    name = attr.ib(default="LatentFactorNet")
    backbone = attr.ib(default="alexnet")
    embd_dim = attr.ib(default=128)
    num_users = attr.ib(default=630)
    loss_weight = attr.ib(factory=dict)
    num_points = attr.ib(default=0)
    num_sab = attr.ib(default=2)
    num_seeds = attr.ib(default=1)
    com_score = attr.ib(default=False)
    num_heads = attr.ib(default=4)
    use_nn_feature = attr.ib(default=False)
    use_semantic = attr.ib(default=False)
    use_visual = attr.ib(default=False)
    cold_start = attr.ib(default=False)


class UserEmbedding(nn.Module):
    def __init__(self, num_users, dim):
        super().__init__()
        self.num_users = num_users
        self.encoder = nn.Linear(num_users, dim, bias=False)
        nn.init.normal_(self.encoder.weight, std=0.01)

    def forward(self, x):
        x = torchutils.one_hot(x, self.num_users)
        h = self.encoder(x)
        return h


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask_a, mask_b):
        # b x n x d
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        # (h b) x n x 1
        mask_a = mask_a.repeat(self.num_heads, 1, 1)
        mask_b = mask_b.repeat(self.num_heads, 1, 1)
        # (h b) x n x d
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        # (n b) x n x n
        dots = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
        mask = torch.bmm(mask_a, mask_b.transpose(1, 2)) == 1.0
        value = -torch.finfo(dots.dtype).max
        dots.masked_fill_(~mask, value)
        A = torch.softmax(dots, dim=2)
        # delta = 1e16 * (1.0 - torch.bmm(mask_a, mask_b.transpose(1, 2)))
        # A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) - delta / math.sqrt(self.dim_V), dim=2)
        # (h b) x n x d -> b x n x (h d)
        H = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        H = H if getattr(self, "ln0", None) is None else self.ln0(H)
        H = H + F.relu(self.fc_o(H))
        H = H if getattr(self, "ln1", None) is None else self.ln1(H)
        return H


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask):
        return self.mab(X, X, mask, mask)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # noqa: E741
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.register_buffer("mask", torch.ones(1, num_seeds, 1))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask):
        b = X.size(0)
        return self.mab(self.S.repeat(b, 1, 1), X, self.mask.repeat(b, 1, 1), mask)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def stack_sab(dim, num_heads, num_sab, num_points=0):
    def _isab():
        return ISAB(dim, dim, num_heads, num_points, ln=True)

    def _sab():
        return SAB(dim, dim, num_heads, ln=True)

    if num_points and num_points > 0:
        sab = _isab
    else:
        sab = _sab
    sabs = [sab() for _ in range(num_sab)]
    return nn.ModuleList(sabs)


def embed_layer(in_features, out_features, dropout=0.2) -> nn.Module:
    x = nn.Sequential(
        nn.Linear(in_features, out_features), nn.LayerNorm(out_features), nn.ReLU(inplace=True), nn.Dropout(dropout)
    )
    x.apply(weights_init)
    return x


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_features,
        embd_dim=128,  # embedding dimension
        num_sab=2,  # num of self-attention blocks
        num_points=16,  # number of inducing points for iSAB
        num_heads=4,  # number of heads for Multi-head Attention
        num_seeds=4,  # number of seed vectors for outputs
    ):
        super().__init__()
        self.embed = embed_layer(in_features, embd_dim)
        # x = iSAB(iSAB(x)) or SAB(SAB(x)) according to the number of points
        self.encoder = stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        self.pma = PMA(embd_dim, num_heads, num_seeds, ln=True)
        # output SAB
        if num_seeds > 1:
            self.sab = SAB(embd_dim, embd_dim, num_heads, ln=True)
        else:
            self.sab = nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: a float tensor with shape [n, b, d].
        """
        h = self.embed(x)
        for encoder in self.encoder:
            h = encoder(h, mask)
        z = h
        # h = SAB(PMA(k, rFF(Z))), shape [b, k, d]
        out = self.sab(self.pma(z, mask))
        return out.permute(1, 0, 2)


class UserMemory(nn.Module):
    """Neighborhood-based prototype memory network."""

    def __init__(self, num_users, embd_dim, num_proto, com_memory=False, logdet=True):
        super().__init__()
        self.d = embd_dim
        self.logdet = logdet
        self.num_users = num_users
        self.num_proto = num_proto
        self.com_memory = com_memory
        self.users = nn.Parameter(torch.zeros(num_users, 1, embd_dim))
        # TODO: to delete since simple linear projection may be not necessary.
        self.trans = nn.Sequential(nn.Linear(embd_dim, embd_dim))
        if com_memory:
            self.out_proto = nn.Parameter(torch.zeros(num_users, num_proto // 2, embd_dim))
            self.com_proto = nn.Parameter(torch.zeros(1, num_proto // 2, embd_dim))
        else:
            self.out_proto = nn.Parameter(torch.zeros(num_users, num_proto, embd_dim))
            self.com_proto = None
        self.init()

    def init(self):
        # TODO: elegant way to initialize the weights
        scale = math.sqrt(3 / self.d)
        nn.init.uniform_(self.out_proto, a=-scale, b=scale)
        nn.init.uniform_(self.users, a=-scale, b=scale)
        if self.com_memory:
            nn.init.uniform_(self.com_proto, a=-scale, b=scale)
        for m in self.trans.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=math.sqrt(2 / self.d))

    def reg(self, uidx: torch.Tensor):
        out_proto = self.out_proto.index_select(0, uidx)
        s = out_proto.size(1)
        n = uidx.size(0)
        g = out_proto.matmul(out_proto.transpose(1, 2))
        # log-determinant divergence
        if self.logdet:
            reg = (g.diagonal(dim1=1, dim2=2).sum() - torch.logdet(g).sum()) / n
        # orthogonality regularization as an alternative
        # when illegal GPU memory access happens
        # TODO: to find that why logdet will rarely raise RuntimeError
        else:
            reg = ((g * (torch.ones_like(g) - torch.eye(s).cuda())) ** 2).sum() / n
        return reg

    def forward(self, uidx: torch.Tensor, x: torch.Tensor, attn_map=False):
        n = x.size(0)
        x = x.unsqueeze(1)
        # n x s x d
        if self.com_memory:
            u_anchors = self.out_proto.index_select(0, uidx)
            g_anchors = self.com_proto.repeat(n, 1, 1)
            anchors = torch.cat((u_anchors, g_anchors), dim=1)
        else:
            anchors = self.out_proto.index_select(0, uidx)
        # cosine similarity
        h = self.trans(x)
        z = F.normalize(h, dim=-1)
        anchors = F.normalize(anchors, dim=-1)
        # n x 1 x s
        cos_sim = z.matmul(anchors.transpose(1, 2))
        # n x 1
        score = cos_sim.sum(dim=-1).view(-1, 1)
        if attn_map:
            return score, cos_sim
        return score


class LPAENet(nn.Module):
    """LPAE-Net."""

    def __init__(self, param: LPAENetParam):
        super().__init__()
        self.param = param
        if param.use_visual:
            backbone, in_features = torchutils.backbone(param.backbone)
            # TODO: replace nn_features with more elegant configuration
            self.visual_feature = nn.Identity() if param.use_nn_feature else backbone
            self.visual_encoder = SetTransformer(
                in_features=in_features,
                embd_dim=param.embd_dim,
                num_sab=param.num_sab,
                num_points=param.num_points,
                num_seeds=param.num_seeds,
            )
        # TODO: add vse loss, do not use another transformer for textual feature
        if param.use_semantic:
            in_features = 2400
            raise NotImplementedError
        self.memory = UserMemory(
            num_users=param.num_users,
            embd_dim=param.embd_dim,
            num_proto=param.num_proto,
            com_memory=param.com_memory,
            logdet=param.logdet,
        )
        if param.cold_start:
            self.cold_start()

    def cold_start(self):
        for param in self.parameters():
            param.requires_grad = False
        self.memory.out_proto.requires_grad = True

    @torch.no_grad()
    def test_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        """Test forward."""
        batch, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num, -1)
        mask = 1.0 * (cate.view(batch, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        scores = self.memory(uidx, feat)
        return scores

    def train_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        """Training forward."""
        batch, _, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch * 2, num, -1)
        mask = 1.0 * (cate.view(batch * 2, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        feat = feat.view(batch, 2, -1)
        pos_feat, neg_feat = feat.split(1, dim=1)
        pos_score = self.memory(uidx, pos_feat.squeeze(), attn_map=False)
        neg_score = self.memory(uidx, neg_feat.squeeze(), attn_map=False)
        diff = pos_score.view(-1, 1) - neg_score.view(1, -1)
        diff = diff.view(-1)
        rank = F.soft_margin_loss(diff, torch.ones_like(diff), reduction="none")
        loss = dict(rank_loss=rank, l1reg=self.memory.reg(uidx))
        accuracy = dict(accuracy=torch.gt(diff, 0.0))
        return loss, accuracy

    def forward(self, *inputs):
        """Forward according to setting."""
        if self.training:
            return self.train_batch(*inputs)
        return self.test_batch(*inputs)


class LatentMatch(nn.Module):
    def __init__(self, embd_dim, com_score=False):
        super().__init__()
        self.d = d = embd_dim
        self.com_score = com_score
        self.dense_g = nn.Parameter(torch.zeros(d, d))
        self.out_g = nn.Parameter(torch.zeros(d, 1))
        self.bias_g = nn.Parameter(torch.zeros(1, d))
        if com_score:
            self.dense_c = nn.Parameter(torch.zeros(d, d))
            self.out_c = nn.Parameter(torch.zeros(d, 1))
            self.bias_c = nn.Parameter(torch.zeros(1, d))
            self.scale = nn.Parameter(torch.ones(1, 1))
        self.init()

    def init(self):
        nn.init.normal_(self.out_g, std=math.sqrt(2 / self.d))
        nn.init.normal_(self.dense_g, std=math.sqrt(2 / self.d))
        if self.com_score:
            nn.init.normal_(self.out_c, std=math.sqrt(2 / self.d))
            nn.init.normal_(self.dense_c, std=math.sqrt(2 / self.d))

    def forward(self, u, z):
        u = F.normalize(u, dim=-1)
        z = F.normalize(z, dim=-1)
        if self.com_score:
            score_u = F.relu((u * z).matmul(self.dense_g) + self.bias_g).matmul(self.out_g)
            score_i = F.relu(z.matmul(self.dense_c) + self.bias_c).matmul(self.out_c)
            score = score_u * self.scale + score_i
        else:
            score = F.relu((u * z).matmul(self.dense_g) + self.bias_g).matmul(self.out_g)
        return score


class LatentFactorNet(nn.Module):
    """Basic outfit transformer.

    Param:
        embd_dim: embedding dimension
        num_sab: number of SABs or iSABs
        num_points: number of induced points, if num_points = 0, then use stacked SABs
            else stacked iSABs
        num_seeds: number of seeds for output

    """

    def __init__(self, param: LatentFactorNetParam):
        super().__init__()
        self.param = param

        if param.use_visual:
            backbone, in_features = torchutils.backbone(param.backbone)
            self.visual_feature = nn.Identity() if param.use_nn_feature else backbone
            self.visual_encoder = SetTransformer(
                in_features=in_features,
                embd_dim=param.embd_dim,
                num_sab=param.num_sab,
                num_points=param.num_points,
                num_seeds=param.num_seeds,
                num_heads=param.num_heads,
            )
        if param.use_semantic:
            raise NotImplementedError
        self.users = UserEmbedding(param.num_users, param.embd_dim)
        self.match = LatentMatch(param.embd_dim, param.com_score)
        if param.cold_start:
            self.cold_start()

    def cold_start(self):
        for param in self.parameters():
            param.requires_grad = False
        self.users.weight.requires_grad = True

    def test_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        """Test forward."""
        batch, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num, -1)
        mask = 1.0 * (cate.view(batch, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        users = self.users(uidx)
        scores = self.match(users, feat)
        return scores

    def train_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        users = self.users(uidx)
        batch, _, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch * 2, num, -1)
        mask = 1.0 * (cate.view(batch * 2, num, -1) != -1)
        feat = self.visual_encoder(feat, mask).reshape(-1, self.param.embd_dim)
        feat = feat.view(batch, 2, -1)
        pos_feat, neg_feat = feat.split(1, dim=1)
        pos_feat = pos_feat.squeeze()
        neg_feat = neg_feat.squeeze()
        pos_score = self.match(users, pos_feat)
        neg_score = self.match(users, neg_feat)

        diff = pos_score - neg_score
        rank = F.soft_margin_loss(diff, torch.ones_like(diff), reduction="none")
        # l2reg = torch.norm(users, p=2, dim=-1) + torch.norm(pos_feat, p=2, dim=-1) + torch.norm(neg_feat, p=2, dim=-1)
        # l2reg /= batch
        loss = dict(rank_loss=rank)
        accuracy = dict(accuracy=torch.gt(diff, 0.0))
        return loss, accuracy

    def forward(self, *inputs):
        """Forward according to setting."""
        if self.training:
            return self.train_batch(*inputs)
        return self.test_batch(*inputs)
