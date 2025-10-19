import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from tqdm import tqdm
torch.cuda.empty_cache()

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# 超图卷积层，用于处理商品、价格和类别之间的关系。它通过多个门控机制（intra_gate 和 inter_gate）来融合不同类型的信息
class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_user, n_price, n_category):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.n_user = n_user  # 加个user

        self.mat_cp = nn.Parameter(torch.Tensor(self.n_category, 1))
        self.mat_pc = nn.Parameter(torch.Tensor(self.n_price, 1))

        # 加了个uv和vu
        self.mat_uv = nn.Parameter(torch.Tensor(self.n_user, 1))
        self.mat_vu = nn.Parameter(torch.Tensor(self.n_node, 1))

        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_cv = nn.Parameter(torch.Tensor(self.n_category, 1))

        self.a_o_g_i = nn.Linear(self.emb_size * 2, self.emb_size)
        self.b_o_gi1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_p = nn.Linear(self.emb_size * 2, self.emb_size)
        self.b_o_gp1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_c = nn.Linear(self.emb_size * 2, self.emb_size)
        self.b_o_gc1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc2 = nn.Linear(self.emb_size, self.emb_size)

        # 新增矩阵
        self.W_user = nn.Linear(3 * emb_size, emb_size)
        self.user_lambda = nn.Parameter(torch.tensor(0.2))  # 用户影响系数

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, adjacency_pv, adjacency_vp, adjacency_uv, adjacency_vu, adjacency_pc, adjacency_cp,
                adjacency_cv, adjacency_vc, embedding, pri_emb, cate_emb, user_emb):
        for i in range(self.layers):
            item_embeddings = self.inter_gate(self.a_o_g_i, self.b_o_gi1, embedding,
                                              self.get_embedding(adjacency_vp, pri_emb)) + self.get_embedding(
                adjacency, embedding) \
                              + self.user_lambda * self.intra_gate(adjacency_vu, self.mat_vu, user_emb)

            price_embeddings = self.inter_gate(self.a_o_g_p, self.b_o_gp1, pri_emb,
                                               self.intra_gate(adjacency_pv, self.mat_pv, embedding))

            # category_embeddings = self.inter_gate(self.a_o_g_c, self.b_o_gc1, self.b_o_gc2, cate_emb,
            #                                       self.intra_gate(adjacency_cp, self.mat_cp, pri_emb),
            #                                       self.intra_gate(adjacency_cv, self.mat_cv, embedding))
            # Inter-type聚合用户节点
            # price_embs = torch.randn(num_price_levels, emb_size).cuda()
            # cate_embs = torch.randn(num_categories, emb_size).cuda()
            # item_price_indices = torch.randint(0, num_price_levels, (num_items,)).cuda()
            # item_cate_indices = torch.randint(0, num_categories, (num_items,)).cuda()

            # 调用聚合函数
            e_price_user = self.inter_aggregate_user(
                adjacency_uv, pri_emb
            )
            user_embeddings = self.inter_gate_user(user_emb, self.intra_gate(adjacency_uv, self.mat_uv, embedding),
                                                   e_price_user)

            embedding = item_embeddings
            pri_emb = price_embeddings
            # cate_emb = category_embeddings
            user_emb = user_embeddings

        return item_embeddings, price_embeddings, user_embeddings

    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse_coo_tensor(i, v, shape)
        embs = embedding
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings

    def intra_gate(self, adjacency, mat_v, embedding2):
        # 构建稀疏矩阵（避免稠密转换）
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices).cuda()
        v = torch.FloatTensor(adjacency.data).cuda()
        sparse_adj = torch.sparse_coo_tensor(i, v, adjacency.shape).coalesce().cuda()

        # 分块处理参数
        batch_size = 4096  # 根据显存调整
        node_dim = sparse_adj.shape[0]
        chunk_num = (node_dim + batch_size - 1) // batch_size
        type_embs = []

        for chunk_idx in range(chunk_num):
            start = chunk_idx * batch_size
            end = min((chunk_idx + 1) * batch_size, node_dim)

            # 提取分块索引
            mask = (sparse_adj.indices()[0] >= start) & (sparse_adj.indices()[0] < end)
            chunk_indices = sparse_adj.indices()[:, mask]
            chunk_values = sparse_adj.values()[mask]

            if chunk_indices.shape[1] == 0:
                chunk_alpha = torch.zeros((batch_size, embedding2.shape[1]), device='cuda')
            else:
                # 调整行索引偏移
                local_rows = chunk_indices[0] - start
                chunk_indices_adj = torch.stack((local_rows, chunk_indices[1]))

                # 构建分块稀疏矩阵
                chunk_adj = torch.sparse_coo_tensor(
                    chunk_indices_adj,
                    chunk_values,
                    (batch_size, sparse_adj.shape[1]),
                    device='cuda'
                )

                # 稀疏矩阵乘法
                with torch.no_grad():  # 减少梯度占用
                    chunk_alpha = torch.sparse.mm(chunk_adj, embedding2)

                # 动态归一化（原位操作）
                row_sum = torch.sparse.sum(chunk_adj, dim=1).to_dense().view(-1, 1) + 1e-8
                chunk_alpha.div_(row_sum)  # 原地除法

            type_embs.append(chunk_alpha)
        # 聚合结果
        type_embs = torch.cat(type_embs, dim=0)[:node_dim]
        item_embeddings = type_embs * mat_v
        # print(f'item_embeddings.shape:{item_embeddings.shape}')
        return self.dropout70(item_embeddings)

    def inter_gate(self, a_o_g, b_o_g1, emb_mat1, emb_mat2):
        # 拼接嵌入矩阵
        all_emb1 = torch.cat([emb_mat1, emb_mat2], 1)
        gate1 = torch.sigmoid(a_o_g(all_emb1) + b_o_g1(emb_mat1))
        h_embedings = emb_mat1 + gate1 * emb_mat2
        return self.dropout50(h_embedings)

    def inter_aggregate_user(self, adjacency, price_emb):
        # Step 1: 将项目映射到对应的价格和类别嵌入
        # 假设 item_price_indices 和 item_cate_indices 是张量
        # 例如: item_price_indices = torch.LongTensor([0, 1, 0, 2, ...]) 形状为 [num_items]
        item_price_indices = torch.randint(0, self.n_price, (self.n_node,)).cuda()
        # item_cate_indices = torch.randint(0, self.n_category, (self.n_node,)).cuda()
        item_price_embs = price_emb[item_price_indices]  # [num_items, emb_size]
        # item_cate_embs = cate_emb[item_cate_indices]  # [num_items, emb_size]

        # Step 2: 将邻接矩阵转换为 PyTorch 稀疏张量（如果尚未转换）
        indices = torch.LongTensor(np.vstack((adjacency.row, adjacency.col)))
        values = torch.FloatTensor(adjacency.data)
        shape = adjacency.shape
        adjacency = torch.sparse_coo_tensor(indices, values, shape).cuda()

        # Step 3: 执行稀疏矩阵乘法（均值池化）
        # adjacency_uv: [num_users, num_items]
        # item_price_embs: [num_items, emb_size]
        # e_price_user: [num_users, emb_size]
        e_price_user = torch.sparse.mm(adjacency, item_price_embs)
        # e_cate_user = torch.sparse.mm(adjacency, item_cate_embs)

        return e_price_user

    def inter_gate_user(self, user_emb, e_item, e_price):
        """
        用户节点的Inter-type聚合：融合项目ID、价格、类别信息
        e_price: 用户关联项目价格的均值池化 [num_users, emb_size]
        e_category: 用户关联项目类别的均值池化 [num_users, emb_size]
        """
        # 拼接特征并动态门控
        combined = torch.cat([user_emb, e_item, e_price], dim=-1)
        gate = torch.sigmoid(self.W_user(combined))  # self.W_user为新增参数矩阵
        h_user = gate * user_emb + (1 - gate) * e_item
        return self.dropout50(h_user)


# 在PGCA类中添加以下组件
class PriceAwareEnhancements(nn.Module):
    def __init__(self, emb_size, n_price_levels, n_categories):
        super().__init__()
        # 1. 动态价格敏感性门控
        self.price_sensitivity_gate = nn.Sequential(
            nn.Linear(emb_size * 3, emb_size),
            nn.Sigmoid()
        )

        # 2. 价格趋势感知模块
        self.trend_conv = nn.Conv1d(emb_size, emb_size, kernel_size=3, padding=1)
        self.trend_attention = nn.Linear(emb_size, 1)

        # 3. 价格-类别联合嵌入
        self.price_category_fusion = nn.Linear(emb_size * 2, emb_size)

        # 4. 多粒度价格对比
        self.price_contrast = nn.ModuleDict({
            'level': nn.Sequential(nn.Linear(emb_size, emb_size)),  # 价格层级对比
            'trend': nn.Sequential(nn.Linear(emb_size, emb_size))})  # 价格趋势对比
        nn.init.xavier_uniform_(self.trend_conv.weight)
        nn.init.zeros_(self.trend_conv.bias)

    def forward(self, price_embeddings, price_seqs, category_embeddings, session_mask):
        """增强价格感知能力"""
        # 0. 安全处理：确保索引有效
        price_seqs = torch.clamp(price_seqs, 0, price_embeddings.size(0) - 1)

        # 1. 动态价格敏感性
        sensitivity = self.price_sensitivity_gate(
            torch.cat([price_embeddings.mean(dim=0),
                       price_embeddings.std(dim=0),
                       category_embeddings.mean(dim=0)], dim=-1)
        )

        # 2. 价格趋势提取
        trend_emb = self.extract_price_trend(price_embeddings, price_seqs, session_mask)

        return sensitivity, trend_emb

    def extract_price_trend(self, price_embeddings, price_seqs, mask):
        """提取会话内价格变化趋势"""
        price_embeddings = price_embeddings.cuda()
        price_seqs = price_seqs.cuda()
        mask = mask.cuda()

        # 添加索引安全检查
        max_index = price_embeddings.size(0) - 1
        if (price_seqs < 0).any() or (price_seqs > max_index).any():
            invalid_min = price_seqs.min().item()
            invalid_max = price_seqs.max().item()
            print(
                f"Error: price_seqs contains invalid indices. Min: {invalid_min}, Max: {invalid_max}, Allowed: [0, {max_index}]")
            # 截断无效索引值
            price_seqs = torch.clamp(price_seqs, 0, max_index)

        # 获取每个会话的实际长度
        valid_lengths = mask.sum(dim=1)  # [batch_size]
        # print(f"valid_lengths: {valid_lengths}")  # 调试信息
        # 序列维度处理 [batch, seq_len, emb_size] -> [batch, emb_size, seq_len]
        seq_emb = price_embeddings[price_seqs].permute(0, 2, 1)

        trend_embs = []  # 存储每个样本的趋势嵌入
        for i in range(seq_emb.size(0)):
            current_length = valid_lengths[i].item()

            if current_length < 3:
                # 使用全局平均池化代替卷积
                pooled = F.adaptive_avg_pool1d(seq_emb[i:i + 1], 1)
                trend_emb = pooled.permute(0, 2, 1)  # [1, 1, emb_size]
            else:
                # 正常卷积处理
                conv_out = F.relu(self.trend_conv(seq_emb[i:i + 1]))  # [1, emb_size, seq_len]
                conv_out = conv_out.permute(0, 2, 1)  # [1, seq_len, emb_size]

                # 趋势重要性加权 - 在样本级别计算
                trend_weights = F.softmax(self.trend_attention(conv_out), dim=1)
                trend_emb = (conv_out * trend_weights).sum(dim=1, keepdim=True)  # [1, 1, emb_size]

            trend_embs.append(trend_emb)

        # 拼接所有样本的趋势嵌入
        trend_emb = torch.cat(trend_embs, dim=0)  # [batch_size, 1, emb_size]
        # 创建会话级掩码并返回
        session_mask = (valid_lengths > 0).float().unsqueeze(-1)  # [batch_size, 1]
        return trend_emb.squeeze(1) * session_mask  # [batch_size, emb_size]

    def contrastive_loss(self, anchor, positive, negative, contrast_type='level', temperature=0.8):
        """多粒度价格对比损失"""
        # 检查输入是否为空
        if anchor.numel() == 0 or positive.numel() == 0 or negative.numel() == 0:
            return torch.tensor(0.0, device=anchor.device)

        # 确保所有输入在相同设备上
        anchor = anchor.to(self.price_contrast[contrast_type][0].weight.device)
        positive = positive.to(anchor.device)
        negative = negative.to(anchor.device)

        proj = self.price_contrast[contrast_type]

        # 检查维度并重塑张量
        if anchor.dim() > 2:
            anchor = anchor.view(-1, anchor.size(-1))
        if positive.dim() > 2:
            positive = positive.view(-1, positive.size(-1))
        if negative.dim() > 2:
            negative = negative.view(-1, negative.size(-1))

        anchor_proj = F.normalize(proj(anchor), dim=-1)
        pos_proj = F.normalize(proj(positive), dim=-1)
        neg_proj = F.normalize(proj(negative), dim=-1)

        # 使用温度参数缩放相似度
        pos_sim = torch.sum(anchor_proj * pos_proj, dim=-1) / temperature
        neg_sim = torch.sum(anchor_proj * neg_proj, dim=-1) / temperature

        return -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))).mean()


class PGCA(Module):
    def __init__(self, adjacency, adjacency_pv, adjacency_vp, adjacency_uv, adjacency_vu, adjacency_pc, adjacency_cp,
                 adjacency_cv, adjacency_vc,
                 n_node, n_user, n_price, n_category, lr, layers, l2, beta, dataset, lambda1, lambda2, num_heads, emb_size=100,
                 batch_size=100):
        super(PGCA, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_user = n_user  # 加个用户
        self.n_price = n_price
        self.n_category = n_category
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta

        self.adjacency = adjacency
        self.adjacency_pv = adjacency_pv
        self.adjacency_vp = adjacency_vp

        # 加个uv和vu
        self.adjacency_uv = adjacency_uv
        self.adjacency_vu = adjacency_vu

        self.adjacency_pc = adjacency_pc
        self.adjacency_cp = adjacency_cp
        self.adjacency_cv = adjacency_cv
        self.adjacency_vc = adjacency_vc

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        # 加个用户
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)

        self.price_embedding = nn.Embedding(self.n_price, self.emb_size)
        self.category_embedding = nn.Embedding(self.n_category, self.emb_size)

        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size, self.n_node, self.n_user, self.n_price,
                                    self.n_category)

        # 添加价格感知增强模块
        self.price_aware = PriceAwareEnhancements(emb_size, n_price, n_category)
        # 增加价格敏感度权重
        self.price_sensitivity_weight = nn.Parameter(torch.tensor(0.5))
        # 新增价格趋势预测头
        self.trend_prediction = nn.Linear(emb_size, 3)  # 预测趋势：上涨/持稳/下降

        self.w_1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # self_attention
        if emb_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
        # parameters setting
        self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16  the dimension of attention head
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # query, key, value
        self.query = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.key = nn.Linear(self.emb_size, self.emb_size)
        self.value = nn.Linear(self.emb_size, self.emb_size)

        # co-guided networks
        self.w_p_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.u_i_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # gate5 & gate6
        self.w_pi_1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_pi_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_i = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.mlp_m_p_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)
        self.mlp_m_i_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)

        self.mlp_m_p_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.mlp_m_i_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

         # ! =========== 新增对比组件 ===========
        self.node_contrast = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.LayerNorm(emb_size * 2),
            nn.GELU(),
            nn.Linear(emb_size * 2, emb_size),
            nn.LayerNorm(emb_size)
        )
        self.sess_contrast = nn.Sequential(
            nn.Linear(emb_size, emb_size * 3),
            nn.LayerNorm(emb_size * 3),
            nn.GELU(),
            nn.Linear(emb_size * 3, emb_size),
            nn.LayerNorm(emb_size)
        )
        # 在模型初始化中添加
        # 定义可学习的对比温度参数
        # 在PGCA类的初始化部分修改
        self.temp_node = nn.Parameter(torch.tensor(0.8))  # 可学习参数
        self.temp_sess = nn.Parameter(torch.tensor(0.8))
        self.temp_price_level = nn.Parameter(torch.tensor(0.8))  # 价格层级对比温度参数
        self.temp_price_trend = nn.Parameter(torch.tensor(0.8))  # 价格趋势对比温度参数
        self.contrast_loss_weight = nn.Parameter(torch.tensor(0.1))  # 初始值为 0.1
        self.item_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.price_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.user_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.cate_loss_weight = nn.Parameter(torch.tensor(1.0))
        # 添加约束（在forward前向传播前插入）
        with torch.no_grad():
            self.temp_node.data.clamp_(0.01, 1.0)  # 限制温度参数范围
            self.temp_sess.data.clamp_(0.01, 1.0)
            self.temp_price_level.data.clamp_(0.01, 1.0)
            self.temp_price_trend.data.clamp_(0.01, 1.0)
            self.contrast_loss_weight.clamp_(0.01, 1.0)

        self.hypergraph_skip = False  # 添加控制标志


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, price_embedding, session_item, price_seqs, session_len,
                          reversed_sess_item, mask):
        zeros = torch.tensor(0, dtype=torch.float, device='cuda').repeat(1, self.emb_size)
        # zeros = torch.zeros(1, self.emb_size)  # for different GPU
        mask = mask.float().unsqueeze(-1)

        price_embedding = torch.cat([zeros, price_embedding], 0)
        get_pri = lambda i: price_embedding[price_seqs[i]]
        seq_pri = torch.tensor(np.zeros((self.batch_size, list(price_seqs.shape)[1], self.emb_size)),
                               dtype=torch.float32, device='cuda')
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size) # for different GPU
        for i in torch.arange(price_seqs.shape[0]):
            seq_pri[i] = get_pri(i)
        # self-attention to get price preference
        attention_mask = mask.permute(0, 2, 1).unsqueeze(1)  # [bs, 1, 1, seqlen]
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(seq_pri)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_pri)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_pri)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores + attention_mask
        # add mask，set padding to -10000
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # [bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)
        item_pos = torch.tensor(range(1, seq_pri.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(price_seqs)

        item_pos = item_pos * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        price_pre = torch.sum(last_interest, 1)

        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)

        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        beta = beta * mask
        interest_pre = torch.sum(beta * seq_h, 1)

        # Co-guided Learning
        m_c = torch.tanh(self.w_pi_1(price_pre * interest_pre))
        m_j = torch.tanh(self.w_pi_2(price_pre + interest_pre))

        r_i = torch.sigmoid(self.w_c_z(m_c) + self.u_j_z(m_j))
        r_p = torch.sigmoid(self.w_c_r(m_c) + self.u_j_r(m_j))

        m_p = torch.tanh(self.w_p(price_pre * r_p) + self.u_p((1 - r_p) * interest_pre))
        m_i = torch.tanh(self.w_i(interest_pre * r_i) + self.u_i((1 - r_i) * price_pre))

        # enriching the semantics of price and interest preferences
        p_pre = (price_pre + m_i) * m_p
        i_pre = (interest_pre + m_p) * m_i

        return i_pre, p_pre

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def forward(self, session_item, price_seqs, session_len, reversed_sess_item, mask):
        # 添加价格索引检查
        # print(f"Price seqs min: {price_seqs.min().item()}, max: {price_seqs.max().item()}")
        # print(f"Price embedding size: {self.price_embedding.weight.size(0)}")
        # 确保价格序列在有效范围内
        price_seqs = torch.clamp(price_seqs, 0, self.n_price - 1)
        # 获取基础嵌入
        item_embeddings_hg = self.embedding.weight
        price_embeddings_hg = self.price_embedding.weight
        user_embeddings_hg = self.user_embedding.weight
        cate_embeddings_hg = self.category_embedding.weight

        # 创建与generate_sess_emb一致的嵌入矩阵 (添加零行)
        zeros = torch.zeros(1, self.emb_size, device=price_embeddings_hg.device)

        if not self.hypergraph_skip:
            # session_item all sessions in a batch [[23,34,0,0],[1,3,4,0]]
            item_embeddings_hg, price_embeddings_hg, user_embeddings_hg = \
                self.HyperGraph(self.adjacency, self.adjacency_pv, self.adjacency_vp,
                                self.adjacency_uv, self.adjacency_vu,
                                self.adjacency_pc, self.adjacency_cp,
                                self.adjacency_cv, self.adjacency_vc,
                                self.embedding.weight,
                                self.price_embedding.weight,
                                self.category_embedding.weight,
                                self.user_embedding.weight)  # updating the item embeddings
        sess_emb_hgnn, sess_pri_hgnn = self.generate_sess_emb(item_embeddings_hg, price_embeddings_hg, session_item,
                                                              price_seqs, session_len, reversed_sess_item,
                                                              mask)  # session embeddings in a batch
        # 处理item-price关系（跳过邻接矩阵依赖）
        if self.hypergraph_skip or self.adjacency_vp is None:
            # 预测模式：使用简单索引（示例方案，需根据实际数据调整）
            item_pri_l = price_embeddings_hg[price_seqs[:, -1]]  # 取最后一个价格级别
        else:
            # get item-price table return price of items
            v_table = self.adjacency_vp.row
            temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
            vp_idx = self.adjacency_vp.col[idx]
            item_pri_l = price_embeddings_hg[vp_idx]

        price_embeddings_with_zero = torch.cat([zeros, price_embeddings_hg], dim=0)
        # 增强价格感知
        sensitivity, trend_emb = self.price_aware(
            price_embeddings_with_zero,
            price_seqs,
            cate_embeddings_hg,
            mask.float()
        )

        # 融合价格趋势信息
        sess_pri_hgnn = sess_pri_hgnn + sensitivity * trend_emb
        # 价格趋势预测辅助任务
        trend_labels = self.calculate_price_trend(price_seqs, mask)
        trend_pred = self.trend_prediction(trend_emb)
        trend_loss = F.cross_entropy(trend_pred, trend_labels)
        # 在调用multi_grain_contrast前添加检查
        if price_embeddings_hg.size(0) < self.n_price:
            print(f"Warning: price_embeddings_hg size {price_embeddings_hg.size(0)} < n_price {self.n_price}")
            # 扩展嵌入矩阵以匹配n_price
            extension = torch.zeros(self.n_price - price_embeddings_hg.size(0),
                                    self.emb_size,
                                    device=price_embeddings_hg.device)
            price_embeddings_hg = torch.cat([price_embeddings_hg, extension], dim=0)

        # 多粒度价格对比学习
        price_contrast_loss = self.multi_grain_contrast(
            price_embeddings_hg,
            price_seqs,
            mask
        )
        ploss = price_contrast_loss + 0.8 * trend_loss

        # 对比学习部分（仅在训练时计算）
        node_loss, sess_loss = 0, 0
        if not self.hypergraph_skip:
            # ! ========== 节点对比分支 ==========
            aug_item = self._mask_augment(item_embeddings_hg)  # 特征掩码增强
            item_loss = self._node_contrast_loss(
                item_embeddings_hg, aug_item
            )
            aug_price = self._mask_augment(price_embeddings_hg)
            price_loss = self._node_contrast_loss(
                price_embeddings_hg, aug_price
            )
            # aug_user = self._mask_augment(user_embeddings_hg)
            # user_loss = self._node_contrast_loss(
            #     user_embeddings_hg, aug_user
            # )
            # 在训练循环中分批次计算损失
            batch_size = 4096  # 根据GPU内存调整
            num_users = user_embeddings_hg.size(0)
            user_loss = 0.0

            for start in range(0, num_users, batch_size):
                end = start + batch_size
                batch_emb = user_embeddings_hg[start:end]
                # 对当前批次进行增强
                aug_batch = self._mask_augment(batch_emb)
                # 计算当前批次的对比损失
                batch_loss = self._node_contrast_loss(batch_emb, aug_batch)
                # 按比例累加损失（考虑最后一批可能较小）
                user_loss += batch_loss * (aug_batch.size(0) / num_users)

            # aug_cate = self._mask_augment(cate_embeddings_hg)
            # cate_loss = self._node_contrast_loss(
            #     cate_embeddings_hg, aug_cate
            # )
            # 将权重转换为概率分布
            weights = torch.softmax(torch.stack([
                self.item_loss_weight,
                self.price_loss_weight,
                self.user_loss_weight,
            ]), dim=0)

            node_loss = (
                    weights[0] * item_loss +
                    weights[1] * price_loss +
                    weights[2] * user_loss
            )

             # 会话对比分支
            # sess_aug = self.subsequence_augment(session_item)
            sess_aug, price_seqs_aug, session_len_aug = self.subsequence_augment(session_item, price_seqs, session_len)
            aug_sess_emb, _ = self.generate_sess_emb(
                item_embeddings_hg, price_embeddings_hg,
                sess_aug,  # 使用增强后的序列
                price_seqs_aug, session_len_aug, reversed_sess_item, mask
            )
            sess_loss = self._sess_contrast_loss(sess_emb_hgnn, aug_sess_emb)

            # get item-price table return price of items
            v_table = self.adjacency_vp.row
            temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
            vp_idx = self.adjacency_vp.col[idx]
            item_pri_l = price_embeddings_hg[vp_idx]


        return item_embeddings_hg, price_embeddings_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, node_loss, sess_loss, ploss

    # ! ========== 新增私有方法 ==========
    def _mask_augment(self, emb, mask_rate=0.3):
        """混合增强策略"""
        # 检查并修复非法值
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            print("Warning: NaN or Inf found in embedding. Replacing with zeros.")
            emb = torch.nan_to_num(emb)

        # 确保张量在正确的设备上
        device = emb.device
        # 策略一：随机掩码
        mask = torch.rand(emb.size(), device=device) < mask_rate
        masked_emb = emb * (~mask)

        # 策略二：高斯噪声
        noise = torch.randn_like(emb, device=device) * 0.1
        noised_emb = emb + noise

        # 策略三：随机裁剪（针对序列数据）
        if len(emb.shape) == 3:  # 序列维度
            seq_len = emb.shape[1]
            crop_len = int(seq_len * 0.7)
            start = torch.randint(0, seq_len - crop_len, (1,), device=device)
            cropped_emb = emb[:, start:start + crop_len]
            cropped_emb = F.pad(cropped_emb, (0, 0, 0, seq_len - crop_len))
        else:
            cropped_emb = emb

        # 随机选择增强方式
        aug_type = torch.randint(0, 3, (1,)).item()
        if aug_type == 0:
            return masked_emb
        elif aug_type == 1:
            return noised_emb
        else:
            return cropped_emb

    # 会话序列对比
    def subsequence_augment(
            self,
            session_item,  # 输入原始序列：[batch_size, max_seq_len]
            price_seqs,  # 对应价格序列：[batch_size, max_seq_len]
            session_len,  # 原始序列有效长度：[batch_size]
            max_crop_ratio=0.1  # 最大裁剪比例
    ):
        batch_size, max_len = session_item.shape
        device = session_item.device

        # 初始化增强后的输出（保持原始张量形状）
        aug_session_item = torch.zeros_like(session_item).to(device)
        aug_price_seqs = torch.zeros_like(price_seqs).to(device)
        aug_session_len = torch.zeros_like(session_len).to(device)

        for i in range(batch_size):
            # 当前会话的实际长度
            actual_len = session_len[i].item()
            if actual_len <= 0:
                # 如果是非法会话，直接跳过保留全零
                continue

            # 生成 [0,1] 之间的随机比例（在同一device上）
            random_ratio = torch.rand(1, device=device).item()
            # 计算裁剪长度（至少保留1个元素）
            crop_len = max(1, int(actual_len * max_crop_ratio * random_ratio))

            # === 关键修正点2：确保随机起始位置生成在相同device ===
            start_pos = torch.randint(
                low=0,
                high=actual_len - crop_len + 1,
                size=(1,),
                device=device
            ).item()


            # 裁剪对应的子序列和价格序列
            cropped_items = session_item[i, start_pos: start_pos + crop_len]
            cropped_prices = price_seqs[i, start_pos: start_pos + crop_len]

            # 将裁剪部分填充到增强序列的头部，其余位置用0填充
            aug_session_item[i, :crop_len] = cropped_items
            aug_price_seqs[i, :crop_len] = cropped_prices
            aug_session_len[i] = crop_len

        return aug_session_item, aug_price_seqs, aug_session_len


    def _node_contrast_loss(self, z1, z2):
        """节点级对比损失"""
        z1 = F.normalize(self.node_contrast(z1), dim=1)
        z2 = F.normalize(self.node_contrast(z2), dim=1)
        logits = torch.mm(z1, z2.T) / self.temp_node
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

    def _sess_contrast_loss(self, s1, s2):
        """会话级对比损失"""
        z1 = F.normalize(self.sess_contrast(s1), dim=1)
        z2 = F.normalize(self.sess_contrast(s2), dim=1)
        sim_matrix = torch.mm(z1, z2.T) / self.temp_sess
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim_matrix, labels)

    def multi_grain_contrast(self, price_emb, price_seqs, mask):
        """多粒度价格对比学习"""
        batch_size, seq_len = price_seqs.shape

        # 1. 价格层级对比 (相同价格层级)
        same_level = price_seqs[:, -1].unsqueeze(1) == price_seqs
        positive = price_emb[price_seqs[:, -1]]
        negative = price_emb[torch.randint(0, self.n_price, (batch_size,), device=price_emb.device)]
        level_loss = self.price_aware.contrastive_loss(
            positive, positive, negative, 'level', self.temp_price_level
        )

        # 2. 价格趋势对比 (相似价格趋势)
        trend_diff = torch.zeros_like(price_seqs, dtype=torch.float, device=price_seqs.device)

        # 计算差异（从第二个元素开始）
        if seq_len > 1:
            trend_diff[:, 1:] = price_seqs.diff(dim=1).sign().float()

        # 获取最后一个价格趋势（每个样本）
        last_trend = trend_diff[:, -1].unsqueeze(1)

        # 创建相同趋势的掩码（排除零值）
        similar_trend = (trend_diff == last_trend) & (trend_diff != 0)
        dissimilar_trend = (trend_diff != last_trend) & (trend_diff != 0)

        # 初始化损失
        trend_loss = torch.tensor(0.0, device=price_emb.device)

        # 对每个样本单独处理
        for i in range(batch_size):
            # 获取当前样本的相似和不相似索引
            similar_indices = similar_trend[i].nonzero(as_tuple=False).squeeze()
            dissimilar_indices = dissimilar_trend[i].nonzero(as_tuple=False).squeeze()

            if similar_indices.numel() == 0 or dissimilar_indices.numel() == 0:
                continue

            # 确保索引是一维的
            if similar_indices.dim() == 0:
                similar_indices = similar_indices.unsqueeze(0)
            if dissimilar_indices.dim() == 0:
                dissimilar_indices = dissimilar_indices.unsqueeze(0)

            # 获取相似和不相似的价格
            similar_prices = price_seqs[i, similar_indices]
            dissimilar_prices = price_seqs[i, dissimilar_indices]

            # 获取锚点价格嵌入
            anchor_emb = price_emb[price_seqs[i, -1]].unsqueeze(0)

            # 获取相似嵌入
            similar_emb = price_emb[similar_prices]
            dissimilar_emb = price_emb[dissimilar_prices]

            # 随机选择一个负样本
            neg_idx = torch.randint(0, dissimilar_emb.size(0), (1,))
            negative_emb = dissimilar_emb[neg_idx].unsqueeze(0)

            # 计算当前样本的对比损失
            sample_loss = self.price_aware.contrastive_loss(
                anchor_emb,
                similar_emb.mean(dim=0, keepdim=True),
                negative_emb,
                'trend',
                self.temp_price_trend
            )
            trend_loss += sample_loss

        # 平均损失
        if batch_size > 0:
            trend_loss /= batch_size

        return level_loss + trend_loss

    def calculate_price_trend(self, price_seqs, mask):
        """计算价格趋势标签 (0:下降, 1:持稳, 2:上涨)"""
        batch_size = price_seqs.size(0)
        device = price_seqs.device
        trend_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            # 获取有效价格序列
            valid_mask = mask[i].bool()
            if valid_mask.sum() < 2:
                continue

            valid_prices = price_seqs[i][valid_mask]

            # 使用线性回归计算趋势
            if len(valid_prices) > 1:
                x = torch.arange(len(valid_prices), device=device).float()
                y = valid_prices.float()

                # 计算斜率
                x_mean = x.mean()
                y_mean = y.mean()
                numerator = ((x - x_mean) * (y - y_mean)).sum()
                denominator = ((x - x_mean) ** 2).sum()

                if denominator > 0:
                    slope = numerator / denominator

                    # 设置趋势标签
                    if slope > 0.1:  # 上涨阈值
                        trend_labels[i] = 2
                    elif slope < -0.1:  # 下降阈值
                        trend_labels[i] = 0
                    else:
                        trend_labels[i] = 1  # 持稳

        return trend_labels


def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs = data.get_slice(
        i)  # obtaining instances from a batch
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, node_loss, sess_loss, ploss = \
        model(session_item, price_seqs, session_len, reversed_sess_item, mask)
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(item_pri_l, 1, 0))
    scores = scores_interest + scores_price
    return tar, scores, node_loss, sess_loss, ploss


def train_test(model, train_data, test_data, lambda1):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    contrast_loss = 0.0  # 新增对比损失记录
    slices = train_data.generate_batch(model.batch_size)

    # Adding progress bar to the training loop
    for i in tqdm(slices, desc="Training", unit="batch"):
        model.zero_grad()
        targets, scores, node_loss, sess_loss, ploss = forward(model, i, train_data)  # 新增两个损失项
        # ! 动态权重调整
        # curr_weight = min(0.5, 0.1 + 0.05 * epoch)  # 随训练逐步增加权重
        # 获取可学习的对比损失权重
        # contrast_loss_weight = model.contrast_loss_weight
        # 计算总损失
        main_loss = model.loss_function(scores + 1e-8, targets)
        l2_reg = 0.01 * (model.item_loss_weight ** 2 + model.price_loss_weight ** 2 + model.user_loss_weight ** 2 + model.cate_loss_weight ** 2)
        total_batch_loss = main_loss + lambda1 * (node_loss + sess_loss + ploss) + l2_reg

        # ! 梯度计算与回传
        total_batch_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # ! 梯度裁剪
        model.optimizer.step()

        # ! 损失统计（保持原有total_loss结构）
        total_loss += main_loss.item()
        contrast_loss += (node_loss.item() + sess_loss.item() + ploss.item())

    print(f'\tMain Loss: {total_loss:.3f} | Contrast Loss: {contrast_loss:.3f}')  # ! 新增输出

    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []

    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)

    # Adding progress bar to the prediction loop
    for i in tqdm(slices, desc="Predicting", unit="batch"):
        tar, scores, _, _, _ = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))

    return metrics, total_loss
