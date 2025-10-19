import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) #count the unique items in a session，delete the repeat items, ranking by item_id
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    # indptr:sum of the session length; indices:item_id - 1
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 10000 * 6558 #sessions * #items H
    return matrix

def data_easy_masks(data_l, n_row, n_col):
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]

    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 10000 * 6558 #sessions * #items H
    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# node就是item
class Data():
    def __init__(self, data, shuffle=False, n_node=None, n_user=None, n_price=None, n_category=None):
        self.raw = np.asarray(data[0], dtype=object) # sessions, item_seq

        self.price_raw = np.asarray(data[1], dtype=object) # price_seq

        H_T = data_easy_masks(data[2], len(data[0]), n_node)  # 10000 * 6558 #sessions * #items
        # items * sessions，对每一行求和，返回每个商品的总和，然后对每一列做归一化
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        # 处理零行，将其对应的结果设为零
        row_sums = H.sum(axis=1).reshape(1, -1)
        row_sums[row_sums == 0] = 1  # 将零行的和设置为1，防止除以零
        DH = H.T.multiply(1.0 / row_sums)
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)  # 这个矩阵可能表示会话和商品之间的加权关系。

        H_pv = data_easy_masks(data[4], n_price, n_node)
        BH_pv = H_pv

        BH_vp = H_pv.T

        # 加一个用户-项目
        H_uv = data_easy_masks(data[5], n_user, n_node)
        BH_uv = H_uv
        BH_vu = H_uv.T

        H_pc = data_easy_masks(data[6], n_price, n_category)
        BH_pc = H_pc

        BH_cp = H_pc.T

        H_cv = data_easy_masks(data[7], n_category, n_node)
        BH_cv = H_cv
        
        BH_vc = H_cv.T

        self.adjacency = DHBH_T.tocoo()

        self.adjacency_pv = BH_pv.tocoo()
        self.adjacency_vp = BH_vp.tocoo()
        # 加一个用户
        self.adjacency_uv = BH_uv.tocoo()
        self.adjacency_vu = BH_vu.tocoo()

        self.adjacency_pc = BH_pc.tocoo()
        self.adjacency_cp = BH_cp.tocoo()
        self.adjacency_cv = BH_cv.tocoo()
        self.adjacency_vc = BH_vc.tocoo()

        self.n_node = n_node
        self.n_user = n_user
        self.n_price = n_price
        self.n_category = n_category
        self.targets = np.asarray(data[8])
        self.length = len(self.raw)
        self.shuffle = shuffle

    # 计算了每一对会话之间的物品集合交集和并集的比例（这个函数整个项目暂时没用到）
    def get_overlap(self, sessions):
        # 初始化一个零矩阵 matrix，它的大小是 len(sessions) x len(sessions)
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            # 将会话 i 转换为集合 seq_a
            seq_a = set(sessions[i])
            seq_a.discard(0)
            # 内层循环遍历会话 i 后面的每一个会话 j
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)  # 两个集合的交集
                ab_set = seq_a | seq_b  # 两个集合的并集
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]  # matrix[i][j] 存储这个重叠度值
        matrix = matrix + np.diag([1.0]*len(sessions))  # 将矩阵的对角线设置为 1.0。每个会话与自身的重叠度当然是 1
        # 度矩阵degree，度是一个会话与其他会话的重叠度的总和（matrux矩阵的每一行的和）
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    # 生成批次数据的切片，通常应用在训练神经网络模型中。其目的是将原始数据按批次分割，并根据需要对数据进行洗牌
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # random session item_seq & price_seq
            self.raw = self.raw[shuffled_arg]
            self.price_raw = self.price_raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    # 从指定索引的原始数据中提取出一个样本
    def get_slice(self, index):
        items, num_node, price_seqs = [], [], []
        inp = self.raw[index]   # 获取该索引对应的原始会话数据，就是用户交互的项目id
        inp_price = self.price_raw[index]  # 获取该索引对应的价格序列数据
        # 表示该会话中的有效交互节点数（例如购买、点击等）
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []   # 存储每个会话的有效节点长度
        reversed_sess_item = []
        mask = []   # 用于存储掩码数据
        for session, price in zip(inp,inp_price):
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, price_seqs


