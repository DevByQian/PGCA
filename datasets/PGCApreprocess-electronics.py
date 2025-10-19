import os
import pickle
import time
import pandas as pd
import numpy as np
import math

# Cosmetics

datasets_name = 'events'
data_path = './electronics/' + datasets_name + '.csv'

# the number of price levels
price_level_num = 19

data_all = pd.read_csv(data_path)

# only retain the records with type 'cart'  and 'purchase'
data_all = data_all[(data_all['event_type'] == 'cart') | (data_all['event_type'] == 'purchase')]
data_all = data_all[['event_time', 'product_id', 'category_id', 'price', 'user_session', 'user_id']]

print(data_all)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def reg_price(price):
    if is_number(price):
        if price == 0:
            results = ''
        else:
            results = float(price)
    else:
        results = ''
    return results


def reg_category(cate):
    results = ''
    if is_number(cate):
        results = cate
    else:
        results = ''
    return results


def reg_user(user):
    if is_number(user):
        results = user
    else:
        results = ''
    return results


def reg_time(envent_time):
    results = int(time.mktime(time.strptime(envent_time[:19], '%Y-%m-%d %H:%M:%S')))
    return results


data_all['price'] = data_all.price.map(reg_price)
data_all['category_id'] = data_all.category_id.map(reg_category)
data_all['event_time'] = data_all.event_time.map(reg_time)
data_all['user_id'] = data_all.user_id.map(reg_user)


# delete records without price tag
data_all = data_all[(data_all['price'] != '')]
# delete records without category tag
data_all = data_all[(data_all['category_id'] != '')]
# delete records without user tag
data_all = data_all[(data_all['user_id'] != '')]


interaction = data_all[['user_session', 'product_id', 'event_time']]

item_all = data_all[['product_id', 'category_id', 'price']]
item_all[['price']] = item_all[['price']].astype(float)
item_all.drop_duplicates(subset=['product_id'], keep='first', inplace=True)

# the number of items in each category
group_cate_num = pd.DataFrame(item_all.groupby(item_all['category_id']).count())
group_num = group_cate_num.reset_index()[['category_id', 'product_id']].rename(columns={'product_id': 'count'})
#  min price in each category
group_cate_min = pd.DataFrame(item_all['price'].groupby(item_all['category_id']).min())
group_min = group_cate_min.reset_index()[['category_id', 'price']].rename(columns={'price': 'min'})
# max price in each category
group_cate_max = pd.DataFrame(item_all['price'].groupby(item_all['category_id']).max())
group_max = group_cate_max.reset_index()[['category_id', 'price']].rename(columns={'price': 'max'})
# average price in each category
group_cate_mean = pd.DataFrame(item_all['price'].groupby(item_all['category_id']).mean())
group_mean = group_cate_mean.reset_index()[['category_id', 'price']].rename(columns={'price': 'mean'})

group_cate_std = pd.DataFrame(item_all['price'].groupby(item_all['category_id']).std())
group_std = group_cate_std.reset_index()[['category_id', 'price']].rename(columns={'price': 'std'})

item_data1 = pd.merge(item_all, group_num, how='left', on='category_id')
item_data2 = pd.merge(item_data1, group_min, how='left', on='category_id')
item_data3 = pd.merge(item_data2, group_max, how='left', on='category_id')
item_data4 = pd.merge(item_data3, group_mean, how='left', on='category_id')
item_data5 = pd.merge(item_data4, group_std, how='left', on='category_id')
# each category must contain at least 10 items
item_data = item_data5[item_data5['count'] > 9]

item_data = item_data[item_data['std'] != 0]


def logistic(t, u, s):
    gama = s * 3 ** (0.5) / math.pi
    results = 1 / (1 + math.exp((t - u) / gama))
    return results


def get_price_level(price, p_min, p_max, mean, std):
    if std == 0:
        print('only one sample')
        return -1
    fenzi = logistic(price, mean, std) - logistic(p_min, mean, std)
    fenmu = logistic(p_max, mean, std) - logistic(p_min, mean, std)
    if fenmu == 0 or price == 0:
        return -1
    results = int(fenzi / fenmu * price_level_num) + 1
    return results


item_data['price_level'] = item_data.apply(
    lambda row: get_price_level(row['price'], row['min'], row['max'], row['mean'], row['std']), axis=1)

item_final = item_data[item_data['price_level'] != -1]

user_item1 = pd.merge(interaction, item_final, how='left', on='product_id')
user_item2 = user_item1.dropna(axis=0)

user_item2.sort_values(by=["user_session", "event_time"], inplace=True, ascending=[True, True])

user_click_num = pd.DataFrame(user_item2.groupby(user_item2['user_session']).count())
click_num = user_click_num.reset_index()[['user_session', 'product_id']].rename(columns={'product_id': 'click_num'})
item_data6 = pd.merge(user_item2, click_num, how='left', on='user_session')
item_data7 = item_data6[item_data6['click_num'] > 1]
data = item_data7[['user_session', 'product_id', 'event_time', 'price', 'category_id', 'price_level']]

# 提取user_session和user_id列
user_id_data = data_all[['user_session', 'user_id']].drop_duplicates(subset=['user_session'])

# 将user_id列合并到最终的data数据框
data = pd.merge(data, user_id_data, how='left', on='user_session')

# 输出最终数据框
print('数据集：')
print('-----------------------------------------')
print(data.head())

# 修改后代码（假设原始数据中包含'user_id'列）
data_all = data.rename(
    columns={
        'user_session': 'sessionID',
        'product_id': 'itemID',
        'event_time': 'time',
        'user_id': 'userID',
        'price_level': 'priceLevel',
        'category_id': 'category',
    })
data_all = data_all[['sessionID', 'itemID', 'time', 'userID', 'price', 'priceLevel', 'category']]  # 包含用户ID

reviewerID2sessionID = {}
asin2itemID = {}
category2categoryID = {}
user2userID = {}

sessionNum = 0
itemNum = 0
categoryNum = 0
userNum = 0


for _, row in data_all.iterrows():
    if row['sessionID'] not in reviewerID2sessionID:
        sessionNum += 1
        reviewerID2sessionID[row['sessionID']] = sessionNum
    if row['itemID'] not in asin2itemID:
        itemNum += 1
        asin2itemID[row['itemID']] = itemNum
    if row['category'] not in category2categoryID:
        categoryNum += 1
        category2categoryID[row['category']] = categoryNum
    if row['userID'] not in user2userID:
        userNum += 1
        user2userID[row['userID']] = userNum

print('#session: ', sessionNum)
print('&item: ', itemNum)
print('#category: ', categoryNum)
print('#user: ', userNum)


def reSession(reviewerID):
    if reviewerID in reviewerID2sessionID:
        return reviewerID2sessionID[reviewerID]
    else:
        print('session is not recorded')
        return 'none'


def reItem(asin):
    if asin in asin2itemID:
        return asin2itemID[asin]
    else:
        print('item is not recorded')
        return 'none'


def reCate(category):
    if category in category2categoryID:
        return category2categoryID[category]
    else:
        print('category is not recorded')
        return 'none'


def reUser(user):
    if user in user2userID:
        return user2userID[user]
    else:
        print('user is not recorded')
        return 'none'


def priceInt(price):
    return int(price)


data_all['sessionID'] = data_all.sessionID.map(reSession)
data_all['itemID'] = data_all.itemID.map(reItem)
data_all['priceLevel'] = data_all.priceLevel.map(priceInt)
data_all['category'] = data_all.category.map(reCate)
data_all['userID'] = data_all.userID.map(reUser)
print(data_all)

data = data_all[['sessionID', 'userID', 'itemID', 'priceLevel', 'category']]

item_inter_num = pd.DataFrame(data.groupby(data['itemID']).count())
item_inter_num = item_inter_num.reset_index()[['sessionID', 'itemID']]
item_num = item_inter_num.rename(columns={'sessionID': 'item_num'})
data = pd.merge(data, item_num, how='left', on='itemID')
data = data[data['item_num'] > 9]
data = data[['sessionID', 'userID', 'itemID', 'priceLevel', 'category']]

# dict (sessionID:[itemID,itemID])
sess_all = {}
# dict (sessionID:[priceLevel, priceLevel])
price_all = {}
# dict (sessionID:[cate, cate])
cate_all = {}
# dict (sessionID:[userID, userID])
user_all = {}
for _, row in data.iterrows():
    sess_id = row['sessionID']
    user_id = row['userID']
    item_id = row['itemID']
    price = row['priceLevel']
    cate = row['category']
    if sess_id in sess_all:
        sess_all[sess_id].append(item_id)
        user_all[sess_id].append(user_id)
        price_all[sess_id].append(price)
        cate_all[sess_id].append(cate)
    else:
        sess_all[sess_id] = []
        sess_all[sess_id].append(item_id)
        user_all[sess_id] = []
        user_all[sess_id].append(user_id)
        price_all[sess_id] = []
        price_all[sess_id].append(price)
        cate_all[sess_id] = []
        cate_all[sess_id].append(cate)

sess_total = data['sessionID'].max()

split_num = int(sess_total / 10 * 9)

tra_sess = dict()  # dict(session_id:[item_id,item_id])
tes_sess = dict()
tra_user = dict()  # # dict(session_id:[user_id,user_id])
tes_user = dict()
tra_price = dict()  # dict(session_id:[price,price])
tes_price = dict()
tra_cate = dict()  # dict(session_id:[cate,cate])
tes_cate = dict()
for sess_temp in sess_all.keys():
    all_seqs = sess_all[sess_temp]
    all_users = user_all[sess_temp]
    all_price = price_all[sess_temp]
    all_cate = cate_all[sess_temp]
    if len(all_seqs) < 2:
        continue
    if len(all_seqs) > 20:
        all_seqs = all_seqs[:20]
        all_users = all_users[:20]
        all_price = all_price[:20]
        all_cate = all_cate[:20]

    if int(sess_temp) < split_num:
        tra_sess[sess_temp] = all_seqs
        tra_user[sess_temp] = all_users
        tra_price[sess_temp] = all_price
        tra_cate[sess_temp] = all_cate
    else:
        tes_sess[sess_temp] = all_seqs
        tes_user[sess_temp] = all_users
        tes_price[sess_temp] = all_price
        tes_cate[sess_temp] = all_cate

item_dict = {}  # dict(old_itemID: new_itemID)
user_dict = {}  # dict(old_userID: new_userID)
cate_dict = {}  # dict(old_cate: new_cate)
price_dict = {}  # dict(old_price: new_price)

user_item = {}  # dict[new_userID: new_itemID]
item_price = {}  # dict[new_itemID: priceLevel]
item_cate = {}  # dict[new_itemID: cate]


# tra_sess tra_user tra_price tra_cate
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_seqs = []
    train_users = []
    train_price = []
    train_cate = []
    item_ctr = 1
    user_ctr = 1
    price_ctr = 1
    cate_ctr = 1
    for s in tra_sess:
        seq = tra_sess[s]
        user_seq = tra_user[s]
        price_seq = tra_price[s]
        cate_seq = tra_cate[s]
        outseq = []
        pri_outseq = []
        user_outseq = []  # 用于存储每个session中的用户ID
        cate_outseq = []
        for i, u, p, c in zip(seq, user_seq, price_seq, cate_seq):
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
            # 处理user_id
            if u in user_dict:
                user_outseq += [user_dict[u]]
            else:
                user_outseq += [user_ctr]
                user_dict[u] = user_ctr
                user_ctr += 1
            if p in price_dict:
                pri_outseq += [price_dict[p]]
            else:
                pri_outseq += [price_ctr]
                price_dict[p] = price_ctr
                price_ctr += 1
            if c in cate_dict:
                cate_outseq += [cate_dict[c]]
            else:
                cate_outseq += [cate_ctr]
                cate_dict[c] = cate_ctr
                cate_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            print('session length is 1')
            continue
        train_seqs += [outseq]
        train_users += [user_outseq]  # 将用户信息添加到train_users中
        train_price += [pri_outseq]
        train_cate += [cate_outseq]
    print("#train_session", len(train_seqs))
    print("#train_users", user_ctr - 1)  # 输出用户总数
    print("#train_items", item_ctr - 1)
    print("#train_price", price_ctr - 1)
    print("#train_category", cate_ctr - 1)
    return train_seqs, train_users, train_price, train_cate


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_seqs = []
    test_users = []
    test_price = []
    test_cate = []
    user_ctr = len(user_dict) + 1

    for s in tes_sess:
        outseq = []
        out_users = []
        out_price = []
        out_cate = []
        for i, m, j, k in zip(tes_sess[s], tes_user[s], tes_price[s], tes_cate[s]):
            if i in item_dict:
                outseq += [item_dict[i]]
                # 检查 m 是否在 user_dict 中
                if m not in user_dict:
                    # 如果 m 不在 user_dict 中，则添加进去，值为 user_ctr
                    user_dict[m] = user_ctr
                    # 然后增加 user_ctr
                    user_ctr += 1
                out_price += [price_dict[j]]
                out_cate += [cate_dict[k]]
        if len(outseq) < 2:
            print('obtain test session length is 1')
            continue
        test_seqs += [outseq]
        test_users += [out_users]
        test_price += [out_price]
        test_cate += [out_cate]
    return test_seqs, test_users, test_price, test_cate


# 函数的作用是从每个历史序列中提取输入特征（去除最后一个元素的序列）和目标标签（序列的最后一个元素，可能是用户下一步的行为、下一个商品等）。
def process_seqs_no(iseqs, iuser, iprice, icate):
    print("no data augment")
    out_seqs = []
    out_users = []
    out_price = []
    out_cate = []
    labs = []  # 每个序列的最后一个元素
    max_length = 19
    for seq, use, pri, cat in zip(iseqs, iuser, iprice, icate):
        labs += [seq[-1]]
        out_seqs += [seq[:-1]]
        out_users += [use[:-1]]
        out_price += [pri[:-1]]
        out_cate += [cat[:-1]]
    return out_seqs, out_users, out_price, out_cate, labs


tra_seqs, tra_use, tra_pri, tra_cat = obtian_tra()
tes_seqs, tes_use, tes_pri, tes_cat = obtian_tes()

tr_seqs, tr_use, tr_pri, tr_cat, tr_labs = process_seqs_no(tra_seqs, tra_use, tra_pri, tra_cat)
te_seqs, te_use, te_pri, te_cat, te_labs = process_seqs_no(tes_seqs, tes_use, tes_pri, tes_cat)

print('train sequence: ', tr_seqs[:5])
print('train users: ', tr_use[:5])
print('train price: ', tr_pri[:5])
print('train category: ', tr_cat[:5])
print('train lab: ', tr_labs[:5])


# construct all matrics whose shape is similar as session-items [[],[]]
def tomatrix(all_seqs, all_user, all_pri, all_cate):
    price_item_dict = {}
    price_item = []

    user_item_dict = {}
    user_item = []

    price_category_dict = {}
    price_category = []

    category_item_dict = {}
    category_item = []

    # price-item dict -> {price_id:[1, 3, 4]}

    for s_seq, u_seq, p_seq, c_seq in zip(all_seqs, all_user, all_pri, all_cate):
        for i_temp, u_temp, p_temp, c_temp in zip(s_seq, u_seq, p_seq, c_seq):
            if p_temp not in price_item_dict:
                # print('price_new: ',p_temp)
                price_item_dict[p_temp] = []
            if p_temp not in price_category_dict:
                price_category_dict[p_temp] = []
            if c_temp not in category_item_dict:
                # print('category_new: ',c_temp)
                category_item_dict[c_temp] = []
            # 记录用户对应商品的列表
            if u_temp not in user_item_dict:
                user_item_dict[u_temp] = []  # 初始化用户对应的商品列表

            price_item_dict[p_temp].append(i_temp)
            user_item_dict[u_temp].append(i_temp)  # 将商品添加到用户对应的商品列表中
            price_category_dict[p_temp].append(c_temp)
            category_item_dict[c_temp].append(i_temp)

    price_item_dict = dict(sorted(price_item_dict.items()))

    user_item_dict = dict(sorted(user_item_dict.items()))

    price_category_dict = dict(sorted(price_category_dict.items()))

    category_item_dict = dict(sorted(category_item_dict.items()))
    print("#price", len(price_item_dict))
    print("#user", len(user_item_dict))
    print("#category", len(category_item_dict))

    price_item = list(price_item_dict.values())
    user_item = list(user_item_dict.values())
    price_category = list(price_category_dict.values())
    category_item = list(category_item_dict.values())
    # print(price_item)
    # print('----------------------------------------------------')
    # print(user_item)
    return price_item, user_item, price_category, category_item


def data_masks(all_sessions):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])  # 获取去重后的会话项
        length = len(session)
        s = indptr[-1]  # 当前行的起始索引位置，将会作为每一行的开始索引
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i] - 1)
            data.append(1)
        results = (data, indices, indptr)
    return results


tra_pi, tra_ui, tra_pc, tra_ci = tomatrix(tra_seqs + tes_seqs, tra_use + tes_use, tra_pri + tes_pri, tra_cat + tes_cat)

tra = (
tr_seqs, tr_pri, data_masks(tr_seqs), data_masks(tr_pri), data_masks(tra_pi), data_masks(tra_ui), data_masks(tra_pc), data_masks(tra_ci),
tr_labs)
tes = (
te_seqs, te_pri, data_masks(te_seqs), data_masks(te_pri), data_masks(tra_pi), data_masks(tra_ui), data_masks(tra_pc), data_masks(tra_ci),
te_labs)
# print(len(tra[3]))
all = 0
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('#interactions: ', all)
print('#session: ', (len(tra_seqs) + len(tes_seqs)))
print('sequence average length: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0))

train_data_path = './electronics'

if not os.path.exists(train_data_path):
    os.makedirs(train_data_path)
path_data_train = train_data_path + "/train.txt"
path_data_test = train_data_path + "/test.txt"

pickle.dump(tra, open(path_data_train, 'wb'))
pickle.dump(tes, open(path_data_test, 'wb'))
print("dataset: ", datasets_name)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("done")
