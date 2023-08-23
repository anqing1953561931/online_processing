import heapq
import os
import pickle
import networkx as nx
import pandas as pd
import time
from tqdm import tqdm

def read_attributes(df):
    header_list = df.columns.tolist()
    return header_list
def read_sencond_column(csv_path):
    # 读取CSV文件，假设第一列是主题列
    df = pd.read_csv(csv_path)
    # 读取第一列数据并存储在一个Series对象中
    second_column_list = df.iloc[:, 1].tolist()
    second_column_list = list(set(second_column_list))
    return second_column_list
def table_label_value(df, n, m):
    label_value = {}
    count = 0
    for row in df.itertuples(index=False):
        count += 1
        key = getattr(row, df.columns[1])
        value = 1.0 / getattr(row, df.columns[2])
        # value = 1.0 / row[2]  # 第三列作为值
        if key in label_value:
            label_value[key] += value
        else:
            label_value[key] = value
    for key, value in label_value.items():
        label_value[key] = (value ** int(n)) / (count ** int(m))
    return label_value
#一对向量的点乘结果
def count_SEP(label_value_set1, label_value_set2):
    common_keys = set(label_value_set1.keys()).intersection(label_value_set2.keys())
    # 对共有的键对应的值进行点乘
    dot_product = sum(label_value_set1[key] * label_value_set2[key] for key in common_keys)
        # print("点乘结果:", dot_product)
    return dot_product


# 计算Jaccard相似度的函数
def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def Max_binaryGraph(list1, list2):
    # 创建一个无向图
    G = nx.Graph()

    # 将列表1中的元素添加到图的一个集合中
    G.add_nodes_from(list1, bipartite=0)

    # 将列表2中的元素添加到图的另一个集合中
    G.add_nodes_from(list2, bipartite=1)

    # 计算并添加边的权重（使用Jaccard相似度）
    for elem1 in list1:
        for elem2 in list2:
            weight = jaccard_similarity(elem1, elem2)
            G.add_edge(elem1, elem2, weight=weight)

    # 最大权重匹配
    matching = nx.algorithms.max_weight_matching(G, maxcardinality=True)

    # 输出一对一映射和相似度值
    matched_pairs = {}

    #W是一对一映射的权重和
    W = 0.0
    for elem1, elem2 in matching:
        similarity = G[elem1][elem2]['weight']
        matched_pairs[(elem1, elem2)] = similarity
        W += similarity
    #存储一对一映射的个数，以及list1,list2的元素个数

    n1 = len(list1)
    n2 = len(list2)
    N = len(matched_pairs)
    # print(W)
    return W / (n1 + n2 - N)

def cal_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2  # 交集运算符
    return intersection

def all_entity_score(query_label_df, query_second_column, fnames, cand_value_labels, second_column, n, m
                     , all_candidate_attribute, query_path_df):
    count = 0
    result = {}
    query_label_value_set = table_label_value(query_label_df, n, m)
    query_att = read_attributes(query_path_df)
    for file_name in tqdm(fnames):
        second = second_column[count]
        inter = cal_intersection(second, query_second_column)

        SEP = count_SEP(query_label_value_set, cand_value_labels[count])
        Sss = Max_binaryGraph(query_att, all_candidate_attribute[count])
        result[file_name] = SEP * Sss
        count += 1
    # print(len(result))
    # for key,value in result.items():
    #     if value != 0:
    #         print(value)
    return result

def find_largest_five(my_dict):
    files_result = []
    largest_items = heapq.nlargest(5, my_dict.items(), key=lambda item: item[1])
    for key, value in largest_items:
        files_result.append(key)
    return files_result


s = time.time()

files_names= '../offline_processing_split1/file_names.pkl'
fnames = []
with open(files_names, 'rb') as f:
    fnames = pickle.load(f)

query_label_path = r'tar1_label.csv'
query_label_df = pd.read_csv(query_label_path)

query_path = r'tar1.csv'
query_path_df = pd.read_csv(query_path)

can_value_label_path = '../offline_processing_split1/candidate_label_value.pkl'
cand_value_labels = []
with open(can_value_label_path, 'rb') as f:
    cand_value_labels = pickle.load(f)

entity_label = '../offline_processing_split1/entity_label.pkl'
second_column = []
with open(entity_label, 'rb') as f:
    second_column = pickle.load(f)


candidate_attributes = '../offline_processing_split1/candidate_attributes_list.pkl'
all_candidate_attribute = []
with open(candidate_attributes, 'rb') as f:
    all_candidate_attribute = pickle.load(f)


query_second_column = read_sencond_column(query_label_path)

start_time = time.time()
result = all_entity_score(query_label_df, query_second_column, fnames, cand_value_labels, second_column, 2, 2
                     , all_candidate_attribute, query_path_df)
end_time = time.time()
print(end_time - start_time)
# with open(file='candidate_attributes_list.pkl', mode='wb') as f:
#     pickle.dump(all_att, f)

file_result = find_largest_five(result)
print(file_result)

e = time.time()
print(e-s)
# print(query_second_column)
# all_entity_score(df, 0,0,0,2,2,)