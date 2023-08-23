import heapq
import pickle
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm


#读取表格的第一列，假设第一列是主题列
def read_first_column(df):
    # 读取CSV文件，假设第一列是主题列
    # 读取第一列数据并存储在一个Series对象中
    first_column_list = df.iloc[:, 0].tolist()
    return first_column_list

#读取表格的第二列，假设第一列是主题列
def read_sencond_column(df):
    # 读取CSV文件，假设第一列是主题列
    # 读取第一列数据并存储在一个Series对象中
    first_column_list = df.iloc[:, 1].tolist()
    return first_column_list

#计算一对表格的SECover得分
def one_SECover(list1, list2):  #计算一对表的SECover得分
    # 先将列表的重复值去除，转换为集合
    set1 = set(list1)
    set2 = set(list2)

    #计算set1，set2的元素个数
    len1 = len(set1)

    # 求两个集合的交集，即相同的元素
    intersection_set = set1.intersection(set2)

    # 计算元素相同的个数
    SECover = len(intersection_set) / len1
    # print(SECover)
    return SECover

def cal_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2  # 交集运算符
    return intersection
# 计算Jaccard相似度的函数
def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# 计算一个候选表的额外属性
def add_attributes(query_att_list, candidate_att_list):
    # 创建一个无向图
    G = nx.Graph()

    # 将列表1中的元素添加到图的一个集合中
    G.add_nodes_from(query_att_list, bipartite=0)

    # 将列表2中的元素添加到图的另一个集合中
    G.add_nodes_from(candidate_att_list, bipartite=1)

    # 计算并添加边的权重（使用Jaccard相似度）
    for elem1 in query_att_list:
        for elem2 in candidate_att_list:
            weight = jaccard_similarity(elem1, elem2)
            G.add_edge(elem1, elem2, weight=weight)

    # 最大权重匹配
    matching = nx.algorithms.max_weight_matching(G, maxcardinality=True)

    # 输出匹配结果，同时存储相似度值
    matched_pairs = {}
    match_att = []
    for elem1, elem2 in matching:
        similarity = G[elem1][elem2]['weight']
        matched_pairs[(elem1, elem2)] = similarity
        match_att.append(elem2)
    #存储一对一映射的个数，以及list1,list2的元素个数
    # print("映射为",matched_pairs)
    add_attr = [x for x in candidate_att_list if x not in match_att]
    return add_attr

#读取表格第一行,即读取表头
def read_att(mytable_path):
    # 读取CSV文件，假设第一行是表头
    df = pd.read_csv(mytable_path)

    # 获取表头，并存放在一个列表中
    headers_list = df.columns.tolist()
    return headers_list


def one_SSB(query_att, one_table_add_att, one_att_freq, two_att_freq):
    end_dict = {}
    length = len(query_att)
    for add in one_table_add_att:
        count = 0.0
        for query in query_att:
            mid = []
            mid.append(add)
            mid.append(query)
            t1 = tuple(mid)
            if t1 in two_att_freq:
                mid1 = two_att_freq[t1]
                # print(mid1)
            else:
                mid1 = 0

            if query in one_att_freq:
                # print(query)
                mid2 = one_att_freq[query]

            else:
                mid2 = 0
            # print(mid1, mid2)
            if mid2 == 0:
                mid_result = 0
            else:
                mid_result = mid1 / mid2
            count += mid_result
        end_dict[add] = count / length
    if end_dict:
        max_item = max(end_dict.items(), key=lambda item: item[1])
        result = max_item[1]
    else:
        result = 0
    return result

def find_largest_five(my_dict):
    files_result = []
    largest_items = heapq.nlargest(5, my_dict.items(), key=lambda item: item[1])
    for key, value in largest_items:
        files_result.append(key)
    print(files_result)
    return files_result
def all_schema(fnames, can_label_second_column, query_label_second_column, can_entity_list, query_entity,
               query_attribute, can_all_att, one_att_freq, two_att_freq):
    count = 0
    result = {}
    for file_name in tqdm(fnames):
        second = can_label_second_column[count]
        inter = cal_intersection(second, query_label_second_column)

        if inter == 0:
            result[file_name] = 0.0
            count += 1
        else:
            SEcover = one_SECover(query_entity, can_entity_list[count])
            add = add_attributes(query_attribute, can_all_att[count])
            SSB = one_SSB(query_attribute, add, one_att_freq, two_att_freq)
            result[file_name] = SEcover * SSB
            count += 1
    # for key, value in result.items():
    #     if value != 0:
    #         print(value)
    find_largest_five(result)

s = time.time()


#获取文件名
fnames = []
files_names = r'../offline_processing_split1/file_names.pkl'
with open(files_names, 'rb') as f:
    fnames = pickle.load(f)

#获得查询表的实体标签
query_label_path = r'tar1_label.csv'
df1 = pd.read_csv(query_label_path)
query_label_second_column = read_sencond_column(df1)

#获取候选表的实体标签
#with open(file='entity_label.pkl',mode='wb') as f:
    # pickle.dump(second_column, f)
can_label_second_column = []
can_entity_label = r'../offline_processing_split1/entity_label.pkl'
with open(can_entity_label, 'rb') as f:
    can_label_second_column = pickle.load(f)

#获取查询表的第一列实体
query_csv_path = r'tar1.csv'
df2 = pd.read_csv(query_csv_path)
query_entity = read_first_column(df2)

#获取候选表所具有的实体
can_entity = r'../offline_processing_split1/can_entity_list.pkl'
can_entity_list = []
with open(can_entity, 'rb') as f:
    can_entity_list = pickle.load(f)

#获取查询表的属性，即第一行
query_attribute = read_att(query_csv_path)

#获取候选表的属性总列表
can_all_att = []
with open('../offline_processing_split1/candidate_attributes_list.pkl', 'rb') as f:
    can_all_att = pickle.load(f)

#获取属性出现的频数
one_att_freq = []
with open('../offline_processing_split1/one_att_freq.pkl', 'rb') as f:
    one_att_freq = pickle.load(f)

two_att_freq = []
with open('../offline_processing_split1/two_att_freq.pkl', 'rb') as f:
    two_att_freq = pickle.load(f)


all_schema(fnames, can_label_second_column, query_label_second_column, can_entity_list, query_entity,
               query_attribute, can_all_att, one_att_freq, two_att_freq)





e = time.time()
print(e-s)