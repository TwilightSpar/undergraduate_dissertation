import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

info_path = r"data/pedata1/"
exit_path = r"data/exit/"


def get_df(cwd):
    filenames = []
    columns = ['invest_comp', 'comp_name', 'intro']
    comp_and_intro = pd.DataFrame(columns=columns)
    get_dir = os.listdir(cwd)

    for i in get_dir:
        filenames.append(cwd + i)

    for file_name in filenames:
        df = pd.read_excel(file_name, header=3)
        df = df[['投资方', '企业名称', '企业简介']]
        df.columns = columns
        comp_and_intro = comp_and_intro.append(df)

    comp_and_intro.loc[:, "intro"] = comp_and_intro.loc[:, "intro"].astype(str).apply(
        lambda x: x.strip().replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', ''))
    # -------del_null
    comp_and_intro = comp_and_intro.drop_duplicates(subset=['comp_name'])
    comp_and_intro.reset_index(drop=True, inplace=True)

    print("total data: %d" % len(comp_and_intro))
    comp_and_intro = comp_and_intro.drop(
        comp_and_intro[comp_and_intro['intro'] == '--'].index)  # del rows that have empty intro('--')
    comp_and_intro.reset_index(drop=True, inplace=True)
    print("total data after del --: %d" % len(comp_and_intro))
    comp_and_intro['comp_name'] = comp_and_intro['comp_name'].str.strip()
    # ----------

    with open('data/comp_and_intro', 'wb') as f:
        pickle.dump(comp_and_intro, f)


def get_year(s):
    return 0 if len(s) < 0 else s[0:4]


def get_province(s):
    if "|" in s:
        return s.split("|")[1]
    else:
        return s


def loc_one_hot(s):
    class_num = 10
    label = np.array([[class_num - 1]])  # init as class_num-1, means 海外
    label = torch.LongTensor(label)
    loc_list = [["北京市"], ["上海市"], ["广东省"], ["黑龙江省", "吉林省", "辽宁省"], ["天津市", "河北省", "内蒙古", "山西省"],
                ["江苏省", "安徽省", "浙江省", "江西省", "福建省", "山东省", "海南省"],
                ["四川省", "湖南省", "湖北省", "广西", "河南省", "重庆市"], ["陕西省", "新疆", "云南省", "贵州省", "甘肃省", "宁夏", "青海省", "西藏"],
                ["台湾", "香港", "澳门", "--"]]
    for i in range(len(loc_list)):
        if s in loc_list[i]:
            label[0][0] = i
    y_one_hot = torch.zeros(1, class_num).scatter_(1, label, 1)
    return y_one_hot


def field_encodding(exit_df):
    corpus = exit_df["field"].drop_duplicates().values.tolist()
    # print(len(corpus))

    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # sentences = ["互联网信息服务", "再生橡胶制造", "环境治理"]
    corpus_embeddings = embedder.encode(corpus)
    # print(corpus_embeddings[0])
    print("finish encoding")

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform glomerative clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=2.5)
    # or: affinity='cosine', linkage='average', distance_threshold=0.4)

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i)
        print(cluster)
        print("")

    with open('data/field_cluster', 'wb') as file:
        pickle.dump(clustered_sentences, file)


def get_field_encoding(s):
    f = open('data/field_cluster', 'rb')
    clustered_sentences = pickle.load(f)

    class_num = len(clustered_sentences)
    label = np.array([[0]])  # init as class_num-1, means 海外
    label = torch.LongTensor(label)

    for j, cluster in clustered_sentences.items():
        if s in cluster:
            label[0][0] = j
            break
    y_one_hot = torch.zeros(1, class_num).scatter_(1, label, 1)
    return y_one_hot


def time_one_hot(s):
    class_num = 4
    label = np.array([[0]])  # init as class_num-1, means 海外
    label = torch.LongTensor(label)
    year = pd.to_numeric(s).astype("int")

    if year in range(1995, 2001):
        label[0][0] = 0
    if year in range(2001, 2010):
        label[0][0] = 1
    if year in range(2010, 2018):
        label[0][0] = 2
    if year in range(2018, 2020):
        label[0][0] = 3

    y_one_hot = torch.zeros(1, class_num).scatter_(1, label, 1)
    return y_one_hot


def get_exit(cwd):
    filenames = []
    columns = ['invest_comp', 'comp_name', 'exit_way', 'return', 'IRR', 'exit_time', 'field', 'loc']
    exit_all = pd.DataFrame(columns=columns)
    get_dir = os.listdir(cwd)

    for i in get_dir:
        filenames.append(cwd + i)

    for file_name in filenames:
        df = pd.read_excel(file_name)
        df = df[['退出方', '企业', '退出方式', '账面回报(倍数)', '内部收益率(%)', '退出时间', '行业(清科)', '地区']]
        df.columns = columns
        exit_all = exit_all.append(df)

    exit_all['comp_name'] = exit_all['comp_name'].str.strip()
    exit_all = exit_all.drop_duplicates(subset=['comp_name'])

    field_encodding(exit_all)  # prepare field encoding
    # encode loc, time, field, add success column
    exit_all['exit_time'] = exit_all['exit_time'].apply(lambda x: get_year(x))
    exit_all.loc[:, "success"] = exit_all.apply(lambda x: get_success(x['exit_way'], x['exit_time'], x['IRR']), axis=1)
    exit_all['exit_time'] = exit_all['exit_time'].apply(lambda x: time_one_hot(x))

    exit_all['loc'] = exit_all['loc'].apply(lambda x: get_province(x)).apply(lambda x: loc_one_hot(x))
    exit_all['field'] = exit_all['field'].apply(lambda x: get_field_encoding(x))

    with open('data/exit_all', 'wb') as file:
        pickle.dump(exit_all, file)

    exit_has_intro = exit_all[exit_all['comp_name'] != '不披露']
    exit_has_intro.reset_index(drop=True, inplace=True)
    print(exit_has_intro.head())
    print(len(exit_has_intro))

    with open('data/exit_has_intro', 'wb') as file:
        pickle.dump(exit_has_intro, file)


def get_success(exit_way, year, IRR):
    # [year][IPO, 股权转让, 并购]
    return_table = [[86.4, 16.3, 11.4], [112.3, 22.8, 19.5], [104.5, 22.9, 11.4], [66.4, 11.6, 11.6],
                    [60.5, 16.4, 14.7], [26.8, 20.8, 20.8], [14.5, 8.7, 16], [31.8, 23.4, 13.4], [17, 20, 17],
                    [16, 16, 32.1], [16, 16, 32.1]]
    if exit_way == "--" or year == "--":
        return 0
    year = pd.to_numeric(year).astype("int")
    if year < 2009:
        return 1 if exit_way in ["IPO", "借壳"] else 0
    exit_level = 0
    if exit_way in ["IPO", "借壳"]:
        exit_level = 0
    elif exit_way == "股权转让":
        exit_level = 1
    elif exit_way == "并购":
        exit_level = 2
    else:
        return 0
    if IRR == "--":
        return 2
    IRR = pd.to_numeric(IRR.replace(',', '').replace('<', ''))
    return 1 if IRR > return_table[year - 2009][exit_level] else 0


def merge_known():
    f = open('data/comp_and_intro', 'rb')
    info_df = pickle.load(f)

    f = open('data/exit_has_intro', 'rb')
    exit_df = pickle.load(f)

    result = pd.merge(info_df, exit_df, on="comp_name")
    result["intro"] = result.intro.fillna('')
    result = result.drop_duplicates(subset=['comp_name'])
    result.reset_index(drop=True, inplace=True)

    print(len(result[result['IRR'] == "--"]))
    result = result[['comp_name', 'intro', 'success', 'exit_time', 'field', 'loc']]
    # print(result.head())

    with open('data/learning_data_exit_has_intro', 'wb') as file:
        pickle.dump(result, file)

    print("intro has %d company, exit has %d company, merge has %d data" % (len(info_df), len(exit_df), len(result)))


def merge_all():
    f = open('data/comp_and_intro', 'rb')
    info_df = pickle.load(f)

    f = open('data/exit_all', 'rb')
    exit_df = pickle.load(f)

    result = pd.merge(exit_df, info_df, how='left', on="comp_name")
    result["intro"] = result.intro.fillna('')
    result = result.drop_duplicates(subset=['comp_name'])
    result.reset_index(drop=True, inplace=True)

    print(len(result[result['IRR'] == "--"]))
    result = result[['comp_name', 'intro', 'success', 'exit_time', 'field', 'loc']]
    # print(result.head())

    with open('data/learning_data_all', 'wb') as file:
        pickle.dump(result, file)

    print("intro has %d company, exit has %d company, merge has %d data" % (len(info_df), len(exit_df), len(result)))


def get_data():
    with open('data/learning_data_all', 'rb') as f:
        data = pickle.load(f)

        text_len = []
        for text in data["intro"]:
            text_len.append(len(text))
        max_len = max(text_len)  # 1890
        max_index = text_len.index(max_len)
        print("max len: %d, text: %s" % (max_len, data["intro"][max_index]))

        plt.hist(text_len, bins=15)
        plt.title("Histogram of the intro of company")
        plt.xlabel("length of paragraph")
        plt.show()


if __name__ == "__main__":
    # pd_display_rows = 10
    # pd_display_cols = 100
    # pd_display_width = 1000
    # pd.set_option('display.max_rows', pd_display_rows)
    # pd.set_option('display.min_rows', pd_display_rows)
    # pd.set_option('display.max_columns', pd_display_cols)
    # pd.set_option('display.width', pd_display_width)
    # pd.set_option('display.max_colwidth', pd_display_width)
    # pd.set_option('display.unicode.ambiguous_as_wide', True)
    # pd.set_option('display.unicode.east_asian_width', True)
    # pd.set_option('expand_frame_repr', False)

    # pd.set_option('display.max_rows', None)
    # get_df(info_path)
    get_exit(exit_path)
    # merge_known()
    # merge_all()
    # get_data()

    # print(time_one_hot("2010"))
    # print(loc_one_hot("江西省"))

    # with open('data/learning_data_all', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data.loc[:, 'success'].value_counts())
    #     # 0        1577
    #     # 1        1407
    #     # 2        1087

    # with open('data/exit_known', 'rb') as f:
    #     data = pickle.load(f)
    #     c = data.loc[:, 'exit_time'].value_counts()
    #     plt.bar(np.arange(len(c)), c, tick_label = c.index)
    #     plt.title("exit year count")
    #     plt.xlabel("year")
    #     plt.xticks(rotation=90, fontsize=11)
    #     plt.show()
    #     print(type(c))

    # with open('data/exit_all', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data.loc[:, 'loc'].value_counts())

    #     l = data["success"]
    #     print(len(l))   # 4072
    #     print(sum(l))   # 2039

    import torch.nn.functional as F
    label = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
    pre = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
    l = F.binary_cross_entropy(pre, label, reduction='none')
    print(l)