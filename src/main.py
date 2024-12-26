import argparse
import os
import random
import time

import numpy as np
import networkx as nx
from py2neo import Graph
import node2vec
from gensim.models import Word2Vec
import torch
from torch_geometric.data import Data


def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='bolt://localhost:7687', help='输入neo4j网址')

    parser.add_argument('--username', nargs='?', default='neo4j', help='输入neo4j用户名')

    parser.add_argument('--password', nargs='?', default='245824', help='输入neo4j密码')

    parser.add_argument('--output_word2vec', nargs='?', default='output_word2vec.emb',
                        help='输出嵌入（embeddings）文件名')

    parser.add_argument('--output_model', nargs='?', default='output_model.model', help='输出模型名')

    """
    微调
    """
    parser.add_argument('--dimensions', type=int, default=128, help='嵌入的维度，默认为 128.')

    parser.add_argument('--walk-length', type=int, default=80, help='每次随机游走的长度，默认为 80.')

    parser.add_argument('--num-walks', type=int, default=10, help='每个节点的随机游走次数，默认为 10.')

    parser.add_argument('--window-size', type=int, default=10, help='Skip gram 模型的窗口大小，默认为 10.')

    parser.add_argument('--iter', default=1, type=int, help='随机梯度下降 (SGD) 迭代次数，默认为 1')

    parser.add_argument('--workers', type=int, default=8, help='并行工作线程数，默认为 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1./Node2Vec 中的超参数，用于控制游走策略，默认为 1')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1/Node2Vec 中的超参数，用于控制游走策略，默认为 1.')
    """
    微调
    """

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='是否使用加权边，默认不加权.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false', help='是否使用加权边，默认不加权.')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='图是否为有向图，默认为无向图.')
    parser.add_argument('--undirected', dest='undirected', action='store_false', help='图是否为有向图，默认为无向图.')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    """
    读取图的数据并返回一个图对象 G
    """
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())  #创建有向图
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:  #如果是无向，转换为无向图
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    """
    训练 Node2Vec 算法并生成节点的嵌入，通过优化 Skip gram 模型（Word2Vec 的一种变体）来学习节点的表示
    """
    walks = [list(map(str, walk)) for walk in walks]  # ensure walks are lists, not generators
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                     workers=args.workers, epochs=args.iter)

    # 获取所有节点的嵌入
    vocab = list(model.wv.index_to_key)  # 所有节点的索引
    embeddings = np.array([model.wv[node] for node in vocab])  # 将嵌入转为 NumPy 数组
    # Save embeddings in Word2Vec format
    model.wv.save_word2vec_format('../emb/' + args.output_word2vec)
    # Save Node2Vec model
    model.save('../emb/' + args.output_model)  # 存储训练好的Node2Vec模型
    return embeddings


def construct_neo4j():  #通过neo4j直接生成图
    graph = Graph(args.input, auth=(args.username, args.password))
    query = """
    MATCH (n)-[r]->(m)
    RETURN ID(n) AS source, ID(m) AS target
    """
    data_list = list(graph.run(query))
    # print(data_list)  # 打印返回的所有记录

    if args.directed:
        print("图结构为有向图")
        nx_graph = nx.DiGraph()
    else:
        print("图结构为无向图")
        nx_graph = nx.Graph()
        for record in data_list:
            source = record["source"]
            target = record["target"]
            # print(f"Adding edge: {source} -> {target}")
            nx_graph.add_edge(source, target)  #根据节点生成边
            # 添加权重属性，默认为1
            nx_graph[source][target]['weight'] = 1  # 如果你的边没有明确的权重，给每条边赋予默认的权重1
        if nx_graph.number_of_nodes() == 0 or nx_graph.number_of_edges() == 0:
            raise ValueError("从 Neo4j 中读取的图为空，请检查数据库或查询语句！")
    return nx_graph


#图中随机边采样，生成不同edge_index，对node_features添加扰动（随机噪声）
def generate_random_data(nx_graph, embeddings, num_samples=100):
    """
    生成包含随机子图的 Data 对象列表。

    参数：
    - nx_graph: NetworkX 图对象（原始图结构）。
    - embeddings: 节点的嵌入（Node2Vec 生成的向量）。
    - num_samples: 要生成的 Data 对象总数。

    返回：
    - data_list: 包含多个随机 Data 对象的列表。
    """
    data_list = []
    for _ in range(num_samples):
        # 随机选择一些边，生成子图
        edges = list(nx_graph.edges)
        sampled_edges = random.sample(edges, int(len(edges) * 0.8))  # 采样 80% 的边
        edge_index = torch.tensor(sampled_edges, dtype=torch.long).t().contiguous()

        # 对节点特征添加随机扰动
        node_features = torch.tensor(embeddings, dtype=torch.float)
        noise = torch.randn_like(node_features) * 0.01  # 添加 0.01 的噪声
        perturbed_node_features = node_features + noise

        # # 随机生成标签
        # y = torch.tensor([random.randint(0, 1)])  # 0 或 1 的标签

        # 构建 Data 对象
        data = Data(edge_index=edge_index, node_features=perturbed_node_features, y=torch.tensor([1]))
        data_list.append(data)

    return data_list


def split_and_save_data(data_list, train_ratio, val_ratio, test_ratio):
    """
    按比例划分数据集并保存为 .pt 文件。

    参数：
    - data_list: 包含所有 Data 对象的列表。
    - train_ratio: 训练集的比例。
    - val_ratio: 验证集的比例。
    - test_ratio: 测试集的比例。
    - output_dir: 输出目录路径。
    """
    # 确保比例之和为 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须等于 1"

    # 打乱数据
    random.shuffle(data_list)

    # 划分数据集
    total_num = len(data_list)
    train_end = int(total_num * train_ratio)
    val_end = train_end + int(total_num * val_ratio)

    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    # 保存数据
    torch.save(train_data, "../predata/" + "train_data.pt")
    print("已生成训练集！")
    torch.save(val_data, "../predata/" + "val_data.pt")
    print("已生成验证集！")
    torch.save(test_data, "../predata/" + "test_data.pt")
    print("已生成测试集！")


def main(args):
    # 1. 构建 NetworkX 图
    nx_G = construct_neo4j()
    start_time = time.time()
    print(f"NetworkX 图构建完成，用时 {time.time() - start_time:.2f} 秒。")
    print(f"NetworkX 图节点数: {nx_G.number_of_nodes()}, 边数: {nx_G.number_of_edges()}")
    # nx_G = read_graph()  #通过调用 read_graph() 来读取图数据并构建图

    # 2. 使用 Node2Vec 生成嵌入
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)  #通过node2vec来初始化图
    G.preprocess_transition_probs()  #预处理转移概率，这些概率会影响随机游走的行为
    walks = G.simulate_walks(args.num_walks, args.walk_length)  #生成随机游走路径
    embedding = learn_embeddings(walks)

    # 3. 使用 NetworkX 图和嵌入生成 Data 对象
    data_list = generate_random_data(nx_G, embedding)

    # 4. 按比例划分数据集并保存
    # 按比例划分数据集 (70% 训练, 15% 验证, 15% 测试)
    split_and_save_data(data_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # print(args.directed)
    # print(args.undirected)
