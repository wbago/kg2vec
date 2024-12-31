import argparse
import random

import numpy as np
import networkx as nx
from py2neo import Graph, Node, Relationship
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
    parser.add_argument('--dimensions', type=int, default=256, help='嵌入的维度，默认为 128.')

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
    # print(data_list)
    if args.directed:
        print("图结构为有向图")
        nx_graph = nx.DiGraph()
    else:
        print("图结构为无向图")
        nx_graph = nx.Graph()

        for record in data_list:
            # 直接使用查询结果中的节点 ID
            source = record["source"]  #获取源节点id,这里的源节点不是指源ip节点，而是流量的出口节点
            # print(source)
            target = record["target"]  #获取目标节点id，这里是流量的入口节点
            # print(target)
            # relation_type = record["relation_type"]     #获取关系类型
            #添加节点，附带属性如标签等
            nx_graph.add_node(source, labels=source, role="source")
            nx_graph.add_node(target, labels=target, role="target")
            nx_graph.add_edge(source, target)  # 根据节点生成边
            # 添加权重属性，默认为1
            nx_graph[source][target]['weight'] = 1  # 如果你的边没有明确的权重，给每条边赋予默认的权重1
            nx_graph[target][source]['role'] = "relation_ship"
        if nx_graph.number_of_nodes() == 0 or nx_graph.number_of_edges() == 0:
            raise ValueError("从 Neo4j 中读取的图为空，请检查数据库或查询语句！")
    return nx_graph


def map_node_ids_to_continuous_index(nx_G):
    """
    将原始节点 ID 映射为从 1 开始的连续整数，跳过 0
    """
    # 获取所有唯一节点 ID（排除0）
    node_ids = [node for node in nx_G.nodes if node != 0]  # 排除 0
    # 创建映射字典，确保从 1 开始
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_ids, start=1)}

    # 保留 0 不做映射
    node_mapping[0] = 0  # 如果 0 是特殊节点，保持它不变

    return node_mapping


def update_edge_index_with_mapping(edge_index, node_mapping):
    """
    根据节点 ID 映射更新 edge_index
    """
    new_edge_index = torch.tensor([
        [
            node_mapping.get(old_id.item(), old_id.item()) if old_id.item() != 0 else 0
            for old_id in edge_index[0]
        ],
        [
            node_mapping.get(old_id.item(), old_id.item()) if old_id.item() != 0 else 0
            for old_id in edge_index[1]
        ]
    ])

    return new_edge_index


def extract_subgraph_from_full_graph(embeddings, nx_G, num_edges_per_subgraph=4):
    """
    提取图的子图，确保每个源节点（source）和目标节点（target）分别生成 edge_index。
    """
    subgraphs = []
    node_list = nx_G.nodes
    node_mapping = map_node_ids_to_continuous_index(nx_G)  # 获取节点ID映射

    # 按照节点类型提取子图
    for node in node_list:
        # 获取当前节点的属性
        node_attributes = nx_G.nodes[node]

        # 初始化 edge_index_0 和 edge_index_1
        edge_index_0 = []
        edge_index_1 = []

        # 处理 source 节点
        if node_attributes.get("role") == "source":
            # 获取与源IP相连的邻居
            neighbors = list(nx_G.neighbors(node))
            # Ensure at least `num_edges_per_subgraph` neighbors
            selected_edges = random.sample(neighbors, min(num_edges_per_subgraph, len(neighbors)))

            # 如果选取的边少于所需数量，填充缺少的边
            edge_index_0 = [[node, neighbor] for neighbor in selected_edges]
            if len(edge_index_0) < num_edges_per_subgraph:
                # 用 0 填充到所需边数
                edge_index_0.extend([[node, 0]] * (num_edges_per_subgraph - len(edge_index_0)))

            edge_index_tensor_0 = torch.tensor(edge_index_0, dtype=torch.long).t().contiguous()
            new_edge_index_0 = update_edge_index_with_mapping(edge_index_tensor_0, node_mapping)
            # print(new_edge_index_0)

        # 处理 target 节点
        if node_attributes.get("role") == "target":
            neighbors = list(nx_G.neighbors(node))
            selected_edges = random.sample(neighbors, min(num_edges_per_subgraph, len(neighbors)))

            edge_index_1 = [[node, neighbor] for neighbor in selected_edges]
            if len(edge_index_1) < num_edges_per_subgraph:
                edge_index_1.extend([[node, 0]] * (num_edges_per_subgraph - len(edge_index_1)))

            edge_index_tensor_1 = torch.tensor(edge_index_1, dtype=torch.long).t().contiguous()
            new_edge_index_1 = update_edge_index_with_mapping(edge_index_tensor_1, node_mapping)
            # print(new_edge_index_1)

        # 如果 node 不是 source 也不是 target，则跳过
        if not edge_index_0 and not edge_index_1:
            print("存在非s非t节点")
            continue

        # 如果某个 edge_index 为空，使用空的张量
        if not edge_index_0:
            new_edge_index_0 = torch.empty((2, 0), dtype=torch.long)
        if not edge_index_1:
            new_edge_index_1 = torch.empty((2, 0), dtype=torch.long)

        # 合并 edge_index_0 和 edge_index_1
        if new_edge_index_0.numel() > 0 and new_edge_index_1.numel() > 0:
            combined_edge_index = torch.cat([new_edge_index_0, new_edge_index_1], dim=1)
        elif new_edge_index_0.numel() > 0:
            combined_edge_index = new_edge_index_0
        elif new_edge_index_1.numel() > 0:
            combined_edge_index = new_edge_index_1
        else:
            # 如果两个 edge_index 都为空，跳过
            continue

        subgraph_data = Data(
            edge_index=combined_edge_index,
            y=torch.tensor([1], dtype=torch.long),  # 标签
            node_features=torch.tensor(embeddings, dtype=torch.float)  # 节点特征
        )
        subgraphs.append(subgraph_data)

    return subgraphs


# def extract_subgraph_from_full_graph(embeddings, nx_G, num_edges_per_subgraph=4):
#     """
#     提取图的子图，确保每个源节点（source）和目标节点（target）分别生成 edge_index[0] 和 edge_index[1]。
#     """
#     subgraphs = []
#     node_list = nx_G.nodes
#     node_mapping = map_node_ids_to_continuous_index(nx_G)  # 获取节点ID映射
#
#     # 初始化 edge_index
#     edge_index_0 = []  # 存储源节点的边
#     edge_index_1 = []  # 存储目标节点的边
#
#     # 按照节点类型提取子图
#     for node in node_list:
#         # 获取当前节点的属性
#         node_attributes = nx_G.nodes[node]
#
#         # 如果当前节点是源节点
#         if node_attributes.get("role") == "source":
#             source_ip_node = node  # 当前源节点
#             # 获取源端口节点 -> 流量特征节点 -> 目标端口节点 -> 目标IP节点     :source_ip_node -> source_port_node -> traffic_feature_node -> target_port_node -> target_ip_node
#             source_port_node = list(nx_G.neighbors(source_ip_node))[0]  # 获取源端口节点
#             traffic_feature_node = list(nx_G.neighbors(source_port_node))[0]  # 获取流量特征节点
#             target_port_node = list(nx_G.neighbors(traffic_feature_node))[0]  # 获取目标端口节点
#             target_ip_node = list(nx_G.neighbors(target_port_node))[0]  # 获取目标IP节点
#
#             # 将生成的边加入到 edge_index_0 中
#             edge_index_0.append([source_ip_node, source_port_node])  # 源IP到源端口
#             edge_index_0.append([source_port_node, traffic_feature_node])  # 源端口到流量特征
#             edge_index_0.append([traffic_feature_node, target_port_node])  # 流量特征到目标端口
#             edge_index_0.append([target_port_node, target_ip_node])  # 目标端口到目标IP
#
#         # 如果当前节点是目标节点
#         elif node_attributes.get("role") == "target":
#             target_ip_node = node  # 当前目标节点
#             # 获取源IP节点 -> 源端口节点 -> 流量特征节点 -> 目标端口节点
#             target_port_node = list(nx_G.neighbors(target_ip_node))[0]  # 获取目标端口节点
#             traffic_feature_node = list(nx_G.neighbors(target_port_node))[0]  # 获取流量特征节点
#             target_port_node = list(nx_G.neighbors(traffic_feature_node))[0]  # 获取目标端口节点
#             target_ip_node = list(nx_G.neighbors(target_port_node))[0]  # 获取源IP节点
#
#             # 将生成的边加入到 edge_index_1 中
#             edge_index_1.append([target_ip_node, target_port_node])  # 目标IP到目标端口
#             edge_index_1.append([target_port_node, traffic_feature_node])  # 目标端口到流量特征
#             edge_index_1.append([traffic_feature_node, source_port_node])  # 流量特征到源端口
#             edge_index_1.append([source_port_node, source_ip_node])  # 源端口到源IP
#
#     # 如果选中的边少于所需数量，填充缺少的边
#     if len(edge_index_0) < num_edges_per_subgraph:
#         edge_index_0.extend([[0, 0]] * (num_edges_per_subgraph - len(edge_index_0)))  # 用 0 填充缺少的边
#     if len(edge_index_1) < num_edges_per_subgraph:
#         edge_index_1.extend([[0, 0]] * (num_edges_per_subgraph - len(edge_index_1)))  # 用 0 填充缺少的边
#
#     # 创建 edge_index，形状为 [2, num_edges_per_subgraph]，即 [2, 4]
#     edge_index_0_tensor = torch.tensor(edge_index_0).t().contiguous()  # 转置后形成 [2, num_edges_per_subgraph] 的形状
#     edge_index_1_tensor = torch.tensor(edge_index_1).t().contiguous()  # 转置后形成 [2, num_edges_per_subgraph] 的形状
#
#     # 更新 edge_index
#     new_edge_index_0 = update_edge_index_with_mapping(edge_index_0_tensor, node_mapping)
#     new_edge_index_1 = update_edge_index_with_mapping(edge_index_1_tensor, node_mapping)
#
#     # 创建子图数据对象
#     subgraph_data = Data(
#         edge_index=[new_edge_index_0, new_edge_index_1],  # 生成两个边集
#         y=torch.tensor([1]),  # 标签
#         node_features=torch.tensor(embeddings, dtype=torch.float)
#     )
#     subgraphs.append(subgraph_data)
#
#     return subgraphs


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
    # 检查该节点是否是源节点
    print(f"NetworkX 图节点数: {nx_G.number_of_nodes()}, 边数: {nx_G.number_of_edges()}")
    # nx_G = read_graph()  #通过调用 read_graph() 来读取图数据并构建图

    # 2. 使用 Node2Vec 生成嵌入
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)  #通过node2vec来初始化图
    G.preprocess_transition_probs()  #预处理转移概率，这些概率会影响随机游走的行为
    walks = G.simulate_walks(args.num_walks, args.walk_length)  #生成随机游走路径
    embedding = learn_embeddings(walks)

    # 3. 使用 NetworkX 图和嵌入生成 Data 对象
    subgraphs = extract_subgraph_from_full_graph(embedding, nx_G)
    # 4. 按比例划分数据集并保存
    # 按比例划分数据集 (70% 训练, 15% 验证, 15% 测试)
    split_and_save_data(subgraphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


if __name__ == "__main__":
    args = parse_args()
    main(args)
