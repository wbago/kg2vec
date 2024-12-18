import networkx as nx
import torch
from py2neo import Graph
from node2vec import Node2Vec
from torch_geometric.data import Data

torch.cuda.empty_cache()
# FILES
EMBEDDING_FILENAME = './embeddings.emb'  #嵌入文件，通常保存为 Word2Vec 格式
EMBEDDING_MODEL_FILENAME = './embeddings.model'  #保存训练后的 Node2Vec 模型
PT_FILE = './diG-KG2Vec.pt'  #保存 PyTorch Geometric 数据的 .pt 文件

# 连接neo4j 替换成自己neo4j的用户名和密码
graph = Graph("bolt://localhost:7687", auth=("neo4j", "245824"))

# 从 Neo4j 中查询节点和关系
# Assuming you have nodes with 'id' and 'name' properties, and relationships with 'type'
query = """
MATCH (n)-[r]->(m)
RETURN ID(n) AS source, ID(m) AS target, type(r) AS relationship_type
"""
data_list = list(graph.run(query))
print(data_list)  # 打印返回的所有记录

# 根据知识图谱构建networkXp[有向]图对象
nx_graph = nx.DiGraph()     #用于生成嵌入的图结构
print(f"Number of nodes: {nx_graph.number_of_nodes()}")
print(f"Number of edges: {nx_graph.number_of_edges()}")

for record in data_list:
    source = record["source"]
    target = record["target"]
    relationship_type = record["relationship_type"]
    print(f"Adding edge: {source} ---type: {relationship_type}---> {target} ")
    nx_graph.add_edge(source, target, relationship_type=relationship_type)  #添加有边属性的边
print(f"Number of nodes: {nx_graph.number_of_nodes()}")
print(f"Number of edges: {nx_graph.number_of_edges()}")

# 是一个图嵌入算法，基于图的随机游走（random walks）生成节点的低维嵌入
node2vec = Node2Vec(nx_graph, dimensions=128, walk_length=30, num_walks=200, workers=4) #图结构，嵌入的维度，随机游走的步长，节点生成的随机游走数量（一个节点会生成多少次游走），四个并行线程
print(f"Number of walks: {len(node2vec.walks)}")
# 训练Node2Vec模型
model = node2vec.fit(window=10, min_count=1, batch_words=4)     #训练Node2Vec模型      #window滑动窗口大小，决定每次随即游走时能不能捕捉到远程节点，min_count忽略出现次数小于min_count的点，batch每次训练时输入的单词数

# 保存嵌入和模型
model.wv.save_word2vec_format(EMBEDDING_FILENAME)
model.save(EMBEDDING_MODEL_FILENAME)

# Convert embeddings to a Data list format
embeddings = model.wv.vectors  # NumPy array of embeddings 返回训练好的嵌入，形状为 (num_nodes, dimensions)，即每个节点有一个由 dimensions 个数值组成的嵌入向量（在这里是 64 维的向量）
#使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 创建节点特征
node_features = torch.tensor(embeddings).to(device)  #将嵌入转换为 PyTorch 张量
# 将图转换为PyTorch Geometric的格式
edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous().to(device)
# 获取边的属性
print(f"Node features shape: {node_features.shape}")
print(f"Edge index shape: {edge_index.shape}")
# 创建多个 Data 对象
data_list = []
count = 0

for i in range(100):  # Assuming we want to generate 5 `Data` objects for illustration
    data = Data(edge_index=edge_index, y=torch.tensor([1]), node_features=node_features)
    data = data.to(device)  # 将整个Data对象移动到GPU
    print(count)
    count = count + 1
    data_list.append(data)

# Save the list of Data objects as a .pt file
torch.save(data_list, PT_FILE)
print(f"图结构向量化结果为： {PT_FILE}")

# Load and print the Data list
loaded_data_list = torch.load(PT_FILE)
for data in loaded_data_list:
    print(data)
