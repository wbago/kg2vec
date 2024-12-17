import networkx as nx
import torch
from py2neo import Graph
from node2vec import Node2Vec

# FILES
EMBEDDING_FILENAME = './embeddings.emb'  #嵌入文件，通常保存为 Word2Vec 格式
EMBEDDING_MODEL_FILENAME = './embeddings.model'  #保存训练后的 Node2Vec 模型
PT_FILE = './embeddings.pt'  #保存 PyTorch Geometric 数据的 .pt 文件

# 连接neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "245824"))

# 从 Neo4j 中查询节点和关系
# Assuming you have nodes with 'id' and 'name' properties, and relationships with 'type'
query = """
MATCH (n)-[r]->(m)
RETURN ID(n) AS source, ID(m) AS target
"""
data = graph.run(query)

for record in data:
    print(record)


# Create a NetworkX graph from Neo4j data
nx_graph = nx.Graph()

for record in data:
    source = record["source"]
    target = record["target"]
    nx_graph.add_edge(source, target)

print(f"Number of nodes: {nx_graph.number_of_nodes()}")
print(f"Number of edges: {nx_graph.number_of_edges()}")
# Precompute probabilities and generate walks
node2vec = Node2Vec(nx_graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Fit the Node2Vec model
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save embeddings in Word2Vec format
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save Node2Vec model
model.save(EMBEDDING_MODEL_FILENAME)

# Convert embeddings to PyTorch tensor and save as .pt
# embeddings = model.wv.vectors  # NumPy array of embeddings
# tensor_embeddings = torch.tensor(embeddings)
#
# # Save the tensor embeddings to .pt file
# torch.save(tensor_embeddings, PT_FILE)
#
# print(f"Embeddings saved to {PT_FILE}")

#**************************
# 将嵌入转换为Python list
embeddings_list = model.wv.vectors.tolist()  # 将嵌入转换为list

# 将list保存为.pt文件
torch.save(embeddings_list, PT_FILE)
#**************************
print(f"Embeddings saved to {PT_FILE}")



watch = torch.load("embeddings.pt")

print(watch)
