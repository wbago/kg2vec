import torch
from sympy.codegen import Print

# watch = torch.load("../dataset/data/DDoS2w/GCN_BS2w_test_data.pt")
# count =1
# for i in watch:
#     print(i)
#     count = count+1
# print(count)
# print(watch[1])
# print(watch[2])
# print(watch[3])
# print(watch[4])
#
# watch1 = torch.load("../dataset/data/DDoS2019syn2w/GCN_BS2w_train_data.pt")
#
# print(watch1[1])
# print(watch1[2])
# print(watch1[3])
# print(watch1[4])


# loaded_data = torch.load('../dataset/data/DDoS2w/GCN_BS2w_test_data.pt')
# print(loaded_data)
# # 查看数据结构
# print(loaded_data)
# # print("Node features shape:", loaded_data.node_features.shape)
# print("Edge index shape:", loaded_data.edge_index.shape)

# print(loaded_data[1].edge_index.shape)
# print(loaded_data[1])
# print(loaded_data[1].node_features.shape)

loaded_data = torch.load('../predata/test_data.pt')

for i, data in enumerate(loaded_data):
    print(f"Data {i}:")
    print(f"Edge Index:\n{data.edge_index}")
    print(f"node_features:\n{data.node_features}")

    # 检查 x 是否为 None
    if data.x is None:
        print("Node features (x) are missing!")
    else:
        print(f"Node Features (first 5 rows):\n{data.x[:5]}")
    print('-' * 50)
