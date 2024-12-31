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


loaded_data = torch.load('../predata/2-2v3/test_data.pt')
print(loaded_data)
# # 查看数据结构
# print(loaded_data)
# # print("Node features shape:", loaded_data.node_features.shape)
# print("Edge index shape:", loaded_data.edge_index.shape)

# print(loaded_data[1].edge_index.shape)
# print(loaded_data[1])
# print(loaded_data[1].node_features.shape)
#
# import torch
#
# # 加载数据
# loaded_data = torch.load('../predata/2-2v3/test_data.pt')
#
# 遍历数据
for i, data in enumerate(loaded_data):
    if i < 50:
        print(f"Data {i}:")
        print(f"Edge Index:\n{data.edge_index}")

        # 尝试通过 'x' 访问节点特征
        if hasattr(data, 'x'):
            print(f"Node Features (x):\n{data.x}")
        elif hasattr(data, 'node_features'):
            print(f"Node Features (node_features):\n{data.node_features}")
        else:
            print("未找到节点特征属性。")

        print('-' * 50)
    else:
        break  # 使用 break 退出循环

