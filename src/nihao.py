import random

#txt生成三个数据集
def split_data(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将 txt 文件中的数据按行分成训练集、验证集和测试集。

    参数：
    - input_file: 输入的txt文件路径
    - train_ratio: 训练集的比例
    - val_ratio: 验证集的比例
    - test_ratio: 测试集的比例

    返回：
    - None: 直接保存到文件
    """
    # 确保比例之和为 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须等于 1"

    # 读取文件中的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # 打乱数据
    random.shuffle(data)

    # 按比例划分数据
    total_num = len(data)
    train_end = int(total_num * train_ratio)
    val_end = train_end + int(total_num * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 保存到不同的文件
    with open('train_data.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    print("训练集已保存为 train_data.txt")

    with open('val_data.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    print("验证集已保存为 val_data.txt")

    with open('test_data.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    print("测试集已保存为 test_data.txt")


# 调用函数
split_data('2-2.txt')  # 这里 'input_data.txt' 是你的输入txt文件路径
