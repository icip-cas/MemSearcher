import torch

def get_position_ids(attention_mask: torch.Tensor):
    # 创建一个和 attention_mask 同形状的 tensor，每一行为按行累加但仅在 mask==1 时递增
    cumsum = torch.cumsum(attention_mask, dim=1)
    # 把没有被 mask 的位置（即为 0 的位置）置为 0，其他位置减去1使得从0开始
    position_ids = (cumsum - 1) * attention_mask
    return position_ids

# 示例
attention_mask = torch.tensor([
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])
print(attention_mask)
position_ids = get_position_ids(attention_mask)
print(position_ids)
