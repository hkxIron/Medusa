import torch
from transformers import PreTrainedModel, PretrainedConfig


class KVCache:
    """
    A key-value cache for the model.
    注意：此处的KVCache的内存是预先分配好的，不会动态增长,初始化时由max_seq_length指定, 而vllm中的paged attention中， kv cache是动态增长的。
    好处是：可以避免动态增长带来的内存碎片化问题，提高内存使用效率, 无需二次分配内存，节省申请内存的时间。
    坏处是：如果max_seq_length设置的过大，可能会浪费内存。

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data:torch.Tensor, current_length:torch.Tensor):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (torch.Tensor): Initial length of the data.
        """
        # data: [num_hidden_layers, batch_size, head_num, max_seq_length, head_dim]
        self.data = data
        # current_length_data: scalar, 是一个标量，表示当前数据的有效长度，而不是返回的max_seq_length
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        # [layer_index, batch_size, cache_seq_length, hidden_size]
        return (
            self.data.shape[0], # layer_index
            self.data.shape[1], # batch_size,
            self.current_length.item(), # cache_seq_length: 当前数据的"有效长度", 而不是返回的max_seq_length=self.data.shape[3]。
            self.data.shape[3], # hidden_size
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        将 self.data 中指定维度 (dim) 的某些索引 (indices) 对应的数据，复制到同一维度的另一个位置（从 prev_length 开始）。

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """

        """
        # 假设 dim=0, indices=[0, 2], data.shape=[3, 4]
        # 结果会选取第0行和第2行的数据
        tgt = data.index_select(0, torch.tensor([0, 2]))
        """
        tgt = self.data.index_select(dim, indices) # 从data中选择指定的dim的指定索引的数据
        """
        作用：在 self.data 的 dim 维度上，从 prev_length 开始切分出长度为 tgt.shape[dim] 的子张量（作为粘贴的目标位置）。
        narrow 方法：
        语法：tensor.narrow(dim, start, length)
        返回一个共享存储的视图（不会复制数据）。

        # 假设 dim=1, prev_length=2, tgt.shape[dim]=2
        # 结果会切出第1维的第2~3列（长度为2）
        dst = data.narrow(1, 2, 2)
        """
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        """
        将 tgt 的数据复制到 dst 的内存位置。
        copy_ 方法： 原地操作（直接修改 dst 的内容）。
        non_blocking=True 表示异步复制（适用于GPU加速，避免阻塞主线程）。
        """
        dst.copy_(tgt, non_blocking=True)

        """
        作用：更新 current_length 为 prev_length + 复制的数据长度。
        意义：记录当前数据的有效长度（例如动态序列的长度）。
        """
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        # 将self.data返回一个视图切片dst
        dst = self.data.narrow(dim, start=self.current_length, length=tensor.shape[dim])
        # 将 tensor 的数据复制到 dst 的内存位置
        dst.copy_(tensor)
        # 更新 current_length 为当前长度加上 tensor 的长度
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(input=self.data, dim=2, start=0, length=self.current_length)


def initialize_past_key_values(model:PreTrainedModel):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers
    # 直接在gpu上分配固定内存，避免后续频繁申请内存
    # past_key_values_data: [num_hidden_layers * 2, batch_size, head_num, max_seq_len, head_dim]
    past_key_values_data: torch.Tensor = torch.zeros(
        config.num_hidden_layers * 2, # layer_num * 2, 因为key和value各一个
        batch_size, # batch_size
        config.num_key_value_heads, # head_num=8
        config.max_position_embeddings, # max_seq_len, 注意：此处直接按最大长度分配GPU内存, 2048
        config.hidden_size // config.num_attention_heads, # head_dim = hidden_size / num_attention_heads = 16
        device=model.device, # 指定在哪个设备上分配内存，一般为GPU
        dtype=model.dtype,
    )

    print(f"past_key_values_data shape: {past_key_values_data.shape}")
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    # current_legth_data: [num_hidden_layers * 2], 奇数和偶数分别对应key和value的长度, 这两个长度一般是相同的
    current_length_of_each_layer = torch.zeros( config.num_hidden_layers * 2, dtype=torch.long, device="cpu") # 每层的kv的有效长度需要存在CPU上，因为需要快速访问和更新

    # Creating a KVCache for each pair of key and value in all layers
    past_key_values_objects = [] * config.num_hidden_layers
    for layer_idx in range(config.num_hidden_layers):
        past_key_values_objects.append(
            [
                # j=0: key, j=1: value, 即偶数为key, 奇数为value
                # 注意：每个KVCache.data的shape为 [batch_size, head_num, max_seq_len, head_dim]
                KVCache(data=past_key_values_data[layer_idx * 2 + j], 
                        current_length=current_length_of_each_layer[layer_idx * 2 + j]) for j in range(2)
            ]
        )
    # past_key_values_objects: [ [KvCache(key), KvCache(value)], [KvCache(key), KvCache(value)], ...], 有 num_hidden_layers 个 key-value kvcache对象
    # 每个KVCache.data的shape为 [batch_size, head_num, max_seq_len, head_dim]
    # past_key_values_data: [num_hidden_layers * 2, batch_size, head_num, max_seq_len, head_dim]
    # 注意：past_key_values_objects和past_key_values_data是两个不同的对象，但它们共享相同的GPU内存空间
    # current_legth_data: [num_hidden_layers * 2]
    return past_key_values_objects, past_key_values_data, current_length_of_each_layer
