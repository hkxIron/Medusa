from typing import List
import torch
import torch.nn.functional as F
#torch.set_printoptions(precision=2, linewidth=120)


#from medusa.model.medusa_model import MedusaModel

TOPK=10 # topk for sparse tree (10 is a placeholder and it is sufficient)

def pad_path(path:List[int], length:int, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices:List[List[int]], device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1 # 63+1=64

    """
    参见attention_mask.txt中的注释
    以mc_sim_7b_63为例

    sorted_medusa_choices中每行均为一个,
    格式为:[head[0],token[i], head[1].token[j], head[2].token[k],...]
    [ 

        # 1~10 列
        [0], # base_model.cur_token+head[0].token[0]
        [1], # base_model.cur_token+head[0].token[1]
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9], # base_model.cur_token+head[0].token[9]

        # 11~20列
        # depth =1， 有28个
        # 格式为:[head[0],token[i], head[1].token[j], head[2].token[k],...]
        # 下面括号的数据,内容为：[head[0].token[0], head[1].token[0:10]]
        [0, 0], # base_model.cur_token+head[0].token[0] + head[1].token[0]
        [0, 1], # base_model.cur_token+head[0].token[0] + head[1].token[1]
        ...
    ]

    ...
    # 49 ~ 51列
    # 下面括号的数据,内容为：[head[0].token[0], head[1].token[1], head[2].token[0:3]]
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 2],
    ...

    depth_counts: [10, 28, 23, 2] 分别表示:
    1阶attention有10个
    2阶attention有28个
    3阶attention有23个
    4阶attention有2个
    """
    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = [] # depth_counts: [10, 28, 23, 2]
    
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len) # 对角矩阵，保证每个token可以attention到自己
    medusa_attn_mask[:, 0] = 1 # 将第0列设置为1, 表示base_model.cur_token在每个tree中必须attention验证
    start = 0
    # 遍历每个depth_counts, 将每个depth_counts中的所有长度的choices取出来
    for i in range(len(depth_counts)):  # depth_counts: [10, 28, 23, 2]
        for j in range(depth_counts[i]):
            # 遍历choice中长度为depth_counts[i]的每个choice
            cur_medusa_choice :List[int]= sorted_medusa_choices[start + j]
            # 若choice长度为1, 1阶attention无祖先，无需查找父节点
            if len(cur_medusa_choice) == 1:
                continue
            # retrieve ancestor position
            # 若choice长度>1, 2~4阶attention查找父节点
            ancestor_idx = []
            """
            cur_medusa_choice: [0, 1, 0]
              => [head[0].token[0], head[1].token[1], head[2].token[0:3]]
              父结点为：[0], [0, 1]
            """
            for c in range(len(cur_medusa_choice) - 1): 
                # 在当前行中查找父节点
                # list.index(): 返回cur_medusa_choice[:c+1]在sorted_medusa_choices中的索引, 即查找父节点
                current_parent_choice_index = sorted_medusa_choices.index(cur_medusa_choice[:c+1])
                # 注意：此处查找的是所有的父结点，而只是一个父结点
                ancestor_idx.append(current_parent_choice_index + 1) #  +1是为了后面索引到本身
            # j + start+1：当前行的索引,其中的+1是因为第一行为base_model.cur_token
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1 # 将所有父节点设置为1
        start += depth_counts[i]

    #torch.set_printoptions(profile="default", linewidth=200, threshold=1000)
    #print(f"{medusa_attn_mask.to(torch.int8).tolist()=}")

    # Generate tree indices for the Medusa structure
    """
    这里的tree_indices，其实就是attention中最后一个head中最后一个token的索引

    #第31～32列：medusa_head[1].token[0:2], attention前缀 base_model.cur_token+medusa_head[0].token[3]
    11, 12,  # 例如：其中11为head[1].token[0]的列索引, 12为head[1].token[1]的列索引

    #第33列：medusa_head[1].token[0], attention前缀 base_model.cur_token+medusa_head[0].token[4]
    #第34列：medusa_head[1].token[0], attention前缀 base_model.cur_token+medusa_head[0].token[5]
    #第35列：medusa_head[1].token[0], attention前缀 base_model.cur_token+medusa_head[0].token[6]
    #第36列：medusa_head[1].token[0], attention前缀 base_model.cur_token+medusa_head[0].token[7]
    #第37列：medusa_head[1].token[0], attention前缀 base_model.cur_token+medusa_head[0].token[8]
    #第38列：medusa_head[1].token[0], attention前缀 base_model.cur_token+medusa_head[0].token[9]
    11, 11, 11, 11, 11, 11,  # 例如：其中11为head[1].token[0]的列索引
    格式为： [base_model.cur_token, head[0], head[1], head[2], head[3]]
    """
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long) # shape:[64]
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):# depth_counts: [10, 28, 23, 2]
        # 遍历choice中长度为depth_counts[i]的每个choice
        for j in range(depth_counts[i]):
            """
            cur_medusa_choice: [0, 1, 0]
              => [head[0].token[0], head[1].token[1], head[2].token[0:3]]
            """
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # TOPK=10, 为每个head预留10个token位置
            last_token_index_of_last_head = cur_medusa_choice[-1] # 最后一个head的最后一个token索引
            depth_start = TOPK * i # 该层的起始位置
            # 最后 +1是因为base_model.cur_token在每个tree中必须attention验证
            medusa_tree_indices[start + j + 1] = last_token_index_of_last_head + depth_start + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    # depth_counts: [10, 28, 23, 2]
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long) # shape:[64]
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    # 注意：这次是反向遍历，从最深的tree开始遍历
    for i in range(len(sorted_medusa_choices)):
        """
        cur_medusa_choice: [0, 1, 0]
            => [head[0].token[0], head[1].token[1], head[2].token[0:3]]
            父结点为：[0], [0, 1]
        """
        cur_medusa_choice = sorted_medusa_choices[-i-1] # 从倒数第一个choices开始遍历
        retrieve_indice = []

        """
        若某个choice的父结点已经存在于retrieve_paths中，说明该choice已经验证过，无需再次验证

        所有待验证的路径，不能是已经出现的路径的前缀编码，否则会导致重复路径验证
        比如head[0]=["the", "a"], head[1]=["last", "day"], head[2]=["day", "choice"]
        若["the", "last", "day"]已出现在验证路径中时, 前缀路径 ["the", "last"]就无需单独验证了
        所以尽管总共有64条路径待验证，但由于相同前缀的原因，去除共同前缀后，只有42条路径待验证
        """
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                # 在所有choices中查找当前的父结点，注意：不是仅在当前层的choices中查找
                parent_of_cur_choice: List[int] = cur_medusa_choice[:c+1]
                current_parent_choice_index: int = sorted_medusa_choices.index(parent_of_cur_choice)
                # 这里的indice为父结点的索引
                retrieve_indice.append(current_parent_choice_index)
                # 父结点的path收集到retrieve_paths中，避免重复
                if parent_of_cur_choice not in retrieve_paths:
                    retrieve_paths.append(parent_of_cur_choice)
        retrieve_indices_nest.append(retrieve_indice)

    max_length = max([len(x) for x in retrieve_indices_nest]) # 最长的路径长度
    
    retrieve_token_indices = [pad_path(path, max_length, -2) for path in retrieve_indices_nest]
    retrieve_token_indices = torch.tensor(retrieve_token_indices, dtype=torch.long)
    retrieve_token_indices = retrieve_token_indices + 1 # 所有indexes +1
    # 将第0列的base_model.cur_token的index拼上
    # retrieve_indices.shape:[42, 1+max_length=5]
    retrieve_token_indices = torch.cat([torch.zeros((retrieve_token_indices.shape[0], 1), dtype=torch.long), retrieve_token_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_token_indices, # 无前缀的路径
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
            if isinstance(v, torch.Tensor) else torch.tensor(v,  device=device)
                for k, v in medusa_buffers.items()
    }
    return medusa_buffers


def initialize_medusa(input_ids, model, # MedusaModel
                      medusa_attn_mask, 
                      past_key_values):
    """
    Initializes the Medusa structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - medusa_attn_mask (torch.Tensor): The attention mask designed specifically for the Medusa structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - medusa_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    medusa_logits, outputs, logits = model.forward(
        input_ids, past_key_values=past_key_values, output_orig=True, medusa_forward=True
    )
    # 注意：更改base_model的medusa_mask
    model.base_model.model.medusa_mask = medusa_attn_mask
    return medusa_logits, logits


def reset_medusa_mode(
    model, #MedusaModel,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values

def get_nucleus_one_token(logit, temperature, top_p):
    """
    核采样

    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    # logits:[batch, vocab_size]
    logit = logit / temperature
    # probs:[batch, vocab_size]
    probs = torch.softmax(logit, dim=-1)
    # entropy:[batch], entropy = -∑p(x)logp(x)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    # entropy:[batch]
    # threshold:[batch]
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha, # exp(-entropy), 即 熵越高，不确定性越高的， threshold概率越低
        )
    # probs: [batch, vocab_size]
    # indices_to_remove:[batch, vocab_size]
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf') # 将概率低于threshold的token的logit置为-inf， 即不采样
    # sampled_tokens: [batch]
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def generate_candidates(medusa_logits, 
                        logits, 
                        tree_indices, 
                        retrieve_indices, 
                        temperature = 0, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = False):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - medusa_logits (torch.Tensor): Logits from a specialized Medusa structure, aiding in candidate selection.
    - logits (torch.Tensor): Standard logits from a language model.
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.
    - temperature (float, optional): Controls the diversity of the sampling process. Defaults to 0.
    - posterior_threshold (float, optional): Threshold for typical sampling. Defaults to 0.3.
    - posterior_alpha (float, optional): Scaling factor for the entropy-based threshold in typical sampling. Defaults to 0.09.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """

    # medusa_logits: [medusa_head=5, batch_size=1, seq_len=input_len, vocab_size]
    # logits: [batch_size=1, seq_len=input_len=6, vocab_size]

    # Greedy decoding: Select the most probable candidate from the original logits.
    if temperature == 0 or fast:
        # logits: [batch_size=1, seq_len, vocab_size]
        # 取最后一个medusa_head的预测结果
        # base_model_candidates_idx: [batch_size=1]
        base_model_candidates_idx = torch.argmax(logits[:, -1]).unsqueeze(0) # 贪婪采样
    else:
        if sampling == 'typical': # 
            base_model_candidates_idx = get_typical_one_token(logits[:, -1], temperature, posterior_threshold, posterior_alpha).squeeze(0)
        elif sampling == 'nucleus': # 核采样
            base_model_candidates_idx = get_nucleus_one_token(logits[:, -1], temperature, top_p).squeeze(0)
        else:
            raise NotImplementedError

    # Extract the TOPK candidates from the medusa logits.
    # medusa_logits: [medusa_head=5, batch_size=1, seq_len, vocab_size]
    # 对每个medusa_head的预测序列的最后一个token取topK, 因为每个step也只预测一次
    # candidates_medusa_logits_idx: [medusa_head=5, vocab_size=topK]
    candidates_medusa_logits_idx = torch.topk(medusa_logits[:, 0, -1], k=TOPK, dim = -1).indices

    # Combine the selected candidate from the original logits with the topk medusa logits.
    # base_model_candidates_idx: [batch_size=1]
    # candidates_medusa_logits_idx: [medusa_head=5, vocab_size=topK]
    # 将base_model_pred和medusa头的猜测拼接起来，作为新的输入
    # candidates_idx: [seq_len=1+medusa_head*top_k=1+5*10=51]
    candidates_token_id = torch.cat([base_model_candidates_idx, candidates_medusa_logits_idx.view(-1)], dim=-1)

    """
    tree_indices: [seq_len=1+medusa_head*top_k=1+5*10=51]
        [0, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
        11, 12, 13, 14, 15, 16, 17, 11, 12, 13, 
        11, 12, 
        11, 11, 11, 11, 11, 11, 
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
        21, 22, 23, 21, 22, 
        21, 21, 21, 21, 21, 
        21, 22, 
        21, 
        31, 32]
    tree_candidates: [seq_len=64]
    """
    # Map the combined candidates to the tree indices to get tree candidates.
    #print(f"{candidates_idx.shape=}")
    #tree_candidates_token_id: [seq_len=64]
    tree_candidates_token_id = candidates_token_id[tree_indices]

    # Extend the tree candidates by appending a zero.
    #tree_candidates_token_id: [seq_len=64]
    #tree_candidates_token_id_ext: [seq_len=64+1=65]
    tree_candidates_token_id_ext = torch.cat([tree_candidates_token_id, torch.zeros((1), dtype=torch.long, device=tree_candidates_token_id.device)], dim=0)

    """
    retrieve_indices: 共42行
    [[0, 1, 11, 39, 63], # base_model.cur_token + head[0].token[0]+head[1].token[0] + head[2].token[1] + head[3].token[1]
    [0, 1, 11, 39, 62],  # base_model.cur_token + head[0].token[0]+head[1].token[0] + head[2].token[1] + head[3].token[0]
    [0, 3, 28, 61, -1],  # base_model.cur_token + head[0].token[2]+head[1].token[0] + head[2].token[0]
    [0, 2, 21, 60, -1],
    ...
    """
    # Retrieve the cartesian candidates using the retrieve indices.
    # cartesian_candidates_token_id: [seq_len=42, head_num=base_model.cur_token+head[0...4]=5]
    # 即从vocab中选出组成tree attention的token_id, 为后面attention作准备 
    cartesian_candidates_token_id = tree_candidates_token_id_ext[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    # cartesian_candidates_token_id: [seq_len=42, head_num=base_model.cur_token+head[0...4]=5]
    # tree_candidates_token_id: [batch=1, seq_len=64]
    tree_candidates_token_id = tree_candidates_token_id.unsqueeze(0)
    return cartesian_candidates_token_id, tree_candidates_token_id


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    # input_ids: [batch_size=1, seq_len]

    # medusa_position_ids: [seq_len=path_choice_num=64]
    # position_ids: [seq_len=64], 为在input_ids的位置上加上偏移 medusa_position_ids
    position_ids = medusa_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.

    # tree_candidates: [batch=1, seq_len=64] 
    # position_ids: [seq_len=64], 为在input_ids的位置上加上偏移 medusa_position_ids

    # tree_medusa_logits: [medusa_head=5, batch_size=1, seq_len=64, vocab_size]
    # outputs: other outputs from the model
    # tree_logits: [batch_size=1, seq_len, vocab_size]
    tree_medusa_logits, outputs, tree_logits = model.forward(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        medusa_forward=True,
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    # tree_logits: [batch_size=1, seq_len=64, vocab_size]
    # retrieve_indices: [seq_len=42, head_position_num=base_model.cur_token+head[0...4]=5]
    # logits: [seq_len=42, head_position_num=base_model.cur_token+head[0...4]=5, vocab_size]
    logits = tree_logits[0, retrieve_indices]

    # tree_medusa_logits: [medusa_head=5, batch_size=1, seq_len=64, vocab_size]
    # medusa_logits: [medusa_head=5, seq_len=42, head_position_num=5, vocab_size]
    medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]
    # logits: [seq_len=42, head_position_num=base_model.cur_token+head[0...4]=5, vocab_size]
    return medusa_logits, logits, outputs


def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    核采样

    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask
    
    

def evaluate_posterior(
    logits, candidate_token_ids, temperature, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = True
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        # logits: [seq_len=42, head_position_num=base_model.cur_token+head[0...4]=5, vocab_size]
        # candidates_token_id: [seq_len=42, head_position_num=base_model.cur_token+head[0...4]=5]
        # posterior_mask: [seq_len=42, 4]
        posterior_mask = (candidate_token_ids[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)).int()
        # 求接受的token个数
        # candidates_accept_length:[seq_len=42]
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate_idx = torch.tensor(0, dtype=torch.long, device=candidate_token_ids.device)
        else:
            # 选择接受最多的candidate所在的index
            best_candidate_idx = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate_idx, accept_length
        
    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidate_token_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate_idx = torch.tensor(0, dtype=torch.long, device=candidate_token_ids.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate_idx = best_candidates[torch.argmax(likelihood)]
            return best_candidate_idx, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(logits, candidate_token_ids, temperature, posterior_threshold, posterior_alpha, fast)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate_idx = torch.tensor(0, dtype=torch.long, device=candidate_token_ids.device)
        else:
            best_candidate_idx = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate_idx, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidate_token_ids, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate_idx = torch.tensor(0, dtype=torch.long, device=candidate_token_ids.device)
        else:
            best_candidate_idx = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate_idx, accept_length
    else:
        raise NotImplementedError

def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]
    medusa_logits = medusa_logits[
        :, None, best_candidate, accept_length : accept_length + 1
    ]
    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, medusa_logits, new_token
