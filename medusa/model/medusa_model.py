import logging
from typing import List
import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
import traceback
from safetensors.torch import load_file
# import transformers
from transformers.utils import logging

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

logger = logging.get_logger(__name__)
"""
注意：
medusa_model.py：每个medusa head都加了一个自己的lm head, 而medusa论文中每个medusa head都加了一个自己的lm head
medusa_model_legacy.py：每个medusa head都没有自己的lm head,大家都复用base model的lm head, 因此被称为legacy，意为“遗留不用的”
"""
class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=2,
        medusa_num_layers=1,
        base_model_name_or_path="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Medusa单独的参数
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)# 注意：权重初始化为0
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()# 如果是其它的模型，可能需要换成其它的激活函数

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        base_model:PreTrainedModel,
        config:MedusaConfig,
        *args,
        **kwargs,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(*args, **kwargs)

        self.base_model: PreTrainedModel = base_model
        base_model_name_or_path = config.base_model_name_or_path

        # For compatibility with the old APIs
        medusa_num_heads = kwargs.pop("medusa_num_heads", None)
        medusa_num_layers = kwargs.pop("medusa_num_layers", None)

        #print(f"{medusa_num_heads=}  {medusa_num_layers=}")
        if medusa_num_heads:
            self.medusa_num_heads = medusa_num_heads # 有多少个Medusa heads
        else:
            self.medusa_num_heads = config.medusa_num_heads # 有多少个Medusa heads

        if medusa_num_layers:
            self.medusa_num_layers = medusa_num_layers # 每个medusa head有多个layers
        else:
            self.medusa_num_layers  = config.medusa_num_layers # 每个medusa head有多个layers

        self.base_model_name_or_path = base_model_name_or_path

        base_model_config = AutoConfig.from_pretrained(base_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)

        self.hidden_size = base_model_config.hidden_size
        self.vocab_size = base_model_config.vocab_size
        #self.current_length_data = torch.zeros(1, dtype=torch.int64) # 用于记录当前的token数量


        # Create a list of Medusa heads
        self.medusa_head:List[nn.Module] = nn.ModuleList(
            [
                nn.Sequential(
                    # 有多个medusa头，每个medusa头有多个ResBlock layer, 直接将medusa_num_layers个ResBlock layer拼接在一个list中
                    *([ResBlock(self.hidden_size)] * self.medusa_num_layers), # 将前面的layers拼在一起
                    # 每个ResBlock layer后面都接一个lm_head
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )
        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(dtype=self.base_model.dtype).to(device=self.base_model.device)
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        MedusaModel的前向只是加了多个medusa head的预测，没有做其它的任何的修改

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward: # 是否启动medusa forward
            return self.base_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )

        with torch.inference_mode(): # 对于base model的forward，不需要计算梯度
            # Pass input through the base model
            # base_model.model仅有decoder,但并不包含lm_head
            outputs = self.base_model.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                # orig_seq_logits: gg[batch_size, seq_len, hidden_size]
                orig_seq_logits = self.base_model.lm_head.forward(outputs[0])

        # Clone the output hidden states
        # hidden_states: [batch_size, seq_len, hidden_size]
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # medusa_logits: list of [batch_size, seq_len, vocab_size], 不同的medusa head的seq logits预测结果
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i].forward(hidden_states))

        # all_medusa_logits: [medusa_num_heads, batch_size, seq_len, vocab_size]
        all_medusa_logits = torch.stack(medusa_logits, dim=0) 
        if output_orig:
            return all_medusa_logits, outputs, orig_seq_logits

        return all_medusa_logits

    def get_medusa_choice(self, model_name:str):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512, # 最大生成多少个token
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        # 只支持batch size=1
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone() # input_ids:[batch_size, seq_len]

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(medusa_choices, device=self.base_model.device)
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            # past_key_values: [ [KvCache(key), KvCache(value)], [KvCache(key), KvCache(value)], ...], 有 num_hidden_layers 个 key-value kvcache对象
            # 每个KVCache.data的shape为 [batch_size, head_num, max_seq_len, head_dim]
            # past_key_values_data: [num_hidden_layers * 2, batch_size, head_num, max_seq_len, head_dim]
            # current_legth_data: [num_hidden_layers * 2]
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            # current_legth_data: [num_hidden_layers * 2], 奇数和偶数分别对应key和value的长度, 这两个长度一般是相同的
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        # 1. 生成原始模型推理的logits，以及medusa的logits
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps): # 最大生成max_steps个token
            # 2. 生成各medusa头的候选集
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # 3. 使用模型再推理一次，使用tree attention验证候选集
            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # 4. 评估候选集的后验概率，选择接受的候选前缀
            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }
            
            # 如果生成了EOS,直接退出
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


    @classmethod
    def from_pretrained(cls, base_model_path:str, medusa_head_path:str, medusa_num_heads:int=None):
        medusa_config: MedusaConfig = MedusaConfig.from_pretrained(medusa_head_path)
        medusa_config.base_model_name_or_path = base_model_path
        if medusa_num_heads is not None:
            print("Overriding medusa_num_heads as:", medusa_num_heads)
            medusa_config.medusa_num_heads = medusa_num_heads
        #base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        #config.model_type = base_model_config.model_type
        base_model = KVLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_path)
        # ---------------
        #args = dict(medusa_num_heads=medusa_config.medusa_num_heads, medusa_num_layers=medusa_config.medusa_num_layers)
        medusa_model = MedusaModel(base_model=base_model, config=medusa_config)
        medusa_model.base_model = base_model

        medusa_head_file = os.path.join(medusa_head_path, "medusa_lm_head.safetensors")
        # 正确设置设备
        device = base_model.device if torch.cuda.is_available() else "cpu"
        if str(device) == "cpu":
            device = "cpu"  # 显式转换为字符串"cpu"
        else:
            device = str(device)  # 如"cuda:0"

        if medusa_head_file.endswith(".safetensors"):
            medusa_head_state_dict = load_file(medusa_head_file, device=device)
        else:
            medusa_head_state_dict = torch.load(medusa_head_file, map_location=device, weights_only=False)
        medusa_model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)

        print(f"inited model:{medusa_model=}")
        return medusa_model


    #Add a link named base_model to self
    # @property
    # def base_model(self):
    #     return self.super()

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path:str,
    #     *args,
    #     **kwargs,
    # ):
    #     # Manually load config to ensure that the medusa_num_heads parameter is loaded
    #     try:
    #         print(f"MedusaModelABC Loading config from: {pretrained_model_name_or_path}")
    #         config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    #         print(f"MedusaModelABC config: {config}")
    #         # 在classmethod中，super()即表示MedusaModel，即MedusaModelABC的父类，即PreTrainedModel
    #         medusa_model: MedusaModel = super().from_pretrained(pretrained_model_name_or_path,
    #             *args,
    #             **kwargs,
    #             config=config,
    #         )
    #         print(f"MedusaModelABC medusa model:{medusa_model=}")
    #         return medusa_model

    #     except Exception as ex:
    #         print(f"MedusaModelABC 加载模型配置出错，{pretrained_model_name_or_path}, 再试加载base_model配置, {traceback.format_exc()}")
    #         config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
    #         print(f"MedusaModelABC meduas config:{config}")
    #         base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    #         base_model_config.medusa_num_heads = config.medusa_num_heads 
    #         base_model_config.medusa_num_layers = config.medusa_num_layers
    #         medusa_model = super().from_pretrained(
    #             config.base_model_name_or_path,
    #             *args,
    #             **kwargs,
    #             config=base_model_config,
    #         )
    #         medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.safetensors")
    #         if os.path.exists(medusa_head_path):
    #             filename = medusa_head_path
    #         else:
    #             filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.safetensors")
    #         medusa_head_state_dict = torch.load(filename, map_location=medusa_model.device)
    #         medusa_model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
    #         return medusa_model



# class MedusaModelLlama(MedusaModel, KVLlamaForCausalLM):
#     pass

# class MedusaModelMistral(MedusaModel, KVMistralForCausalLM):
#     pass


# class MedusaModel():

#     @classmethod
#     def from_pretrained(
#         cls,
#         pretrained_model_name_or_path,
#         *args,
#         **kwargs,
#     ) -> MedusaModelLlama | MedusaModelMistral:
#         # Manually load config to ensure that the medusa_num_heads parameter is loaded
#         try:
#             print(f"MedusaModel.from_pretrained 加载预训练模型:{pretrained_model_name_or_path}")
#             config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
#         except Exception as ex:
#             print(f"MedusaModel.from_pretrained 加载预训练模型出错:{pretrained_model_name_or_path}, 重新初始化模型配置")
#             # MEDUSA-v0.1 load
#             config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
#             base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
#             config.model_type = base_model_config.model_type

#         print(f"MedusaModel.from_pretrained config:{config}")
#         if config.model_type == "llama":
#             return MedusaModelLlama.from_pretrained(
#                 pretrained_model_name_or_path,
#                 *args,
#                 **kwargs,
#             )
#         elif config.model_type == "mistral":
#             return MedusaModelMistral.from_pretrained(
#                 pretrained_model_name_or_path,
#                 *args,
#                 **kwargs,
#             )
#         else:
#             raise ValueError("Only support llama and mistral for now!!")
