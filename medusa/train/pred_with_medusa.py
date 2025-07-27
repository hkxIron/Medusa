# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py
import sys
#sys.path.append("../../") # 否则找不到medusa/model/medusa_model.py
import pandas as pd
from typing import List

import torch
from transformers.trainer_pt_utils import LabelSmoother

#from fastchat.conversation import SeparatorStyle
#from fastchat.model.model_adapter import get_conversation_template
import os
from medusa.model.medusa_model import MedusaModel, MedusaConfig, MedusaModel
from transformers import (PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM)

from medusa.model.modeling_llama_kv import LlamaForCausalLM
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
# 设置打印选项，显示1000行
torch.set_printoptions(profile="default", linewidth=200, threshold=1000)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def show_id_token_mask(tokenizer:PreTrainedTokenizer, 
                       input_id:List[int]|torch.Tensor, 
                       attention_mask:List[int]|torch.Tensor, 
                       labels:List[int]|torch.Tensor=None,
                       name=""):
    if isinstance(input_id, torch.Tensor):
        input_id = input_id.tolist()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.tolist()

    tokens = [f"{tokenizer.convert_ids_to_tokens(i)}" if i>=0 else str(i) for i in input_id]
    df = pd.DataFrame({
        "input_ids":input_id,
        "token":tokens,
        "attention_mask":attention_mask,
    })
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        df["labels"]=pd.Series(labels)
    print(f"name:{name}\n{df.to_string()}")
    #print(tabulate(df, headers='keys', tablefmt='grid'))

def show_diff_ids(ids_left:List[int], ids_right:List[int], tokenizer, ids_left_name='left', ids_right_name='right'):
    pre_n = 10
    post_n = 10
    for (ind,(a, b)) in enumerate(zip(ids_left, ids_right)):
        if a!=b:
            pre_ids = ids_left[max(ind-pre_n,0):ind]
            post_ids1 = ids_left[ind: min(ind+post_n,len(ids_left))]
            post_ids2 = ids_right[ind: min(ind+post_n,len(ids_right))]
            print(f"{ids_left_name}:\n{[(i, tokenizer.convert_ids_to_tokens(i) if i>=0 else '====') for i in pre_ids+[-1]+post_ids1]}")
            print(f"{ids_right_name}:\n{[(i, tokenizer.convert_ids_to_tokens(i)if i>=0 else '====') for i in pre_ids+[-1]+post_ids2]}")
            break


def pred():
    base_model_path = "/home/hkx/data/work/hf_data_and_model/models/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA"
    #medusa_model_path = "/home/hkx/data/work/open/Medusa/llama_medusa_output_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1" 
    medusa_model_path = "/home/hkx/data/work/open/Medusa/llama_medusa_output2_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1" 
    # medusa_config: MedusaConfig = MedusaConfig.from_pretrained(medusa_model_path)
    # medusa_config.base_model_name_or_path = base_model_path
    # #base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    # #config.model_type = base_model_config.model_type
    # base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_path)
    # ---------------
    #args = dict(medusa_num_heads=medusa_config.medusa_num_heads, medusa_num_layers=medusa_config.medusa_num_layers)
    medusa_model = MedusaModel.from_pretrained(base_model_path=base_model_path, medusa_head_path=medusa_model_path)
    print(f"{medusa_model=}")
    prompt="为以下关键词生成一条广告语。类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结"   
    inputs = medusa_model.tokenizer(prompt, return_tensors="pt")
    pred = medusa_model.forward(**inputs, medusa_forward=False)
    print(f"{pred=}")

    medusa_pred_gen = medusa_model.medusa_generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], temperature=0.1)
    medusa_pred_text = [x for x in medusa_pred_gen]
    print(f"{medusa_pred_text=}")

def pred_step():
    base_model_path = "/home/hkx/data/work/hf_data_and_model/models/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA"
    #medusa_model_path = "/home/hkx/data/work/open/Medusa/llama_medusa_output_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1" 
    medusa_model_path = "/home/hkx/data/work/open/Medusa/llama_medusa_output2_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1" 
    # medusa_config: MedusaConfig = MedusaConfig.from_pretrained(medusa_model_path)
    # medusa_config.base_model_name_or_path = base_model_path
    # #base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    # #config.model_type = base_model_config.model_type
    # base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_path)

    # ---------------
    #args = dict(medusa_num_heads=medusa_config.medusa_num_heads, medusa_num_layers=medusa_config.medusa_num_layers)
    model = MedusaModel.from_pretrained(base_model_path=base_model_path, medusa_head_path=medusa_model_path)
    print(f"{model=}")
    prompt="为以下关键词生成一条广告语。类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结"   
    inputs = model.tokenizer(prompt, return_tensors="pt")
    #pred = model.forward(**inputs, medusa_forward=False)

    tokenizer = model.get_tokenizer()
    medusa_choices = mc_sim_7b_63
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    # ==================================
    model.current_length_data.zero_() # this is for rerun
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
    print(prompt)
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    print('Input token length:', len(input_ids[0]))
    print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

    return 
    inference_count = 0
    accept_lengths = []
    with torch.inference_mode():
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids) #.cuda()
        model.current_length_data.zero_() # this is for rerun
        medusa_logits, outputs, logits = model.forward(input_ids, output_orig = True, past_key_values=model.past_key_values)
        inference_count += 1

        medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim = -1)
        pred = torch.argmax(logits[..., -1, :], dim = -1)
        preds = torch.cat([pred, medusa_pred[:, 0 ]], dim = -1)
        print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')
        cur_length = input_len
        accept_lengths.append(1)
        for _ in range(1024):
            medusa_logits, outputs, logits = model.forward(preds.unsqueeze(0), output_orig = True, past_key_values = model.past_key_values)
            inference_count += 1

            medusa_pred = torch.argmax(medusa_logits[..., -5:, :], dim = -1)
            pred = torch.argmax(logits[..., :, :], dim = -1)
            posterior_mask = (
                        preds[1:] == pred[0, :-1]
                    ).int()
            accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
            cur_length = cur_length + accept_length + 1
            # update kv cache
            model.current_length_data.fill_(cur_length)
            # create new input
            preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
            print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
            accept_lengths.append(accept_length + 1)
            if tokenizer.eos_token_id in pred[0, :accept_length + 1]:
                break

def test_medusa():
    base_model_path = "/home/hkx/data/work/hf_data_and_model/models/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA"
    #medusa_model_path = "/home/hkx/data/work/open/Medusa/llama_medusa_output_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1" 
    medusa_model_path = "/home/hkx/data/work/open/Medusa/llama_medusa_output2_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1" 
    # medusa_config: MedusaConfig = MedusaConfig.from_pretrained(medusa_model_path)
    # medusa_config.base_model_name_or_path = base_model_path
    # #base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    # #config.model_type = base_model_config.model_type
    # base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_path)

    # ---------------
    #args = dict(medusa_num_heads=medusa_config.medusa_num_heads, medusa_num_layers=medusa_config.medusa_num_layers)
    model = MedusaModel.from_pretrained(base_model_path=base_model_path, medusa_head_path=medusa_model_path)
    print(f"{model=}")

    tokenizer = model.get_tokenizer()

    medusa_choices = mc_sim_7b_63

    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    print(f"{model.base_model.config.max_position_embeddings=}") # 2048
    print(f"第0层的key的cache分配的内存张量大小:{past_key_values[0][0].data.shape=}") # (1, 8, 2048, 16)
    print(f"第0层的key的cache实际有效数据的张量大小:{past_key_values[0][0].shape=}") # (1, 8, 0, 16)

    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
    print(prompt)
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    print('Input token length:', len(input_ids[0]))
    # 打印第0层的KV cache 中的key, value的shape, shape: [batch_size, head_num, cache_seq_len, head_dim]
    print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)
    print(f'当前各层的缓存长度:{model.current_length_data}  layer_number:{model.current_length_data.shape[0]}')

    torch.set_printoptions(profile="default", linewidth=200, threshold=1000)

    with torch.inference_mode():
        new_token = 0
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids) #.cuda()
        model.current_length_data.zero_() # this is for rerun
        reset_medusa_mode(model)
        medusa_buffers: dict[str, torch.Tensor] = generate_medusa_buffers( medusa_choices, device=model.base_model.device)
        for k, v in medusa_buffers.items():
            print(f"{k}:")
            v_int = v.to(torch.int8).tolist()
            print(f"{v_int}")
        medusa_logits, logits = initialize_medusa(input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values)

if __name__ == "__main__":
    #pred_step()
    test_medusa()
