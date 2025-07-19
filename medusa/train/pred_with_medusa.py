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
import pandas as pd
from typing import List

import torch
from transformers.trainer_pt_utils import LabelSmoother

#from fastchat.conversation import SeparatorStyle
#from fastchat.model.model_adapter import get_conversation_template
import os
from medusa.model.medusa_model import MedusaModel, MedusaConfig, MedusaModel
from transformers import (PreTrainedTokenizer)

from medusa.model.modeling_llama_kv import LlamaForCausalLM


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


# def load_medusa_model(base_model_path, medusa_head_path):
#     medusa_config: MedusaConfig = MedusaConfig.from_pretrained(medusa_head_path)
#     medusa_config.base_model_name_or_path = base_model_path
#     #base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
#     #config.model_type = base_model_config.model_type
#     base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_path)
#     # ---------------
#     #args = dict(medusa_num_heads=medusa_config.medusa_num_heads, medusa_num_layers=medusa_config.medusa_num_layers)
#     medusa_model = MedusaModel(config=medusa_config)
#     medusa_model.base_model = base_model

#     medusa_head_file = os.path.join(medusa_head_path, "medusa_lm_head.safetensors")
#     medusa_head_state_dict = torch.load(medusa_head_file, map_location=base_model.device, weights_only=False)
#     medusa_model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)

    # print(f"inited model:{medusa_model=}")
    # return medusa_model

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

    medusa_pred = medusa_model.medusa_generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], temperature=0.1)
    print(f"{medusa_pred=}")

if __name__ == "__main__":
    pred()
