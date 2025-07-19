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
from dataclasses import dataclass, field
import datasets
from datasets import load_dataset, Dataset, IterableDataset
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file

#from fastchat.conversation import SeparatorStyle
#from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from medusa.model.medusa_model import MedusaModel, MedusaConfig, MedusaModel
from transformers import (AutoModelForCausalLM, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer)

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

"""

1. 手动去除前缀空格（推荐）
在 tokenize 后手动处理：
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "Hello world"
inputs = tokenizer(text, add_special_tokens=False)

# 检查第一个 token 是否是空格（Llama 的空格符为 '▁'）
if inputs["input_ids"] and tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == "▁":
    inputs["input_ids"] = inputs["input_ids"][1:]  # 移除第一个 token
    inputs["attention_mask"] = inputs["attention_mask"][1:]  # 同步处理 mask

print(tokenizer.decode(inputs["input_ids"]))  # 输出: "Hello world"（无前缀空格）


2. 替换 tokenizer 的初始化行为
通过修改 tokenizer.json 的配置（需重新保存 tokenizer）：

python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 强制设置（不保证完全兼容）
tokenizer.add_prefix_space = False  

# 重新保存并测试
tokenizer.save_pretrained("./modified_tokenizer")
new_tokenizer = AutoTokenizer.from_pretrained("./modified_tokenizer")

3. 预处理输入文本
在输入前手动移除首部空格（简单但可能不通用）：

python
text = " Hello world".lstrip()  # 确保文本开头无空格
inputs = tokenizer(text, add_special_tokens=False)

测试:
tokens = tokenizer.tokenize("Hello")
print(tokens)  # 期望输出: ['Hello'] 而非 ['▁Hello']
"""


"""
1. debug发现是拼接的多了空格token
some token is merged in encode once, please check last token of input_text, len(all_text_encode_once)=140 != len(input_ids)=141, first token of output_text, 
input_text='类型#裙*风格#简约*图案#条纹*图案#线条*图案#撞色*裙型#鱼尾裙*裙袖长#无袖' 
output_text='圆形领口修饰脖颈线条，适合各种脸型，耐看有气质。无袖设计，尤显清凉，简约横条纹装饰，使得整身人鱼造型更为生动立体。加之撞色的鱼尾下摆，深邃富有诗意。收腰包臀,修饰女性身体曲线，结合别出心裁的鱼尾裙摆设计，勾勒出自然流畅的身体轮廓，展现了婀娜多姿的迷人姿态。'
once_enc:
[(60312, '鱼'), (60715, '尾'), (61811, '裙'), (59379, '*'), (61811, '裙'), (61525, '袖'), (59503, '长'), (59377, '#'), (59569, '无'), (61525, '袖'), (-1, '===='), (17132, '圆形'), (59928, '领'), (59718, '口'), (42784, '修饰'), (62593, '脖'), (61379, '颈'), (29900, '线条'), (65, '，'), (5982, '适合'), (4023, '各种')]
add_enc:
[(60312, '鱼'), (60715, '尾'), (61811, '裙'), (59379, '*'), (61811, '裙'), (61525, '袖'), (59503, '长'), (59377, '#'), (59569, '无'), (61525, '袖'), (-1, '===='), (59320, '▁'), (17132, '圆形'), (59928, '领'), (59718, '口'), (42784, '修饰'), (62593, '脖'), (61379, '颈'), (29900, '线条'), (65, '，'), (5982, '适合')]

2. 也有的是因为tokenizer在有无前文时，部分分词不一样，如下
some token is merged in encode once, please check last token of input_text, len(all_text_encode_once)=100 != len(input_ids)=101, first token of output_text, input_text='类型#上衣*材质#针织*颜色#粉色*风格#文艺*风格#清新*衣样式#外套*衣领型#v领*衣门襟#系带' output_text='有美式独家风的一款小外套，春秋季节作为防晒服小巧可爱。细腻的针织材质柔软舒适，搭配上裸粉色更显清新文艺。大大的v领加上合身的版型自由舒适，前面的系带增加立体感和层次感。'
once_enc:
[(59683, '型'), (59377, '#'), (59343, 'v'), (59928, '领'), (59379, '*'), (60320, '衣'), (59752, '门'), (62806, '襟'), (59377, '#'), (59574, '系'), (-1, '===='), (19313, '带有'), (59639, '美'), (59585, '式'), (35209, '独家'), (59722, '风'), (26010, '的一款'), (59472, '小'), (44895, '外套'), (65, '，'), (27307, '春秋')]
add_enc:
[(59683, '型'), (59377, '#'), (59343, 'v'), (59928, '领'), (59379, '*'), (60320, '衣'), (59752, '门'), (62806, '襟'), (59377, '#'), (59574, '系'), (-1, '===='), (59822, '带'), (59395, '有'), (59639, '美'), (59585, '式'), (35209, '独家'), (59722, '风'), (26010, '的一款'), (59472, '小'), (44895, '外套'), (65, '，')]
"""
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

"""
训练Medusa模型的MedusaHead
"""
# Customized for training Medusa heads
class CustomizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.steps = 0
    
    def get_original_model(self, model):
        return model.module if hasattr(model, "module") else model

    """计算loss
    """
    def compute_loss(self, model:MedusaModel, inputs:Dict[str, Any], return_outputs=False, **kwargs):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"): # DDP
            medusa_heads = model.module.medusa_head
        else: # 单卡
            medusa_heads = model.medusa_head
        #medusa = model.medusa_head


        # logits:[medusa_head, batch, seq_len, vocab_size]
        logits = model.forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], medusa_forward=True)
        # labels:[batch, seq_len]
        #print(f"{inputs.keys()=}, {inputs}")
        labels = inputs["labels"]

        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(len(medusa_heads)): # 遍历所有的Medusa头
            """
            原始的llama模型的loss计算 
            # logits:[head, batch_size, seq_len, vocab_size], 取0～seq_len-1的token logits
            # labels:[head, batch_size, seq_len], 对应label为1～seq_len
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            """
            #print(f"{logits.shape=}")
            medusa_logits = logits[i, :, : -(2 + i)].contiguous() # 取出第i个Medusa头的logits, 预测的是next next token, 即向左移2位
            medusa_labels = labels[..., 2 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1]) # [batch_size * (seq_len - 2), vocab_size]
            medusa_labels = medusa_labels.view(-1) # [batch_size * (seq_len - 2)]
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore] # [batch_size * (seq_len - 2)], 只取label不为IGNORE_TOKEN_ID的token, shape可能会变小

            # Add top-k accuracy
            for k in range(1, 2):# 这里只计算top-1的准确率
                # medusa_logits: [batch_size * (seq_len - 2), vocab_size], vocab_size维取topK
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item() # item()将tensor转换为python标量
        self.log(log)

        current_step = self.state.global_step
        if current_step==0:
            origin_model = self.get_original_model(model) # ddp -> unwrapped model
            show_id_token_mask(origin_model.tokenizer, inputs["input_ids"][0], inputs["attention_mask"][0], labels=labels[0], name="原始标签")
            show_id_token_mask(origin_model.tokenizer, inputs["input_ids"][0], inputs["attention_mask"][0], labels=medusa_labels[0], name="medusa标签")

        #self.step +=1

        return (loss, logits) if return_outputs else loss # hf Trainer框架要求返回标量


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     """
#     Save the model's state dictionary to a specified directory.

#     Args:
#         trainer (transformers.Trainer): The Hugging Face Trainer object.
#         output_dir (str): The directory where the model state dictionary will be saved.
#     """
#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class MySFTDataset:
    """Dataset for supervised fine-tuning.
    注意，并不是hf的Dataset
    """
    def __init__(
        self,
        data_path:str,
        tokenizer,
        max_seq_length=512,
        hint=False
    ):

        df = pd.read_json(path_or_buf=data_path, lines=True)
        #df = df.head(2)
        if hint:
            print(f"data columns:{list(df.columns)}")
            print(f"data len:{df.shape[0]}")
            print(f"head data:{df.head(1)=}")

        self.tokenizer = tokenizer
        self.model_max_length = max_seq_length
        # 构建hf dataset数据集
        sft_dataset = datasets.Dataset.from_pandas(df)\
            .map(self.my_convert_tokens_to_ids)\
            .filter(lambda x:len(x["input_ids"])>0 and len(x["labels"])>0)

        self.dataset = sft_dataset.map(remove_columns=["prompt", "input", "output", "in_text", "out_text"])
        if hint:
            print(f"head data:{self.dataset[0:2]=}")
            print(f"dataset:{data_path} build done!")
    
    def my_convert_tokens_to_ids(self, example:Dict[str, str], add_enter_before_output:bool=True):
        #input_ids = [self.tokenizer.bos_token_id]
        #label_ids = [ignore_index]
        
        prompt_text = example['prompt'].strip()
        input_text = example['input'].strip() 
        # 防止token merge
        # 1.去掉ouput_text前面的空白字符
        # 2.前面加上"\n", 以防止token merge, 这样之后，就不会出现意外的token_merge
        if add_enter_before_output:
            output_text = "\n"+example['output'].strip() 
        else:
            output_text = example['output'].strip() 


        # prompt部分不需要计算loss,因此使用ignore_index=-100 
        # 注意：
        # 1. tokenizer.__call__可以得到 input_ids+ attention_mask的Dict[str, List[int]]
        # 2.而tokenizer.encode只能得到 List[int], 所以此处只需要用encode
        prompt_input_ids = self.tokenizer.encode(prompt_text+input_text, add_special_tokens=False)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        input_ids = prompt_input_ids 
        label_ids = [IGNORE_TOKEN_ID] * len(input_ids)
        
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False) + [eos_id]
        if self.tokenizer.convert_ids_to_tokens(output_ids[0]) == "▁":
            output_ids = output_ids[1:] # 移除第一个 space token, 会导致label_id错位, 可以解决文本合并后tokenizer的token merge的问题

        input_ids+=output_ids

        # 有个疑问是：如果在llm sft中，恰好input_text的最后一个token与output_text中第一个token可以组成一个新的token,如input_text最后一个token为"你"，output第一个token为“好”，
        # 两个文本拼接在一起时"你好"恰好也为一个新的token，但整体上token数变少了，
        # 那么len(tokenizer(input_text+output_text))<len(tokenizer(input_text))+len(tokenizer(output_text))了，就会产生id的错位，label_id也会错位，
        # 答案：见上面的token merge
        all_text_encode_once = self.tokenizer.encode(prompt_text+input_text+output_text, add_special_tokens=False) + [eos_id]
        # 不知为何llama的tokenizer在有无前文时，总是会加前缀空格
        if len(all_text_encode_once) != len(input_ids):
            print(f"some token is merged in encode once, please check last token of input_text, {len(all_text_encode_once)=} != {len(input_ids)=}, first token of output_text, {input_text=} {output_text=}")
            show_diff_ids(all_text_encode_once, input_ids, self.tokenizer, ids_left_name="一次性编码", ids_right_name="编码后相加")
            # 不能返回None, None在dataset中不可遍历, 'NoneType' object is not subscriptable, 要求返回的key也必须与正常的相同
            return {
                "in_text": "",
                "out_text": "",
                "input_ids": [],
                "labels": [],
                "attention_mask":[]
            }

        # output需要计算loss, 使用原始的token_id
        label_ids+=output_ids

        #attention_mask = [1] * len(input_ids+label_ids)
        feature = {
            "in_text": prompt_text + input_text,
            "out_text": output_text,
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask":[1]*len(input_ids)
        }
        return feature
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index:int):
        return self.dataset[index]

"""
将List[Dict]转为Dict[str, Tensor]
"""
def my_batch_padding_collator(examples: List[Dict[str, Any]], tokenizer, padding_side='left' , max_seq_len=1024) -> Dict[str, torch.Tensor]:
    """
        将List[Dict[str, Any]] 进行padding后转成Dict[str, Tensor]  = { 'labels':Tensor[...], 'attention_mask':Tensor[...], 'input_ids':Tensor[...]}
        1. 将batch list中的多条样本变为一个batch中的单条样本
        2. 对一个batch中的样本进行left padding
    """
    max_len_in_batch = min(max_seq_len, max([len(x["input_ids"]) for x in examples]))
    padded_output = {}

    for example in examples:
        for key, value in example.items():
            if key == "labels":
                pad_id = IGNORE_TOKEN_ID
            elif key=="attention_mask":
                pad_id = 0
            else:  # input token ids
                pad_id = tokenizer.pad_token_id
            # 截断
            value = value[:max_len_in_batch]
            # padding
            to_pad_ids = [pad_id]*(max_len_in_batch-len(value))
            if padding_side == "left": # casual_attention, 从左往右padding
                padded_value = to_pad_ids + value
            else:
                padded_value = value + to_pad_ids
            padded_output.setdefault(key, []).append(padded_value)

    # 转为tensor_ids
    padded_tensor = {k:torch.LongTensor(v) for k,v in padded_output.items()} # 均为torch.int64
    #print(f"{padded_tensor=}")
    return padded_tensor     



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    # config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,)
    # origin_context_length = getattr(config, "max_position_embeddings", None)
    # if origin_context_length and training_args.model_max_length > origin_context_length:
    #     scaling_factor = float(math.ceil(training_args.model_max_length / origin_context_length))
    #     config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # config.use_cache = False # 不使用KV缓存

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_prefix_space = False  

    # Making sure the tokenizer works before loading the model.
    #print(tokenizer(["This is a test", "secondary"], padding=True))
    #print(tokenizer.apply_chat_template([{"role": "user", "content": "This is a test"}]))

    # Load model and tokenizer
    #config.medusa_num_heads=training_args.medusa_num_heads
    #config.medusa_num_layers=training_args.medusa_num_layers
    # medusa_model:MedusaModel = MedusaModel.from_pretrained(
    #     pretrained_model_name_or_path=model_args.model_name_or_path,
    #     medusa_num_heads=training_args.medusa_num_heads,
    #     medusa_num_layers=training_args.medusa_num_layers,
    #     cache_dir=training_args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    # )
    base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path,)
    medusa_config = MedusaConfig(medusa_num_heads=2, medusa_num_layers=1, base_model_name_or_path=model_args.model_name_or_path,)
    medusa_model = MedusaModel(base_model=base_model, config=medusa_config)
    print(f"inited model:{medusa_model=}")

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}_medusa_mlp_{model_args.model_name_or_path.split('/')[-1]}_medusa_{training_args.medusa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    train_dataset = MySFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_seq_length=training_args.model_max_length,
        hint=True
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = MySFTDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=training_args.model_max_length,
            hint=False
        )

    # 在创建train_dataset后添加
    print("验证训练数据...")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample
        assert len(sample["input_ids"]) > 0
        assert len(sample["labels"]) > 0
    print("数据验证通过")

    # Generate Medusa config for pushing to HF hub
    # medusa_config = MedusaConfig(
    #     medusa_num_heads=training_args.medusa_num_heads,
    #     medusa_num_layers=training_args.medusa_num_layers,
    #     base_model_name_or_path=model_args.model_name_or_path,
    #     version="2"
    # )

    # # Save Medusa config
    # medusa_config.save_pretrained(training_args.output_dir)

    # Start trainner
    training_args.remove_unused_columns = False # 要设置为false,否则transformers会将labels列删除导致报错
    trainer = CustomizedTrainer(
        model=medusa_model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda x: my_batch_padding_collator(x, tokenizer, max_seq_len=training_args.model_max_length), 
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    #medusa_model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Save MedusaHead seperately
    if hasattr(medusa_model, "module"):
        lm_head = medusa_model.module.medusa_head
    else:
        lm_head = medusa_model.medusa_head

    # import deepspeed
    # with deepspeed.zero.GatheredParameters(lm_head.parameters()):
    #     state_dict = lm_head.state_dict()
    
    state_dict = trainer.accelerator.get_state_dict(lm_head)

    # Save Medusa heads
    if local_rank == 0:
        # Modify the tokenizer internal state before saving.
        tokenizer.encode("Test", truncation=None, padding="do_not_pad")
        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
        )

        print(f"save medusa head done, path:{training_args.output_dir}")


if __name__ == "__main__":
    train()
