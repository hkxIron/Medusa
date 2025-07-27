import traceback
import gradio as gr
import time
import torch
from medusa.model.medusa_model import MedusaModel
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM

from medusa.model.modeling_llama_kv import LlamaForCausalLM

# Global variables
chat_history = ""
model = None
tokenizer = None
conv = None


def load_model_function(model_name:str, load_in_8bit=False, load_in_4bit=False):
    #model_name = model_name or "FasterDecoding/medusa-vicuna-7b-v1.3"
    #model_name = model_name or "/home/hkx/data/work/open/Medusa/llama_medusa_output_medusa_mlp_TinyStories-LLaMA2-20M-256h-4l-GQA_medusa_2_lr_0.001_layers_1"
    base_model_path = "/home/hkx/data/work/hf_data_and_model/models/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA"
    global model, tokenizer, conv

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            # torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            #device_map="auto",
            # load_in_8bit=load_in_8bit,
            # load_in_4bit=load_in_4bit
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        conv = get_conversation_template("vicuna")
        return "Model loaded successfully!"
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        return "Error loading the model. Please check the model name and try again."


def reset_conversation():
    """
    Reset the global conversation and chat history
    """
    global conv, chat_history
    conv = get_conversation_template("vicuna")
    chat_history = ""


def model_chat_interface(user_input, temperature, max_steps, no_history):
    global model, tokenizer, conv, chat_history

    # Reset the conversation if no_history is checked
    if no_history:
        reset_conversation()

    if not model or not tokenizer:
        return "Error: Model not loaded!", chat_history

    chat_history += "\nYou: " + user_input
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], '')
    prompt = conv.get_prompt()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)

    outputs = model.generate(input_ids, temperature=temperature)
    print(outputs)
    response = ""
    for output in outputs:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #response = output['text']
        yield response, chat_history
        time.sleep(0.01)

    chat_history += "\nChatBot: " + response.strip()

    return response, chat_history


if __name__ == "__main__":
    load_model_interface = gr.Interface(
        load_model_function,
        [
            gr.components.Textbox(placeholder="FasterDecoding/medusa-vicuna-7b-v1.3", label="Model Name"),
            gr.components.Checkbox(label="Use 8-bit Quantization"),
            gr.components.Checkbox(label="Use 4-bit Quantization"),
        ],
        gr.components.Textbox(label="Model Load Status", type="text"),
        description="Load Medusa Model",
        title="Medusa Model Loader",
        live=False,
        api_name="load_model"
    )

    # Chat Interface
    chat_interface = gr.Interface(
        model_chat_interface,
        [
            gr.components.Textbox(placeholder="你好", label="User Input"),
            gr.components.Slider(minimum=0, maximum=1.5, label="Temperature"),
            gr.components.Slider(minimum=50, maximum=1000, label="Max Steps"),
            gr.components.Checkbox(label="No History"),
        ],
        [
            gr.components.Textbox(label="Medusa's Response", type="text"),
            gr.components.Textbox(label="Chat History", type="text")
        ],
        live=False,
        description="Chat with Medusa",
        title="Medusa Chatbox",
        api_name="chat"
    )

    # Combine the interfaces in a TabbedInterface
    combined_interface = gr.TabbedInterface([load_model_interface, chat_interface], ["Load Model", "Chat"])

    # Launch the combined interface
    #combined_interface.queue().launch(share=True)
    combined_interface.queue().launch(share=False)
