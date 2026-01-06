import torch
from dotenv import dotenv_values
from peft import PeftModel
from transformers import AutoTokenizer
from src.fine_tuning import description_len, signal_token_id, get_bearllm, get_reference_retriever, mod_xt_for_qwen
import numpy as np
from functions.dcn import dcn
import json

env = dotenv_values()
mbhm_dataset = env['MBHM_DATASET']
qwen_weights = env['QWEN_WEIGHTS']
bearllm_weights = env['BEARLLM_WEIGHTS']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

demo_data = json.load(open(f'{mbhm_dataset}/demo_data.json'))

def create_cache():
    query_data = np.array(demo_data['vib_data'])
    if 'ref_pool' in demo_data:
        retriever = get_reference_retriever(device)
        ref_pool = np.array(demo_data['ref_pool'])
        ref_tensor = retriever.retrieve(
            torch.from_numpy(query_data).to(device),
            torch.from_numpy(ref_pool).to(device),
        )
        ref_data = ref_tensor.detach().cpu().numpy()
    else:
        ref_data = np.array(demo_data['ref_data'])
    query_data = dcn(query_data)
    ref_data = dcn(ref_data)
    rv = np.array([query_data, ref_data])
    np.save('./cache.npy', rv)


def run_demo():
    create_cache()

    place_holder_ids = torch.ones(description_len, dtype=torch.long) * signal_token_id
    text_part1, text_part2 = mod_xt_for_qwen(demo_data['instruction'])

    tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids])
    user_ids = user_ids.to(device)
    attention_mask = torch.ones_like(user_ids)
    attention_mask = attention_mask.to(device)

    model = get_bearllm(train_mode=False)
    model = PeftModel.from_pretrained(model, f'{bearllm_weights}')
    model.eval()

    output = model.generate(user_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_new_tokens=2048)
    output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    run_demo()
