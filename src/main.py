import os
# os.environ["HF_HOME"] = "/hdd/hjl8708/saved_models"
# os.environ["TRANSFORMERS_CACHE"] = "/hdd/hjl8708/saved_models"

from generation import predict_answer
import fire 
import json

def start(
    test_data_path=f'../data/input.json', 
    output_filename=f"../results/output.json",
    batchsize=2, 
    model_name='mixtral-instruct-8x7b',
    pretrained_model_name_or_path='mistralai/Mixtral-8x7B-Instruct-v0.1',
    model_precision="bf16",
    max_new_tokens=150,
):
    
    data = predict_answer(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                     bsz=batchsize, data_fname=test_data_path,
                     max_new_tokens=max_new_tokens, model_precision=model_precision)

    json.dump(data, open(
        f'{output_filename}', 'w'), indent=4)

if __name__ == '__main__':
    fire.Fire(start)