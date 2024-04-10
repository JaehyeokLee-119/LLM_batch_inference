import json
import numpy as np
from model_wrapper import CLM_wrapper
from transformers import AutoTokenizer

def predict_answer(model_name, pretrained_model_name_or_path,
                     data_fname,
                     model_parallel=True, bsz=16, do_sample=False, 
                     num_beams=1, top_p=None, num_return_sequences=1,
                     max_new_tokens=600, model_precision='bf16'):
    # load data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    eos_token_id = tokenizer.eos_token_id
    
    data = json.load(open(data_fname))    
    eval_inputs = [d['input'] for d in data]
    
    # eval_inputs의 길이 statistics
    eval_inputs_lens = np.array([len(input1) for input1 in eval_inputs])
    print(f"Max input length: {np.max(eval_inputs_lens)}, Min input length: {np.min(eval_inputs_lens)}, Mean input length: {np.mean(eval_inputs_lens)}")
    
    clm_wrapper = CLM_wrapper(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                                model_parallel=model_parallel, use_lora=False, model_precision=model_precision)
    
    # << Length test >>: eval_inputs 중에서 가장 긴 것들을 bsz 개수만큼 뽑아서 터지는지 보기
    eval_inputs_long = eval_inputs.copy()
    eval_inputs_long.sort(key=lambda x: len(x), reverse=True)
    eval_inputs_long = eval_inputs_long[:bsz]
    
    # eval_inputs_long에 대해서 predict 시킨다
    try:
        long_test_outputs = clm_wrapper.predict(inputs=eval_inputs_long, eos_token_id=eos_token_id, bsz=bsz,
                                            do_sample=do_sample, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences,
                                            max_new_tokens=max_new_tokens)
        print("Longest input test passed!")
    except:
        print(f"Error occurred! too long (not enough GPU memory)")
        return None
    
    # Prediction
    output_texts = clm_wrapper.predict(inputs=eval_inputs, eos_token_id=eos_token_id, bsz=bsz,
                                       do_sample=do_sample, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences,
                                       max_new_tokens=max_new_tokens)
    
    for i in range(len(output_texts)):
        data[i]['output'] = output_texts[i]
    
    return data