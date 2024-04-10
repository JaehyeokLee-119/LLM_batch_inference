from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class CLM_wrapper:    
    def __init__(self, model_name, pretrained_model_name_or_path, load_model_weight_dir=None, model_parallel=False, use_lora=False, model_precision='bf16', hf_home=None):
        device_map = None if model_parallel is False else 'auto'
        print(f'load_model_weight_dir: {load_model_weight_dir}')
        
        def load_model_with_precision(pretrained_model_name_or_path, device_map, model_precision):
            if model_precision == 'bf16':
                return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map, cache_dir=hf_home)
            elif model_precision == 'fp16':
                return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, device_map=device_map, cache_dir=hf_home)
            elif model_precision == '8-bit':
                if pretrained_model_name_or_path.startswith('mistralai'):
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, load_in_8bit=True, device_map=device_map, cache_dir=hf_home)
                else:
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.int8, device_map=device_map, cache_dir=hf_home)
            elif model_precision == '4-bit':
                if pretrained_model_name_or_path.startswith('mistralai'):
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, load_in_4bit=True, device_map=device_map, cache_dir=hf_home)
                else:
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32, device_map=device_map, cache_dir=hf_home)
            else:
                raise ValueError(f"Invalid model_precision: {model_precision}")
                    
        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path) #use_auth_token=API_KEY
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        # setup model
        self.model = load_model_with_precision(pretrained_model_name_or_path, device_map, model_precision)
        
        if device_map is None:
            self.model.cuda()
            
        self.model_name = model_name
        

    def collate_fn_nolabel(self, inputs):
        # inputs is a list of input (str)
        inputs = [x.strip() for x in inputs]
        input_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")
        return input_dict


    def predict(self, inputs, bsz, eos_token_id, do_sample, num_beams=None, top_p=None, num_return_sequences=1, max_new_tokens=600):
        if do_sample:
            assert (top_p is not None) and (num_beams is None)
        else:
            assert (num_beams is not None) and (top_p is None)
        if num_return_sequences > 1:
            assert do_sample
            
        self.model.eval()
        outputs = []
        
        dataloader = DataLoader(inputs, batch_size=bsz, collate_fn=self.collate_fn_nolabel)
        for batch in tqdm(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            if 'token_type_ids' in batch:
                del batch['token_type_ids']
            with torch.no_grad():
                if do_sample: # sampling
                    if self.model_name.startswith('mixtral'):
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, num_return_sequences=num_return_sequences,
                                                        early_stopping=True, eos_token_id=eos_token_id, pad_token_id=2)
                    else:
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, num_return_sequences=num_return_sequences,
                                                        early_stopping=True, eos_token_id=eos_token_id)
                    
                else: # greedy/beam search
                    if self.model_name.startswith('mixtral'):
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                                        early_stopping=True, eos_token_id=eos_token_id, pad_token_id=2)
                    else:
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                                        early_stopping=True, eos_token_id=eos_token_id)
            assert len(input_outputs) == len(batch['input_ids']) * num_return_sequences
            # duplicate batch['input_ids'] by num_return_sequences number of times (to match input_outputs)
            batch_input_ids = torch.stack([batch['input_ids'][ex_idx] for ex_idx in range(len(batch['input_ids'])) for _ in range(num_return_sequences)])
            assert len(batch_input_ids) == len(input_outputs)
            assert torch.all(input_outputs[:, : batch_input_ids.shape[1]] == batch_input_ids)
            batch_outputs = input_outputs[:, batch_input_ids.shape[1]:].cpu().tolist()
            for output in batch_outputs:
                if eos_token_id not in output:
                    outputs.append(output)
                else:
                    outputs.append(output[: output.index(eos_token_id)])
            
        output_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert len(output_texts) == len(inputs) * num_return_sequences
        output_texts = [text.strip() for text in output_texts]
        if num_return_sequences == 1:
            return output_texts
        else:
            assert len(output_texts) % num_return_sequences == 0
            output_texts = [output_texts[pos: pos + num_return_sequences] for pos in range(0, len(output_texts), num_return_sequences)]
            assert len(output_texts) == len(inputs)
            return output_texts