import torch
import transformers
import datasets
import onnx
from ppq.samples.LLM.tasks import WikiEvaluator
import threading
import numpy as np

done = False

def checkGpuMemory():
    file = open('gpu_mem.txt', 'w')
    max_mem_allocated = np.zeros(8)
    max_mem_reserved = np.zeros(8)
    while not done:
        for i in range(8):
            max_mem_allocated[i] = max(max_mem_allocated[i], torch.cuda.memory_allocated(i))
            max_mem_reserved[i] = max(max_mem_reserved[i], torch.cuda.memory_reserved(i))
            file.write('%d: %f/%f\n' % (i, max_mem_allocated[i]/1024/1024, max_mem_reserved[i]/1024/1024))
    print('done')
    print('max_mem_allocated:', max_mem_allocated/1024/1024)
    print('max_mem_reserved:', max_mem_reserved/1024/1024)
    file.close()

# gpu_thread = threading.Thread(target=checkGpuMemory)

# model_checkpoint = 'facebook/opt-350m'
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

model_checkpoint = 'huggyllama/llama-65b'
tokenizer = transformers.LlamaTokenizer.from_pretrained(model_checkpoint, use_fast=False)

CACHE_DIR = '/gptq_hub'
CFG_DEVICE = 'cuda'

model_fp16 = transformers.AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto", cache_dir=CACHE_DIR)
print('-----device map-----')
print(model_fp16.hf_device_map)
dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1',split='validation')
evaluator = WikiEvaluator(dataset, tokenizer, CFG_DEVICE, model_fp16, 100)

batch = evaluator.sample_batch()
input_ids = batch.to(CFG_DEVICE)
input_shape=input_ids.shape
input_dtype=input_ids.dtype
dummy_input = torch.zeros(size=input_shape, device=CFG_DEVICE, dtype=input_dtype)

print('-----export start-----')
# gpu_thread.start()

torch.onnx.export(
    model=model_fp16, args=dummy_input, 
    verbose=False, f='onnx.model', opset_version=11,
)
print('-----export end-----')
# done = True
# gpu_thread.join()