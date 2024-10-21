from typing import Optional, List
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
  
  def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
    self.model = model
    self.tokenizer = tokenizer
    self.args = model_args
  
  @staticmethod
  def build(checkpoints_dir: str, tokenizer_path: str , load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
    prev_time = time.time()
    if load_model:
      checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
      assert len(checkpoints) > 0, "No checkpoints files found"
      chk_path = checkpoints[0]
      print(f'Loading checkpoint {chk_path}')
      checkpoint = torch.load(chk_path, map_location="cpu", weights_only=True)
      print(f'Loaded checkpoint in {(time.time() - prev_time):.2f}')
      prev_time = time.time()
      
    with open(Path(checkpoints_dir) / "params.json", "r") as f:
      params = json.loads(f.read())
    
    model_args: ModelArgs = ModelArgs(
      max_seq_len=max_seq_len,
      max_batch_size=max_batch_size,
      device=device,
      **params
    )  
    
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    model_args.vocab_size = tokenizer.vocab_size()
    
    if device == "cuda":
      torch.torch.set_default_tensor_type(torch.HalfTensor)
    else:
      torch.torch.set_default_tensor_type(torch.BFloat16Tensor)
    
    model = Transformer(model_args).to(device)
    
    if load_model:
      del checkpoint['rope.freqs']
      model.load_state_dict(checkpoint, strict=True)
      print(f'Loaded state dict in {(time.time() - prev_time):.2f}s')
    
    # # If load_model is True, load weights lazily to avoid memory issues
    # if load_model:
    #   print("Loading model weights lazily...")
    #   for key in tqdm(checkpoint.keys(), desc="Loading weights"):
    #     model.state_dict()[key].copy_(checkpoint[key].half() if device == 'cuda' else checkpoint[key].bfloat16())
    #   print(f'Loaded state dict in {(time.time() - prev_time):.2f}s')
      
    return LLaMA(model, tokenizer, model_args)

  def text_completion(self, prompts: List[str], temprature: float=0.6, top_p: float=0.8, max_gen_len: Optional[int]=None):
    if max_gen_len is None:
      max_gen_len = self.args.max_seq_len - 1
    # Convert each prompt into tokens
    prompt_tokens = [self.tokenizer.Encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
    # Make sure the batch size is not too large
    batch_size = len(prompt_tokens)
    assert batch_size <= self.args.max_batch_size
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    # Make sure the prompt length is not larger than the maximum seq len
    assert max_prompt_len <= self.args.max_seq_len
    total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
    
    # Create the list that will contain the generated tokens, along with the intial prompt tokens
    pad_id = self.tokenizer.pad_id()
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
      # Populate the initial tokens with the prompt tokens
      tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    eos_reached = torch.tensor([False] * batch_size, device=device)
    # eos_reached = torch.full((batch_size,), False, dtype=torch.bool, device=device)
    prompt_tokens_mask = (tokens != pad_id).to(torch.bool) # True if the token is a prompt token, False otherwise
    for cur_pos in tqdm(range(1, total_len), desc="Genearating tokens"):
      with torch.no_grad():
        logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
      if temprature > 0:
        # The temprature is applied BEFORE the softmax
        # print(f"{logits.shape=}")
        probs = torch.softmax(logits[:, -1] / temprature, dim=-1)
        next_token = self._sample_top_p(probs, top_p)
      else:
        # Greedily select the token with the maximum probability
        next_token = torch.argmax(logits[:, -1], dim=-1)
      
      next_token = next_token.reshape(-1)
      # Only replace the token if it is a padding token
      next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token) # if first param is True return the second param. else 3th.
      tokens[:, cur_pos] = next_token
      # EOS is reached onyl of we found an EOS token for padding position
      eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
      # eos_reached = eos_reached.to(torch.bool)
      if all(eos_reached):
        break
      
    
    out_tokens = []
    out_text = []
    for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
      # Cut to the EOS token, if present
      if self.tokenizer.eos_id() in current_prompt_tokens:
        eos_index = current_prompt_tokens.index(self.tokenizer.eos_id())
        current_prompt_tokens = current_prompt_tokens[:eos_index]
      out_tokens.append(current_prompt_tokens)
      out_text.append(self.tokenizer.decode(current_prompt_tokens))
    return (out_tokens, out_text)


  def _sample_top_p(self, probs, p):
    probs_sort, probbs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # print(f'Cumulative sum of probabilities: {probs_sum=}')
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probbs_idx, -1, next_token)
    return next_token      
        

if __name__ == "__main__":
  torch.manual_seed(0)
  
  allow_cuda = True
  device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
  
  
  prompts = [
    "Simply put, the theory of relativity states that ",
    "If Google was an Italian company founded in Milan, it would",
    # Few shot promt
    """Translate English to French:
    
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>""",
    # Zero shot prompt
    """Tell me if the following person is actually Doraemon disguised as human:
    Name: Emir Akagunduz
    Decision: 
    """
  ]
  
  model = LLaMA.build(
    checkpoints_dir='llama-2-7b/',
    tokenizer_path='tokenizer.model',
    load_model=True,
    max_seq_len=1024,
    max_batch_size=len(prompts),
    device=device
  )

  # Inference the model
  out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
  assert len(out_text) == len(prompts)
  for i in range(len(out_text)):
    print(f'{out_text[i]}')
    print('-' * 50)