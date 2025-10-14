import os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def load_model(model_id="google/flan-t5-base"):
    device = "mps" if torch.backends.mps.is_available() \
        else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device in ("mps","cuda") else torch.float32
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    return tok, mod, device

def call_model(tok, mod, device, prompt, max_new_tokens=120):
    x = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.inference_mode():
        y = mod.generate(
            **x,
            max_new_tokens=max_new_tokens,
            min_new_tokens=40,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(y[0], skip_special_tokens=True).strip()
