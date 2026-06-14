"""Smoke test: load Gemma-4-E4B-it, locate yes/no tokens, extract answer-slot logits, benchmark."""
import time, sys
import torch

MODEL_ID = "google/gemma-4-E4B-it"

def load_model():
    from transformers import AutoTokenizer, AutoProcessor
    tok = None
    proc = None
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        tok = getattr(proc, "tokenizer", None)
    except Exception as e:
        print("AutoProcessor failed:", e)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)

    model = None
    last_err = None
    for cls_name in ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModel"]:
        try:
            import transformers
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
            print(f"Loaded with {cls_name}: {type(model).__name__}")
            break
        except Exception as e:
            last_err = e
            print(f"{cls_name} failed: {repr(e)[:200]}")
    if model is None:
        raise RuntimeError(f"Could not load model. Last error: {last_err}")
    model.eval()
    return tok, proc, model


def build_prompt(tok, text, question):
    msgs = [{"role": "user", "content": f"{text}\n\nQuestion: {question}\nAnswer with only 'yes' or 'no'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def yes_no_ids(tok):
    def ids_for(words):
        s = set()
        for w in words:
            for variant in (w, " " + w):
                enc = tok.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    s.add(enc[0])
        return sorted(s)
    yes = ids_for(["yes", "Yes", "YES"])
    no = ids_for(["no", "No", "NO"])
    return yes, no


def main():
    t0 = time.time()
    tok, proc, model = load_model()
    print(f"load time: {time.time()-t0:.1f}s")
    print("device:", next(model.parameters()).device, "dtype:", next(model.parameters()).dtype)

    yes_ids, no_ids = yes_no_ids(tok)
    print("yes ids:", yes_ids, [tok.decode([i]) for i in yes_ids])
    print("no  ids:", no_ids, [tok.decode([i]) for i in no_ids])

    text = "The home team scored in the final minute to win the championship game 3 to 2."
    for q in ["Does this text smell smoky?", "Is this text purple?", "Does this text taste salty?", "Is this text loud?"]:
        prompt = build_prompt(tok, text, q)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        topk = torch.topk(probs, 5)
        ly = torch.logsumexp(logits[yes_ids], 0).item()
        ln = torch.logsumexp(logits[no_ids], 0).item()
        top_str = ", ".join(f"{tok.decode([i])!r}:{p:.3f}" for p, i in zip(topk.values.tolist(), topk.indices.tolist()))
        print(f"\nQ: {q}\n  top: {top_str}\n  logit(yes)-logit(no) = {ly-ln:+.3f}")

    # Benchmark batched throughput
    print("\n--- benchmark ---")
    prompts = [build_prompt(tok, text, "Does this text smell smoky?")] * 32
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    print("seq len:", enc.input_ids.shape)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        for _ in range(3):
            model(**enc)
    torch.cuda.synchronize()
    dt = (time.time()-t)/3
    print(f"batch=32 forward: {dt*1000:.0f} ms -> {32/dt:.1f} seq/s")
    print(f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

if __name__ == "__main__":
    main()
