import argparse, ast, json, os, pandas as pd, re, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTER_MAP = ["a", "b", "c", "d"]


def extract_answer_letter(text: str):
    """
    Look for:
      Answer: a
    or a standalone 'a','b','c','d' at the end.
    """
    m = re.search(r"Answer:\s*([a-dA-D])", text)
    if m:
        return m.group(1).lower()
    m2 = re.findall(r"\b([a-dA-D])\b", text)
    return m2[-1].lower() if m2 else None


def make_prompt(question: str, choices: list[str]) -> str:
    """
    Prefix each choice with a), b), c), d).
    Ask the model to respond with exactly the letter.
    """
    prompt = (
        "You are a helpful assistant. Read the question and the options below.\n"
        "Respond with exactly one lowercase letter: a, b, c, or d—no extra text.\n\n"
        f"Question: {question.strip()}\n"
    )
    for letter, option in zip(LETTER_MAP, choices):
        prompt += f"{letter}) {option.strip()}\n"
    prompt += "\nAnswer:"
    return prompt


@torch.inference_mode()
def predict(model, tokenizer, device, prompts, batch_size):
    preds = []
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="inference"):
        batch = prompts[i : i + batch_size]
        toks = tokenizer(batch,
                         return_tensors="pt",
                         padding=True,
                         truncation=True).to(device)

        out = model.generate(**toks,
                             max_new_tokens=4,
                             do_sample=False,
                             temperature=0.0,
                             eos_token_id=tokenizer.eos_token_id)

        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for text in texts:
            letter = extract_answer_letter(text)
            if letter in LETTER_MAP:
                idx = LETTER_MAP.index(letter) + 1
            else:
                idx = None
            preds.append((letter, idx))
    return preds


def evaluate_csv(path, model_name, model, tok, device, batch_size):
    df = pd.read_csv(path)
    df["choices"] = df["choices"].apply(ast.literal_eval)

    # build prompts
    orig_prompts = [
        make_prompt(q, ch)
        for q, ch in zip(df["original_question"], df["choices"])
    ]
    reph_prompts = [
        make_prompt(q, ch)
        for q, ch in zip(df["rephrased_question"], df["choices"])
    ]

    # run
    orig_preds = predict(model, tok, device, orig_prompts, batch_size)
    reph_preds = predict(model, tok, device, reph_prompts, batch_size)

    results = []
    corr_orig = corr_reph = 0
    for row, (let1, idx1), (let2, idx2) in zip(
        df.itertuples(), orig_preds, reph_preds
    ):
        # ground‑truth index
        try:
            correct_idx = row.choices.index(str(row.answer).strip()) + 1
        except ValueError:
            correct_idx = None

        rec = {
            "model": model_name,
            "original_question": row.original_question,
            "rephrased_question": row.rephrased_question,
            "choices": row.choices,
            "correct_answer_text": row.answer,
            "predicted_letter_original": let1,
            "predicted_index_original": idx1,
            "predicted_letter_rephrased": let2,
            "predicted_index_rephrased": idx2,
            "original_is_correct": (idx1 == correct_idx),
            "rephrased_is_correct": (idx2 == correct_idx),
        }
        corr_orig += rec["original_is_correct"]
        corr_reph += rec["rephrased_is_correct"]
        results.append(rec)

    out = {
        "model": model_name,
        "total": len(results),
        "original_accuracy": corr_orig / len(results),
        "rephrased_accuracy": corr_reph / len(results),
        "results": results
    }
          
    safe_model = model_name.replace("/", "_")
    base = os.path.basename(path).replace(".csv", "")
    fn = f"results_{safe_model}_{base}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n▶ {path}: orig acc {out['original_accuracy']:.3f}, "
          f"reph acc {out['rephrased_accuracy']:.3f} → {fn}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True,
                   help="HF model name or local path")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("csvs", nargs="+", help="CSV(s) to eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    for csv in args.csvs:
        evaluate_csv(csv, args.model, mdl, tok, device, args.batch_size)


if __name__ == "__main__":
    main()
