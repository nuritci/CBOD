import os
import json
import argparse
import re
import csv
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def extract_answer(generated_text):
    # Try a direct "Answer: X" pattern first
    match = re.search(r'Answer:\s*([A-D])', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Otherwise pick the last standalone letter A–D
    letters = re.findall(r'\b([A-D])\b', generated_text.upper())
    return letters[-1].upper() if letters else None

def format_prompt(question, choices):
    choice_letters = ['A', 'B', 'C', 'D'][:len(choices)]
    prompt = (
        "You are a helpful assistant. Read the question and the provided options. "
        "Select the best answer from the given options. Respond with just the letter of the correct choice.\n\n"
    )
    prompt += f"Question: {question}\n"
    for i, option in enumerate(choices):
        prompt += f"{choice_letters[i]}. {option}\n"
    prompt += "Answer:"
    return prompt

def batch_predict(model, tokenizer, device, prompts, batch_size=1):
    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Running inference in batches"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in decoded:
            predictions.append(extract_answer(text))
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model identifier (Hugging Face Hub name or local path)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--input_file", "-i", type=str, default="rephrased_mmlu_test.csv",
                        help="Path to the input CSV file")
    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batch_size
    input_filename = args.input_file

    # --- Load first 1000 rows from CSV ---
    data = []
    with open(input_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # parse the stringified list in 'choices' column
            choices = ast.literal_eval(row["choices"])
            data.append({
                "subject": row.get("subject"),
                "original_question": row.get("original_question"),
                "rephrased_question": row.get("rephrased_question"),
                "choices": choices,
                "answer": int(row.get("answer"))
            })

    # Set up model + tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).to(device)
    model.eval()

    # Build prompts
    orig_prompts = [format_prompt(d["original_question"], d["choices"]) for d in data]
    reph_prompts = [format_prompt(d["rephrased_question"], d["choices"]) for d in data]

    # Inference
    orig_preds = batch_predict(model, tokenizer, device, orig_prompts, batch_size)
    reph_preds = batch_predict(model, tokenizer, device, reph_prompts, batch_size)

    # Collate results
    results = []
    for entry, op, rp in zip(data, orig_preds, reph_preds):
        correct_idx = entry["answer"]
        correct_letter = ['A', 'B', 'C', 'D'][correct_idx]
        results.append({
            "subject": entry["subject"],
            "original_question": entry["original_question"],
            "rephrased_question": entry["rephrased_question"],
            "choices": entry["choices"],
            "answer": correct_idx,
            "original_question_is_correct": (op == correct_letter),
            "rephrased_question_is_correct": (rp == correct_letter)
        })

    # Write out
    out_name = f"results_{model_name.replace('/', '_')}.json"
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"Done! Results saved to {out_name}")

if __name__ == "__main__":
    main()
