#!/usr/bin/env python3
import argparse
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------- Loaders --------

def load_teacher_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    return tokenizer, model, model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_student_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


# -------- Generation helpers --------

# --------------------------
# Cleaning utilities
# --------------------------
def clean_response(response: str) -> str:
    lines = response.splitlines()
    filtered_lines = [line for line in lines if "[Teacher]" not in line and "[Student]" not in line]
    return " ".join(filtered_lines).strip()


def get_banned_tokens(tokenizer):
    banned_strings = [
        ".", ",", "!", "?", ";", ":", "'", "\"", "-", "–", "—", "(", ")",
        "[", "]", "{", "}", "…", "\n", "\t", "\r", "/", "\\", "*", "_",
        "[Teacher]", "[Student]"
    ]
    banned_ids = []
    for s in banned_strings:
        try:
            token_ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            banned_ids.extend(token_ids)
        except Exception:
            # ignore tokens that tokenizer cannot encode
            continue
    return [[tid] for tid in set(banned_ids)]


def clean_response_final(response: str) -> str:
    # remove role tokens, punctuation, keep alphanumerics and spaces
    response = re.sub(r"\[Teacher\]|\[Student\]", " ", response)
    response = re.sub(r"[^\w\s]", "", response)
    response = re.sub(r"\s+", " ", response)
    return response.strip()


# --------------------------
# Generation functions
# --------------------------
def generate_teacher_response(prompt, tokenizer, model, device, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + max_length,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_response(response)


def generate_student_response(prompt, tokenizer, model, device, max_length=50, max_retries=3):
    bad_words_ids = get_banned_tokens(tokenizer)

    for attempt in range(max_retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids
        )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        response = clean_response_final(response)

        if any(c.isalnum() for c in response):
            return response

    return ""

# -------- Chat loop --------

def chat_teacher_student(teacher_tokenizer, teacher_model, teacher_device,
                         student_tokenizer, student_model, student_device,
                         meta_prompt, num_turns=5):
    transcript_lines = []

    teacher_reply = generate_teacher_response(meta_prompt, teacher_tokenizer, teacher_model, teacher_device)
    transcript_lines.append(f"Teacher: {teacher_reply}")

    dialogue = f"[Teacher]: {teacher_reply}"

    for _ in range(num_turns):
        student_input = dialogue + "\n[Student]:"
        student_reply = generate_student_response(student_input, student_tokenizer, student_model, student_device)
        transcript_lines.append(f"Student: {student_reply}")
        dialogue += f"\n[Student]: {student_reply}"

        teacher_input = dialogue + "\n[Teacher]:"
        teacher_reply = generate_teacher_response(teacher_input, teacher_tokenizer, teacher_model, teacher_device)
        transcript_lines.append(f"Teacher: {teacher_reply}")
        dialogue += f"\n[Teacher]: {teacher_reply}"

    return transcript_lines


# -------- Main --------

def main():
    parser = argparse.ArgumentParser(description="Teacher-Student Chat Simulation")
    parser.add_argument("--student_model", type=str, required=True,
                        help="Hugging Face model name or path for the student model")
    parser.add_argument("--num_turns", type=int, default=2,
                        help="Number of conversation turns per dialogue")
    parser.add_argument("--num_dialogues", type=int, default=5,
                        help="Number of dialogues to generate")
    parser.add_argument("--output_file", type=str, default="./dialogues/dialogues.txt",
                        help="Path to save the dialogues")
    args = parser.parse_args()

    # Prepare output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_tokenizer, teacher_model, teacher_device = load_teacher_model()
    student_tokenizer, student_model, student_device = load_student_model(args.student_model)

    meta_prompt = (
        "You are an expert dialogue assistant initiating a conversation with a child language model "
        "that has the linguistic abilities of a competent learner.\n\n"
        "Generate the first message to begin the dialogue. The message should:\n"
        "- Be concise: 1 to 2 short sentences, no more than 30 words total.\n"
        "- Use a friendly, positive, age-appropriate tone.\n"
        "- Mention or draw attention to a few familiar objects (e.g., ball, cup, dog, book).\n"
        "- Avoid abstract or complex concepts.\n\n"
        "Tone:\n"
        "- Conversational and nurturing\n"
        "- Suitable for a preverbal or babbling child\n\n"
        "Only output the first utterance to the child model to start the conversation."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(1, args.num_dialogues + 1):
            transcript = chat_teacher_student(
                teacher_tokenizer, teacher_model, teacher_device,
                student_tokenizer, student_model, student_device,
                meta_prompt, num_turns=args.num_turns
            )
            f.write(f"Dialogue {i}\n")
            f.write("\n".join(transcript))
            f.write("\n\n")  # Blank line between dialogues

    print(f"✅ Saved {args.num_dialogues} dialogues to {output_path}")


if __name__ == "__main__":
    main()

