#!/usr/bin/env python3
"""
multiturn_dialogue_generator.py

Generates multi-turn dialogs between a teacher and student model.

Usage example:
python multiturn_dialogue_generator.py \
  --teacher_model meta-llama/Llama-3.2-3B-Instruct \
  --student_model babylm-seqlen/opt-1024-warmup-v2 \
  --student_tokenizer facebook/opt-125m \
  --num_turns 6 \
  --max_length 100 \
  --num_samples 10 \
  --output_dir ./multiturn_dialogues
"""

import argparse
import random
import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------
# Conversation starters
# --------------------------
STARTERS = [
    "Have you been on any trips recently? Where did you go, and did anything interesting happen there?",
    "What kind of music do you usually listen to? Do you have a favorite artist or concert experience you remember?",
    "Do you enjoy cooking at home? What's the best meal you've made recently, or do you prefer eating out?",
    "Do you have any pets? How long have you had them, and what do you like most about them?",
    "Do you play any sports or keep active? Have you joined any teams or tried something new lately?",
    "What's the weather usually like where you live? Does it affect your plans or the way you spend your weekends?",
    "Have you watched any shows or movies recently? Did you enjoy them, and would you recommend them to others?",
    "How's work going these days? Have you faced any interesting challenges or had any funny moments?",
    "Do you have any hobbies you like to spend time on? How did you get into them, and what keeps you interested?",
    "Do you celebrate any holidays with your family? Are there any special traditions or funny stories from past celebrations?"
]


# --------------------------
# Model + tokenizer loading with tokenizer override and fallbacks
# --------------------------
def _ensure_tokenizer_pad_token(tokenizer):
    # make sure there's a pad token (some tokenizers don't set one)
    if getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            # best-effort fallback: set pad_token_id to eos_token_id if available
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id


def load_model_and_tokenizer(model_name, tokenizer_name=None, device=None, role_name="model"):
    """
    Load model and tokenizer. If tokenizer_name is None, try loading tokenizer from model_name.
    If tokenizer load fails and tokenizer_name was provided, try alternate fallbacks:
      - try loading tokenizer from model_name (if tokenizer_name failed),
      - if that fails, raise a clear error with suggestions.
    Returns: (used_tokenizer_name, tokenizer, model, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {role_name} model weights: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # determine tokenizer name priority list
    tokenizer_candidates = []
    if tokenizer_name:
        tokenizer_candidates.append(tokenizer_name)
    tokenizer_candidates.append(model_name)

    tokenizer = None
    used_tokenizer_name = None
    last_exc = None

    for cand in tokenizer_candidates:
        try:
            print(f"Attempting to load tokenizer '{cand}' for {role_name}...")
            tokenizer = AutoTokenizer.from_pretrained(cand)
            used_tokenizer_name = cand
            _ensure_tokenizer_pad_token(tokenizer)
            break
        except Exception as e:
            print(f"  -> failed to load tokenizer '{cand}': {e}")
            last_exc = e
            tokenizer = None
            used_tokenizer_name = None
            continue

    if tokenizer is None:
        # final attempt: try a very common compatible tokenizer families (best-effort)
        fallback_candidates = ["gpt2", "facebook/opt-125m", "EleutherAI/gpt-neo-125M"]
        for cand in fallback_candidates:
            try:
                print(f"Attempting fallback tokenizer '{cand}' for {role_name}...")
                tokenizer = AutoTokenizer.from_pretrained(cand)
                used_tokenizer_name = cand
                _ensure_tokenizer_pad_token(tokenizer)
                print(f"Using fallback tokenizer '{cand}' for {role_name}.")
                last_exc = None
                break
            except Exception as e:
                print(f"  -> fallback '{cand}' failed: {e}")
                last_exc = e
                tokenizer = None

    if tokenizer is None:
        print("\nCould not load a tokenizer for model '{}'.\n"
              "Suggestions:\n"
              " - If the model was trained with a custom tokenizer, pass --{}_tokenizer <tokenizer_path_or_hf_id>\n"
              " - If the tokenizer files are local, pass the local directory path to --{}_tokenizer\n"
              " - Ensure 'protobuf' and 'tokenizers' packages are installed in your environment\n"
              .format(model_name, role_name, role_name))
        raise RuntimeError(f"Tokenizer load failed for candidates. Last exception: {last_exc}")

    return used_tokenizer_name, tokenizer, model, device


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
def generate_teacher_response(prompt, tokenizer, model, device, max_length=100):
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


def generate_student_response(prompt, tokenizer, model, device, max_length=100, max_retries=3):
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


# --- Main multiturn generator ---
def generate_multiturn_samples(
    num_samples, num_turns, teacher_tokenizer, teacher_model, teacher_device,
    student_tokenizer, student_model, student_device,
    teacher_model_name, student_model_name,
    max_length
):
    samples = []
    for i in range(num_samples):
        starter = random.choice(STARTERS)
        conversation = starter  # Start with starter as metaprompt

        transcript_lines = [f"[Starter]: {starter}"]
        student_turns = []
        teacher_turns = []

        pairs = num_turns // 2
        for turn in range(pairs):  # student-teacher pairs
            # Student turn
            student_text = generate_student_response(
                conversation, student_tokenizer, student_model, student_device, max_length=max_length
            )
            student_turns.append({"turn_index": 2*turn + 1, "text": student_text})
            transcript_lines.append(f"[Student]: {student_text}")
            conversation += f" {student_text}"

            # Teacher turn
            teacher_text = generate_teacher_response(
                conversation, teacher_tokenizer, teacher_model, teacher_device, max_length=max_length
            )
            teacher_turns.append({"turn_index": 2*turn + 2, "text": teacher_text})
            transcript_lines.append(f"[Teacher]: {teacher_text}")
            conversation += f" {teacher_text}"

        transcript = "\n".join(transcript_lines)

        samples.append({
            "teacher_model": teacher_model_name,
            "student_model": student_model_name,
            "id": str(i + 1),
            "STARTER": starter,
            "level": "Default (No Age Meta-Prompt)",
            "text": transcript,
            "student_turns": student_turns,
            "teacher_turns": teacher_turns
        })

    return samples


# --------------------------
# Command-line interface
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate multi-turn dialogues between teacher and student models.")
    p.add_argument("--teacher_model", type=str, required=True, help="Hugging Face model id for the teacher")
    p.add_argument("--teacher_tokenizer", type=str, default=None, help="Optional override tokenizer for the teacher")
    p.add_argument("--student_model", type=str, required=True, help="Hugging Face model id for the student")
    p.add_argument("--student_tokenizer", type=str, default=None, help="Optional override tokenizer for the student")
    p.add_argument("--num_turns", type=int, default=6, help="Total number of turns (student+teacher). Will be bumped to even if odd.")
    p.add_argument("--max_length", type=int, default=100, help="Max generation length per reply (tokens)")
    p.add_argument("--num_samples", type=int, default=10, help="How many conversations to generate")
    p.add_argument("--output_dir", type=str, default="./multiturn_dialogues", help="Directory to save JSON dialogues")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default=None, help="Device to place models on (e.g. cpu, cuda). Auto-detect if not set")
    return p.parse_args()


def _sanitize_name_for_filename(name: str) -> str:
    # replace slashes with hyphens and remove spaces
    safe = name.replace("/", "-").replace(" ", "_")
    # keep only filename-safe characters
    safe = re.sub(r"[^A-Za-z0-9_.-]", "", safe)
    return safe


def main():
    args = parse_args()
    random.seed(args.seed)

    # Ensure even number of turns
    num_turns = args.num_turns
    if num_turns % 2 != 0:
        print(f"Warning: num_turns={num_turns} is odd — incrementing to {num_turns+1} to keep student-teacher pairs.")
        num_turns += 1

    # Load teacher (allow tokenizer override)
    teacher_used_tok_name, teacher_tokenizer, teacher_model, teacher_device = load_model_and_tokenizer(
        args.teacher_model, tokenizer_name=args.teacher_tokenizer, device=args.device, role_name="teacher"
    )

    # Load student (allow tokenizer override)
    student_used_tok_name, student_tokenizer, student_model, student_device = load_model_and_tokenizer(
        args.student_model, tokenizer_name=args.student_tokenizer, device=args.device, role_name="student"
    )

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} conversations with {num_turns} turns, max_length={args.max_length}...")

    samples = generate_multiturn_samples(
        num_samples=args.num_samples,
        num_turns=num_turns,
        teacher_tokenizer=teacher_tokenizer,
        teacher_model=teacher_model,
        teacher_device=teacher_device,
        student_tokenizer=student_tokenizer,
        student_model=student_model,
        student_device=student_device,
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        max_length=args.max_length
    )

    safe_teacher = _sanitize_name_for_filename(args.teacher_model)
    safe_student = _sanitize_name_for_filename(args.student_model)

    output_file = os.path.join(
        args.output_dir,
        f"dialogues_{num_turns}_turns_len{args.max_length}_{safe_teacher}_{safe_student}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} dialogues to {output_file}")


if __name__ == "__main__":
    main()
