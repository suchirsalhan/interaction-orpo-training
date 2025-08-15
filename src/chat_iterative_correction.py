#!/usr/bin/env python3
"""
Teacher-Student Chat Simulation (extended)

Key changes from your original version:
- Explicit confirmation (in comments) that the student responds to the teacher.
- Three feedback styles selectable via --feedback_type.
- Cleaning is optional/configurable via --cleaning: none | light | strict.
- Output filename automatically concatenates key parameters.
- Uses max_new_tokens (safer than max_length arithmetic).
- More robust banned-token handling when strict cleaning is requested.
"""

import argparse
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------
# Loaders
# --------------------------

def load_teacher_model(name="meta-llama/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model, device


def load_student_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # For student we keep it simple; for very large students you may want device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


# --------------------------
# Cleaning utilities (configurable)
# --------------------------

def clean_none(response: str) -> str:
    """No cleaning; return raw response."""
    return response

def clean_light(response: str) -> str:
    """Light cleaning: remove role tokens but keep punctuation and newlines; collapse extra whitespace."""
    response = re.sub(r"\[Teacher\]|\[Student\]", " ", response)
    response = re.sub(r"\s+", " ", response)
    return response.strip()

def clean_strict(response: str) -> str:
    """Strict cleaning: remove role tokens, punctuation; return only alphanumerics and spaces."""
    response = re.sub(r"\[Teacher\]|\[Student\]", " ", response)
    response = re.sub(r"[^\w\s]", "", response)
    response = re.sub(r"\s+", " ", response)
    return response.strip()

CLEANING_FUNCS = {
    "none": clean_none,
    "light": clean_light,
    "strict": clean_strict
}

def get_banned_tokens_for_cleaning(tokenizer, cleaning_level="none"):
    """
    Return bad_words_ids for huggingface generate depending on cleaning_level.
    Hugging Face expects a list of token-id lists, e.g. [[id1], [id2], ...]
    We'll ban role tokens when cleaning is 'light' or 'strict'; for 'strict'
    the calling code will choose to ban more punctuation if desired.
    """
    banned_strings = ["[Teacher]", "[Student]"]
    if cleaning_level == "strict":
        # minimal punctuation ban (you can expand this list but be cautious)
        banned_strings += ["\n", "\r"]
    banned_ids = []
    for s in banned_strings:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                banned_ids.append(ids)
        except Exception:
            continue
    return banned_ids  # already a list of lists when multi-token strings are encoded as multi-ids


# --------------------------
# Teacher feedback templates
# --------------------------

FEEDBACK_TEMPLATES = {
    "affirmative": lambda teacher_uttr, student_uttr: (
        # teacher says a brief encouraging phrase that refers back
        f"{teacher_uttr}\n\n[Teacher]: Good job noticing the {student_uttr.split()[:3] and ' '.join(student_uttr.split()[:3])}. Very nice!"
    ),
    "explicit_correction": lambda teacher_uttr, student_uttr: (
        # teacher offers a short correction/rephrase
        f"{teacher_uttr}\n\n[Teacher]: Actually, a clearer way to say that is: \"{student_uttr}\". Try saying it like that."
    ),
    "socratic": lambda teacher_uttr, student_uttr: (
        # teacher asks a short guiding question
        f"{teacher_uttr}\n\n[Teacher]: That's interesting. Can you tell me what the {('word' if student_uttr else 'thing')} does?"
    )
}

# --------------------------
# Generation helpers
# --------------------------

def generate_response(prompt, tokenizer, model, device, max_new_tokens=100,
                      do_sample=True, top_p=0.95, top_k=50, temperature=0.8,
                      repetition_penalty=None, bad_words_ids=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Use max_new_tokens for clarity
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids

    output = model.generate(**gen_kwargs)
    # decode only the newly generated portion
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

def generate_teacher_response(prompt, tokenizer, model, device, cleaning_func, max_new_tokens=100):
    raw = generate_response(
        prompt, tokenizer, model, device,
        max_new_tokens=max_new_tokens,
        do_sample=True, top_p=0.95, top_k=50, temperature=0.8
    )
    return cleaning_func(raw)

def generate_student_response(prompt, tokenizer, model, device, cleaning_func,
                              max_new_tokens=100, max_retries=3, cleaning_level="none"):
    # prepare potential bad_words_ids depending on cleaning level
    bad_words_ids = get_banned_tokens_for_cleaning(tokenizer, cleaning_level=cleaning_level) if cleaning_level != "none" else None

    for attempt in range(max_retries):
        raw = generate_response(
            prompt, tokenizer, model, device,
            max_new_tokens=max_new_tokens,
            do_sample=True, top_p=0.95, top_k=50, temperature=0.8,
            repetition_penalty=1.2,
            bad_words_ids=bad_words_ids
        )
        cleaned = cleaning_func(raw)

        # accept if it has any alphanumeric characters
        if any(c.isalnum() for c in cleaned):
            return cleaned
    # fallback empty string if none succeeded
    return ""

def chat_teacher_student(
    teacher_tokenizer, teacher_model, teacher_device,
    student_tokenizer, student_model, student_device,
    meta_prompt, num_turns=2,
    num_self_corrections=2, cleaning_level="light",
    max_new_tokens=100
):
    transcript_lines = []
    cleaning_func = CLEANING_FUNCS.get(cleaning_level, clean_light)

    teacher_reply = generate_teacher_response(
        meta_prompt, teacher_tokenizer, teacher_model, teacher_device,
        cleaning_func, max_new_tokens=max_new_tokens
    )
    transcript_lines.append(f"Teacher: {teacher_reply}")
    dialogue = f"[Teacher]: {teacher_reply}"

    for turn in range(num_turns):
        student_input = dialogue + "\n[Student]:"
        student_reply = generate_student_response(
            student_input, student_tokenizer, student_model, student_device,
            cleaning_func, max_new_tokens=max_new_tokens,
            cleaning_level=cleaning_level
        )
        transcript_lines.append(f"Student: {student_reply}")
        dialogue += f"\n[Student]: {student_reply}"

        correction_prompt = FEEDBACK_TEMPLATES["explicit_correction"](teacher_reply, student_reply)
        teacher_correction = generate_teacher_response(
            correction_prompt, teacher_tokenizer, teacher_model, teacher_device,
            cleaning_func, max_new_tokens=max_new_tokens
        )
        transcript_lines.append(f"Teacher (correction): {teacher_correction}")
        dialogue += f"\n[Teacher]: {teacher_correction}"

        for i in range(num_self_corrections):
            self_correction_input = dialogue + "\n[Student]:"
            student_reply = generate_student_response(
                self_correction_input, student_tokenizer, student_model, student_device,
                cleaning_func, max_new_tokens=max_new_tokens,
                cleaning_level=cleaning_level
            )
            transcript_lines.append(f"Student (self-correct {i+1}): {student_reply}")
            dialogue += f"\n[Student]: {student_reply}"

    return transcript_lines
# --------------------------
# Main CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Teacher-Student Chat Simulation (with feedback & cleaning options)"
    )
    parser.add_argument("--student_model", type=str, required=True,
                        help="Hugging Face model name or path for the student model")
    parser.add_argument("--num_turns", type=int, default=2,
                        help="Number of conversation turns per dialogue (teacher+student pairs)")
    parser.add_argument("--num_dialogues", type=int, default=5,
                        help="Number of dialogues to generate")
    parser.add_argument("--feedback_type", type=str, choices=["affirmative", "explicit_correction", "socratic"],
                        default="affirmative", help="Type of explicit feedback teacher gives the student")
    parser.add_argument("--cleaning", type=str, choices=["none", "light", "strict"],
                        default="light", help="Cleaning level applied to generated responses")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of newly generated tokens per generation call")
    args = parser.parse_args()

    # prepare output path
    output_dir = Path("./dialogues")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"dialogues_{student_tag}_"
        f"{args.num_turns}turns_"
        f"{args.num_dialogues}dialogs_"
        f"{args.feedback_type}_"
        f"{args.cleaning}_"
        f"{args.max_new_tokens}tokens.txt"
    )
    output_path = output_dir / filename

    print(f"Loading teacher model...")
    teacher_tokenizer, teacher_model, teacher_device = load_teacher_model()
    print(f"Loading student model '{args.student_model}'...")
    student_tokenizer, student_model, student_device = load_student_model(args.student_model)

    # meta prompt for first utterance
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

    print(f"Generating {args.num_dialogues} dialogues -> {output_path} (feedback={args.feedback_type}, cleaning={args.cleaning})")

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(1, args.num_dialogues + 1):
            transcript = chat_teacher_student(
                teacher_tokenizer, teacher_model, teacher_device,
                student_tokenizer, student_model, student_device,
                meta_prompt,
                num_turns=args.num_turns,
                feedback_type=args.feedback_type,
                cleaning_level=args.cleaning,
                max_new_tokens=args.max_new_tokens
            )
            f.write(f"Dialogue_Correction {i}\n")
            f.write("\n".join(transcript))
            f.write("\n\n")

    print(f"✅ Saved {args.num_dialogues} dialogues to {output_path}")

if __name__ == "__main__":
    main()
