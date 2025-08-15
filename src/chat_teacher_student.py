#!/usr/bin/env python3
import argparse
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

def generate_student_response(prompt, tokenizer, model, device, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_length = inputs.input_ids.shape[1]
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    output_ids = inputs.input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(output_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)

            if next_token_id.item() < 0 or next_token_id.item() >= logits.shape[-1]:
                break

            output_ids = torch.cat([output_ids, next_token_id], dim=-1)

            if eos_token_id != -1 and next_token_id.item() == eos_token_id:
                break

    generated_tokens = output_ids[0, input_length:].tolist()
    generated_text = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True)
    return generated_text.strip()


def generate_teacher_response(prompt, tokenizer, model, device, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0, inputs.input_ids.shape[1]:].tolist()
    text = tokenizer.decode(generated, clean_up_tokenization_spaces=True).strip()
    return text


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
        "that has the linguistic abilities of a 6 to 11 month old infant.\n\n"
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

