import torch

# ----------------------------
# Cleaning functions
# ----------------------------
def clean_light(text):
    return text.strip()

CLEANING_FUNCS = {
    "light": clean_light,
}

# ----------------------------
# Feedback template
# ----------------------------
FEEDBACK_TEMPLATES = {
    "explicit_correction": lambda teacher_msg, student_msg: (
        f"Student said: '{student_msg}'\n"
        f"As the teacher, provide an explicit correction to improve the student's answer."
    )
}

# ----------------------------
# Response generation helpers
# ----------------------------
def generate_teacher_response(prompt, tokenizer, model, device, cleaning_func, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return cleaning_func(text)

def generate_student_response(prompt, tokenizer, model, device, cleaning_func, max_new_tokens=100, cleaning_level="light"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return CLEANING_FUNCS.get(cleaning_level, clean_light)(text)

# ----------------------------
# Teacher-Student Chat Loop with Iterative Self-Correction
# ----------------------------
def chat_teacher_student_self_correction(
    teacher_tokenizer, teacher_model, teacher_device,
    student_tokenizer, student_model, student_device,
    meta_prompt, num_turns=2,
    num_self_corrections=2, cleaning_level="light",
    max_new_tokens=100
):
    """
    Teacher-student dialogue with iterative self-correction.
    """
    transcript_lines = []
    cleaning_func = CLEANING_FUNCS.get(cleaning_level, clean_light)

    # Teacher initiates dialogue
    teacher_reply = generate_teacher_response(
        meta_prompt, teacher_tokenizer, teacher_model, teacher_device,
        cleaning_func, max_new_tokens=max_new_tokens
    )
    transcript_lines.append(f"Teacher: {teacher_reply}")
    dialogue = f"[Teacher]: {teacher_reply}"

    for turn in range(num_turns):
        # Student responds to teacher
        student_input = dialogue + "\n[Student]:"
        student_reply = generate_student_response(
            student_input, student_tokenizer, student_model, student_device,
            cleaning_func, max_new_tokens=max_new_tokens,
            cleaning_level=cleaning_level
        )
        transcript_lines.append(f"Student: {student_reply}")
        dialogue += f"\n[Student]: {student_reply}"

        # Teacher issues explicit correction
        correction_prompt = FEEDBACK_TEMPLATES["explicit_correction"](teacher_reply, student_reply)
        teacher_correction = generate_teacher_response(
            correction_prompt, teacher_tokenizer, teacher_model, teacher_device,
            cleaning_func, max_new_tokens=max_new_tokens
        )
        transcript_lines.append(f"Teacher (correction): {teacher_correction}")
        dialogue += f"\n[Teacher]: {teacher_correction}"

        # Iterative self-correction by student
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

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Example models (use small ones for testing)
    teacher_model_name = "gpt2"
    student_model_name = "gpt2"

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model.to(teacher_device)

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
    student_device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(student_device)

    # Run chat
    meta_prompt = "Teacher introduces the topic: Explain the concept of photosynthesis."
    transcript = chat_teacher_student_self_correction(
        teacher_tokenizer, teacher_model, teacher_device,
        student_tokenizer, student_model, student_device,
        meta_prompt, num_turns=1, num_self_corrections=2
    )

    # Print transcript
    for line in transcript:
        print(line)
