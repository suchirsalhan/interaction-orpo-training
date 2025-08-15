import argparse
from pathlib import Path

# --------------------------
# Helper functions
# --------------------------

def sanitize_for_filename(s: str) -> str:
    """Replace unsafe filename characters with underscores."""
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")

# Placeholder for actual loading functions
def load_teacher_model():
    # Replace with actual teacher model/tokenizer loading
    return None, None, "cpu"

def load_student_model(model_name):
    # Replace with actual student model/tokenizer loading
    return None, None, "cpu"

def chat_teacher_student(
    teacher_tokenizer, teacher_model, teacher_device,
    student_tokenizer, student_model, student_device,
    meta_prompt, num_turns, feedback_type, cleaning_level, max_new_tokens
):
    # Replace with your real chat loop implementation
    transcript = []
    for turn in range(num_turns):
        transcript.append(f"Teacher: Turn {turn+1}")
        transcript.append(f"Student: Response {turn+1}")
        # Optional: implement iterative self-correction here
    return transcript

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

    student_tag = sanitize_for_filename(args.student_model)
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
            f.write(f"Dialogue {i}\n")
            f.write("\n".join(transcript))
            f.write("\n\n")

    print(f"✅ Saved {args.num_dialogues} dialogues to {output_path}")

if __name__ == "__main__":
    main()
