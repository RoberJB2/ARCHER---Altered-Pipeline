import os
import json
import litellm
from typing import List
from typing import Tuple
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence
from dotenv import load_dotenv

load_dotenv(override=True)

INPUT_DIR = Path("data/wiki_qa/PURE_txt")

HF_REPO_ID = "Ichorv/KGGen_PURE"
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------- READ ME ------------------ #
# This file uploads a dataset of .txt files to
# a hugging face database set with the HF_TOKEN
# and REPO ID variables above.
# 
# The format is customizable by changing the 
# feature names.
# ------------------------------------------- #

def upload_questions_to_huggingface(
    rows: list[dict],
    repo_id: str,
    hf_token: str,
) -> Dataset:

    features = Features({
        "id": Value("int64"),
        "essay_topic": Value("string"),
        "essay_content": Value("string"),
        "generated_queries": Sequence(Value("string")),
        "generated_answers": Sequence(Value("string")),
        "num_generated_queries": Value("int64"),
    })

    dataset = Dataset.from_list(
        rows,
        features=features
    )

    dataset.push_to_hub(
        repo_id,
        token=hf_token,
        split="train"
    )

    return dataset


def generate_questions(
    article: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    prompt: str = "Generate questions that encompass the article as a whole."
) -> Tuple[str, List[str]]:
    
    full_prompt = f"""
{prompt}

Read the following article.

Your task:
1. Determine the title of the article.
2. Generate exactly 15 questions about the article.
3. The questions should cover the article as a whole rather than focusing
   heavily on only one section.
4. Questions should address the article's major topics, ideas, events,
   people, arguments, or conclusions where applicable.
5. Only use information contained in the article.

Return ONLY valid JSON using this format:

{{
    "title": "Article Title",
    "questions": [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?",
        "Question 6?",
        "Question 7?",
        "Question 8?",
        "Question 9?",
        "Question 10?",
        "Question 11?",
        "Question 12?",
        "Question 13?",
        "Question 14?",
        "Question 15?"
    ]
}}

ARTICLE:
{article}
"""

    response = litellm.completion(
        model=model,
        api_key=api_key,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze articles and generate comprehensive "
                    "questions about their content."
                )
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        temperature=temperature
    )

    # Get the entire text response from the LLM
    response_text = response.choices[0].message.content

    # Convert the JSON string into a Python dictionary
    result = json.loads(response_text)

    title = result["title"]
    questions = result["questions"]

    # Make sure exactly 15 questions were returned
    if len(questions) != 15:
        raise ValueError(
            f"Expected 15 questions, but the LLM returned {len(questions)}."
        )

    return title, questions


def answer_questions(
    questions: List[str],
    model: str,
    api_key: str,
    temperature: float = 0.0,
) -> List[str]:

    if len(questions) != 15:
        raise ValueError(
            f"Expected 15 questions, but received {len(questions)}."
        )

    # Convert the questions into a numbered string
    questions_text = "\n".join(
        f"{i}. {question}"
        for i, question in enumerate(questions, start=1)
    )

    prompt = f"""
Answer each of the following 15 questions.

Return ONLY a valid JSON list containing the answers.

The answers must be in the exact same order as the questions.

Do not include:
- question numbers
- the questions themselves
- labels such as "Answer 1"
- explanations outside the JSON list

Example output format:

[
    "First answer",
    "Second answer",
    "Third answer"
]

QUESTIONS:

{questions_text}
"""

    response = litellm.completion(
        model=model,
        api_key=api_key,
        messages=[
            {
                "role": "system",
                "content": "Answer the provided questions accurately and concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    response_text = response.choices[0].message.content

    # Convert JSON response to Python list
    answers = json.loads(response_text)

    if len(answers) != 15:
        raise ValueError(
            f"Expected 15 answers, but the LLM returned {len(answers)}."
        )

    return answers


def main(
    model: str = "openai/gpt-4.1-mini",
    api_key: str = os.getenv("OPENAI_API_KEY"),
    temperature: float = 0.0,
    prompt: str = (
        "Generate exactly 15 questions that encompass the article as a whole."
    ),
):

    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set.")

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN is not set.")

    rows = []

    # Go through every .txt article in the directory
    for item in sorted(INPUT_DIR.glob("*.txt")):

        print(f"Processing: {item.name}")

        # Read the TXT file into a string
        article = item.read_text(
            encoding="utf-8"
        ).strip()

        # Skip empty files
        if not article:
            print(f"Skipping empty file: {item.name}")
            continue

        # ----------------------------------
        # 1. Generate title + 15 questions
        # ----------------------------------

        title, questions = generate_questions(
            article=article,
            model=model,
            api_key=api_key,
            temperature=temperature,
            prompt=prompt
        )

        # ----------------------------------
        # 2. Answer the 15 questions
        # ----------------------------------

        answers = answer_questions(
            questions=questions,
            model=model,
            api_key=api_key,
            temperature=temperature
        )

        # ----------------------------------
        # 3. Create Hugging Face row
        # ----------------------------------

        row = {
            "id": len(rows) + 1,
            "essay_topic": title,
            "essay_content": article,
            "generated_queries": questions,
            "generated_answers": answers,
            "num_generated_queries": len(questions),
        }

        rows.append(row)

        print(
            f"Finished: {title} "
            f"({len(questions)} questions, "
            f"{len(answers)} answers)"
        )

    # ----------------------------------
    # 4. Upload after ALL articles finish
    # ----------------------------------

    if not rows:
        raise ValueError(
            f"No TXT articles were found in {INPUT_DIR}"
        )

    print(
        f"\nUploading {len(rows)} articles "
        "to Hugging Face..."
    )

    dataset = upload_questions_to_huggingface(
        rows=rows,
        repo_id=HF_REPO_ID,
        hf_token=HF_TOKEN
    )

    print("Upload complete.")

    return dataset

if __name__ == "__main__":
    main()