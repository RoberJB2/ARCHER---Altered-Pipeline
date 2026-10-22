import os
from typing import List
import nltk
from openai import OpenAI
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompts
WIKIPEDIA_PROMPT = "Give only the concept and factual sentences. Retain as much information as possible. No other output. This is a wikipedia article about {concept}:"
KG_PROMPT = "Give only the concept and factual sentences. Retain as much information as possible. No other output. This is a knowledge graph about {concept}:"

# Paths
ARTICLE_DIR = Path("selfcheckgpt")

def generate_samples(
    article: str,
    prompt: str = None,
    llm_model: str = "gpt-4.1-mini",
    kg: bool = False,
    json: bool = False,
) -> List[str]:
    
    # Input and Output Directories
    article_dir = ARTICLE_DIR / article
    # outpath = Path(os.getenv("OUTPUT_DIR"), "output") # ============

    # Read the article content from the specified directory
    article = article_dir.read_text(encoding="utf-8")

    # Insert the article into the prompt
    prompt = prompt.format(concept=article) if prompt else article

    samples = []

    if llm_model == 'gpt-4.1-mini':
        for i in range(20):
            # GPT-4.1-Mini
            response = client.chat.completions.create(model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0, # 1.0 = stochastic responses
            )
            sample = response.choices[0].message.content

            # Performance improvement from selfcheckgpt
            sample = sample.replace("\n", " ")
            print("gpt-4.1-mini:", sample)
            samples.append(sample)

        print(f"{samples}")
    else:
        raise Exception("LLM not found")

    return samples

def score_sentence(
    article: str,
    sentence: str,
    llm_model: str = "gpt-4.1-mini",
) -> float:
    """
    Returns:
        0.0 when the sentence is supported
        1.0 when the sentence is not supported
        0.5 when the answer cannot be determined
    """

    prompt = f"""
Original article:
{article}

Sentence:
{sentence}

Is the sentence supported by the original article?

Answer using only one of these values:
yes
no
unsure
"""

    response = client.responses.create(
        model=llm_model,
        input=prompt,
        temperature=0.0,
    )

    answer = response.output_text.strip().lower()

    if answer == "yes":
        return 0.0

    if answer == "no":
        return 1.0

    return 0.5

def score_sample(
    article: str,
    sample: str,
    llm_model: str = "gpt-4.1-mini",
) -> dict:
    sentences: list[str] = nltk.sent_tokenize(sample)

    sentence_results: list[dict] = []

    for sentence in sentences:
        score = score_sentence(
            article=article,
            sentence=sentence,
            llm_model=llm_model,
        )

        sentence_results.append(
            {
                "sentence": sentence,
                "score": score,
            }
        )

    if sentence_results:
        sample_score = sum(
            result["score"] for result in sentence_results
        ) / len(sentence_results)
    else:
        sample_score = 0.0

    return {
        "sample": sample,
        "sample_score": sample_score,
        "sentence_results": sentence_results,
    }

def compare_samples(
    article_path: str | Path,
    samples: list[str],
    llm_model: str = "gpt-4.1-mini",
) -> list[dict]:
    article = ARTICLE_DIR / article_path
    article = article.read_text(encoding="utf-8")

    results: list[dict] = []

    for sample_number, sample in enumerate(samples):
        result = score_sample(
            article=article,
            sample=sample,
            llm_model=llm_model,
        )

        result["sample_number"] = sample_number
        results.append(result)

    return results

# For the normal SelfCheckGPT pipeline
def generate_passage(
    article: str,
    prompt: str,
    llm_model: str = "gpt-4.1-mini",
) -> str:
    response = client.chat.completions.create(model=llm_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.0, # 0.0 = deterministic responses
    )
    
    return response

def kg_to_text(path: str) -> str:
    input_path = Path(path)

    # Creates the output path in the same directory:
    # kg_0.json -> kg_0.txt
    output_path = "article_" +input_path.with_suffix(".txt")

    with input_path.open("r", encoding="utf-8") as file:
        kg_file = json.load(file)

    statements: list[str] = []

    for relation in kg_file.get("relations", []):
        if len(relation) != 3:
            continue

        subject, predicate, obj = relation

        statements.append(
            f"Subject: {subject}; "
            f"Relation: {predicate}; "
            f"Object: {obj}."
        )

    output_path.write_text(
        "\n".join(statements),
        encoding="utf-8",
    )

    return str(output_path)

def print_results(results: list[dict]) -> None:
    for result in results:
        print(
            f"Sample {result['sample_number']}: "
            f"{result['sample_score']}"
        )

        for sentence_result in result["sentence_results"]:
            print(
                sentence_result["score"],
                sentence_result["sentence"],
            )

def calculate_hallucination_rates(
    results: list[dict],
    threshold: float = 0.5,
) -> dict:
    sample_scores = [
        result["sample_score"]
        for result in results
    ]

    sentence_scores = [
        sentence_result["score"]
        for result in results
        for sentence_result in result["sentence_results"]
    ]

    sample_rate = (
        sum(score >= threshold for score in sample_scores)
        / len(sample_scores)
        if sample_scores
        else 0.0
    )

    sentence_rate = (
        sum(score >= threshold for score in sentence_scores)
        / len(sentence_scores)
        if sentence_scores
        else 0.0
    )

    return {
        "sample_rate": sample_rate,
        "sentence_rate": sentence_rate,
        "sample_scores": sample_scores,
        "sentence_scores": sentence_scores,
    }

def main(
        kg_file: str = "kg_0.txt",
        article_file: str = "article_0.txt",
) -> None:
    # The OPENAI_API_KEY environment variable must be defined.
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not defined. "
            "Set it before running this program."
        )

    # If kg txt file doesn't exist:
    if not Path("selfcheckgpt/kg_0.txt").exists():
        answer = input("Do you want to create a readable knowledge graph text file from the JSON file? (y/n): ")
        if answer.lower() == "y":
            # Load a readable knowledge graph JSON file
            kg_to_text("selfcheckgpt/kg_0.json")
            print("Creating a readable KG text file from the JSON file.")
        raise FileNotFoundError(
            "Knowledge graph file wasn't found."
        )
    

# ==================== Sampling and results ====================  #
    kg_samples = generate_samples(
        article=kg_file,
        prompt=KG_PROMPT,
        llm_model="gpt-4.1-mini",
        kg=True,
    )

    kg_results = compare_samples(
        article_path=kg_file,
        samples=kg_samples,
        llm_model="gpt-4.1-mini",
    )
    
    article_samples = generate_samples(
        article=article_file,
        prompt=WIKIPEDIA_PROMPT,
        llm_model="gpt-4.1-mini",
        json=True,
    )

    article_results = compare_samples(
        article_path=article_file,
        samples=article_samples,
        llm_model="gpt-4.1-mini",
    )

    article_rates = calculate_hallucination_rates(article_results)
    kg_rates = calculate_hallucination_rates(kg_results)

    print(
        f"Article sentence hallucination rate: "
        f"{article_rates['sentence_rate']:.2%}"
    )
    print(
        f"Article sample hallucination rate: "
        f"{article_rates['sample_rate']:.2%}"
    )

    print(
        f"KG sentence hallucination rate: "
        f"{kg_rates['sentence_rate']:.2%}"
    )
    print(
        f"KG sample hallucination rate: "
        f"{kg_rates['sample_rate']:.2%}"
    )


if __name__ == "__main__":
    main(kg_file = "kg_0.txt",
         article_file = "article_0.txt"
    )