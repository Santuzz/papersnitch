import json
import os
import sys

from openai import OpenAI

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
from web.settings import BASE_DIR

# from webApp.models import Paper, Dataset, Conference
from dotenv import load_dotenv, set_key

sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env.local")
PDF_DIR = BASE_DIR / "media" / "pdf"


api_key = os.getenv("MOONSHOT_API_KEY")
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2-0905-preview"
# Set as True means the API are called providing the pdf file in input
is_pdf = False


# ================================= KIMI K2 ========================================
# structured answer
"""{
    "id": "cmpl-04ea926191a14749b7f2c7a48a68abc6",
    "object": "chat.completion",
    "created": 1698999496,
    "model": "kimi-k2-turbo-preview",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, Li Lei! 1+1 equals 2. If you have any other questions, feel free to ask!"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 21,
        "total_tokens": 40,
        "cached_tokens": 10  # The number of tokens hit by the cache, only models that support automatic caching will return this field
    }
}"""


# ========== STANDARD USE ============
def kimi_standard(
    client: OpenAI,
    pdf_text: str,
    system_prompt: str,
    model: str,
    temperature: float = 0.0,
):

    messages = [
        {
            "role": "system",
            "content": pdf_text,
        },
        {"role": "user", "content": system_prompt},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={
            "type": "json_object"
        },  # <-- Use the response_format parameter to specify the output format as json_object
    )

    return completion


# ============ UPLOAD FILES ============


def kimi_files(
    client: OpenAI, file: str, system_prompt: str, model: str, temperature: float = 0.0
):

    file_object = client.files.create(file=file, purpose="file-extract")

    # Retrieve the result
    file_content = client.files.content(file_id=file_object.id).text

    # Include it in the request
    messages = [
        {
            "role": "system",
            "content": file_content,
        },
        {
            "role": "user",
            "content": system_prompt,
        },
    ]

    # Then call chat-completion to get Kimi's response

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    return completion


def _update_token_kimi(tokens_used: int) -> int:
    current_total = int(os.getenv("TOTAL_TOKEN_KIMI", 0))
    new_total = current_total + tokens_used
    set_key(str(BASE_DIR / ".env.local"), "TOTAL_TOKEN_KIMI", str(new_total))
    return new_total


def main():

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    criterions = """
{
  "annotation": {
    "definition": "Information describing how data annotations or labels were created, validated, and quality-controlled. This includes: how labels were produced, who performed the annotation, what tools or platforms were used, what annotation guidelines were followed, the expertise or training of annotators, reported inter-annotator agreement metrics (e.g., Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, percent agreement), how disagreements were resolved (consensus, arbitration, majority vote), and any procedures used to check or enforce annotation reliability.",
    "keywords": [
      "annotation",
      "labeling",
      "human raters",
      "dataset creation",
      "labeling guidelines",
      "agreement",
      "crowdworkers",
      "expert annotators",
      "quality control"
    ]
  },

  "preprocessing": {
    "definition": "Detailed description of all transformations applied to raw data before training or evaluation. This includes explicit preprocessing steps such as resampling, cropping, normalization, and augmentation; numerical parameters such as input dimensions, value ranges, and augmentation probabilities; description of the augmentation pipeline; motivations for each preprocessing step; and any software libraries, frameworks, or pipelines used.",
    "keywords": [
      "data preprocessing",
      "data preparation",
      "normalization",
      "augmentation",
      "resampling",
      "cropping",
      "feature scaling",
      "input processing"
    ]
  },

  "evaluation": {
    "definition": "Completeness and transparency of the experimental evaluation setup. This includes train/validation/test data splits, stratification, subject-wise splits, cross-validation strategies (e.g., k-fold, leave-one-out), performance metrics used, uncertainty or variability reporting (standard deviations, confidence intervals, repeated runs, random seeds), details of the experimental setup such as hyperparameters, epochs, hardware, search methods, and any robustness, ablation, or sensitivity analyses.",
    "keywords": [
      "experimental evaluation",
      "results",
      "validation protocol",
      "test split",
      "cross-validation",
      "performance reporting",
      "uncertainty",
      "experimental setup"
    ]
  },

  "licensing_and_ethical_transparency": {
    "definition": "Legal, ethical, and compliance information for datasets, code, and experiments involving human subjects. This includes dataset and code licensing terms (e.g., CC-BY, CC-BY-NC, MIT, GPL), any usage restrictions, ethical approvals or IRB statements, informed consent procedures, anonymization or pseudonymization methods, and discussions of privacy protection, risks, societal impact, bias, or limitations.",
    "keywords": [
      "ethical considerations",
      "IRB approval",
      "informed consent",
      "anonymization",
      "privacy",
      "licensing",
      "data release terms",
      "usage restrictions",
      "responsible use"
    ]
  }
}
"""

    json_format = """
    {
        "annotation": {
            "extracted": "Extracted text related to the annotation",
            "score": "Score based on annotation"
        },
        "preprocessing": {
            "extracted": "Extracted text related to the preprocessing",
            "score": "Score based on preprocessing"
        },
        "evaluation": {
            "extracted": "Extracted text related to the evaluation",
            "score": "Score based on evaluation "
        },
        "licensing_and_ethical_transparency": {
            "extracted": "Extracted text related to the licensing_and_ethical_transparency",
            "score": "Score based on licensing_and_ethical_transparency"
        }
    }
    """
    system_prompt = f"""
    You are an intelligent scientific paper evaluator, responsible for analyzing scientific papers provided and giving them a different score based on given criterions. Your reply can be only text in JSON format as specified below.
    Based on the criterion provided, one or more, you have to find and put in the answer the text related to the CRITERION (`extracted` field) and a integer score from 0 to 2 (`score` field) based on how well the paper addresses that CRITERION.
    CRITERION - {criterions}
    
    Please output your reply in the following JSON format:
    
    {json_format}
    
    Note: Your answer MUST contain ONLY the text extracted from the paper and the score, no other words that are not in the input text. Please place the information extracted from the paper in the `extracted` field and the corresponding score in the `score` field.
        For the score values: Put 0 if there is no information related to the criterion, values from 1 to 4 if the informations found are incomplete or something is missing, and 5 if there is detailed and clear information about the annotation process, including the expertise of annotators, metrics used, and inter-annotator agreement.
        If you didn't find any information related to a specific criterion, set the `extracted` field to an empty string and the `score` field to 0.
    """

    # system_prompt = f"set 'ciao' as a value for 'extracted' and 2 as a value for 'score'. Please output your reply in the following JSON format: {json_format}"
    if is_pdf:
        completion = kimi_files(
            client, PDF_DIR / "miccai_2025_0308_paper.pdf", system_prompt, model
        )
    else:
        with open(PDF_DIR / "miccai_2025_0308_paper.txt", "r") as f:
            pdf_text = f.read()
            completion = kimi_standard(client, pdf_text, system_prompt, model)

    # save the response (json file)
    response = json.loads(completion.choices[0].message.content)
    with open(PDF_DIR / "miccai_2025_0308_kimi.json", "w") as f:
        json.dump(response, f, indent=2)

    output_tokens = completion.usage.completion_tokens
    input_tokens = completion.usage.prompt_tokens
    total_tokens = completion.usage.total_tokens
    cumulative_tokens = _update_token_kimi(total_tokens)

    print(
        f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cumulative tokens: {cumulative_tokens}"
    )


if __name__ == "__main__":
    main()
