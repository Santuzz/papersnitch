"""
LLM Analysis Service for Paper Evaluation

This module provides functions to extract text from PDFs and analyze them
using various LLM models based on specific criteria.
"""

import json
import logging
import os
from pydoc import text
import re
import time
import uuid
import threading
from typing import Dict, Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv(".env.local")

from django.contrib.auth.models import User
from webApp.models import (
    Paper,
    Analysis,
    Criterion,
    AnalysisCriterion,
    Dataset,
    LLMModelConfig,
    Prompt,
)
from webApp.functions import (
    get_code,
    update_token,
    check_token_limit,
    TokenLimitExceededError,
)
from pydantic import BaseModel, Field
from typing import List


class Code(BaseModel):
    url: str
    extracted: str
    score_explanation: str
    score: int


class Datasets(BaseModel):
    extracted: List[str]


class GeneralCriteria(BaseModel):
    extracted: str
    score_explanation: str
    score: int


class PaperAnalysis(BaseModel):
    code: Code
    datasets: Datasets
    annotation: GeneralCriteria
    preprocessing: GeneralCriteria
    evaluation: GeneralCriteria
    licensing: GeneralCriteria

    class Config:
        extra = "forbid"


def get_model_configs() -> Dict[str, Any]:
    """Get model configurations from the database."""
    return LLMModelConfig.get_all_configs()


def get_criterions() -> str:
    """Get criteria from the database and format as a string for the prompt.

    This is called at runtime rather than module import time to avoid
    database access during Django's initialization phase (e.g., makemigrations).
    """
    criterions = "CRITERIONS:\n{"
    crit = Criterion.objects.all().values("key", "description").order_by("id")
    for c in crit:
        criterions = (
            criterions
            + f'\n  "{c["key"]}": {{\n    "definition": "{c["description"]}"\n  }},'
        )
    criterions = criterions[:-1] + "\n}"  # remove last comma and close bracket
    return criterions


# JSON Schema for structured model outputs
JSON_SCHEMA = {
    "format": {
        "type": "json_schema",
        "name": "paper_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "object",
                    "description": "Code repository information extracted from the paper",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Code repository URL (e.g., GitHub link). Empty string if not found.",
                        },
                        "extracted": {
                            "type": "string",
                            "description": "Extracted text related to code availability, documentation, or instructions to access the code",
                        },
                        "score_explanation": {
                            "type": "string",
                            "description": "Explanation and motivation for the score given regarding code availability and documentation",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score indicating how well the paper addresses code availability and documentation",
                        },
                    },
                    "required": ["url", "extracted", "score_explanation", "score"],
                    "additionalProperties": False,
                },
                "datasets": {
                    "type": "object",
                    "description": "Datasets used in the paper",
                    "properties": {
                        "extracted": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of dataset names extracted from the paper",
                        }
                    },
                    "required": ["extracted"],
                    "additionalProperties": False,
                },
                "annotation": {
                    "type": "object",
                    "description": "Information about data annotation methodology",
                    "properties": {
                        "extracted": {
                            "type": "string",
                            "description": "Extracted text related to annotation methodology",
                        },
                        "score_explanation": {
                            "type": "string",
                            "description": "Explanation and motivation for the score given",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score indicating how well the paper addresses annotation",
                        },
                    },
                    "required": ["extracted", "score_explanation", "score"],
                    "additionalProperties": False,
                },
                "preprocessing": {
                    "type": "object",
                    "description": "Information about data preprocessing steps",
                    "properties": {
                        "extracted": {
                            "type": "string",
                            "description": "Extracted text related to preprocessing methodology",
                        },
                        "score_explanation": {
                            "type": "string",
                            "description": "Explanation and motivation for the score given",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score indicating how well the paper addresses preprocessing",
                        },
                    },
                    "required": ["extracted", "score_explanation", "score"],
                    "additionalProperties": False,
                },
                "evaluation": {
                    "type": "object",
                    "description": "Information about experimental evaluation setup",
                    "properties": {
                        "extracted": {
                            "type": "string",
                            "description": "Extracted text related to evaluation methodology",
                        },
                        "score_explanation": {
                            "type": "string",
                            "description": "Explanation and motivation for the score given",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score indicating how well the paper addresses evaluation",
                        },
                    },
                    "required": ["extracted", "score_explanation", "score"],
                    "additionalProperties": False,
                },
                "licensing": {
                    "type": "object",
                    "description": "Information about licensing and ethical considerations",
                    "properties": {
                        "extracted": {
                            "type": "string",
                            "description": "Extracted text related to licensing and ethical transparency",
                        },
                        "score_explanation": {
                            "type": "string",
                            "description": "Explanation and motivation for the score given",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score indicating how well the paper addresses licensing and ethics",
                        },
                    },
                    "required": ["extracted", "score_explanation", "score"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "code",
                "datasets",
                "annotation",
                "preprocessing",
                "evaluation",
                "licensing",
            ],
            "additionalProperties": False,
        },
    }
}

json_object = {
    "code": {
        "url": "Code repository URL (e.g., GitHub link). Empty string if not found.",
        "extracted": "Extracted text related to code availability, documentation, or instructions to access the code",
        "score_explanation": "Explanation and motivation for the score given regarding code availability and documentation",
        "score": "INTEGER Score indicating how well the paper addresses code availability and documentation",
    },
    "datasets": {"extracted": "List of dataset names extracted from the paper"},
    "annotation": {
        "extracted": "Extracted text related to annotation methodology",
        "score_explanation": "Explanation and motivation for the score given",
        "score": "INTEGER Score indicating how well the paper addresses annotation",
    },
    "preprocessing": {
        "extracted": "Extracted text related to preprocessing methodology",
        "score_explanation": "Explanation and motivation for the score given",
        "score": "INTEGER Score indicating how well the paper addresses preprocessing",
    },
    "evaluation": {
        "extracted": "Extracted text related to evaluation methodology",
        "score_explanation": "Explanation and motivation for the score given",
        "score": "INTEGER Score indicating how well the paper addresses evaluation",
    },
    "licensing": {
        "extracted": "Extracted text related to licensing and ethical transparency",
        "score_explanation": "Explanation and motivation for the score given",
        "score": "INTEGER Score indicating how well the paper addresses licensing and ethics",
    },
}
# Store for active analysis tasks
_analysis_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def code_tool():
    """Define the code documentation tool for LLM."""
    tools = [
        {
            "type": "function",
            "name": "get_code_documentation",
            "description": 'Download and get code documentation from a repository URL formatted in a JSON with summary, tree and content fields. {"summary": summary, "tree": tree, "content": content}',
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the code repository.",
                    }
                },
                "required": ["url"],
            },
        }
    ]
    tool_instruction = """\n When you find a code repository link from ONLY github (eg. https://github.com/openai) call the function get_code_documentation with the url parameter set to the link found. The text returned from the tool is the documentation of the code. Use the documentation to provide a score in the `score` field and an explaination for the score given in the `score_explanation` field."""

    return tools, tool_instruction


def llm_analysis(
    client: OpenAI,
    system_prompt: str,
    pdf_text: str,
    config: dict,
    code_text: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Get JSON response from the LLM model.
    Args:
        client: The OpenAI client instance.
        system_prompt: The system prompt to set the context.
        pdf_text: The extracted text from the PDF.
        config: Configuration dictionary containing model details.
        progress_callback: Optional callback function to report progress.
    Returns:
        Parsed JSON response from the LLM.
    """
    input_list = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"PAPER TEXT:\n{pdf_text}\nCODE REPOSITORY:\n{code_text or 'Not provided'}",
        },
    ]

    if "reasoning" in config:
        reasoning = {"effort": config["reasoning"].get("effort", "none")}

    if progress_callback:
        progress_callback("Checking token limits and preparing request...")

    # Check token limit before making API call
    try:
        check_token_limit(config["token_var"])
    except TokenLimitExceededError as e:
        if progress_callback:
            progress_callback("Token limit exceeded.")

    code_text = None

    if progress_callback:
        progress_callback("Waiting for LLM response...")

    iteration_start = time.perf_counter()

    # GPT MODELS

    if config["model"].startswith("gpt"):
        ##Tool request:  WORK IN PROGRESS - DO NNOT USE IT YET ##
        if config["code_tool"]:
            tools, tool_instruction = code_tool()
            input_list[0]["content"] = input_list[0]["content"] + tool_instruction

            response = client.responses.create(
                model=config["model"],
                input=input_list,
                temperature=config.get("temperature", 1.0),
                reasoning=reasoning,
                # text=JSON_SCHEMA,
                text_format=PaperAnalysis,
                tools=tools,
            )

            input_list += response.output

            # Handle tool call
            for item in response.output:
                if item.type == "function_call":
                    if item.name == "get_code":
                        if progress_callback:
                            progress_callback(
                                "LLM requesting code, starting ingestion process..."
                            )
                        args = json.loads(item.arguments)
                        print("=" * 50)
                        print(args)
                        code_text = get_code(**args)
                        paper = Paper.objects.filter(id=config["paper_id"]).first()
                        paper.code_text = code_text
                        paper.save(update_fields=["code_text"])
                        if progress_callback:
                            progress_callback(
                                "Ingestion completed successfully, continuing analysis..."
                            )

                        input_list.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps({"code": code_text}),
                            }
                        )

            response = client.responses.create(
                model=config["model"],
                input=input_list,
                temperature=config.get("temperature", 1.0),
                reasoning=reasoning,
                # text=JSON_SCHEMA,
                text_format=PaperAnalysis,
                tools=tools,
            )

        else:
            # regular request - Code already ingested or not available

            response = client.responses.parse(
                model=config["model"],
                input=input_list,
                temperature=config.get("temperature", 1.0),
                reasoning=reasoning,
                # text=JSON_SCHEMA,
                text_format=PaperAnalysis,
            )

        output_tokens = response.usage.output_tokens
        input_tokens = response.usage.input_tokens
        total_tokens = response.usage.total_tokens
        response = response.output_parsed

    # NON GPT MODELS ## WORK IN PROGRESS - DO NOT USE (C'è da creare un prompt per cui i modelli rispondano in formato JSON)##
    # elif config["model"].startswith("kimi-k2"):
    #     completion = client.chat.completions.create(
    #         model=config["model"],
    #         messages=input_list,
    #         temperature=config.get("temperature", 1.0),
    #     )
    #     response = completion.choices[0].message.content
    #     output_tokens = completion.usage.completion_tokens
    #     input_tokens = completion.usage.prompt_tokens
    #     total_tokens = completion.usage.total_tokens

    else:
        time.sleep(1)
        # create a dummy response as PaperAnalysis with empty fields
        result = PaperAnalysis(
            code=Code(
                url="",
                extracted="No information extracted for this criterion",
                score_explanation="No information extracted for this criterion",
                score=0,
            ),
            datasets=Datasets(extracted=[]),
            annotation=GeneralCriteria(
                extracted="No information extracted for this criterion",
                score_explanation="No information extracted for this criterion",
                score=0,
            ),
            preprocessing=GeneralCriteria(
                extracted="No information extracted for this criterion",
                score_explanation="No information extracted for this criterion",
                score=0,
            ),
            evaluation=GeneralCriteria(
                extracted="No information extracted for this criterion",
                score_explanation="No information extracted for this criterion",
                score=0,
            ),
            licensing=GeneralCriteria(
                extracted="No information extracted for this criterion",
                score_explanation="No information extracted for this criterion",
                score=0,
            ),
        )

        return {
            "model": config.get("visual_name", config["model_key"]),
            "result": result,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration": round(0.0, 2),
        }
    #     input_list[0]["content"] = (
    #         input_list[0]["content"]
    #         + "\n Respond in JSON format only. following this schema:"
    #         + str(json_object)
    #     )
    #     completion = client.chat.completions.create(
    #         model=config["model"],
    #         messages=input_list,
    #         temperature=config.get("temperature", 1.0),
    #         response_format={"type": "json_object"},
    #     )
    #     response = completion.choices[0].message.content
    #     output_tokens = completion.usage.completion_tokens
    #     input_tokens = completion.usage.prompt_tokens
    #     total_tokens = completion.usage.total_tokens

    duration = time.perf_counter() - iteration_start

    if progress_callback:
        progress_callback(
            f"LLM response received in {round(duration, 2)} seconds. Parsing response... "
        )

    # upload code to DB
    if "paper_id" in config and code_text is not None:
        paper = Paper.objects.filter(id=config["paper_id"]).first()
        paper.code_text = code_text
        paper.save(update_fields=["code_text"])

    # try:
    #     json_response = json.loads(response)
    # except json.JSONDecodeError as e:
    #     response = {
    #         "error": f"Error decoding JSON (LLM response not well formatted): {e}",
    #         "raw_content": response,
    #     }

    update_token(total_tokens, config["token_var"])

    return {
        "model": config.get("visual_name", config["model_key"]),
        "result": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration": round(duration, 2),
    }


def get_available_models() -> Dict[str, str]:
    """Get a dictionary of available models for selection.
    Returns:
        Dictionary with model_key as key and display name as value.
    """
    return LLMModelConfig.get_available_models()


def _save_analysis_to_db(
    paper_id: int, model_key: str, config: dict, result: dict, user_id: int = None
):
    """Save analysis results to the database.
    Args:
        paper_id: The ID of the Paper model instance.
        model_key: The key of the model used for analysis.
        config: Model configuration dictionary.
        result: The analysis result from llm_analysis or error dict.
        user_id: The ID of the user who initiated the analysis.
    """
    try:
        paper = Paper.objects.get(id=paper_id)
        user = User.objects.get(id=user_id) if user_id else None

        # Check if this is an error result
        if "error" in result and "result" not in result:
            Analysis.objects.create(
                paper=paper,
                user=user,
                model_name=config.get("visual_name", model_key),
                model_key=model_key,
                error=result["error"],
            )
            return

        # Extract data from successful result
        llm_response = result.get("result", {})

        # Save code_url to Paper if extracted
        code_data = llm_response.get("code", {})
        if isinstance(code_data, dict):
            code_url = code_data.get("url")
            if (
                code_url
                and isinstance(code_url, str)
                and code_url.startswith("http")
                and paper.code_url == ""
            ):
                paper.code_url = code_url
                paper.save(update_fields=["code_url"])

        # Save datasets to Dataset model
        datasets_data = llm_response.get("datasets", {})
        if isinstance(datasets_data, dict):
            datasets_list = datasets_data.get("extracted", [])
            if isinstance(datasets_list, list):
                for dataset_name in datasets_list:
                    if dataset_name and isinstance(dataset_name, str):
                        dataset, _ = Dataset.objects.get_or_create(
                            name=dataset_name, defaults={"from_pdf": True}
                        )
                        dataset.papers.add(paper)

        # Create the Analysis instance
        analysis = Analysis.objects.create(
            paper=paper,
            user=user,
            model_name=config.get("visual_name", model_key),
            model_key=model_key,
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            duration=result.get("duration"),
            raw_response=llm_response,
        )

        # Save each criterion result
        for criterion_key in llm_response.keys():
            try:
                criterion = Criterion.objects.get(key=criterion_key)
                criterion_data = llm_response.get(criterion_key, {})

                if isinstance(criterion_data, dict):
                    AnalysisCriterion.objects.create(
                        analysis=analysis,
                        criterion=criterion,
                        extracted=criterion_data.get("extracted"),
                        score_explanation=criterion_data.get("score_explanation"),
                        score=criterion_data.get("score"),
                    )
            except Criterion.DoesNotExist:
                # Criterion not found in database, skip
                logger.warning(f"Criterion '{criterion_key}' not found in database")
                continue

    except Exception as e:
        logger.exception(f"Error saving analysis to database: {e}")
        return None

    return analysis


def create_analysis_task(
    paper: Paper, selected_models: list = None, user_id: int = None
) -> str:
    """Create a new analysis task and return its ID.
    Args:
        paper: Paper model instance containing the PDF text.
        selected_models: List of model keys to use. If None, uses all models.
        user_id: The ID of the user who initiated the analysis.
    Returns:
        Task ID for tracking progress.
    """
    task_id = str(uuid.uuid4())

    # Get model configs from database
    model_configs = get_model_configs()

    # Filter models
    if selected_models:
        models_to_use = {k: v for k, v in model_configs.items() if k in selected_models}
    else:
        models_to_use = model_configs

    with _tasks_lock:
        _analysis_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "current_step": "Initializing...",
            "total_steps": len(models_to_use),
            "completed_steps": 0,
            "results": {},
            "error": None,
            "paper_id": paper.id,
            "paper_text": paper.text,
            "user_id": user_id,
            "selected_models": list(models_to_use.keys()),
        }

    return task_id


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get the current status of an analysis task.
    Args:
        task_id: The task ID to check.
    Returns:
        Task status dictionary or None if not found.
    """
    with _tasks_lock:
        task = _analysis_tasks.get(task_id)
        if task:
            return task.copy()
        return None


def run_analysis_task(task_id: str):
    """Run the analysis task in a background thread.
    Args:
        task_id: The task ID to run.
    """

    def _update_progress(step: str, increment: bool = True):
        with _tasks_lock:
            task = _analysis_tasks.get(task_id)
            if task:
                task["current_step"] = step
                if increment:
                    task["completed_steps"] += 1
                task["progress"] = int(
                    (task["completed_steps"] / task["total_steps"]) * 100
                )

    def _run():
        try:
            with _tasks_lock:
                task = _analysis_tasks.get(task_id)
                if not task:
                    return
                task["status"] = "running"
                pdf_text = task["paper_text"]
                paper_id = task["paper_id"]
                user_id = task.get("user_id")
                selected_models = task.get("selected_models", [])
                total_steps = task.get("total_steps", 0)
                completed_steps = task.get("completed_steps", 0)

            # Get model configs from database
            model_configs = get_model_configs()

            # Get system prompt
            system_prompt = Prompt.objects.filter(name="system_prompt").first()
            system_prompt = system_prompt.template + "\n" + get_criterions()

            # Analyze with selected LLM models
            results = {}

            for model_key in selected_models:
                config = model_configs.get(model_key)
                if not config:
                    continue
                visual_name = config.get("visual_name", model_key)
                _update_progress(f"Analyzing with {visual_name}...", increment=False)

                client = OpenAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                )
                # check if the code already exists in the paper
                _update_progress(f"Checking for pre-existing code...", increment=False)
                code_text = None
                paper = Paper.objects.filter(id=paper_id).first()
                if paper.code_text == "":
                    if paper.code_url == "":
                        # GITHUB url regex
                        url_pattern = re.compile(
                            r"https?://(?:www\.)?github\.com/[A-Za-z0-9-]{1,39}/[A-Za-z0-9_.-]+(?:\.git)?"
                        )
                        match = url_pattern.search(paper.text)
                        url = match.group(0) if match else None
                        if url:
                            url = url.rstrip(".,;:!?)\"]'")
                            print("MATCH:", url)
                            paper.code_url = url

                    # GITHUB ingestion
                    if "github" in paper.code_url:
                        _update_progress(
                            f"Code not ingested, starting ingestion process...",
                            increment=False,
                        )
                        code_text = get_code(paper.code_url)
                        paper.code_text = code_text
                        paper.save(update_fields=["code_text", "code_url"])
                        config["code_tool"] = False

                        _update_progress(
                            f"Ingestion completed successfully, continuing analysis...",
                            increment=False,
                        )

                    else:
                        # TODO Abilitare code tool di GPT
                        config["code_tool"] = True
                        config["paper_id"] = paper_id

                else:
                    code_text = paper.code_text
                    config["code_tool"] = False

                _update_progress(
                    f"Sending request to {visual_name} model...", increment=False
                )

                # Create a progress callback for detailed updates (no increment to prevent overflow)
                def model_progress(msg):
                    _update_progress(
                        f"[{completed_steps+1}/{total_steps}] {visual_name}: {msg}",
                        increment=False,
                    )

                result = llm_analysis(
                    client,
                    system_prompt,
                    pdf_text,
                    config,
                    code_text=code_text,
                    progress_callback=model_progress,
                )

                result["result"] = result["result"].dict()

                results[model_key] = result

                # Save analysis to database
                _update_progress(
                    f"Saving analysis results to database...", increment=False
                )
                analysis = _save_analysis_to_db(
                    paper_id, model_key, config, result, user_id
                )

                if analysis:
                    results[model_key]["final_score"] = analysis.final_score()

                _update_progress(f"Completed {visual_name}")

            # Mark as completed
            with _tasks_lock:
                task = _analysis_tasks.get(task_id)
                if task:
                    task["status"] = "completed"
                    task["progress"] = 100
                    task["current_step"] = "Analysis complete"
                    task["results"] = results

        except Exception as e:
            logger.exception(f"Error in analysis task {task_id}: {e}")
            with _tasks_lock:
                task = _analysis_tasks.get(task_id)
                if task:
                    task["status"] = "error"
                    task["error"] = str(e)
                    task["current_step"] = f"Error: {str(e)}"

    # Start background thread
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def cleanup_task(task_id: str):
    """Remove a completed task from memory.
    Args:
        task_id: The task ID to remove.
    """
    with _tasks_lock:
        if task_id in _analysis_tasks:
            del _analysis_tasks[task_id]
