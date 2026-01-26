import json
from pathlib import Path
import sys
import os
import time

import argparse

from dotenv import load_dotenv
from pypdf import PdfReader

from google import genai
from pydantic import BaseModel, Field
from typing import Optional, Dict


from gitingest import ingest_async, ingest


api_key = os.getenv("GEMINI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env.llm")
PDF_DIR = Path(__file__).resolve().parent / "media" / "pdf"


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Load papers from JSON file into the database"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to text file",
    )
    args = parser.parse_args()
    return args


class Paper(BaseModel):
    """Schema for extracting information from scientific papers"""

    authors_mail: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping author names to their email addresses, the amount of emails always correspond to the number of authors, the authors should be saved in the same order as you found in the paper",
    )
    datasets: Optional[Dict[str, str]] = Field(
        default=None, description="Dictionary mapping dataset names to their citation"
    )
    code_url: Optional[str] = Field(
        default=None, description="URL of the code repository"
    )


def get_code_documentation(url, file=None):

    summary, tree, content = ingest(url, include_patterns=["*.md", "docs/", "*.ipynb"])
    return {"summary": summary, "tree": tree, "content": content}


def get_prompt(category):
    if category == "cleaning":
        return "### INSTRUCTION: You are an expert data cleaner. Follow these steps: 1. Analyse the DATA. 2. Clean the DATA by removing any noise related the fact the text provided comes from a pdf 3. Fix the synthax of all the formulas and math characters using a mathjax one 4. Do not provide any explanation or further text. Only provide the cleaned information. —- ### DATA:"
    elif category == "extraction":
        return '### INSTRUCTION: You are an expert at reading scientific papers. Follow these steps: 1. Extract these information from the PAPER, the author mail "authors_mail" (as a dictionary), the datasets used in the paper "datasets" (as a dictionary), the code of the repository "code_url" 2. Structure your response using the following JSON schema: { "authors_mail":{"Name Surname":"mail@example.com"},"datasets":{"name_1":"url","name_2":"url",...}, "code_url": "..."} 3. If a value is not found put null 4. Do not provide any explanation or further text Only provide the JSON. —- ### PAPER:'


def read_pdf(pdf_name):
    reader = PdfReader(PDF_DIR / pdf_name)
    pages = reader.pages
    text = ""
    for page in pages:
        text = text + page.extract_text()
    return text
