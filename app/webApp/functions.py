from gitingest import ingest

import django
from django.core.files.base import ContentFile
import os
from datetime import date
from pathlib import Path
from pypdf import PdfReader

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
django.setup()

from webApp.models import Paper, TokenUsage


def update_token(tokens_used: int, token_var: str) -> int:
    """Update the cumulative token usage for a given LLM model.

    Resets the counter at the beginning of each day and stores usage in the database.

    Args:
        tokens_used: The number of tokens used in the current API call.
        token_var: The environment variable name that tracks the cumulative token usage.
    Returns:
        The updated cumulative token usage for today.
    """
    today = date.today()

    # Get or create today's token usage record
    token_record, created = TokenUsage.objects.get_or_create(
        date=today, model_var=token_var, defaults={"tokens_used": 0}
    )

    # Update the token count
    token_record.tokens_used += tokens_used
    token_record.save()

    return token_record.tokens_used


# Token limits for free tier
TOKEN_LIMITS = {
    "TOTAL_TOKEN_OPENAI": 500000,
    "TOTAL_TOKEN_OPENAI_MINI": 1250000,
}


class TokenLimitExceededError(Exception):
    """Raised when the token usage exceeds the allowed limit."""

    pass


def check_token_limit(token_var: str) -> bool:
    """Check if the token usage is within the allowed limit.
    Args:
        token_var: The token variable name to check.
    Returns:
        True if within limit, raises TokenLimitExceededError if exceeded.
    Raises:
        TokenLimitExceededError: If the token limit has been exceeded.
    """

    if token_var not in TOKEN_LIMITS:
        return True  # No limit for this token variable

    today = date.today()
    limit = TOKEN_LIMITS[token_var]

    try:
        token_record = TokenUsage.objects.filter(
            date=today, model_var=token_var
        ).first()

        current_usage = token_record.tokens_used if token_record else 0

        if current_usage >= limit:
            raise TokenLimitExceededError(
                f"Maximum amount of free tokens exhausted for {token_var}. "
                f"Used: {current_usage:,} / Limit: {limit:,}. "
                "Contact the admin to get access to more tokens."
            )
    except TokenUsage.DoesNotExist:
        pass  # No usage yet, within limit

    return True


def get_text(pdf_file):
    """Extract text from a PDF file object.
    Args:
        pdf_file: File object of the PDF.
    Returns:
        Extracted text from all pages.
    """
    reader = PdfReader(pdf_file)
    pages = reader.pages
    text = ""
    for page in pages:
        text = text + page.extract_text()
    return text


def get_code(url: str):
    """Ingest code from a given URL.
    Args:
        url: URL of the code repository.
    Returns:
        A dictionary with summary, tree, and content of the ingested code.
    """

    summary, tree, content = ingest(url, include_patterns=["*.md", "docs/", "*.ipynb"])
    return f"SUMMARY\n{summary}\nTREE\n{tree}\nCONTENT\n{content}"


def read_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file.
    Args:
        pdf_path: Path to the PDF file.
    Returns:
        Extracted text from all pages.
    """
    reader = PdfReader(pdf_path)
    pages = reader.pages
    text = ""
    for page in pages:
        text = text + page.extract_text()
    return text


def upload_pdf(pdf_file):
    """Upload a PDF file in the DB
    pdf_file comes from request.FILES["pdf_file"]"""

    # Read first bytes to check if it's a valid PDF
    first_bytes = pdf_file.read(4)
    pdf_file.seek(0)  # Reset file pointer

    if not first_bytes.startswith(b"%PDF"):
        print(f"Downloaded file is not a valid PDF")
        return None

    text = get_text(pdf_file)
    # take the first line of the text as the paper title
    name = text.split("\n")[0]

    # get the paper from db using name
    existing_paper = Paper.objects.filter(title=name).first()

    if existing_paper:
        return existing_paper

    # Extract text from the PDF
    pdf_file.seek(0)  # Reset file pointer for saving

    # Read file content for storage
    pdf_content = pdf_file.read()
    pdf_content = ContentFile(pdf_content, name=name + ".pdf")

    paper = Paper.objects.create(title=name, file=pdf_content, text=text)

    return paper
