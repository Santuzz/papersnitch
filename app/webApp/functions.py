from gitingest import ingest

import django
import os
from datetime import date
from pathlib import Path
from pypdf import PdfReader

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    os.getenv("DJANGO_SETTINGS_MODULE", "web.settings.development"),
)

django.setup()

from webApp.models import Paper, TokenUsage
from bs4 import BeautifulSoup
import requests


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

    summary, tree, content = ingest(
        url, include_patterns=["*.md", "docs/", "*.ipynb", "LICENSE*", "main*"]
    )
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


def get_pdf_content(pdf_path):
    """
    Extract clean title and text from a PDF using Grobid and BeautifulSoup.
    Args:
        pdf_path: Path to the PDF file.
    Returns:
        A tuple containing the title and cleaned text."""
    # Use host.docker.internal to reach GROBID running on host from Docker container
    base_url = os.environ.get("GROBID_URL", "http://grobid:8070")
    grobid_url = f"{base_url}/api/processFulltextDocument"
    params = {"consolidateHeader": "0", "consolidateCitations": "0"}

    try:
        with open(pdf_path, "rb") as pdf_file:
            files = {"input": ("paper.pdf", pdf_file, "application/pdf")}
            response = requests.post(grobid_url, files=files, data=params, timeout=60)

        if response.status_code != 200:
            print(f"Error Grobid: {response.status_code}")
            return None, None

        xml_content = response.text

    except Exception as e:
        print(f"Error: {e}")
        return None, None

    # 2. BeautifulSoup parsing
    soup = BeautifulSoup(xml_content, "xml")

    # Title extraction
    title_tag = soup.find("title", level="a") or soup.find("title", type="main")
    # Title Fallback
    if not title_tag:
        title_tag = soup.find("title")

    title = title_tag.get_text(strip=True) if title_tag else "Title not Found"

    # Abstract extraction
    abstract_tag = soup.find("abstract")
    abstract_text = ""
    if abstract_tag:
        # Get all paragraph text from abstract
        abstract_paragraphs = abstract_tag.find_all("p")
        if abstract_paragraphs:
            abstract_text = "\n\n".join(
                p.get_text(strip=True) for p in abstract_paragraphs
            )
        else:
            abstract_text = abstract_tag.get_text(strip=True)

    # Text extraction

    body = soup.find("body")

    text_content = []

    # Add abstract at the beginning
    if abstract_text:
        text_content.append("Abstract\n\n" + abstract_text)

    if body:
        # find all paragraph (<p>), section title (<head>), figures and tables
        for tag in body.find_all(["head", "p", "figure"]):
            # Skip head tags that are inside figures (they're handled with the figure)
            if tag.name == "head" and tag.find_parent("figure"):
                continue

            # Remove bibliographic references
            for ref in tag.find_all("ref", type="bibr"):
                ref.decompose()

            # For figures, extract the caption/description
            if tag.name == "figure":
                fig_head = tag.find("head")
                fig_desc = tag.find("figDesc")
                table = tag.find("table")

                fig_text_parts = []

                # Add label (e.g., "Table 1" or "Fig. 1")

                if fig_head and table:
                    fig_text_parts.append(fig_head.get_text(strip=True))

                if fig_desc:
                    fig_text_parts.append(fig_desc.get_text(strip=True))

                # Extract table content with structure
                if table:
                    table_rows = []
                    for row in table.find_all("row"):
                        cells = [
                            cell.get_text(strip=True) for cell in row.find_all("cell")
                        ]
                        table_rows.append(" | ".join(cells))
                    if table_rows:
                        fig_text_parts.append("\n".join(table_rows))

                if fig_text_parts:
                    text_content.append(" ".join(fig_text_parts))
            else:
                text_content.append(tag.get_text(strip=True))

    # concatenate all text parts
    text = "\n\n".join(text_content)

    return title, text


# def upload_pdf(pdf_file):
#     """Upload a PDF file in the DB
#     pdf_file comes from request.FILES["pdf_file"]"""

#     # Read first bytes to check if it's a valid PDF
#     first_bytes = pdf_file.read(4)
#     pdf_file.seek(0)  # Reset file pointer

#     if not first_bytes.startswith(b"%PDF"):
#         print(f"Downloaded file is not a valid PDF")
#         return None

#     title, text = get_pdf_content(pdf_file)

#     # get the paper from db using name
#     existing_paper = Paper.objects.filter(title=title).first()

#     if existing_paper:
#         return existing_paper

#     # Extract text from the PDF
#     pdf_file.seek(0)  # Reset file pointer for saving

#     # Read file content for storage
#     pdf_content = pdf_file.read()
#     pdf_content = ContentFile(pdf_content, name=title + ".pdf")

#     paper = Paper.objects.create(title=title, file=pdf_content, text=text)

#     return paper
