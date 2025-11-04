import os
import json
import re

import requests
from pathlib import Path
from dotenv import load_dotenv

from crawl4ai import *
import asyncio

load_dotenv(".env.llm")
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"


def get_response(url, timeout=10):
    """
    Fetch a web page from the given URL
    Returns:
        requests.Response: The response object containing the page content
    """
    if not url.startswith("https://"):
        url = "https://" + url

    try:
        page = requests.get(url, timeout=timeout)
        page.raise_for_status()

        return page
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


text_content = {}


def extract_sections(markdown_text):
    """
    Automatically extract all H1 sections from markdown text.

    Returns:
        dict: Dictionary with section names as keys and content as values
    """
    sections = {}
    # Find all H1 headers
    h1_pattern = r"^# (.+)$"
    matches = list(re.finditer(h1_pattern, markdown_text, re.MULTILINE))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_pos = match.end()

        # Determine end position (next H1 or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(markdown_text)

        content = markdown_text[start_pos:end_pos].strip()
        key = section_name.lower().replace(" ", "_")
        sections[key] = content

    return sections


def get_schema(html_sample=None, model=None):

    if model is not None:
        schema_file = f"{MEDIA_DIR}/home_schema_{model}.json"
    else:
        schema_file = f"{MEDIA_DIR}/home_schema.json"

    try:
        with open(schema_file, "r") as f:
            schema = json.load(f)
            print(f"Loaded Schema from {schema_file}")

    except FileNotFoundError:
        schema = JsonCssExtractionStrategy.generate_schema(
            html=html_sample,
            llm_config=LLMConfig(
                provider="gemini/gemini-2.5-pro",
                api_token=os.getenv("GEMINI_API_KEY"),
            ),
            query="From https://papers.miccai.org/miccai-2025/, i shared a sample html structure of a paper listing. Please generate a schema for this div extracting only the paper title, authors list, and the link to the paper information and reviews",
        )
        print(f"Generated Schema with GEMINI")

        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

    return schema


async def home_crawling(url, schema, model=None):
    try:
        with open(f"{MEDIA_DIR}/home_claude.json", "r") as f:
            data = json.load(f)
            print(f"Loaded Paper list")
            return data
    except FileNotFoundError:
        extraction_strategy = JsonCssExtractionStrategy(schema)
        config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

        async with AsyncWebCrawler() as crawler:
            result: CrawlResult = await crawler.arun(
                url=url,
                config=config,
            )

            if model is not None:
                file_name = f"{MEDIA_DIR}/home_{model}.json"
            else:
                file_name = f"{MEDIA_DIR}/home.json"

            if result.success:
                data = json.loads(result.extracted_content)
                with open(file_name, "w") as f:
                    json.dump(data, f, indent=2)
                return data
            else:
                raise Exception(f"Crawling failed for {url}")


async def paper_crawling(url, schema):
    extraction_strategy = JsonCssExtractionStrategy(schema)
    config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

    async with AsyncWebCrawler() as crawler:
        result: CrawlResult = await crawler.arun(
            url=url,
            config=config,
        )

        paper_id = url.split("/")[-1].replace(".html", "")
        file_name = f"{MEDIA_DIR}/{paper_id}.json"

        paper_md = result.markdown
        return extract_sections(paper_md)


def clean_section(paper, section, section_name):

    if section_name == "link_to_the_dataset(s)":
        dataset_pattern = r"([^:]+):\s*<([^>]+)>"
        matches = re.findall(dataset_pattern, section)
        urls = [url.strip() for name, url in matches]
        paper["datasets"] = ", ".join(urls) if urls else None

    elif section_name == "link_to_the_code_repository":
        code_pattern = r"<([^>]+)>"
        match = re.search(code_pattern, section)
        if match:
            paper["code_url"] = match.group(1).strip()
        else:
            paper["code_url"] = None

    elif section_name == "links_to_paper_and_supplementary_materials":
        # Extract DOI value (everything after "SpringerLink (DOI):" until newline)
        doi_pattern = r"SpringerLink \(DOI\):\s*(.+?)(?:\n|$)"
        doi_match = re.search(doi_pattern, section)

        # Extract Supplementary Material value (everything after "Supplementary Material:" until newline or end)
        supp_pattern = r"Supplementary Material:\s*(.+?)(?:\n|$)"
        supp_match = re.search(supp_pattern, section)

        if doi_match:
            doi_value = doi_match.group(1).strip()
            if doi_value.lower() in ["not yet available", "not available"]:
                paper["doi"] = None
            else:
                paper["doi"] = doi_value
        else:
            paper["doi"] = None

        if supp_match:
            supp_value = supp_match.group(1).strip()
            if supp_value.lower() in ["not submitted", "not available"]:
                paper["supp_materials"] = None
            else:
                paper["supp_materials"] = supp_value
        else:
            paper["supp_materials"] = None

    elif section_name == "meta-review":
        paper["meta_review"] = section

    else:
        paper[section_name] = section


async def get_data(base_url, url, model=None, html_sample=None):

    # Generate schema if not provided
    schema = get_schema(html_sample, model)

    # Home crawling
    home_data = await home_crawling(url, schema, model)

    # Paper crawling
    papers_list = []
    count = 0  # TESTING
    for paper in home_data:
        if count >= 1:  # TESTING
            break
        count += 1  # TESTING
        if paper["paper_url"]:
            if not paper.get("paper_url").startswith("https://"):
                paper["paper_url"] = base_url + paper.get("paper_url")
            paper_sections = await paper_crawling(paper["paper_url"], schema)
            valid_sections = [
                "abstract",
                "links_to_paper_and_supplementary_materials",
                "link_to_the_code_repository",
                "link_to_the_dataset(s)",
                "reviews",
                "author_feedback",
                "meta-review",
            ]
            # prendere solo le sezioni in sections e aggiungerle agli altri dati
            for paper_section in paper_sections:
                if paper_section in valid_sections:
                    clean_section(paper, paper_sections[paper_section], paper_section)

            papers_list.append(paper)

    # Write all papers as a list of dictionaries
    with open(f"{MEDIA_DIR}/papers_info.json", "w", encoding="utf-8") as f:
        json.dump(papers_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    url = "https://papers.miccai.org/miccai-2025"

    asyncio.run(get_data("https://papers.miccai.org", url, model="claude"))
    # print("\nAll sections:", all_sections)
