import asyncio
from crawl4ai import *
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import re

load_dotenv(".env.llm")
BASE_DIR = Path(__file__).resolve().parent


async def get_list(model):

    # schema = JsonCssExtractionStrategy.generate_schema(
    #     html=html_schema,
    #     llm_config=LLMConfig(
    #         provider="gemini/gemini-2.5-pro",
    #         api_token=os.getenv("GEMINI_API_KEY"),
    #     ),
    #     query="From https://papers.miccai.org/miccai-2025/, i shared a sample html structure of a paper listing. Please generate a schema for this div extracting only the paper title, authors list, and the link to the paper information and reviews",
    # )
    # print(f"Generated Schema: {json.dumps(schema,indent =2)}")
    # with open("home_schema.json", "w") as f:
    #     json.dump(schema, f, indent=2)

    with open(f"{BASE_DIR}/media/home_schema_{model}.json", "r") as f:
        schema = json.load(f)

    extraction_strategy = JsonCssExtractionStrategy(schema)
    config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

    async with AsyncWebCrawler() as crawler:
        results: list[CrawlResult] = await crawler.arun(
            url="https://papers.miccai.org/miccai-2025/",
            config=config,
        )

        for result in results:
            file_name = f"{BASE_DIR}/media/test_home_{model}.json"
            if result.success:
                data = json.loads(result.extracted_content)
                with open(file_name, "w") as f:
                    json.dump(data, f, indent=2)


async def get_paper(paper, model):
    # Extract the paper ID or filename from the paper_info_link
    # Example: /miccai-2025/0001-Paper0308.html -> 0001-Paper0308
    paper_id = paper["paper_info_link"].split("/")[-1].replace(".html", "")
    md_file = f"{BASE_DIR}/media/{paper_id}.md"

    if not os.path.exists(md_file):
        print(f"Markdown file not found: {md_file}")
        return

    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract abstract
    abstract_match = re.search(r"#\s*Abstract\s*\n(.+?)(?=\n#|\Z)", content, re.DOTALL)
    paper["abstract"] = abstract_match.group(1).strip() if abstract_match else None

    # Extract DOI
    doi_match = re.search(r"SpringerLink \(DOI\):\s*(.+)", content)
    if doi_match:
        paper["doi"] = doi_match.group(1).strip()
    else:
        paper["doi"] = None

    # Extract supplementary material
    supp_match = re.search(r"Supplementary Material:\s*(.+)", content)
    paper["supplementary_material"] = (
        supp_match.group(1).strip() if supp_match else None
    )

    # Extract code repository
    code_match = re.search(
        r"#\s*Link to the Code Repository\s*\n\[(.+?)\]\((.+?)\)", content
    )
    paper["code_repository"] = code_match.group(2) if code_match else None

    # Extract datasets as a list
    datasets = []
    dataset_section = re.search(
        r"#\s*Link to the Dataset\(s\)\s*\n(.+?)(?=\n#|\Z)", content, re.DOTALL
    )
    if dataset_section:
        dataset_links = re.findall(
            r"(.+?):\s*\[(.+?)\]\((.+?)\)", dataset_section.group(1)
        )
        for name, text, url in dataset_links:
            datasets.append(
                {"description": f"{name.strip()}: {text.strip()}", "url": url.strip()}
            )
    paper["datasets"] = datasets if datasets else None

    # Extract reviews as a list
    reviews = []
    review_pattern = re.compile(
        r"###\s*Review #(\d+)\s*\n(.+?)(?=\n###\s*Review #|\n---\n#\s*Author Feedback|\Z)",
        re.DOTALL,
    )
    for match in review_pattern.finditer(content):
        reviews.append(
            {
                "review_number": match.group(1),
                "review_full_text": match.group(2).strip(),
            }
        )
    paper["reviews"] = reviews if reviews else None

    # Extract author feedback
    feedback_match = re.search(
        r"#\s*Author Feedback\s*\n>(.+?)(?=\n---\n#|\Z)", content, re.DOTALL
    )
    paper["author_feedback"] = (
        feedback_match.group(1).strip() if feedback_match else None
    )

    # Extract meta-reviews as a list
    meta_reviews = []
    meta_pattern = re.compile(
        r"##\s*Meta-review #(\d+)\s*\n(.+?)(?=\n##\s*Meta-review #|\Z)", re.DOTALL
    )
    for match in meta_pattern.finditer(content):
        meta_reviews.append(
            {
                "meta_review_number": match.group(1),
                "meta_review_full_text": match.group(2).strip(),
            }
        )
    paper["meta_reviews"] = meta_reviews if meta_reviews else None


if __name__ == "__main__":
    model = "claude"
    # asyncio.run(get_list(model))

    file_name = f"{BASE_DIR}/media/test_home_{model}.json"
    with open(file_name, "r") as f:
        data = json.load(f)
    for paper in data:
        asyncio.run(get_paper(paper, model))
        break

    file_name = f"{BASE_DIR}/media/test_completo_{model}.json"
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)


"""async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://papers.miccai.org/miccai-2025/0001-Paper0308.html",
        )
        file_name = f"{BASE_DIR}/media/0001-Paper0308.md"
        with open(file_name, "w") as f:
            f.write(result.markdown)


if __name__ == "__main__":
    asyncio.run(main())"""
