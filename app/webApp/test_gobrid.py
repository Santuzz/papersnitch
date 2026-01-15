from bs4 import BeautifulSoup
import requests
from pathlib import Path


def get_clean_content(pdf_path):
    """
    Extract clean title and text from a PDF using Grobid and BeautifulSoup.
    Args:
        pdf_path: Path to the PDF file.
    Returns:
        A tuple containing the title and cleaned text."""
    grobid_url = "http://localhost:8070/api/processFulltextDocument"
    params = {"consolidateHeader": "0", "consolidateCitations": "0"}

    try:
        with open(pdf_path, "rb") as f:
            files = {"input": f}
            response = requests.post(grobid_url, files=files, data=params)

        if response.status_code != 200:
            print("Errore Grobid")
            return None, None

        xml_content = response.text

    except Exception as e:
        print(f"Errore di connessione: {e}")
        return None, None

    # 2. BeautifulSoup parsing
    soup = BeautifulSoup(xml_content, "xml")

    # Title extraction
    title_tag = soup.find("title", level="a") or soup.find("title", type="main")
    # Title Fallback
    if not title_tag:
        title_tag = soup.find("title")

    title = title_tag.get_text(strip=True) if title_tag else "Titolo not Found"

    # Text extraction

    body = soup.find("body")

    text_content = []

    if body:
        # find all paragraph (<p>) and the section title (<head>)
        for tag in body.find_all(["head", "p"]):
            # Remove bibliographic references
            for ref in tag.find_all("ref", type="bibr"):
                ref.decompose()

            text_content.append(tag.get_text(strip=True))

    # concatenate all text parts
    full_text = "\n\n".join(text_content)

    return title, full_text


# --- Utilizzo ---
if __name__ == "__main__":
    path = Path(
        "/home/dsantoli/papersnitch/app/media/pdf/20251128163055_0308_paper.pdf"
    )
    titolo, testo = get_clean_content(path)

    print(f"--- TITLE ---\n{titolo}\n")
    print(
        f"--- TEXT ---\n{testo[:500]}..."
    )  # Stampo solo i primi 500 caratteri per prova
