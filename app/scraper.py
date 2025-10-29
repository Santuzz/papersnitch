import re
import requests
import os
import json


def get_response(url, timeout=10):
    """
    Fetch a web page from the given URL

    Args:
        url (str): The URL to fetch

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


def download_json(url, output_filename):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, output_filename)

    print(f"Attempting to download from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"File successfully saved to: {save_path}")

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")
    except IOError as e:
        print(f"Error writing file to disk: {e}")


def read_jina(url):
    url = "https://r.jina.ai/" + url
    response = get_response(url, timeout=1000)
    # print(response.text)

    text_content = {}
    if response and response.status_code == 200:
        lines = response.text.splitlines()

    # get abstract
    offset = 3
    marker = "Abstract"
    target_index = None

    for index, line in enumerate(lines):
        if marker in line:
            target_index = index + offset

            if target_index < len(lines):
                text_content[marker.lower()] = lines[target_index].strip()
            else:
                print(f"Found '{marker}' but offset {offset} is out of bounds.")
                return None
            break

    # get reviews
    lines = lines[target_index:]
    n = 1
    offset = 2
    marker = f"### Review #{n}"
    marker_next = f"### Review #{n+1}"
    marker_break = "Author Feedback"
    target_start = None
    target_end = None

    for index, line in enumerate(lines):
        if marker_break in line:
            print(marker, target_start, index)
            if target_start is not None and index is not None:
                text_content[f"review_{n}"] = "\n".join(
                    lines[target_start:index]
                ).strip()
            target_start = index + offset
            break
        if marker in line:
            target_start = index + offset

        if marker_next in line:
            target_end = index
            if target_start is not None and target_end is not None:
                text_content[f"review_{n}"] = "\n".join(
                    lines[target_start:target_end]
                ).strip()
                n += 1
                marker = f"### Review #{n}"
                marker_next = f"### Review #{n+1}"
                target_start = target_end
                target_end = None
            # print(text_content)

    print(text_content)


def change_json(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)
    base_url = "https://papers.miccai.org"
    no_url = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # get only the first entry
        if data:
            data = [data[0]]
        for entry in data:
            if "url" in entry:
                entry["url"] = base_url + entry.get("url")
                text_content = read_jina(entry["url"])
                for key, value in text_content.items():
                    entry[key] = value
            else:
                no_url += 1

        if no_url > 0:
            print(f"Number of entries without URL: {no_url}")

        # with open(file_path, "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"File successfully updated: {file_path}")

    except IOError as e:
        print(f"Error reading or writing file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


if __name__ == "__main__":

    url = "https://papers.miccai.org/miccai-2025/js/search.json"
    filename = "miccai_2025_papers.json"
    # download_json(url, filename)

    change_json(filename)


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) > 1:
#         url = sys.argv[1]
#         response = main(url)
#         if response:
#             print(f"Successfully fetched {url}")
#             print(f"Status code: {response.status_code}")
#             print(f"Content length: {len(response.content)} bytes")
#     else:
#         print("Usage: python scraper.py <url>")
