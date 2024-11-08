import requests
from bs4 import BeautifulSoup
import os
import urllib

# The URL to scrape
url = "https://docs.llamaindex.ai/en/stable/"

# The directory to store files in
output_dir = "./llamaindex-docs/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Fetch the page with headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links
links = soup.find_all("a", href=True)

for link in links:
    href = link["href"]

    # Check if the link ends with .html or is a directory
    if href.endswith(".html") or "/" in href:
        # Make a full URL if necessary
        if not href.startswith("http"):
            href = urllib.parse.urljoin(url, href)

        # Fetch the file
        print(f"downloading {href}")
        file_response = requests.get(href, headers=headers)

        # Construct the file name
        base_name = os.path.basename(href.strip("/"))

        # If it's a directory or doesn't have .html, add .html extension
        if not base_name.endswith(".html"):
            base_name = base_name + ".html"

        file_name = os.path.join(output_dir, base_name)

        # Write it to a file
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(file_response.text)
