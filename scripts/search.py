import re
import asyncio
import aiohttp
import pmc_python
from lxml import etree

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARTICLE_RE = re.compile(
    r"<article\b[^>]*?>.*?</article>",
    re.IGNORECASE | re.DOTALL
)

async def get_open_access_xml(query: str, n: int = 3,) -> str | None:
    esearch_params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": str(n)
    }

    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"{BASE}/esearch.fcgi", params=esearch_params) as r:
            r.raise_for_status()
            data = await r.json()
            ids = data["esearchresult"]["idlist"]

        if not ids:
            return None

        efetch_params = {
            "db": "pmc",
            "id": ",".join(ids),
        }
        async with sess.get(f"{BASE}/efetch.fcgi", params=efetch_params) as r:
            r.raise_for_status()
            xml_text = await r.text()

    xml_list = ARTICLE_RE.findall(xml_text)
    markdown_list = []
    for xml in xml_list:
        doc = etree.fromstring(xml.encode())
        markdown = pmc_python.to_markdown(doc)
        markdown_list.append(markdown)
    return markdown_list


if __name__ == "__main__":
    # avoid searching for non-commercial and non-derivative licenses
    query = """
    diabetes AND open access[filter] NOT (
    "cc by-nc license"[filter] OR
    "cc by-nc-sa license"[filter] OR
    "cc by-nc-nd license"[filter] OR
    "cc by-nd license"[filter]
    )
    """
    markdown_list = asyncio.run(get_open_access_xml(query, n=3))
    for mk in markdown_list:
        print(mk[:1000] + f" ({len(mk)-1000} characters truncated...)")