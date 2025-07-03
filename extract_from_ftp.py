import re
import tarfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from lxml import etree

import pmc_python
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED

# Constants ---------------------------------------------------------------

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
# Regex for files like: oa_comm_xml.PMC0001234.baseline.2025-06-26.tar.gz
TARFILE_RE = re.compile(r"oa_comm_xml\.PMC[^\s\"'>]+\.tar\.gz", re.IGNORECASE)

LICENSE_FILTER = ["BY-ND", "BY ND"]  # Licence substrings to filter
OBESITY_DIABETES_WEIGHT_LOSS_FILTER = True
_TOPIC_RE = re.compile(r"\b(obesity|weight[\s\-]?loss|diabetes|diabetic)\b", re.I)


# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------

def _matches_topic(doc: etree._Element) -> bool:
    """Return True when the article title / abstract / keywords mention the target topics."""
    title = " ".join(doc.xpath(".//article-title//text()"))
    abstract = " ".join(doc.xpath(".//abstract//text()"))
    keywords = " ".join(doc.xpath(".//kwd-group//kwd//text()"))

    haystack = f"{title} {abstract} {keywords}".lower()
    if OBESITY_DIABETES_WEIGHT_LOSS_FILTER:
        return bool(_TOPIC_RE.search(haystack))
    return True


def _robust_session(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """Return a requests.Session configured with retry logic."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def list_remote_tarfiles(session: requests.Session) -> List[str]:
    """Return a list of tar-file names available under BASE_URL that match TARFILE_RE."""
    resp = session.get(BASE_URL, timeout=30)
    resp.raise_for_status()

    # Extract href targets from simple directory index (each entry is on its own line)
    matches = TARFILE_RE.findall(resp.text)
    return sorted(set(matches))


def download_tarfile(session: requests.Session, filename: str) -> BytesIO:
    """Download *filename* under BASE_URL into a BytesIO buffer with progress bar."""
    url = BASE_URL + filename
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        buf = BytesIO()
        desc = filename.split("/")[-1]
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {desc}") as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    buf.write(chunk)
                    pbar.update(len(chunk))
        buf.seek(0)
        return buf


def _process_xml(xml_bytes: bytes, member_name: str, output_dir_str: str) -> str:
    """Worker function executed in a separate process.

    Parses *xml_bytes*, applies filtering logic, and writes markdown to *output_dir_str*.

    Returns one of the status strings: "saved", "filtered", "error".
    """
    try:
        output_dir = Path(output_dir_str)
        doc = etree.fromstring(xml_bytes)

        metadata = pmc_python.to_metadata(doc) or {}
        license = metadata.get("License", "")

        # Skip filtered licences / topics
        if not license or any(l in license.upper() for l in LICENSE_FILTER):
            return "filtered"

        # Skip non research-article types (e.g. corrections)
        article_type = (doc.get("article-type") or "").lower()
        if article_type != "research-article":
            return "filtered"

        if not _matches_topic(doc):
            return "filtered"

        markdown = pmc_python.to_markdown(doc)

        base_name = Path(member_name).stem
        output_path = output_dir / f"{base_name}.md"
        with output_path.open("w+", encoding="utf-8") as fh:
            fh.write(markdown)
        return "saved"
    except Exception:
        return "error"


def process_tar(buf: BytesIO, output_dir: Path, max_workers: Optional[int] = None) -> None:
    """Extract eligible XML articles from *buf* to markdown in *output_dir* using multiple processes."""
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".xml")]

        saved = filtered = errors = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for m in tqdm(members, desc="Reading XML"):
                with tf.extractfile(m) as f:
                    xml_bytes = f.read()
                futures.append(
                    executor.submit(_process_xml, xml_bytes, m.name, str(output_dir))
                )

            # Process futures with a watchdog so that a single misbehaving task
            # cannot block the entire run.  We allow up to 60 s of silence; any
            # tasks that are still pending after that will be cancelled and
            # counted as errors.
            pending = set(futures)
            with tqdm(total=len(futures), desc="Processing XML", leave=False) as pbar:
                while pending:
                    # Wait until at least one future completes or the timeout expires.
                    done, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)

                    # If the timeout elapsed with *no* future completing, cancel the oldest one.
                    if not done:
                        # Cancel all remaining stuck futures to unblock the loop.
                        for fut in list(pending):
                            fut.cancel()
                            errors += 1
                            pbar.update(1)
                        break

                    for fut in done:
                        try:
                            status = fut.result()
                        except FuturesTimeoutError:
                            status = "error"
                        except Exception:
                            status = "error"

                        if status == "saved":
                            saved += 1
                        elif status == "filtered":
                            filtered += 1
                        else:
                            errors += 1

                        pbar.update(1)
                        pbar.set_postfix(saved=saved, filtered=filtered, errors=errors)

    print(f"Summary => saved: {saved}, filtered: {filtered}, errors: {errors}")


# ------------------------------------------------------------------------
# Main script logic
# ------------------------------------------------------------------------

def main() -> None:
    output_dir = Path("data/markdown_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    sess = _robust_session()

    print("Fetching file list…")
    tar_files = list_remote_tarfiles(sess)[12:]
    if not tar_files:
        print("No matching tar files found at remote location.")
        return

    for fname in tar_files:
        print(f"\n=== Handling archive: {fname} ===")
        buf = download_tarfile(sess, fname)
        process_tar(buf, output_dir)


if __name__ == "__main__":
    main() 