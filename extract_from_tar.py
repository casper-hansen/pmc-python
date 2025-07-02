import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tarfile
from typing import Optional

import pmc_python
from lxml import etree


LICENSE_FILTER = ["BY-ND", "BY ND"]  # Licence substrings to filter
OBESITY_DIABETES_WEIGHT_LOSS_FILTER = True
_TOPIC_RE = re.compile(r"\b(obesity|weight[\s\-]?loss|diabetes|diabetic)\b", re.I)

# Global TarFile instance for worker processes (set in initializer)
_TAR: Optional[tarfile.TarFile] = None


def _init_tar(tar_path: str) -> None:
    """Initializer called once in each worker process to open the tar archive."""
    global _TAR
    _TAR = tarfile.open(tar_path, "r:gz")


def _matches_topic(doc: etree._Element) -> bool:
    # 1. Article title
    title = " ".join(doc.xpath(".//article-title//text()"))
    # 2. Abstract text
    abstract = " ".join(doc.xpath(".//abstract//text()"))
    # 3. Author-supplied keywords (if any)
    keywords = " ".join(doc.xpath(".//kwd-group//kwd//text()"))

    haystack = f"{title} {abstract} {keywords}".lower()
    if OBESITY_DIABETES_WEIGHT_LOSS_FILTER:
        return bool(_TOPIC_RE.search(haystack))
    else:
        return True

def _process_xml_member(member_name: str, output_dir: Path) -> str:
    """Convert a single PMC XML file to Markdown unless it is CC BY-ND.

    Parameters
    ----------
    member_name : str
        Name of the XML file in the tar archive.
    output_dir : Path
        Directory where the resulting Markdown file should be written.
    """
    try:
        # Use the globally opened tarfile per process
        member = _TAR.getmember(member_name)
        with _TAR.extractfile(member) as f:
            xml_bytes = f.read()

        doc = etree.fromstring(xml_bytes)

        metadata = pmc_python.to_metadata(doc) or {}
        licence = metadata.get("License", "")

        # Skip files with a filtered licence
        if any(l in licence.upper() for l in LICENSE_FILTER):
            return "filtered"
        
        # Skip non research-article types (e.g. corrections)
        article_type = (doc.get("article-type") or "").lower()
        if article_type != "research-article":
            return "filtered"
        
        if not _matches_topic(doc):
            return "filtered"

        markdown = pmc_python.to_markdown(doc)

        # Determine output filename from member name (strip directories and .xml extension)
        base_name = Path(member_name).stem
        output_path = output_dir / f"{base_name}.md"
        with output_path.open("w+", encoding="utf-8") as f:
            f.write(markdown)

        return "saved"
    except Exception:
        # Any unexpected error
        return "error"


def main() -> None:
    tar_dir = Path("data")  # Directory containing one or more tar archives
    tar_paths = sorted([p for p in tar_dir.iterdir() if p.suffixes[-2:] == [".tar", ".gz"]])

    if not tar_paths:
        print(f"No .tar.gz files found in {tar_dir.resolve()}")
        return

    overall_saved = overall_filtered = overall_errors = 0

    output_dir = Path("data/markdown_outputs")
    output_dir.mkdir(exist_ok=True)

    for tar_path in tar_paths:
        print(f"\n=== Processing archive: {tar_path.name} ===")

        # Collect XML member names for this archive
        with tarfile.open(tar_path, "r:gz") as tf:
            member_names = [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".xml")]

        if not member_names:
            print("No XML files found in this archive; skipping.")
            continue

        with ProcessPoolExecutor(initializer=_init_tar, initargs=(str(tar_path),)) as executor:
            futures = {executor.submit(_process_xml_member, name, output_dir): name for name in member_names}

            saved = filtered = errors = 0
            pbar = tqdm(total=len(futures), desc="Processing", leave=False)

            for future in as_completed(futures):
                status = future.result()

                if status == "saved":
                    saved += 1
                    overall_saved += 1
                elif status == "filtered":
                    filtered += 1
                    overall_filtered += 1
                else:
                    errors += 1
                    overall_errors += 1

                pbar.update(1)
                pbar.set_postfix({"saved": saved, "filtered": filtered, "errors": errors})

            pbar.close()

        print(f"Archive summary => saved: {saved}, filtered: {filtered}, errors: {errors}")

    print("\n=== Overall summary ===")
    print(f"Saved: {overall_saved}, Filtered: {overall_filtered}, Errors: {overall_errors}")


if __name__ == "__main__":
    main()
