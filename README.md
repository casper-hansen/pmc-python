# pmc - Python

Parse full text XML documents from PubMed Central's Open Access subset into structured data.

This is a complete rewrite in Python that achieves similar functionality to [tidypmc](https://github.com/ropensci/tidypmc) but which can generate Markdown.

## Examples

### 1. Convert a single XML file to Markdown

Run the helper script pointing at a local PMC XML file and it will generate a Markdown document next to it:

`python extract_from_xml.py`

(The sample script is pre-configured to read `data/PMC2231364.xml` and write `sample_output.md`. Feel free to adjust the paths inside the file.)

### 2. Process a local TAR archive of PMC XMLs

Have one or more `.tar.gz` archives in the `data/` directory? Convert every eligible article inside them with:

`python extract_from_tar.py`

The script automatically skips papers that are licensed *CC-BY-ND* and – by default – keeps **only** articles that mention *obesity*, *weight-loss* or *diabetes* in their title, abstract or author keywords. You can tweak the regular expression or turn the filter off by editing the constants at the top of the script.

### 3. Download & extract the latest PMC OA archives from FTP

To work with the newest content without downloading the huge full dataset first, use:

`python extract_from_ftp.py`

The script will:
1. Query the NCBI FTP mirror for the most recent **Open Access – comm** archives.
2. Stream-download each `*.tar.gz` file, decompress it on the fly and apply the same licence/topic filters as above.
3. Write the resulting Markdown files to `data/markdown_outputs/`.

Adjust the topic regular expression or disable filtering in `extract_from_ftp.py` just like in the TAR example above.