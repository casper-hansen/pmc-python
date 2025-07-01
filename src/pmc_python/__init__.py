from .text import to_text
from .table import to_table
from .reference import to_reference
from .metadata import to_metadata
from .markdown import to_markdown
from .search import separate_text, separate_refs, separate_genes, separate_tags
from .utils import collapse_rows, repeat_sub
from .__version__ import __version__

__all__ = [
    "download_xml",
    "to_text", 
    "to_table",
    "to_reference",
    "to_metadata",
    "to_markdown",
    "separate_text",
    "separate_refs", 
    "separate_genes",
    "separate_tags",
    "collapse_rows",
    "repeat_sub",
    "__version__",
]
