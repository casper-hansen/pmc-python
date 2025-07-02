import re
import pandas as pd
from lxml import etree
from typing import Optional, Dict, Any
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from .utils import path_string

# Ensure the pretrained English Punkt model is available.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ---------------------------------------------------------------------------
# Custom sentence tokenizer
# ---------------------------------------------------------------------------
# "et al." is extremely common in biomedical and other scientific texts, but
# the default Punkt parameters tend to treat the period after "al." as a
# sentence boundary.  We patch the pretrained English model by adding "al" to
# its abbreviation list.  This prevents unwanted splits inside the phrase
# "et al." while still allowing the period to end a sentence when there is no
# following lowercase token (e.g. at the very end of a sentence).

# Load the standard English Punkt model once and modify its parameters.
_PUNKT = nltk.data.load('tokenizers/punkt/english.pickle')
_PUNKT._params.abbrev_types.update({'al'})  # handles "et al." correctly

# Expose a simple callable for code readability.
_SENT_TOKENIZE = _PUNKT.tokenize

def to_text(doc: etree._Element) -> Optional[pd.DataFrame]:
    """
    Extract structured text from PubMed Central XML
    
    Parameters
    ----------
    doc : lxml.etree._Element
        XML document from PubMed Central
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: section, paragraph, sentence, text
        Returns None if no text sections found
    """
    if not isinstance(doc, etree._Element):
        raise ValueError("doc should be an XML document from PubMed Central")
    
    # ---------------------------------------------------------------
    # Collect structural sections
    # ---------------------------------------------------------------
    # 1. Root <body> (for articles that place <p> directly inside it)
    body_elem = doc.xpath('//*[local-name()="body"]')

    # 2. All <sec> and <abstract> elements in document order.
    sec_elems = doc.xpath('//*[local-name()="sec" or local-name()="abstract"]')

    # Preserve overall article order: body first (if present), then all other
    # sections as they appear.
    sections = []
    if body_elem:
        sections.append(body_elem[0])
    sections.extend(sec_elems)
    
    if not sections:
        return None
    
    results = []
    
    for section in sections:
        section_info = _extract_section_info(section)
        
        # Ensure the main "Abstract" heading is present even when the abstract
        # element only contains nested <sec> sub-sections (Background, Results,
        # etc.) and no direct <p> children. We inject a dummy row so that the
        # markdown formatter later outputs the heading, while leaving the
        # content area blank.
        if _tags_equal(section, 'abstract'):
            results.append({
                'section': section_info['path'],
                'paragraph': 0,
                'sentence': 0,
                'text': ''
            })
        
        _process_section_content(section, section_info, results)
    
    if not results:
        return None
        
    return pd.DataFrame(results)

def _process_section_content(section: etree._Element, section_info: dict, results: list):
    """Process content within a section (paragraphs, figures, tables)"""
    para_idx = 0
    
    for child in section:
        tag = _local_name(child)
        if tag == 'p':
            para_idx += 1
            para_text = _extract_paragraph_text(child)
            if not para_text.strip():
                continue
                
            sentences = _SENT_TOKENIZE(para_text)
            
            for sent_idx, sentence in enumerate(sentences, 1):
                if sentence.strip():
                    results.append({
                        'section': section_info['path'],
                        'paragraph': para_idx,
                        'sentence': sent_idx,
                        'text': sentence.strip()
                    })
        
        elif tag == 'fig':
            label = child.xpath('.//*[local-name()="label"]')
            caption = child.xpath('.//*[local-name()="caption"]')
            label = label[0] if label else None
            caption = caption[0] if caption else None
            if label is not None and caption is not None:
                label_text = etree.tostring(label, method="text", encoding="unicode").strip()
                caption_text = etree.tostring(caption, method="text", encoding="unicode").strip()
                results.append({
                    'section': section_info['path'],
                    'paragraph': 0,
                    'sentence': 0,
                    'text': f"**{label_text}:** *{caption_text}*"
                })
        
        elif tag == 'table-wrap':
            label = child.xpath('.//*[local-name()="label"]')
            if label:
                label_text = etree.tostring(label[0], method="text", encoding="unicode").strip()
                results.append({
                    'section': section_info['path'],
                    'paragraph': 0,
                    'sentence': 0,
                    'text': f"TABLE_PLACEHOLDER:{label_text}"
                })



def _extract_section_info(section: etree._Element) -> Dict[str, Any]:
    """Extract section title and hierarchical path"""
    titles = []
    levels = []
    
    current = section
    level = 1
    
    while current is not None:
        title_candidates = current.xpath('./*[local-name()="title"]')
        title_elem = title_candidates[0] if title_candidates else None
        if title_elem is not None:
            title_text = etree.tostring(title_elem, method="text", encoding="unicode").strip()
            if title_text:
                titles.insert(0, title_text)
                levels.insert(0, level)
        
        parent = current.getparent()
        if parent is not None and _local_name(parent) in ["sec", "abstract"]:
            current = parent
            level += 1
        else:
            break
    
    if not titles:
        if _tags_equal(section, 'abstract'):
            return {"path": "Abstract"}
        if _tags_equal(section, 'body'):
            return {"path": "Body"}
        else:
            return {"path": "Unknown"}
    
    if len(titles) == 1:
        path = titles[0]
    else:
        path = path_string(titles, levels)

    # If this section is nested within an <abstract>, prefix the path so that
    # it is clearly grouped under the Abstract heading when converted to
    # markdown. The root <abstract> element itself should remain just
    # "Abstract".
    parent = section.getparent()
    while parent is not None:
        if _tags_equal(parent, "abstract"):
            # Only prepend for nested sections, not for the abstract element itself
            if not _tags_equal(section, "abstract"):
                path = f"Abstract; {path}"
            break
        parent = parent.getparent()

    return {"path": path}


def _extract_paragraph_text(para: etree._Element) -> str:
    """Extract and tidy plain text from a paragraph element."""

    # Use lxml's text serialization to capture all nested text content in the
    # paragraph while ignoring markup. This approach is both simpler and less
    # error-prone than manually iterating over every child element.
    text = etree.tostring(para, method="text", encoding="unicode")

    # Collapse any run of whitespace (spaces, tabs, newlines) down to a single
    # space so that the downstream sentence tokenizer sees clean text.
    text = re.sub(r"\s+", " ", text)

    # Convert purely numeric parenthetical citations to square-bracket form,
    # e.g. "(12, 15)" -> "[12, 15]". This makes them easier to read in the
    # generated markdown and avoids false sentence breaks inside parentheses.
    text = re.sub(r"\(\s*([\d\s,-]+)\s*\)", r"[\1]", text)

    # Remove unnecessary spaces directly inside square brackets, e.g. "[ 33]"
    # -> "[33]".
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)

    # Likewise, remove spaces just inside parentheses so that patterns like
    # "( Vionnet et al. 1992)" become "(Vionnet et al. 1992)".
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # Remove stray whitespace that can appear before punctuation (e.g. "level ."
    # -> "level.").
    text = re.sub(r"\s+([\.,;:?!])", r"\1", text)

    return text.strip()

# ---------------------------------------------------------------------------
# XML namespace helpers (internal use only)
# ---------------------------------------------------------------------------


def _local_name(elem: etree._Element) -> str:
    """Return the element's tag name without namespace, e.g. '{ns}sec' -> 'sec'."""

    tag = elem.tag  # type: ignore[attr-defined]
    return tag.split('}', 1)[1] if '}' in tag else tag  # type: ignore[return-value]


def _tags_equal(elem: etree._Element, tag_name: str) -> bool:
    """Case-sensitive check that *elem*'s local tag equals *tag_name*."""

    return _local_name(elem) == tag_name
