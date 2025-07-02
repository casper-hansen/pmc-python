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
    
    sections = doc.xpath("//sec | //abstract")
    
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
        if section.tag == 'abstract':
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
        if child.tag == 'p':
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
        
        elif child.tag == 'fig':
            label = child.find('.//label')
            caption = child.find('.//caption')
            if label is not None and caption is not None:
                label_text = etree.tostring(label, method="text", encoding="unicode").strip()
                caption_text = etree.tostring(caption, method="text", encoding="unicode").strip()
                results.append({
                    'section': section_info['path'],
                    'paragraph': 0,
                    'sentence': 0,
                    'text': f"**{label_text}:** *{caption_text}*"
                })
        
        elif child.tag == 'table-wrap':
            label = child.find('.//label')
            if label is not None:
                label_text = etree.tostring(label, method="text", encoding="unicode").strip()
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
        title_elem = current.find("title")
        if title_elem is not None:
            title_text = etree.tostring(title_elem, method="text", encoding="unicode").strip()
            if title_text:
                titles.insert(0, title_text)
                levels.insert(0, level)
        
        parent = current.getparent()
        if parent is not None and parent.tag in ["sec", "abstract"]:
            current = parent
            level += 1
        else:
            break
    
    if not titles:
        if section.tag == "abstract":
            return {"path": "Abstract"}
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
        if parent.tag == "abstract":
            # Only prepend for nested sections, not for the abstract element itself
            if section.tag != "abstract":
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
