import re
import pandas as pd
from lxml import etree
from typing import Optional, Dict, Any
import nltk
from .utils import path_string

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


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
        
        if section.tag == 'abstract':
            nested_sections = section.xpath(".//sec")
            if nested_sections:
                for nested_sec in nested_sections:
                    nested_info = _extract_section_info(nested_sec)
                    nested_info['path'] = f"Abstract; {nested_info['path']}"
                    _process_section_content(nested_sec, nested_info, results)
            else:
                _process_section_content(section, section_info, results)
        else:
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
                
            sentences = nltk.sent_tokenize(para_text)
            
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
        return {"path": titles[0]}
    else:
        return {"path": path_string(titles, levels)}


def _extract_paragraph_text(para: etree._Element) -> str:
    """Extract clean text from paragraph, handling nested elements"""
    text_parts = []
    
    for elem in para.iter():
        if elem.text:
            text_parts.append(elem.text)
        if elem.tail:
            text_parts.append(elem.tail)
    
    text = "".join(text_parts)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[\d\s,-]+\]', '', text)
    text = re.sub(r'\([\d\s,-]+\)', '', text)
    
    return text.strip()
