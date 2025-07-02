import pandas as pd
from lxml import etree
from typing import Dict, Any, List, Optional
import re
from .text import to_text
from .table import to_table
from .metadata import to_metadata
from .reference import to_reference
import yaml  # type: ignore


def to_markdown(doc: etree._Element) -> str:
    """
    Convert PubMed Central XML to formatted markdown with integrated tables and figures
    
    Parameters
    ----------
    doc : lxml.etree._Element
        XML document from PubMed Central
    include_references : bool, default True
        Whether to include references section
    include_tables : bool, default True
        Whether to include tables inline
    include_figures : bool, default True
        Whether to include figure captions inline
        
    Returns
    -------
    str
        Formatted markdown document with integrated content
    """
    if not isinstance(doc, etree._Element):
        raise ValueError("doc should be an XML document from PubMed Central")
    
    markdown_parts = []
    
    metadata = to_metadata(doc)
    if metadata:
        markdown_parts.append(_format_metadata_header(metadata))
    
    text_data = to_text(doc)
    tables = to_table(doc)
    
    if text_data is not None:
        integrated_content = _format_integrated_content(text_data, tables)
        markdown_parts.append(integrated_content)
    
    references = to_reference(doc)
    if references is not None:
        markdown_parts.append(_format_references(references))
    
    return "\n\n".join(filter(None, markdown_parts))


def _format_metadata_header(metadata: Dict[str, Any]) -> str:
    """Return YAML front-matter block for the article metadata.

    The block follows the common Markdown convention:
    --- (YAML) ---
    Optionally we append the article title as an H1 heading so the document
    still renders nicely in plain Markdown viewers that ignore YAML metadata.
    """

    # Map internal metadata keys to YAML keys (lower-case)
    meta_map = {
        "Title": "title",
        "Authors": "authors",
        "Journal": "journal",
        "Year": "year",
        "PMCID": "pmcid",
        "DOI": "doi",
        "License": "license",
    }

    yaml_meta: Dict[str, Any] = {}

    for src_key, yaml_key in meta_map.items():
        if src_key in metadata and metadata[src_key]:
            value = metadata[src_key]
            # Coerce single author string into list for YAML consistency
            if src_key == "Authors" and not isinstance(value, list):
                value = [value]
            yaml_meta[yaml_key] = value

    # Dump YAML without sorting keys and in block style
    yaml_str = yaml.safe_dump(
        yaml_meta, sort_keys=False, default_flow_style=False, allow_unicode=True
    ).strip()

    header_parts = ["---", yaml_str, "---"]

    # Add a visible H1 title for renderers that ignore YAML
    if "Title" in metadata and metadata["Title"]:
        header_parts.extend(["", f"# {metadata['Title']}"])

    return "\n".join(header_parts)


def _format_integrated_content(text_data: pd.DataFrame, tables) -> str:
    """Format text content with integrated tables and figures.

    The *text_data* frame coming from ``to_text`` has one row per sentence
    and contains ``section``, ``paragraph`` and ``sentence`` positional
    markers.  The previous implementation concatenated rows directly which
    led to paragraph breaks after **every sentence**.  Here we first bundle
    rows belonging to the same paragraph (identified by the combination of
    *section* and *paragraph* index) so that we only insert a blank line
    between paragraphs while fully preserving the original article order.
    """
    if text_data.empty:
        return ""

    # -----------------------------------------------------------
    # Build a lookup so that "TABLE_PLACEHOLDER:Table 1" markers can be
    # replaced later on.  This is unchanged behaviour from the previous
    # implementation.
    # -----------------------------------------------------------
    table_lookup: Dict[str, Any] = {}
    if tables is not None:
        if isinstance(tables, dict):
            table_lookup = tables
        else:
            table_lookup = {f"Table {i}": tbl for i, tbl in enumerate(tables, 1)}

    markdown_parts: List[str] = []

    # -----------------------------------------------------------
    # Iterate over the dataframe while coalescing sentences that belong to
    # the same paragraph.  *text_data* is already in document order, so we
    # can stream over it without additional sorting.
    # -----------------------------------------------------------
    current_section: Optional[str] = None
    current_paragraph_id: Optional[int] = None
    current_sentences: List[str] = []

    def _flush_paragraphs(section: str, sentences_acc: List[str], out: List[str]):
        """Helper to emit a completed paragraph list for *section*."""
        if not sentences_acc:
            return
        paragraph_text = " ".join(sentences_acc).strip()
        if paragraph_text or paragraph_text.startswith("TABLE_PLACEHOLDER:"):
            # Store paragraphs per section until the section changes – the
            # final transformation to markdown is done in one go for each
            # section so that headings are handled only once.
            out.append(paragraph_text)
        sentences_acc.clear()

    # A mapping of section -> list[paragraph_str]; we build it on the fly
    section_to_paragraphs: Dict[str, List[str]] = {}

    for _, row in text_data.iterrows():
        section = row["section"]
        para_idx = row.get("paragraph", 0)
        sentence_text = row["text"]

        # -------------------------------------------------------
        # Detect a change in section or paragraph and flush the
        # accumulated sentences accordingly.
        # -------------------------------------------------------
        if (section != current_section) or (para_idx != current_paragraph_id):
            if current_section is not None:
                # Ensure there is a container for the current section.
                section_to_paragraphs.setdefault(current_section, [])
                _flush_paragraphs(current_section, current_sentences, section_to_paragraphs[current_section])

            # On section change we also need to reset the paragraph list so
            # that paragraphs are grouped under their respective section
            # heading in the original order.
            if section != current_section:
                current_section = section
            current_paragraph_id = para_idx

        # Accumulate sentences for the current paragraph.
        current_sentences.append(sentence_text)

    # Flush any remaining buffered sentences from the last loop iteration.
    if current_section is not None:
        section_to_paragraphs.setdefault(current_section, [])
        _flush_paragraphs(current_section, current_sentences, section_to_paragraphs[current_section])

    # -----------------------------------------------------------
    # With paragraphs now grouped by section we can format each section
    # individually, preserving the original ordering stored in
    # *section_to_paragraphs*.
    # -----------------------------------------------------------
    for section_name, paragraphs in section_to_paragraphs.items():
        section_content = _format_section(section_name, paragraphs, table_lookup)
        markdown_parts.append(section_content)

    return "\n\n".join(markdown_parts)


def _format_single_table(table: pd.DataFrame, table_name: str) -> str:
    """Format a single table as markdown"""
    if not isinstance(table, pd.DataFrame) or table.empty:
        return ""
    
    table_copy = table.copy()
    table_copy.columns = [str(col).replace('\n', ' ').strip() for col in table_copy.columns]
    
    headers = " | ".join(table_copy.columns)
    separator = " | ".join(["---"] * len(table_copy.columns))
    
    rows = []
    for _, row in table_copy.iterrows():
        row_data = [str(val).replace('\n', ' ').strip() if pd.notna(val) else "" for val in row]
        rows.append(" | ".join(row_data))
    
    table_md = f"**{table_name}**\n\n| {headers} |\n| {separator} |\n" + "\n".join(f"| {row} |" for row in rows)
    
    return table_md


def _format_section(section_name: str, paragraphs: List[str], table_lookup: dict = None) -> str:
    """Format a section with appropriate heading level"""
    if ';' in section_name:
        parts = [part.strip() for part in section_name.split(';')]
        if parts[0] == "Abstract":
            if len(parts) == 1:
                section_title = "Abstract"
                heading_level = 2
            else:
                section_title = parts[-1].strip()
                heading_level = 3  # ### for first-level subheadings inside Abstract
                if len(parts) > 2:
                    heading_level = min(3 + (len(parts) - 2), 6)
        else:
            heading_level = min(len(parts) + 1, 6)  # Max heading level is 6
            section_title = parts[-1].strip()
    else:
        heading_level = 2
        section_title = section_name
    
    heading = "#" * heading_level + " " + section_title
    
    formatted_paragraphs = []
    current_para = []
    
    for sentence in paragraphs:
        if sentence.startswith("**") and sentence.endswith("*") and ":" in sentence:
            if current_para:
                formatted_paragraphs.append(" ".join(current_para))
                current_para = []
            formatted_paragraphs.append(sentence)
        elif sentence.startswith("TABLE_PLACEHOLDER:"):
            if current_para:
                formatted_paragraphs.append(" ".join(current_para))
                current_para = []
            if table_lookup:
                table_name = sentence.replace("TABLE_PLACEHOLDER:", "")
                if table_name in table_lookup:
                    table_md = _format_single_table(table_lookup[table_name], table_name)
                    if table_md:
                        formatted_paragraphs.append(table_md)
        else:
            current_para.append(sentence)
            if len(current_para) >= 4 or sentence.endswith(('.', '!', '?')):
                formatted_paragraphs.append(" ".join(current_para))
                current_para = []
    
    if current_para:
        formatted_paragraphs.append(" ".join(current_para))
    
    processed_paragraphs = []
    for para in formatted_paragraphs:
        if para.startswith("**") and ":" in para and para.endswith("*"):
            processed_paragraphs.append(para)
        elif para.startswith("**") and "|" in para:
            processed_paragraphs.append(para)
        else:
            para = _process_math_formatting(para)
            para = _process_text_formatting(para)
            processed_paragraphs.append(para)
    
    content = "\n\n".join(processed_paragraphs).strip()
    if content:
        return f"{heading}\n\n{content}"
    else:
        # No paragraph content; return just the heading string. The caller will
        # insert section separators between blocks, so returning the bare
        # heading avoids accumulating extra blank lines (which previously
        # produced four newlines between headings).
        return heading


def _process_math_formatting(text: str) -> str:
    """Convert mathematical expressions to KaTeX format"""
    text = re.sub(r'\b([a-zA-Z])\s*=\s*([0-9.]+)', r'$\1 = \2$', text)
    text = re.sub(r'\b(p|P)\s*[<>=]\s*([0-9.]+)', r'$\1 \\leq \2$', text)
    text = re.sub(r'\b(n|N)\s*=\s*([0-9,]+)', r'$n = \2$', text)
    
    text = re.sub(r'([0-9.]+)%', r'$\1\\%$', text)
    
    text = re.sub(r'([0-9.]+)/([0-9.]+)', r'$\\frac{\1}{\2}$', text)
    
    return text


def _process_text_formatting(text: str) -> str:
    """Apply text formatting (bold, italic, etc.)"""
    text = re.sub(r'\b([A-Z][a-z]+[0-9]*)\b(?=\s+gene|protein)', r'*\1*', text)
    
    text = re.sub(r'(?<!^)(?<!\. )\b([A-Z][a-z]+\s+[a-z]+)\b(?=\s)', 
                  lambda m: f'*{m.group(1)}*' if _is_likely_species_name(m.group(1)) else m.group(1), text)
    
    return text


def _is_likely_species_name(text: str) -> bool:
    """Check if text looks like a scientific species name"""
    words = text.split()
    if len(words) != 2:
        return False
    
    genus, species = words
    
    common_genera = {
        'Yersinia', 'Escherichia', 'Salmonella', 'Bacillus', 'Staphylococcus', 
        'Streptococcus', 'Pseudomonas', 'Mycobacterium', 'Clostridium', 'Vibrio',
        'Listeria', 'Campylobacter', 'Helicobacter', 'Legionella', 'Chlamydia'
    }
    
    sentence_starters = {
        'Those genes', 'These genes', 'Several genes', 'Many genes', 'Some genes',
        'Classification of', 'Analysis of', 'Studies of', 'Results of', 'Effects of',
        'Genes responsible', 'Proteins involved', 'Factors affecting', 'Methods for',
        'Data from', 'Evidence for', 'Comparison of', 'Expression of', 'Regulation of'
    }
    
    if text in sentence_starters:
        return False
    
    if genus in common_genera:
        return True
    
    species_suffixes = ['is', 'us', 'um', 'a', 'ae', 'ii', 'ensis', 'icus', 'alis']
    if any(species.endswith(suffix) for suffix in species_suffixes):
        return True
    
    return False


def _format_references(references: pd.DataFrame) -> str:
    """Format references section"""
    if references.empty:
        return ""
    
    markdown_parts = ["## References"]
    
    for i, (_, ref) in enumerate(references.iterrows(), 1):
        ref_parts = []
        
        if pd.notna(ref.get('authors')) and ref['authors']:
            ref_parts.append(f"{ref['authors']}")
        
        if pd.notna(ref.get('title')) and ref['title']:
            ref_parts.append(f"**{ref['title']}**")
        
        journal_parts = []
        if pd.notna(ref.get('journal')) and ref['journal']:
            journal_parts.append(f"*{ref['journal']}*")
        
        if pd.notna(ref.get('year')) and ref['year']:
            journal_parts.append(f"({ref['year']})")
        
        if pd.notna(ref.get('volume')) and ref['volume']:
            journal_parts.append(f"**{ref['volume']}**")
        
        if pd.notna(ref.get('pages')) and ref['pages']:
            journal_parts.append(f"{ref['pages']}")
        
        if journal_parts:
            ref_parts.append(" ".join(journal_parts))
        
        if pd.notna(ref.get('doi')) and ref['doi']:
            ref_parts.append(f"DOI: {ref['doi']}")
        elif pd.notna(ref.get('pmid')) and ref['pmid']:
            ref_parts.append(f"PMID: {ref['pmid']}")
        
        reference_text = ". ".join(filter(None, ref_parts))
        markdown_parts.append(f"{i}. {reference_text}")
    
    return "\n\n".join(markdown_parts)
