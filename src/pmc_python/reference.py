import pandas as pd
from lxml import etree
from typing import Optional
import re


def to_reference(doc: etree._Element) -> Optional[pd.DataFrame]:
    """
    Format references cited from PubMed Central XML
    
    Parameters
    ----------
    doc : lxml.etree._Element
        XML document from PubMed Central
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: id, pmid, authors, year, title, journal, volume, pages, doi
        Returns None if no references found
    """
    if not isinstance(doc, etree._Element):
        raise ValueError("doc should be an XML document from PubMed Central")
    
    refs = doc.xpath("//ref")
    
    if not refs:
        return None
    
    tag_counts = {}
    for ref in refs:
        child_tags = ref.xpath(".//*")
        for tag in child_tags:
            tag_name = tag.tag
            if tag_name not in ['label', 'note']:
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
    
    if tag_counts:
        tag_summary = " and ".join([f"{count} {tag}" for tag, count in tag_counts.items()])
    
    results = []
    
    for i, ref in enumerate(refs, 1):
        ref_data = _extract_reference_data(ref, i)
        results.append(ref_data)
    
    df = pd.DataFrame(results)
    
    missing_both = df['authors'].isna() & df['title'].isna()
    if missing_both.any():
        for i, ref in enumerate(refs):
            if missing_both.iloc[i]:
                ref_text = etree.tostring(ref, method="text", encoding="unicode").strip()
                df.loc[i, 'authors'] = ref_text
    
    return df


def _extract_reference_data(ref: etree._Element, ref_id: int) -> dict:
    """Extract reference data from a single ref element"""
    pmid_elem = ref.xpath(".//pub-id[@pub-id-type='pmid']")
    pmid = pmid_elem[0].text.strip() if pmid_elem and pmid_elem[0].text else None
    
    doi_elem = ref.xpath(".//pub-id[@pub-id-type='doi']")
    doi = doi_elem[0].text.strip() if doi_elem and doi_elem[0].text else None
    
    surnames = [elem.text.strip() for elem in ref.xpath(".//surname") if elem.text]
    given_names = [elem.text.strip() for elem in ref.xpath(".//given-names") if elem.text]
    
    if surnames and given_names:
        authors = []
        for surname, given in zip(surnames, given_names):
            authors.append(f"{surname} {given}")
        authors_str = ", ".join(authors)
    elif surnames:
        authors_str = ", ".join(surnames)
    else:
        authors_str = None
    
    year_elem = ref.xpath(".//year")
    year = year_elem[0].text.strip() if year_elem and year_elem[0].text else None
    if year and year.isdigit():
        year = int(year)
    
    title_elem = ref.xpath(".//article-title")
    title = title_elem[0].text.strip() if title_elem and title_elem[0].text else None
    if title:
        title = re.sub(r'\n\s*', ' ', title)
    
    journal_elem = ref.xpath(".//source")
    journal = journal_elem[0].text.strip() if journal_elem and journal_elem[0].text else None
    
    volume_elem = ref.xpath(".//volume")
    volume = volume_elem[0].text.strip() if volume_elem and volume_elem[0].text else None
    
    fpage_elem = ref.xpath(".//fpage")
    fpage = fpage_elem[0].text.strip() if fpage_elem and fpage_elem[0].text else None
    
    lpage_elem = ref.xpath(".//lpage")
    lpage = lpage_elem[0].text.strip() if lpage_elem and lpage_elem[0].text else None
    
    if fpage and lpage:
        pages = f"{fpage}-{lpage}"
    elif fpage:
        pages = fpage
    else:
        pages = None
    
    return {
        'id': ref_id,
        'pmid': pmid,
        'authors': authors_str,
        'year': year,
        'title': title,
        'journal': journal,
        'volume': volume,
        'pages': pages,
        'doi': doi
    }
