from lxml import etree
from typing import Optional, Dict, Any
import re

# XML namespace for xlink
XLINK_NS = "{http://www.w3.org/1999/xlink}href"

def _parse_license_url(url: str) -> str:
    """Convert a Creative Commons/public domain license URL to a human-readable identifier.

    Examples
    --------
    >>> _parse_license_url('http://creativecommons.org/licenses/by/4.0')
    'CC BY 4.0'
    >>> _parse_license_url('http://creativecommons.org/publicdomain/zero/1.0')
    'CC0 1.0'
    """
    if not url:
        return ""
    url = url.lower()

    # Handle CC0 which sits under publicdomain path
    if "publicdomain/zero" in url or "creativecommons.org/zero" in url:
        version_match = re.search(r"zero/(\d+\.\d+)", url)
        version = version_match.group(1) if version_match else "1.0"
        return f"CC0 {version}"

    # Handle standard CC licenses
    m = re.search(r"licenses/([a-z\-]+)/([\d\.]+)", url)
    if m:
        license_code, ver = m.groups()
        return f"CC {license_code.upper()} {ver}"

    # Fallback to raw URL when no standard pattern matches
    return url

def to_metadata(doc: etree._Element) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from PubMed Central XML
    
    Parameters
    ----------
    doc : lxml.etree._Element
        XML document from PubMed Central
        
    Returns
    -------
    dict or None
        Dictionary with metadata fields
        Returns None if no metadata found
    """
    if not isinstance(doc, etree._Element):
        raise ValueError("doc should be an XML document from PubMed Central")
    
    front = doc.find(".//front")
    if front is None:
        return None
    
    metadata = {}
    
    article_meta = front.find(".//article-meta")
    if article_meta is not None:
        pmc_id_elem = article_meta.xpath(".//article-id[@pub-id-type='pmc']")
        if pmc_id_elem:
            metadata['pmc_id'] = pmc_id_elem[0].text.strip()
        
        pmid_elem = article_meta.xpath(".//article-id[@pub-id-type='pmid']")
        if pmid_elem:
            metadata['pmid'] = pmid_elem[0].text.strip()
        
        doi_elem = article_meta.xpath(".//article-id[@pub-id-type='doi']")
        if doi_elem:
            metadata['doi'] = doi_elem[0].text.strip()
        
        title_elem = article_meta.find(".//article-title")
        if title_elem is not None:
            metadata['title'] = etree.tostring(title_elem, method="text", encoding="unicode").strip()
        
        authors = []
        contrib_group = article_meta.find(".//contrib-group")
        if contrib_group is not None:
            for contrib in contrib_group.xpath(".//contrib[@contrib-type='author']"):
                name_elem = contrib.find(".//name")
                if name_elem is not None:
                    surname_elem = name_elem.find("surname")
                    given_names_elem = name_elem.find("given-names")
                    
                    if surname_elem is not None and given_names_elem is not None:
                        surname = surname_elem.text.strip() if surname_elem.text else ""
                        given_names = given_names_elem.text.strip() if given_names_elem.text else ""
                        authors.append(f"{given_names} {surname}".strip())
        
        if authors:
            metadata['authors'] = authors
        
        pub_date = article_meta.find(".//pub-date[@pub-type='epub']")
        if pub_date is None:
            pub_date = article_meta.find(".//pub-date")
        
        if pub_date is not None:
            year_elem = pub_date.find("year")
            month_elem = pub_date.find("month")
            day_elem = pub_date.find("day")
            
            if year_elem is not None:
                metadata['year'] = int(year_elem.text.strip())
            if month_elem is not None:
                metadata['month'] = int(month_elem.text.strip())
            if day_elem is not None:
                metadata['day'] = int(day_elem.text.strip())
    
    # Extract license information (usually found under <permissions>)
    license_elem = front.find(".//permissions/license")
    if license_elem is not None:
        license_href = license_elem.get(XLINK_NS) or license_elem.get("xlink:href")
        if license_href:
            metadata['license_url'] = license_href
            metadata['license'] = _parse_license_url(license_href)

    journal_meta = front.find(".//journal-meta")
    if journal_meta is not None:
        journal_title_elem = journal_meta.find(".//journal-title")
        if journal_title_elem is not None:
            metadata['journal'] = journal_title_elem.text.strip()
        
        journal_id_elem = journal_meta.find(".//journal-id[@journal-id-type='nlm-ta']")
        if journal_id_elem is not None:
            metadata['journal_abbrev'] = journal_id_elem.text.strip()
        
        issn_elem = journal_meta.find(".//issn[@pub-type='epub']")
        if issn_elem is None:
            issn_elem = journal_meta.find(".//issn")
        if issn_elem is not None:
            metadata['issn'] = issn_elem.text.strip()
    
    if not metadata:
        return None
    
    # Duplicate select fields with capitalised keys expected by markdown formatter
    key_map = {
        'title': 'Title',
        'authors': 'Authors',
        'journal': 'Journal',
        'year': 'Year',
        'pmc_id': 'PMCID',
        'pmid': 'PMID',
        'doi': 'DOI',
        'license': 'License',
    }

    for k_src, k_dest in key_map.items():
        if k_src in metadata and k_dest not in metadata:
            metadata[k_dest] = metadata[k_src]

    return metadata
