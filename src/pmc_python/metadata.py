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
    
    # Constants for additional namespaces that may contain license references
    ALI_LICENSE_REF = "{http://www.niso.org/schemas/ali/1.0/}license_ref"

    license_url: Optional[str] = None

    # 1) Search any <license> element anywhere in the <front> matter (covers most cases)
    for lic_elem in front.findall(".//license"):
        # Direct xlink:href attribute (namespaced or with prefix)
        license_url = (
            lic_elem.get(XLINK_NS)
            or lic_elem.get("xlink:href")
            or lic_elem.get("href")
        )
        if license_url:
            license_url = license_url.strip()
            break

        # ali:license_ref element inside <license>
        lic_ref_elem = lic_elem.find(f".//{ALI_LICENSE_REF}")
        if lic_ref_elem is not None and lic_ref_elem.text:
            license_url = lic_ref_elem.text.strip()
            break

        # <ext-link> as a fall-back holder of the URL
        ext_link_elem = lic_elem.find(".//ext-link")
        if ext_link_elem is not None:
            license_url = (
                ext_link_elem.get(XLINK_NS)
                or ext_link_elem.get("xlink:href")
                or (ext_link_elem.text.strip() if ext_link_elem.text else None)
            )
            if license_url and license_url.strip():
                license_url = license_url.strip()
                break

    # 2) Search for ali:license_ref elements that live outside a <license> wrapper
    if not license_url:
        lic_ref_elem = front.find(f".//{ALI_LICENSE_REF}")
        if lic_ref_elem is not None and lic_ref_elem.text:
            license_url = lic_ref_elem.text.strip()

    # 3) Fallback: look for any xlink:href attributes under <permissions> containing a CC/public domain URL
    if not license_url:
        try:
            hrefs = front.xpath(
                ".//permissions//@xlink:href",
                namespaces={"xlink": "http://www.w3.org/1999/xlink"},
            )
            for href in hrefs:
                if "creativecommons" in href or "publicdomain" in href:
                    license_url = href.strip()
                    break
        except Exception:
            # If xpath fails (e.g., no namespace map), silently ignore
            pass

    if license_url:
        metadata['license_url'] = license_url
        metadata['license'] = _parse_license_url(license_url)

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
