import pandas as pd
from lxml import etree
from typing import List, Optional
from io import StringIO


def to_table(doc: etree._Element) -> Optional[List[pd.DataFrame]]:
    """
    Parse tables from PubMed Central XML
    
    Parameters
    ----------
    doc : lxml.etree._Element
        XML document from PubMed Central
        
    Returns
    -------
    List[pd.DataFrame] or None
        List of DataFrames, one for each table
        Returns None if no tables found
    """
    if not isinstance(doc, etree._Element):
        raise ValueError("doc should be an XML document from PubMed Central")
    
    table_wraps = doc.xpath("//table-wrap")
    
    if not table_wraps:
        return None
    
    tables = []
    table_names = []
    
    for i, table_wrap in enumerate(table_wraps, 1):
        label_elem = table_wrap.find(".//label")
        caption_elem = table_wrap.find(".//caption")
        
        if label_elem is not None:
            label_text = etree.tostring(label_elem, method="text", encoding="unicode").strip()
        else:
            label_text = f"Table {i}"
        
        table_elem = table_wrap.find(".//table")
        if table_elem is None:
            continue
            
        df = _parse_html_table(table_elem)
        if df is not None and not df.empty:
            if caption_elem is not None:
                caption_text = etree.tostring(caption_elem, method="text", encoding="unicode").strip()
                df.attrs['caption'] = caption_text
            
            tables.append(df)
            table_names.append(label_text)
    
    if not tables:
        return None
    
    return dict(zip(table_names, tables)) if table_names else tables


def _parse_html_table(table_elem: etree._Element) -> Optional[pd.DataFrame]:
    """Parse HTML table element into DataFrame"""
    try:
        table_html = etree.tostring(table_elem, encoding="unicode")
        
        dfs = pd.read_html(StringIO(table_html), header=0)
        if dfs:
            df = dfs[0]
            
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            df.columns = [str(col) for col in df.columns]
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).replace('nan', '')
            
            return df
    except Exception as e:
        return None
    
    return None


def _handle_rowspan_colspan(table_elem: etree._Element) -> List[List[str]]:
    """Handle rowspan and colspan in table parsing"""
    rows = table_elem.xpath(".//tr")
    if not rows:
        return []
    
    grid = []
    max_cols = 0
    
    for row in rows:
        cells = row.xpath(".//td | .//th")
        row_data = []
        
        for cell in cells:
            cell_text = etree.tostring(cell, method="text", encoding="unicode").strip()
            rowspan = int(cell.get("rowspan", "1"))
            colspan = int(cell.get("colspan", "1"))
            
            for _ in range(colspan):
                row_data.append(cell_text)
        
        grid.append(row_data)
        max_cols = max(max_cols, len(row_data))
    
    for row in grid:
        while len(row) < max_cols:
            row.append("")
    
    return grid
