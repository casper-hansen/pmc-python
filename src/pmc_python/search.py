import pandas as pd
import re
from typing import Optional, List, Union


def separate_text(df: pd.DataFrame, pattern: Union[str, List[str]], column: str = "text") -> Optional[pd.DataFrame]:
    """
    Separate all matching text into multiple rows
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame, usually results from pmc_text
    pattern : str or List[str]
        Either a regular expression or a list of words to find in text
    column : str, default "text"
        Column name to search in
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with matches, or None if no matches found
        
    Examples
    --------
    >>> text_df = pmc_text(doc)
    >>> separate_text(text_df, "[ATCGN]{5,}")
    >>> separate_text(text_df, ["hmu", "ybt", "yfe", "yfu"])
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a DataFrame")
    
    if column not in df.columns:
        raise ValueError(f"column {column} is not found")
    
    if isinstance(pattern, list):
        pattern = r"\b" + r"\b|\b".join(pattern) + r"\b"
    
    matches_mask = df[column].str.contains(pattern, regex=True, na=False)
    matched_rows = df[matches_mask]
    
    if matched_rows.empty:
        return None
    
    results = []
    
    for idx, row in matched_rows.iterrows():
        text = row[column]
        found_matches = re.findall(pattern, text)
        
        unique_matches = list(dict.fromkeys(found_matches))
        
        for match in unique_matches:
            result_row = row.copy()
            result_row['match'] = match
            results.append(result_row)
    
    if not results:
        return None
    
    result_df = pd.DataFrame(results)
    match_col = result_df.pop('match')
    result_df.insert(0, 'match', match_col)
    
    return result_df.reset_index(drop=True)


def separate_refs(df: pd.DataFrame, column: str = "text") -> Optional[pd.DataFrame]:
    """
    Separate references cited into multiple rows
    
    Separates references cited in brackets or parentheses into multiple rows and
    splits the comma-delimited numeric strings and expands ranges like 7-9 into new rows
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with text data
    column : str, default "text"
        Column name to search in
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with reference IDs, or None if no references found
        
    Examples
    --------
    >>> df = pd.DataFrame({'row': [1], 'text': ['some important studies [7-9,15]']})
    >>> separate_refs(df)
    """
    pattern = r"(\(|\[)[0-9, -]+(\]|\))"
    matched_df = separate_text(df, pattern, column)
    
    if matched_df is None:
        return None
    
    results = []
    
    for idx, row in matched_df.iterrows():
        match_text = row['match']
        
        clean_match = re.sub(r'[)([\]\s]', '', match_text)
        
        parts = clean_match.split(',')
        
        ref_ids = []
        for part in parts:
            part = part.strip()
            if '-' in part:
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    try:
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        ref_ids.extend(range(start, end + 1))
                    except ValueError:
                        continue
            else:
                try:
                    ref_ids.append(int(part))
                except ValueError:
                    continue
        
        for ref_id in ref_ids:
            result_row = row.copy()
            result_row['id'] = ref_id
            results.append(result_row)
    
    if not results:
        return None
    
    result_df = pd.DataFrame(results)
    id_col = result_df.pop('id')
    result_df.insert(0, 'id', id_col)
    
    return result_df.reset_index(drop=True)


def separate_genes(df: pd.DataFrame, pattern: str = r"\b[A-Za-z][a-z]{2}[A-Z0-9]+\b", 
                   genes: Optional[List[str]] = None, operon: int = 6, 
                   column: str = "text") -> Optional[pd.DataFrame]:
    """
    Separate genes and operons into multiple rows
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with text data
    pattern : str, default r"\b[A-Za-z][a-z]{2}[A-Z0-9]+\b"
        Regular expression to match genes
    genes : List[str], optional
        An optional list of genes, set pattern to None to only match this list
    operon : int, default 6
        Operon length. Split genes with 6 or more letters into separate genes
    column : str, default "text"
        Column name to search in
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with gene names, or None if no genes found
        
    Examples
    --------
    >>> df = pd.DataFrame({'row': [1], 'text': ['Genes like YacK, hmu and sufABC']})
    >>> separate_genes(df)
    >>> separate_genes(df, genes=['hmu'])
    """
    if operon <= 4:
        raise ValueError("Operon length should be 5 or more")
    
    if genes is not None:
        gene_pattern = r"\b" + r"\b|\b".join(genes) + r"\b"
        if pattern in ["", None]:
            pattern = gene_pattern
        else:
            pattern = f"{pattern}|{gene_pattern}"
    
    matched_df = separate_text(df, pattern, column)
    
    if matched_df is None:
        return None
    
    excluded_matches = {"TraDIS", "taqDNA", "log2", "log10", "ecoRI", "bamHI", "chr1", "chr2"}
    matched_df = matched_df[~matched_df['match'].isin(excluded_matches)]
    
    if matched_df.empty:
        raise ValueError("No match to genes")
    
    results = []
    
    for idx, row in matched_df.iterrows():
        match = row['match']
        
        if len(match) >= operon and not re.match(r'^[0-9]+$', match[3:]):
            prefix = match[:3].lower()
            suffix_chars = list(match[3:])
            gene_names = [prefix + char for char in suffix_chars]
        else:
            gene_names = [match[0].lower() + match[1:]]
        
        for gene_name in gene_names:
            result_row = row.copy()
            result_row['gene'] = gene_name
            results.append(result_row)
    
    if not results:
        return None
    
    result_df = pd.DataFrame(results)
    gene_col = result_df.pop('gene')
    result_df.insert(0, 'gene', gene_col)
    
    return result_df.reset_index(drop=True)


def separate_tags(df: pd.DataFrame, pattern: str, column: str = "text") -> Optional[pd.DataFrame]:
    """
    Separate locus tag into multiple rows
    
    Separates locus tags mentioned in full text and expands ranges like
    YPO1970-74 into new rows
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with text data
    pattern : str
        Regular expression to match locus tags like YPO[0-9-]+ or the locus tag prefix like YPO
    column : str, default "text"
        Column name to search in
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with locus tags, or None if no tags found
        
    Examples
    --------
    >>> df = pd.DataFrame({'row': [1], 'text': ['some genes like YPO1002 and YPO1970-74']})
    >>> separate_tags(df, "YPO")
    """
    if not re.search(r'[0-9]', pattern):
        pattern = f"{pattern}[0-9{pattern}-]+"
    
    matched_df = separate_text(df, pattern, column)
    
    if matched_df is None:
        return None
    
    dash_counts = matched_df['match'].str.count('-')
    if (dash_counts > 1).any():
        raise ValueError("pattern matches 3 or more tags")
    
    matched_df['match'] = matched_df['match'].str.rstrip('-')
    
    results = []
    
    for idx, row in matched_df.iterrows():
        match = row['match']
        
        if '-' in match:
            prefix = re.match(r'^[^0-9]+', match).group()
            number_part = re.sub(r'^[^0-9]+', '', match)
            range_parts = number_part.split('-')
            
            if len(range_parts) == 2:
                start_num = int(range_parts[0])
                end_num = int(range_parts[1])
                
                if end_num < start_num:
                    start_str = str(start_num)
                    end_str = str(end_num)
                    end_num = int(start_str[:-len(end_str)] + end_str)
                
                num_digits = len(range_parts[0])
                
                for num in range(start_num, end_num + 1):
                    tag_id = f"{prefix}{num:0{num_digits}d}"
                    result_row = row.copy()
                    result_row['id'] = tag_id
                    results.append(result_row)
            else:
                result_row = row.copy()
                result_row['id'] = match
                results.append(result_row)
        else:
            result_row = row.copy()
            result_row['id'] = match
            results.append(result_row)
    
    if not results:
        return None
    
    result_df = pd.DataFrame(results)
    id_col = result_df.pop('id')
    result_df.insert(0, 'id', id_col)
    
    return result_df.reset_index(drop=True)
