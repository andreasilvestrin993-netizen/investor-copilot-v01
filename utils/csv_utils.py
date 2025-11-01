"""
CSV utilities for reading, writing, and format detection
"""
import pandas as pd
import streamlit as st
import re
from pathlib import Path
from docx import Document

def detect_csv_format(file_content):
    """
    Detect if CSV uses European format (semicolon delimiter, comma decimal separator)
    or standard format (comma delimiter, dot decimal separator).
    Returns (delimiter, decimal_separator)
    """
    # Read first few lines to detect format
    first_lines = file_content[:2000]  # Check first 2000 chars
    
    # Count delimiters
    semicolon_count = first_lines.count(';')
    comma_count = first_lines.count(',')
    
    # European format typically has more semicolons than commas in header/data
    # (semicolons separate columns, commas are in decimal numbers)
    if semicolon_count > comma_count:
        return ';', ','  # European format
    else:
        return ',', '.'  # Standard format

def read_csv_smart(uploaded_file):
    """
    Smart CSV reader that handles both European and standard formats.
    Returns DataFrame with properly parsed numeric values.
    """
    # Read file content
    file_content = uploaded_file.getvalue().decode('utf-8')
    
    # Detect format
    delimiter, decimal_sep = detect_csv_format(file_content)
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Read CSV with detected format - explicitly set index_col=False to prevent first column being used as index
    if decimal_sep == ',':
        # European format: need to convert commas to dots in numeric columns
        df = pd.read_csv(uploaded_file, sep=delimiter, dtype=str, na_values=['', 'None', 'NaN', 'null'], index_col=False)
        
        # Convert numeric columns: replace comma with dot
        for col in df.columns:
            # Try to detect if column contains numeric data with commas
            if df[col].dtype == 'object':
                # Check if values look like European decimals (e.g., "123,45")
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    # If any value contains comma followed by digits, treat as European decimal
                    if any(pd.notna(val) and isinstance(val, str) and re.search(r',\d', val) for val in sample):
                        df[col] = df[col].str.replace(',', '.', regex=False)
    else:
        # Standard format
        df = pd.read_csv(uploaded_file, sep=delimiter, dtype=str, na_values=['', 'None', 'NaN', 'null'], index_col=False)
    
    return df, delimiter, decimal_sep

def show_column_mapping_ui(uploaded_df, expected_cols, csv_type="portfolio", csv_format_info=None):
    """
    Show column mapping interface for CSV uploads.
    Returns mapped DataFrame if confirmed, None if still mapping.
    """
    st.info(f"📋 **Column Mapping for {csv_type.title()}**")
    
    # Show detected format
    if csv_format_info:
        delimiter, decimal_sep = csv_format_info
        format_type = "European (semicolon delimiter, comma decimal)" if delimiter == ';' else "Standard (comma delimiter, dot decimal)"
        st.success(f"✓ Detected CSV format: **{format_type}**")
    
    st.write("Map your CSV columns to the expected format:")
    
    # Get uploaded CSV columns
    uploaded_cols = uploaded_df.columns.tolist()
    
    # Create mapping interface
    mapping = {}
    st.write("**Your CSV columns → Expected columns:**")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.write("**Your CSV Column**")
    with col2:
        st.write("**→**")
    with col3:
        st.write("**Maps to Expected Column**")
    
    # Create mapping dropdowns for each expected column
    for i, expected_col in enumerate(expected_cols):
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            # Try to find best match
            best_match = None
            expected_lower = expected_col.lower().replace("_", "").replace(" ", "")
            
            for uploaded_col in uploaded_cols:
                uploaded_lower = uploaded_col.lower().replace("_", "").replace(" ", "")
                if expected_lower == uploaded_lower:
                    best_match = uploaded_col
                    break
            
            # If no exact match, try partial match
            if not best_match:
                for uploaded_col in uploaded_cols:
                    uploaded_lower = uploaded_col.lower().replace("_", "").replace(" ", "")
                    if expected_lower in uploaded_lower or uploaded_lower in expected_lower:
                        best_match = uploaded_col
                        break
            
            # Default to first column if still no match
            if not best_match and uploaded_cols:
                best_match = uploaded_cols[0] if i < len(uploaded_cols) else uploaded_cols[0]
            
            default_idx = uploaded_cols.index(best_match) if best_match in uploaded_cols else 0
            
            selected = st.selectbox(
                f"Column for '{expected_col}'",
                options=["(skip)"] + uploaded_cols,
                index=default_idx + 1 if best_match else 0,
                key=f"map_{csv_type}_{i}",
                label_visibility="collapsed"
            )
            
            if selected != "(skip)":
                mapping[expected_col] = selected
        
        with col2:
            st.write("→")
        
        with col3:
            st.write(f"**{expected_col}**")
    
    # Create preview DataFrame based on current mapping
    preview_df = pd.DataFrame()
    for expected_col, uploaded_col in mapping.items():
        if uploaded_col in uploaded_df.columns:
            preview_df[expected_col] = uploaded_df[uploaded_col]
    
    # Fill missing columns with empty values for preview
    for col in expected_cols:
        if col not in preview_df.columns:
            preview_df[col] = None
    
    # Show live preview with mapped columns
    with st.expander("📊 Preview mapped data (first 5 rows)", expanded=True):
        st.caption("This preview updates as you change the column mappings above")
        st.dataframe(preview_df.head(), use_container_width=True)
    
    # Action buttons
    col_confirm, col_cancel = st.columns(2)
    
    with col_confirm:
        if st.button("✅ Confirm Mapping", type="primary", use_container_width=True):
            return preview_df
    
    with col_cancel:
        if st.button("❌ Cancel Upload", use_container_width=True):
            st.session_state.pending_upload = None
            st.session_state.upload_type = None
            st.session_state.csv_format_info = None
            st.rerun()
    
    return None

def ensure_columns(df: pd.DataFrame, cols: list):
    """Ensure DataFrame has all required columns"""
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def load_csv(path, cols):
    """Load CSV with proper delimiter detection"""
    if path.exists():
        try:
            # Try to read with semicolon delimiter first (European format)
            df = pd.read_csv(path, sep=';', index_col=False)
            # If only one column, try comma delimiter
            if len(df.columns) == 1:
                df = pd.read_csv(path, sep=',', index_col=False)
        except Exception:
            df = pd.DataFrame(columns=cols)
        return ensure_columns(df, cols)
    return pd.DataFrame(columns=cols)

def save_csv(path, df):
    """Save CSV with semicolon delimiter (European format compatible)"""
    df.to_csv(path, index=False, sep=';')

def write_docx(text: str, out_path: Path):
    """Write text to a Word document"""
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(str(out_path))
