import os
from pathlib import Path
from langchain_core.tools import tool


@tool
def dobavi_podatke_aplikaciji() -> str:
    """
    Vraca podatke aplikacije iz tekstualne datoteke. Kad korisnik trazi    
    Returns:
        str: Podatke aplikacije iz tekstualne datoteke
    """
    try:
        data_file_path = Path(os.getenv("APP_DATA_FILE", "tool.txt"))
        
        if not data_file_path.exists():
            return f"Error: Application data file not found at {data_file_path}"
        
        with open(data_file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
        if not content:
            return "Error: Application data file is empty."
            
        return f"Application Data:\n{content}"
            
    except Exception as e:
        return f"Error reading application data: {str(e)}"


available_tools = [dobavi_podatke_aplikaciji]