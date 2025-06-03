from pathlib import Path


def load_system_prompt(file_path: str = "prompt.txt") -> str:
    """
    Load system prompt from a file.
    
    Args:
        file_path: Path to the system prompt file
        
    Returns:
        str: The system prompt content
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        prompt_path = Path(file_path)
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {file_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
        if not content:
            raise ValueError(f"System prompt file is empty: {file_path}")
            
        return content
        
    except Exception as e:
        print(f"Error loading system prompt from {file_path}: {e}")
        raise