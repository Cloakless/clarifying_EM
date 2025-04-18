import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key(key_name):
    """
    Get API key from environment variables
    
    Args:
        key_name: Name of the environment variable
        
    Returns:
        API key as string
    
    Raises:
        ValueError: If the API key is not found
    """
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"{key_name} not found in environment variables. Please add it to your .env file.")
    return api_key