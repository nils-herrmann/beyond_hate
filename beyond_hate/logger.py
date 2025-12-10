import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

def get_logger(name: str, logs_dir: Optional[Path] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        logs_dir: Directory to save log files (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (only if logs_dir is provided)
        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract script name from logger name (e.g., 'beyond_hate.analysis.validate_annotations' -> 'validate_annotations')
            script_name = name.split('.')[-1]
            log_file = logs_dir / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
