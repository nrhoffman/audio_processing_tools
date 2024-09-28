import logging

def logger(log_name, log_file, file_level=logging.WARNING, console_level=logging.INFO):
    logger = logging.getLogger()

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # File handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger