import logging


class LoggerMixin:
    def _setup_logger(self, name, level=logging.INFO):
        """Set up a logger with consistent formatting"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Add console handler if none exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.propagate = False

        return logger
