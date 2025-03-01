import threading
from pathlib import Path
import urllib.request
import inspect
import os
from datetime import datetime
import logging


class UpdateDownloader(threading.Thread):
    def __init__(self, downloads, save_dir, callback=None, logging_level=logging.INFO):
        """
        downloads: dict of {type: url} pairs
        save_dir: directory to save files
        callback: callback function to call when all downloads complete
        logging_level: logging level for this class
        """
        super().__init__()
        self._downloads = downloads  # private as it's internal state
        self._save_dir = save_dir  # private as it's internal state
        self._callback = callback  # private as it's internal state
        self._results = {}  # private as it's internal state

        # Configure logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging_level)

        # Add console handler if none exists
        if not self._logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
            self._logger.propagate = False

    def _call_callback(self):  # already private, correct
        if not self._callback:
            return

        # Check number of parameters the callback accepts
        params = inspect.signature(self._callback).parameters
        success = all(self._results.values())
        self._logger.debug(f"Callback triggered with success={success}")
        if len(params) == 1:
            self._callback(success)
        else:
            self._callback(success, self._results)

    def _download_file(
        self, update_type, url
    ):  # make private as it's an internal helper
        try:
            self._logger.info(f"Starting download for {update_type} from {url}")

            # Create save directory if it doesn't exist
            Path(self._save_dir).mkdir(parents=True, exist_ok=True)
            self._logger.debug(f"Save directory ensured: {self._save_dir}")

            # Extract filename from URL
            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{update_type}_{timestamp}"
                self._logger.debug(f"Generated filename: {filename}")

            # Combine directory path with filename
            full_path = os.path.join(self._save_dir, filename)
            self._logger.debug(f"Full save path: {full_path}")

            # Create request with headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            request = urllib.request.Request(url, headers=headers)

            # Download and save file
            with urllib.request.urlopen(request) as response:
                self._logger.debug(f"Connection established, downloading file")
                with open(full_path, "wb") as f:
                    f.write(response.read())

            self._logger.info(
                f"Successfully downloaded {update_type} update to {full_path}"
            )
            return True
        except Exception as e:
            self._logger.error(f"Failed to download {update_type} update: {str(e)}")
            return False

    def run(self):  # public as it's part of Thread interface
        self._logger.info(
            f"Starting download threads for {len(self._downloads)} updates"
        )
        threads = []
        for update_type, url in self._downloads.items():
            thread = threading.Thread(
                target=lambda t=update_type, u=url: self._results.update(
                    {t: self._download_file(t, u)}
                )
            )
            thread.start()
            threads.append(thread)
            self._logger.debug(f"Started download thread for {update_type}")

        # Wait for all downloads to complete
        for thread in threads:
            thread.join()

        self._logger.info("All download threads completed")
        self._call_callback()
