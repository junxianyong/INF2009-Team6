import threading
from pathlib import Path
import urllib.request
import inspect
import os
from datetime import datetime


class UpdateDownloader(threading.Thread):
    def __init__(self, downloads, save_dir, callback=None):
        """
        downloads: dict of {type: url} pairs
        save_dir: directory to save files
        callback: callback function to call when all downloads complete
        """
        super().__init__()
        self.downloads = downloads
        self.save_dir = save_dir
        self.callback = callback
        self.results = {}

    def _call_callback(self):
        if not self.callback:
            return

        # Check number of parameters the callback accepts
        params = inspect.signature(self.callback).parameters
        if len(params) == 1:
            self.callback(all(self.results.values()))
        else:
            self.callback(all(self.results.values()), self.results)

    def download_file(self, update_type, url):
        try:
            # Create save directory if it doesn't exist
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

            # Extract filename from URL
            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if not filename:
                # If no filename in URL, create one with timestamp to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{update_type}_{timestamp}"

            # Combine directory path with filename
            full_path = os.path.join(self.save_dir, filename)

            # Create request with headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            request = urllib.request.Request(url, headers=headers)

            # Download and save file
            with urllib.request.urlopen(request) as response:
                with open(full_path, "wb") as f:
                    f.write(response.read())

            print(f"{update_type} update downloaded successfully to {full_path}")
            return True
        except Exception as e:
            print(f"Failed to download {update_type} update: {str(e)}")
            return False

    def run(self):
        threads = []
        for update_type, url in self.downloads.items():
            thread = threading.Thread(
                target=lambda t=update_type, u=url: self.results.update(
                    {t: self.download_file(t, u)}
                )
            )
            thread.start()
            threads.append(thread)

        # Wait for all downloads to complete
        for thread in threads:
            thread.join()

        self._call_callback()
