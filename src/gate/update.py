import threading
from pathlib import Path
import urllib.request
import inspect
import os


class UpdateDownloader(threading.Thread):
    def __init__(self, url, save_dir, callback=None):
        super().__init__()
        self.url = url
        self.save_dir = save_dir
        self.callback = callback

    def _call_callback(self, success, error=None):
        if not self.callback:
            return

        # Check number of parameters the callback accepts
        params = inspect.signature(self.callback).parameters
        if len(params) == 1:
            self.callback(success)
        else:
            self.callback(success, error)

    def run(self):
        try:
            # Create save directory if it doesn't exist
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

            # Extract filename from URL
            filename = os.path.basename(urllib.parse.urlparse(self.url).path)
            if not filename:
                filename = "update.bin"  # Default filename if none in URL

            # Combine directory path with filename
            full_path = os.path.join(self.save_dir, filename)

            # Create request with headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            request = urllib.request.Request(self.url, headers=headers)

            # Download and save file
            with urllib.request.urlopen(request) as response:
                with open(full_path, "wb") as f:
                    f.write(response.read())

            print(f"Update downloaded successfully to {full_path}")
            self._call_callback(True)
        except Exception as e:
            error_msg = f"Failed to download update: {str(e)}"
            print(error_msg)
            self._call_callback(False, error_msg)
