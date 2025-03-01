from queue import Queue
from network.api.download_client import UpdateDownloader
from utils.logger_mixin import LoggerMixin


class UpdateManager(LoggerMixin):
    def __init__(self, gate, update_url: str, update_save_dir: str):
        self.gate = gate
        self.update_url = update_url
        self.update_save_dir = update_save_dir
        self.update_thread = None
        self.update_queue = Queue()
        self.logger = self._setup_logger(__name__, self.gate.logger.level)

    def handle_update(self, updates):
        """Handle multiple update processes in parallel"""
        if self.gate.is_busy:
            self.logger.info("System is busy, queueing updates")
            self.update_queue.put(updates)
            return True

        if self.update_thread and self.update_thread.is_alive():
            self.gate.logger.warning("Update already in progress")
            return False

        # Create URLs dictionary
        download_urls = {
            update_type: f"{self.update_url}{filename}"
            for update_type, filename in updates.items()
        }

        def update_callback(success, results=None):
            if success:
                self.gate.logger.info("All updates completed successfully")
            else:
                self.gate.logger.error("Some updates failed")
                if results:
                    for update_type, result in results.items():
                        self.gate.logger.info(
                            f"{update_type} update: {'Success' if result else 'Failed'}"
                        )
            self.update_thread = None

        self.update_thread = UpdateDownloader(
            download_urls,
            self.update_save_dir,
            update_callback,
            self.gate.logger.level,
        )
        self.update_thread.start()
        return True

    def process_pending_updates(self):
        """Process any pending updates in the queue"""
        if not self.update_queue.empty() and not self.gate.is_busy:
            updates = self.update_queue.get()
            self.gate.logger.info("Processing queued updates")
            self.handle_update(updates)
