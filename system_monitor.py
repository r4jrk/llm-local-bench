class SystemMonitor:
    def __init__(self):
        self.running = False
        self.samples = []
        self.prev_disk = None
        self.prev_time = None

    def start(self):
        self.running = True
        threading.Thread(target=self._collect, daemon=True).start()

    def stop(self):
        self.running = False

    def _collect(self):
        while self.running:
            now = time.time()

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            disk = psutil.disk_io_counters()
            read_bytes = disk.read_bytes
            write_bytes = disk.write_bytes

            read_bps = 0
            write_bps = 0

            if self.prev_disk is not None:
                dt = now - self.prev_time
                if dt > 0:
                    read_bps = (read_bytes - self.prev_disk.read_bytes) / dt
                    write_bps = (write_bytes - self.prev_disk.write_bytes) / dt

            self.prev_disk = disk
            self.prev_time = now

            gpu, gpu_mem = get_gpu_usage()

            self.samples.append({
                "cpu": cpu,
                "ram": ram,
                "gpu": gpu,
                "gpu_mem": gpu_mem,
                "disk_read_bps": read_bps,
                "disk_write_bps": write_bps,
            })

            time.sleep(0.5)