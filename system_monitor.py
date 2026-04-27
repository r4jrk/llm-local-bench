import time
import threading
import psutil

#NVML, but if no NVIDIA GPU(s) then no-no. Extend to handle AMD's and Intel's perhaps?
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class SystemMonitor:
    def __init__(self, gpu_index=0, track_process_vram=True):
        self.running = False
        self.samples = []

        self.prev_disk = None
        self.prev_time = None

        self.gpu_index = gpu_index
        self.track_process_vram = track_process_vram
        self.handle = None

        if _NVML_AVAILABLE:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                self.handle = None

    def start(self):
        self.running = True
        threading.Thread(target=self._collect, daemon=True).start()

    def stop(self):
        self.running = False

    def _collect(self):
        while self.running:
            now = time.time()

            # --- CPU / RAM ---
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            # --- Disk throughput ---
            disk = psutil.disk_io_counters()
            read_bps = 0
            write_bps = 0

            if self.prev_disk is not None:
                dt = now - self.prev_time
                if dt > 0:
                    read_bps = (disk.read_bytes - self.prev_disk.read_bytes) / dt
                    write_bps = (disk.write_bytes - self.prev_disk.write_bytes) / dt

            self.prev_disk = disk
            self.prev_time = now

            # --- GPU ---
            gpu_util, gpu_mem, proc_mem = self._get_gpu_metrics()

            self.samples.append({
                "cpu": cpu,
                "ram": ram,
                "gpu": gpu_util,
                "gpu_mem": gpu_mem,              # MB (device total used)
                "gpu_proc_mem": proc_mem,        # MB (sum over processes)
                "disk_read_bps": read_bps,
                "disk_write_bps": write_bps,
            })

            time.sleep(0.1)  # high-resolution sampling

    def _get_gpu_metrics(self):
        if not _NVML_AVAILABLE or self.handle is None:
            return 0.0, 0.0, 0.0

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            gpu_util = float(util.gpu)  # %
            gpu_mem = float(mem.used / (1024 * 1024))  # MB

            proc_mem = 0.0
            if self.track_process_vram:
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                    proc_mem = sum(p.usedGpuMemory for p in procs if p.usedGpuMemory) / (1024 * 1024)
                except Exception:
                    proc_mem = 0.0

            return gpu_util, gpu_mem, proc_mem

        except Exception:
            return 0.0, 0.0, 0.0

    def summary(self):
        if not self.samples:
            return {}

        def avg(key):
            return sum(s[key] for s in self.samples) / len(self.samples)

        def peak(key):
            return max(s[key] for s in self.samples)

        def median(key):
            vals = sorted(s[key] for s in self.samples)
            n = len(vals)
            return vals[n // 2] if n else 0

        return {
            "cpu_avg": avg("cpu"),
            "ram_avg": avg("ram"),

            "gpu_avg": avg("gpu"),
            "gpu_peak": peak("gpu"),

            "gpu_mem_avg": avg("gpu_mem"),
            "gpu_mem_peak": peak("gpu_mem"),

            "gpu_proc_mem_avg": avg("gpu_proc_mem"),
            "gpu_proc_mem_peak": peak("gpu_proc_mem"),

            "disk_read_avg": avg("disk_read_bps"),
            "disk_read_peak": peak("disk_read_bps"),
            "disk_read_median": median("disk_read_bps"),

            "disk_write_avg": avg("disk_write_bps"),
            "disk_write_peak": peak("disk_write_bps"),
            "disk_write_median": median("disk_write_bps"),
        }