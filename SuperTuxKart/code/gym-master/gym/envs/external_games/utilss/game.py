import re
import os
import subprocess
import threading
import signal
import random
import psutil
import numpy as np

from gym import logger


class GameRegion:
    """A rectangle. Represents the game region.
    """
    
    def __init__(self, x: int = 0, y: int = 0, width: int = None, height: int = None):
        """Creates a new game region
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def as_tuple(self) -> tuple:
        """Returns the game region as a tuple (x, y, width, height)
        """
        return (self.x, self.y, self.width, self.height)


class Action:
    def __init__(self, action, parallel=False):
        self._action = action
        self.parallel = parallel
        
        self._pre  = lambda: None
        self._post = lambda: None
    
    def set_pre(self, action):
        self._pre = action
    
    def set_post(self, action):
        self._post = action
    
    def pre(self):
        self._pre()
    
    def run(self):
        if type(self._action) == list:
            if not self.parallel:
                for action in self._action:
                    action()
            else:
                threads = []
                for action in self._action:
                    thread = threading.Thread(target=action)
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
        else:
            self._action()
    
    def post(self):
        self._post()


class GameHandler:
    """A wrapper that allows to easily handle the game process and window. It allows to read the process standard
    output, to focus the window, and so on.
    """
    SPACE_SPLITTER = re.compile("\s+")
    
    def run_process(command: list, monitor_stdout=True) -> subprocess.Popen:
        """Runs the command specified and spawns a process. It also pipes the standard output so that it can be
        easily handled, if needed.
        """
        kwargs = {}
        if monitor_stdout:
            kwargs['stdout'] = subprocess.PIPE
        
        return subprocess.Popen(command, **kwargs)
        
    def find_window_by_class(klass: str) -> str:
        """Returns the window ID of the specified class. Returns None if no window is present for the given class.
        If more than a window matches the class, this method returns the last one (usually, the correct one).
        """
        
        window_id = subprocess.check_output(['xdotool', 'search', '--class', klass])
        window_ids = window_id.decode("ascii").rstrip().split("\n")
        
        if len(window_ids) > 0:
            return window_ids[-1]
        else:
            return None
    
    def find_windows_by_pid(pid: int) -> str:
        """Returns the window ID of the specified process ID. Returns None if no window is present for the given class.
        This method assumes that there is exactly a window for the searched class: if more than a window if found,
        the assertion fails.
        """
        window_id = subprocess.check_output(['xdotool', 'search', '--pid', str(pid)])
        window_ids = window_id.decode("ascii").rstrip().split("\n")
        assert len(window_ids) <= 1
        
        if len(window_ids) == 1:
            return window_ids
        else:
            return None
    
    def __init__(self, process, window:str = None, **kwargs):
        """Creates a new handler for a given process (subprocess.Popen) and a window ID (string). If the window ID is 
        not specified, it uses the process ID to automatically detect the game window.
        """
        self._process = process
        self._process_info = psutil.Process(self._process.pid)
        
        self._ram = None
        self._vram = None
        
        if window:
            self._windows = [window]
        else:
            self._windows = GameHandler.find_windows_by_pid(self._process.pid)
        
        self._stdout = []
        
        self.autofocus = True
        if 'autofocus' in kwargs:
            self.autofocus = kwargs['autofocus']
        
        if self._process.stdout is not None:
            self._process_log_thread = threading.Thread(target=lambda: self._process_log_mainloop())
            self._process_log_thread.start()

        self.reset_metrics()

        logger.info(f"New game process: {window}")
    
    def screenshot(self):
        return Screenshot.take_window(self._windows[0])
    
    def focus_window(self) -> None:
        """Focuses the game window.
        """
        if self.autofocus:
            for window in self._windows:
                logger.info(f"Focusing window: {window}")
                subprocess.call(['xdotool', 'windowfocus', window])
        else:
            logger.info("Avoiding autofocus")
            
    def move_window(self, x, y):
        for window in self._windows:
            logger.info(f"Moving window {window} to {x}, {y}")
            subprocess.call(['xdotool', 'windowmove', window, str(x), str(y)])
    
    def get_window_region(self) -> GameRegion:
        """Returns the region of the game region (GameRegion).
        """
        raw = subprocess.check_output(['xwininfo', '-id', self._windows[0]])
        txt = raw.decode("utf8")
        
        x       = int(re.findall("Absolute upper\-left X\:\s+([0-9]+)\n", txt)[0])
        y       = int(re.findall("Absolute upper\-left Y\:\s+([0-9]+)\n", txt)[0])
        width   = int(re.findall("Width\:\s+([0-9]+)\n", txt)[0])
        height  = int(re.findall("Height\:\s+([0-9]+)\n", txt)[0])
        
        logger.info(f"Retrieved window region: {x}, {y}, {width}, {height}")
        return GameRegion(x, y, width, height)
    
    def _used_ram(self) -> float:
        return self.used_total_ram()
    
    def used_total_ram(self) -> float:
        """Returns the RAM used by the main process in bytes
        """
        return self._process_info.memory_info().vms
    
    def used_data_ram(self) -> float:
        """Returns the data RAM used by the main process in bytes
        """
        return self._process_info.memory_info().data
    
    def _used_cpu(self) -> float:
        """Returns the CPU clocks used by the main process
        """
        
        result = 0
        with open(f"/proc/{self._process.pid}/stat") as f:
            parts = f.read().split(" ")
            result = float(parts[13]) + float(parts[14]) + float(parts[15]) + float(parts[16])
        
        return result
    
    def _used_vram(self) -> float:
        """Returns the VRAM used by the main process on the VGA
        """
        try:
            dump = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1', '-s', 'm']).decode("utf8")
            lines = dump.split("\n")
            for i in range(2, len(lines)):
                line = lines[i].strip()
                if len(line) == 0:
                    continue
                stats = GameHandler.SPACE_SPLITTER.split(line)
                if int(stats[1]) == self._process.pid:
                    return float(stats[3])
                
            return None
        except subprocess.CalledProcessError:
            return None
    
    def _used_gpu(self) -> float:
        """Returns the total GPU percentage used
        """
        return 0
    
    def used_cpu(self) -> float:
        current = self._used_cpu()
        if self._base_cpu is None or current is None:
            return None
        else:
            return current - self._base_cpu
    
    def used_gpu(self) -> float:
        current = self._used_gpu()
        if self._base_gpu is None or current is None:
            return None
        else:
            return current - self._base_gpu
    
    def used_ram(self) -> float:
        current = self._used_ram()
        if self._base_ram is None or current is None:
            return None
        else:
            return current - self._base_ram
    
    def used_vram(self) -> float:
        current = self._used_vram()
        if self._base_vram is None or current is None:
            return None
        else:
            return current - self._base_vram
    
    def reset_metrics(self) -> None:
        self._base_cpu = self._used_cpu()
        self._base_gpu = self._used_gpu()
        self._base_ram  = self._used_ram()
        self._base_vram = self._used_vram()
    
    def process_id(self) -> int:
        return self._process.pid
    
    def terminate(self) -> None:
        """Terminates the game process.
        """
        self._process.terminate()
        self._process.returncode = 0
    
    def suspend(self) -> None:
        self._process.send_signal(signal.SIGSTOP)
    
    def resume(self) -> None:
        self._process.send_signal(signal.SIGCONT)
    
    def alive(self) -> bool:
        return self._process.returncode is None
    
    def read_log_line(self) -> str:
        """Returns the last process output line not read yet. Returns None if no new line is available.
        """
        if len(self._stdout) > 0:
            return self._stdout.pop(0)
        else:
            return None
    
    def read_log_lines(self) -> list:
        """Returns all the last process output lines not read yet. Returns an empty list if no new line is available.
        """
        result = []
        while len(self._stdout) > 0:
            result.append(self._stdout.pop(0))
        return result
    
    def _process_log_mainloop(self) -> None:
        """Helper method: it is the main loop of the thread that fetches the process output.
        """
        for line in self._process.stdout:
            self._stdout.append(line.decode("utf8"))
    
    def _process_performance_mainloop(self) -> None:
        while self.alive():
            self._ram = self._used_ram()
            self._vram = self._used_vram()


class ProcessWrapper:
    def __init__(self, pid):
        self.pid = pid
        self.stdout = None
        self.stderr = None
        self.stdin  = None
        self.returncode = None
    
    def terminate(self):
        os.kill(self.pid, signal.SIGINT)
    
    def send_signal(self, sig):
        os.kill(self.pid, sig)


class Screenshot:
    TEMP_SCREENSHOT_NAME = f"/tmp/gym-screenshot-{random.randint(0, 100000000)}.ppm"
    
    @classmethod
    def take_window(cls, window):
        subprocess.call(['import', '-window', window, '-silent', cls.TEMP_SCREENSHOT_NAME])
        
        array = None
        with open(cls.TEMP_SCREENSHOT_NAME, 'rb') as f:
            array = np.fromstring(f.read(), dtype='B')
        
        # Reads the header
        header = [b'', b'', b'']
        header_line = 0
        index = 0
        while header_line < 3:
            header[header_line] += array[index]
            if array[index] == 10:
                header_line += 1
            index += 1
        
        # Reads the image pixels based on the maxval header value
        maxval = int(header[2].decode('ascii'))
        if maxval == 255:
            img_data = array[index:]
        elif maxval == 65535:
            img_data = array[index::2]
        else:
            raise WrongPPMFileException(f"Invalid maxvalue {maxval}")
        
        wh = header[1].decode('ascii').split(' ')
        
        return Screenshot(int(wh[0]), int(wh[1]), img_data)
    
    def __init__(self, width, height, data):
        if len(data) != width * height * 3:
            raise WrongImageDataException("Image data not matching declared size")
        
        self.width = width
        self.height = height
        self.pixels = data
    
    def pixel(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise WrongPixelException(f"Invalid pixel {x}, {y}")
        
        base = (x * self.height + y) * 3
        return (self.pixels[base], self.pixels[base + 1], self.pixels[base + 2])
    
    def save(self, filename, overwrite=False, fast=True):
        if os.path.exists(filename) and not overwrite:
            return False
        
        with open(filename, 'wb') as f:
            f.write(f"P6\n{self.width} {self.height}\n255\n".encode())
            if not fast:
                for x in range(self.width):
                    for y in range(self.height):
                        f.write(bytes(self.pixel(x, y)))
            else:
                f.write(bytes(self.pixels))


class SlidingWindowWithOverlap:
    def __init__(self, size=20, overlap=10):
        self._observations = []
        self._size = size
        self._overlap = overlap
        self._current_overlap = -1
    
    def add(self, value):
        self._observations.append(value)
        
        if len(self._observations) >= self._size:
            self._observations.pop(0)
            self._current_overlap += 1
    
    def get(self):
        print(f"Overlap: {self._current_overlap}; Size: {len(self._observations)}")
        overlap = self._current_overlap
        
        if len(self._observations) < self._size - 1:
            return None
        
        if overlap % self._overlap == 0:
            return sum(self._observations)
        else:
            return None
    
    def force_get(self):
        return sum(self._observations)
    
    def clear(self):
        self._observations.clear()
        self._current_overlap = -1


class WrongPPMFileException(Exception):
    pass


class WrongImageDataException(Exception):
    pass


class WrongPixelException(Exception):
    pass
