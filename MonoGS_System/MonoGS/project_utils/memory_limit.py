import os
import sys

_windows_job_handle = None

def get_total_cpu_memory():
    """Returns the total CPU RAM in bytes."""
    if sys.platform == 'win32':
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys
    elif sys.platform.startswith('linux'):
        # from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    return 0


def limit_cpu_memory(limit_gb: float):
    if limit_gb is None:
        return
    if limit_gb < 1.0:
        raise ValueError("Limit must be >= 1GB")
        
    limit_bytes = int(limit_gb * (1024 ** 3))
    total_memory = get_total_cpu_memory()
    
    if total_memory > 0 and limit_bytes >= total_memory:
        # If the limit is greater than the available memory on the hardware,
        # the system has unlimited memory access.
        return
    
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        
        global _windows_job_handle
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_ulonglong),
                ('WriteOperationCount', ctypes.c_ulonglong),
                ('OtherOperationCount', ctypes.c_ulonglong),
                ('ReadTransferCount', ctypes.c_ulonglong),
                ('WriteTransferCount', ctypes.c_ulonglong),
                ('OtherTransferCount', ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', ctypes.c_ulong),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', ctypes.c_ulong),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', ctypes.c_ulong),
                ('SchedulingClass', ctypes.c_ulong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("Failed to create job object")
            
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        res = kernel32.QueryInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None
        )
        
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = limit_bytes
        
        res = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        if not res:
            raise OSError("Failed to set job object information")
            
        current_process = kernel32.GetCurrentProcess()
        res = kernel32.AssignProcessToJobObject(job, current_process)
        if not res:
            raise OSError(f"Failed to assign process to job object. Error: {ctypes.GetLastError()}")
        
        _windows_job_handle = job
    
    elif sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    else:
        print(f"CPU memory limiting is not supported on platform {sys.platform}")


def limit_gpu_memory(limit_gb: float):
    if limit_gb is None:
        return
    if limit_gb < 1.0:
        raise ValueError("Limit must be >= 1GB")
        
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed, skipping GPU memory limit.")
        return
        
    if not torch.cuda.is_available():
        print("CUDA is not available, skipping GPU memory limit.")
        return
        
    limit_bytes = int(limit_gb * (1024 ** 3))
    
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        if limit_bytes >= total_memory:
            # Unlimited access if limit is greater than available
            pass
        else:
            fraction = limit_bytes / total_memory
            torch.cuda.set_per_process_memory_fraction(fraction, i)


def set_memory_limit(gpu_limit_gb=None, cpu_limit_gb=None):
    """
    Limits the available GPU VRAM or CPU RAM to a specified limit in GB.
    Limits must be >= 1GB.
    If the limit is greater than the available memory on the hardware,
    the system has unlimited memory access.
    """
    if gpu_limit_gb is not None:
        limit_gpu_memory(gpu_limit_gb)
    if cpu_limit_gb is not None:
        limit_cpu_memory(cpu_limit_gb)
