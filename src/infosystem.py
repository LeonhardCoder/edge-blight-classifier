import torch
import platform
import subprocess
import sys
import psutil
import os

def get_system_info():
    """
    Recupera información completa del sistema, GPU, librerías y entorno de ejecución
    """

    print("="*70)
    print("INFORMACIÓN DEL ENTORNO DE EXPERIMENTACIÓN")
    print("="*70)

    # ==========================================
    # INFORMACIÓN DE PLATAFORMA
    # ==========================================
    print("\n📋 PLATAFORMA Y SISTEMA OPERATIVO")
    print("-"*70)
    print(f"Sistema Operativo    : {platform.system()} {platform.release()}")
    print(f"Versión              : {platform.version()}")
    print(f"Arquitectura         : {platform.machine()}")
    print(f"Procesador           : {platform.processor()}")

    # Verificar si estamos en Colab
    try:
        import google.colab
        print(f"Entorno              : Google Colaboratory")
    except:
        print(f"Entorno              : Local/Otro")

    # ==========================================
    # INFORMACIÓN DE GPU
    # ==========================================
    print("\n🎮 INFORMACIÓN DE GPU")
    print("-"*70)

    if torch.cuda.is_available():
        print(f"GPU Disponible       : Sí")
        print(f"Nombre GPU           : {torch.cuda.get_device_name(0)}")
        print(f"Número de GPUs       : {torch.cuda.device_count()}")
        print(f"CUDA Disponible      : Sí")
        print(f"Versión CUDA         : {torch.version.cuda}")
        print(f"cuDNN Versión        : {torch.backends.cudnn.version()}")

        # Memoria GPU
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Memoria GPU Total    : {gpu_mem:.2f} GB")

        # Memoria disponible actual
        mem_alloc = torch.cuda.memory_allocated(0) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"Memoria Asignada     : {mem_alloc:.2f} GB")
        print(f"Memoria Reservada    : {mem_reserved:.2f} GB")

        # Capacidad de computación
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"Compute Capability   : {compute_capability[0]}.{compute_capability[1]}")
    else:
        print("GPU Disponible       : No")

    # ==========================================
    # INFORMACIÓN DE MEMORIA RAM
    # ==========================================
    print("\n💾 INFORMACIÓN DE MEMORIA RAM")
    print("-"*70)

    mem = psutil.virtual_memory()
    print(f"RAM Total            : {mem.total / (1024**3):.2f} GB")
    print(f"RAM Disponible       : {mem.available / (1024**3):.2f} GB")
    print(f"RAM Utilizada        : {mem.used / (1024**3):.2f} GB")
    print(f"Porcentaje Uso       : {mem.percent}%")

    # ==========================================
    # VERSIONES DE LIBRERÍAS
    # ==========================================
    print("\n📚 VERSIONES DE LIBRERÍAS PRINCIPALES")
    print("-"*70)

    libraries = {
        'PyTorch': torch.__version__,
        'Python': platform.python_version(),
    }

    # Intentar importar y obtener versiones de otras librerías
    try:
        import timm
        libraries['timm'] = timm.__version__
    except:
        libraries['timm'] = 'No instalado'

    try:
        import numpy as np
        libraries['NumPy'] = np.__version__
    except:
        libraries['NumPy'] = 'No instalado'

    try:
        import pandas as pd
        libraries['Pandas'] = pd.__version__
    except:
        libraries['Pandas'] = 'No instalado'

    try:
        import sklearn
        libraries['scikit-learn'] = sklearn.__version__
    except:
        libraries['scikit-learn'] = 'No instalado'

    try:
        import cv2
        libraries['OpenCV'] = cv2.__version__
    except:
        libraries['OpenCV'] = 'No instalado'

    try:
        import albumentations
        libraries['Albumentations'] = albumentations.__version__
    except:
        libraries['Albumentations'] = 'No instalado'

    try:
        import matplotlib
        libraries['Matplotlib'] = matplotlib.__version__
    except:
        libraries['Matplotlib'] = 'No instalado'

    try:
        import seaborn
        libraries['Seaborn'] = seaborn.__version__
    except:
        libraries['Seaborn'] = 'No instalado'

    try:
        import scipy
        libraries['SciPy'] = scipy.__version__
    except:
        libraries['SciPy'] = 'No instalado'

    try:
        import PIL
        libraries['Pillow'] = PIL.__version__
    except:
        libraries['Pillow'] = 'No instalado'

    for lib, version in libraries.items():
        print(f"{lib:20s} : {version}")

    # ==========================================
    # INFORMACIÓN DE ALMACENAMIENTO
    # ==========================================
    print("\n💿 INFORMACIÓN DE ALMACENAMIENTO")
    print("-"*70)

    disk = psutil.disk_usage('/')
    print(f"Disco Total          : {disk.total / (1024**3):.2f} GB")
    print(f"Disco Disponible     : {disk.free / (1024**3):.2f} GB")
    print(f"Disco Utilizado      : {disk.used / (1024**3):.2f} GB")
    print(f"Porcentaje Uso       : {disk.percent}%")

    # Verificar montaje de Google Drive
    if os.path.exists('/content/drive'):
        print(f"Google Drive         : Montado en /content/drive")
    else:
        print(f"Google Drive         : No montado")

    # ==========================================
    # CONFIGURACIÓN DE PyTorch
    # ==========================================
    print("\n⚙️  CONFIGURACIÓN DE PyTorch")
    print("-"*70)
    print(f"Versión PyTorch      : {torch.__version__}")
    print(f"Versión TorchVision  : {torch.__version__}")  # Reemplazar si tienes torchvision
    print(f"Dispositivo Default  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Num Threads CPU      : {torch.get_num_threads()}")
    print(f"cuDNN Habilitado     : {torch.backends.cudnn.enabled}")
    print(f"cuDNN Benchmark      : {torch.backends.cudnn.benchmark}")

    print("\n" + "="*70)
    print("✅ Recopilación de información completada")
    print("="*70)

# Ejecutar la función
get_system_info()
