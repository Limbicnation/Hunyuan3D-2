# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import logging
import os
from functools import wraps

import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger('hy3dgen.shapgen')


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.

        Supports both context manager and decorator usage.

        Example as context manager:
        ```python
        with synchronize_timer('name') as t:
            run()
        ```

        Example as decorator:
        ```python
        @synchronize_timer('Export to trimesh')
        def export_to_trimesh(mesh_output):
            pass
        ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f'{self.name} takes {self.time} ms')

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    
    # ===============================================================
    # FIRST: Try to use models from hardcoded path
    # ===============================================================
    if model_path.startswith("tencent/"):
        # Try direct path to models in the parent directory
        local_model_path = f"/home/gero/GitHub/Hunyuan3D-2/models/{model_path}/{subfolder}"
        logger.info(f"Checking for models at: {local_model_path}")
        
        if os.path.exists(local_model_path):
            extension = 'ckpt' if not use_safetensors else 'safetensors'
            variant_str = '' if variant is None else f'.{variant}'
            ckpt_name = f'model{variant_str}.{extension}'
            config_path = os.path.join(local_model_path, 'config.yaml')
            ckpt_path = os.path.join(local_model_path, ckpt_name)
            
            if os.path.exists(config_path) and os.path.exists(ckpt_path):
                logger.info(f"Using local model from {local_model_path}")
                return config_path, ckpt_path
            else:
                logger.info(f"Files missing in {local_model_path}. Config: {os.path.exists(config_path)}, Model: {os.path.exists(ckpt_path)}")
    
    # ===============================================================
    # SECOND: Try environment variable path or default cache 
    # ===============================================================
    base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
    base_dir = os.path.expanduser(base_dir)
    env_model_path = os.path.join(base_dir, model_path, subfolder)
    logger.info(f'Checking environment path: {env_model_path}')
    
    if os.path.exists(env_model_path):
        extension = 'ckpt' if not use_safetensors else 'safetensors'
        variant_str = '' if variant is None else f'.{variant}'
        ckpt_name = f'model{variant_str}.{extension}'
        config_path = os.path.join(env_model_path, 'config.yaml')
        ckpt_path = os.path.join(env_model_path, ckpt_name)
        
        if os.path.exists(config_path) and os.path.exists(ckpt_path):
            logger.info(f"Using model from environment path: {env_model_path}")
            return config_path, ckpt_path
    
    # ===============================================================
    # THIRD: Download from HuggingFace as last resort
    # ===============================================================
    logger.info('No local models found - downloading from HuggingFace')
    try:
        from huggingface_hub import snapshot_download
        # Only download specified subdirectory
        path = snapshot_download(
            repo_id=original_model_path,
            allow_patterns=[f"{subfolder}/*"],
        )
        model_path = os.path.join(path, subfolder)
        logger.info(f'Downloaded model to: {model_path}')
    except ImportError:
        logger.warning("HuggingFace Hub not available - install with 'pip install huggingface-hub'")
        raise RuntimeError(f"Model not found locally and HuggingFace Hub not available")
    except Exception as e:
        logger.warning(f"Error downloading model: {e}")
        raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {original_model_path} not found")

    extension = 'ckpt' if not use_safetensors else 'safetensors'
    variant_str = '' if variant is None else f'.{variant}'
    ckpt_name = f'model{variant_str}.{extension}'
    config_path = os.path.join(model_path, 'config.yaml')
    ckpt_path = os.path.join(model_path, ckpt_name)
    
    # Verify that the files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file not found at {ckpt_path}")
        
    return config_path, ckpt_path