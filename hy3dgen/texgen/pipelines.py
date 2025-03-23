import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import Union, Optional

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path using environment variable or default cache
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    logger.info(f"Downloading models from huggingface for {original_model_path}")
                    
                    # Download both model components with more robust error handling
                    try:
                        # First try downloading with a single call that gets everything
                        model_path = huggingface_hub.snapshot_download(
                            repo_id=original_model_path,
                            allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"]
                        )
                    except Exception as e:
                        logger.warning(f"Error downloading full model, trying individual components: {e}")
                        # If that fails, try downloading each component separately
                        huggingface_hub.snapshot_download(
                            repo_id=original_model_path,
                            allow_patterns=["hunyuan3d-delight-v2-0/*"],
                            local_dir=model_path
                        )
                        huggingface_hub.snapshot_download(
                            repo_id=original_model_path,
                            allow_patterns=["hunyuan3d-paint-v2-0/*"],
                            local_dir=model_path
                        )
                    
                    # Check if the download succeeded
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
                    
                    if not os.path.exists(delight_model_path):
                        raise RuntimeError(f"Failed to download delight model to {delight_model_path}")
                    if not os.path.exists(multiview_model_path):
                        raise RuntimeError(f"Failed to download multiview model to {multiview_model_path}")
                        
                    logger.info(f"Successfully downloaded models to {model_path}")
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))
                    
                except ImportError:
                    logger.warning("You need to install HuggingFace Hub to load models from the hub.")
                    raise RuntimeError(f"Model path {model_path} not found and huggingface_hub not installed")
                except Exception as e:
                    logger.error(f"Error downloading models: {e}")
                    raise RuntimeError(f"Failed to download models: {e}")
            else:
                logger.info(f"Using models from local cache: {model_path}")
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))
        else:
            # Direct path was provided and exists
            logger.info(f"Using directly provided model path: {model_path}")
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
            
            if not os.path.exists(delight_model_path):
                raise RuntimeError(f"Model path {delight_model_path} not found")
            if not os.path.exists(multiview_model_path):
                raise RuntimeError(f"Model path {multiview_model_path} not found")
                
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cuda cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    # Rest of the class methods remain unchanged
    # ...