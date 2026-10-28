import os
import math
import torch
import numpy as np
from PIL import Image
import folder_paths

class RefModCachePersistent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generated_image": ("IMAGE", {"lazy": True}),
            "reuse_saved": ("BOOLEAN", {"default": False}),
            "slot": (["rostro", "cuerpo"], {"default": "rostro"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "RefMod/Cache"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-evaluate on every queued request, so a changed switch or saved image is always seen.
        return float("nan")

    def check_lazy_status(self, generated_image=None, reuse_saved=False, slot="rostro"):
        # In reuse mode ComfyUI never traverses the expensive photo generation branches.
        if reuse_saved:
            return []
        if generated_image is None:
            return ["generated_image"]
        return []

    @staticmethod
    def target_path(slot):
        if slot not in ("rostro", "cuerpo"):
            raise ValueError("slot invalido")
        root = os.path.join(folder_paths.get_output_directory(), "refmod_cache_persistente")
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"ultimo_{slot}.png")

    def execute(self, generated_image=None, reuse_saved=False, slot="rostro"):
        path = self.target_path(slot)
        if reuse_saved:
            if not os.path.exists(path):
                raise RuntimeError(f"No hay foto guardada para {slot}: {path}. Cambia el switch a FALSE y genera una vez.")
            with Image.open(path) as pil:
                arr = np.asarray(pil.convert("RGB"), dtype=np.float32).copy()/255.0
            print(f"[RefModCache] REUTILIZANDO {slot}: {path}")
            return (torch.from_numpy(arr).unsqueeze(0),)
        if generated_image is None:
            raise RuntimeError(f"No llegó la foto RefMod {slot}; verifica conexión de generación.")
        a=(generated_image[0].detach().float().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        temp=path+'.tmp'
        Image.fromarray(a).convert('RGB').save(temp,format='PNG')
        os.replace(temp,path)
        print(f"[RefModCache] GUARDADO {slot}: {path}")
        return (generated_image,)

NODE_CLASS_MAPPINGS={"RefModCachePersistent":RefModCachePersistent}
NODE_DISPLAY_NAME_MAPPINGS={"RefModCachePersistent":"RefMod Cache persistente - TRUE reutiliza / FALSE guarda"}
