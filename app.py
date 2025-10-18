import os
import torch
import numpy as np
from typing import Tuple
from PIL import Image
from easydict import EasyDict as edict
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import postprocessing_utils

# --- Pipeline init ---
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")

# --- Constantes ---
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

# --- FastAPI app ---
app = FastAPI()

@app.on_event("startup")
def create_tmp_dir():
    os.makedirs(TMP_DIR, exist_ok=True)

# --- Fonctions ---
def preprocess_image(image: Image.Image) -> Image.Image:
    """Prétraitement officiel Trellis (correction couleurs, masque, luminosité)."""
    return pipeline.preprocess_image(image)

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh

def image_to_3d(
    image: Image.Image,
    seed: int = 42,
    ss_guidance_strength: float = 1.0,
    ss_sampling_steps: int = 20,
    slat_guidance_strength: float = 1.0,
    slat_sampling_steps: int = 20,
) -> str:
    user_dir = TMP_DIR
    os.makedirs(user_dir, exist_ok=True)

    with torch.no_grad():
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,  # déjà fait avant
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

    glb_path = os.path.join(user_dir, 'output.glb')
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.96,
        texture_size=512,
        verbose=False
    )
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path

@app.post("/to_3d/")
async def to_3d(file: UploadFile = File(...)):
    """Reçoit une image, la prétraite et génère le modèle GLB."""
    image = Image.open(file.file)

    # --- Gestion du mode et correction de luminosité ---
    if image.mode == "RGBA":
        # Conserve l'alpha pour éviter l'assombrissement
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    # --- Étape clé : prétraitement officiel Trellis ---
    image = preprocess_image(image)

    # --- Génération ---
    glb_path = image_to_3d(image)
    return FileResponse(glb_path, media_type="model/gltf-binary", filename="output.glb")
