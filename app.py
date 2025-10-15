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

# --- Constantes ---
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

# --- Pipeline global (corrigé) ---
_pipeline = None  # Doit être déclaré avant preload_model()

# --- Pipeline loader ---
def preload_model() -> TrellisImageTo3DPipeline:
    """Charge le modèle TRELLIS sur GPU si disponible et retourne le pipeline."""
    global _pipeline
    if _pipeline is not None:
        print("✅ Modèle déjà chargé, skip preload.")
        return _pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Initialisation du pipeline sur le device: {device}")

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    if pipeline is None:
        raise RuntimeError("❌ Échec du chargement du pipeline TRELLIS.")

    pipeline = pipeline.to(device)

    # Vérifier que pipeline a bien l'attribut device avant d'assigner
    if hasattr(pipeline, 'device'):
        pipeline.device = device
    else:
        print("⚠️ pipeline n'a pas d'attribut device, utilisation directe du device lors de l'appel")

    _pipeline = pipeline  # assignation correcte
    print(f"✅ Modèle TRELLIS chargé sur {device.upper()}")
    return _pipeline

# --- FastAPI app ---
app = FastAPI()


@app.on_event("startup")
def on_startup():
    """Création du dossier tmp et préchargement du modèle au démarrage FastAPI."""
    os.makedirs(TMP_DIR, exist_ok=True)
    print("🔹 Démarrage FastAPI : création du dossier tmp et préchargement du modèle...")
    preload_model()
    print("✅ Modèle TRELLIS prêt à l'utilisation.")


# --- Fonctions utilitaires ---
def preprocess_image(pipeline: TrellisImageTo3DPipeline, image: Image.Image) -> Image.Image:
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
    pipeline: TrellisImageTo3DPipeline,
    image: Image.Image,
    seed: int = 42,
    ss_guidance_strength: float = 1.0,
    ss_sampling_steps: int = 20,
    slat_guidance_strength: float = 1.0,
    slat_sampling_steps: int = 20,
) -> str:
    """Génère un fichier GLB à partir d'une image en utilisant le pipeline fourni."""

    os.makedirs(TMP_DIR, exist_ok=True)

    with torch.no_grad():
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

    glb_path = os.path.join(TMP_DIR, 'output.glb')
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

# --- Endpoint FastAPI ---
async def to_3d(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGBA")
    pipeline = preload_model()
    glb_path = image_to_3d(pipeline, image)
    return FileResponse(glb_path, media_type="model/gltf-binary", filename="output.glb")
