import os
import torch
import numpy as np
import threading
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

# --- Pipeline loader ---
GLOBAL_PIPELINE: TrellisImageTo3DPipeline | None = None  # ✅ Singleton global
_PIPELINE_LOCK = threading.Lock()  # protège la création du pipeline

def preload_model() -> TrellisImageTo3DPipeline:
    """Charge le modèle TRELLIS sur GPU si disponible et retourne le pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Initialisation du pipeline sur le device: {device} (pid={os.getpid()})")

    try:
        print("🔹 Tentative de chargement du modèle via TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        print(f"🔹 Résultat du chargement from_pretrained: {type(pipeline)}")

        if isinstance(pipeline, type):
            print("⚠️ from_pretrained() a retourné une CLASSE, instanciation manuelle...")
            pipeline = pipeline()
            print(f"✅ Pipeline instancié manuellement : {pipeline} (id={id(pipeline)})")

        if pipeline is None:
            raise RuntimeError("❌ TrellisImageTo3DPipeline.from_pretrained() a retourné None")

        # Assigne le device
        try:
            pipeline = pipeline.to(device)
        except Exception as e:
            print(f"⚠️ Erreur pendant pipeline.to({device}): {repr(e)} — continuation")

        print(f"✅ Modèle TRELLIS chargé sur {device.upper()} (pipeline id={id(pipeline)})")
        return pipeline

    except Exception as e:
        print(f"❌ Exception pendant preload_model: {repr(e)}")
        raise RuntimeError(f"❌ Échec du chargement du modèle TRELLIS : {e}")


def get_pipeline() -> TrellisImageTo3DPipeline:
    """Retourne le pipeline global, le charge si nécessaire (thread-safe)."""
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is None:
        with _PIPELINE_LOCK:
            if GLOBAL_PIPELINE is None:
                print("🔹 Pipeline non chargé — initialisation maintenant...")
                try:
                    GLOBAL_PIPELINE = preload_model()
                    print(f"✅ Pipeline global assigné (id={id(GLOBAL_PIPELINE)})")
                except Exception as e:
                    print(f"❌ Erreur pendant le chargement du pipeline global : {e}")
                    GLOBAL_PIPELINE = None
                    raise RuntimeError("❌ Pipeline non disponible") from e
    else:
        print(f"🔹 Pipeline global déjà chargé (id={id(GLOBAL_PIPELINE)})")

    return GLOBAL_PIPELINE


# --- FastAPI app ---
app = FastAPI()


@app.on_event("startup")
def on_startup():
    """Création du dossier tmp uniquement. Ne pas précharger le pipeline ici."""
    os.makedirs(TMP_DIR, exist_ok=True)
    print("🔹 Démarrage FastAPI : création du dossier tmp")
    print("🔹 Pipeline TRELLIS sera chargé à la première requête.")


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
    """Génère un fichier GLB à partir d'une image."""
    # ⚡ Si le pipeline est None, tenter de récupérer le pipeline global
    if pipeline is None:
        print("⚠️ Aucun pipeline fourni, tentative de récupération via get_pipeline()...")
        pipeline = get_pipeline()
        if pipeline is None:
            raise RuntimeError("❌ Aucun pipeline disponible pour image_to_3d")

    print(f"🔹 Pipeline prêt pour génération 3D (id={id(pipeline)})")
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
        if outputs is None:
            raise RuntimeError("❌ pipeline.run() a retourné None")

    glb_path = os.path.join(TMP_DIR, "output.glb")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.96,
        texture_size=512,
        verbose=False,
    )
    glb.export(glb_path)
    torch.cuda.empty_cache()
    print(f"✅ Génération 3D terminée : {glb_path}")
    return glb_path


# --- Endpoint FastAPI corrigé ---
from fastapi import APIRouter

router = APIRouter()

@router.post("/to_3d/")
async def to_3d(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGBA")
    print(f"🔹 Image reçue, taille: {img.size}, mode: {img.mode}")

    # ✅ Récupération sûre du pipeline
    pipeline = get_pipeline()
    if pipeline is None:
        raise RuntimeError("❌ Impossible de récupérer le pipeline TRELLIS")

    glb_path = image_to_3d(pipeline, img, seed=42)
    return FileResponse(glb_path, media_type="model/gltf-binary", filename="output.glb")
