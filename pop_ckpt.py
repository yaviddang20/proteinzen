import torch
from pathlib import Path

# ckpt = torch.load(f"{Path(__file__).parent}/proteinzen_weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch=3278-step=780000.ckpt", map_location="cpu", weights_only=False)
ckpt_path = Path(f"{Path(__file__).parent}/outputs/geom_identityRot_256_conformer_3std_bondlength/train/lightning_logs/version_29/checkpoints/last.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
ckpt.pop("optimizer_states", None)
ckpt.pop("lr_schedulers", None)                                                                                                                                      
ckpt["optimizer_states"] = []                                                                                                                                        
ckpt["lr_schedulers"] = []     
torch.save(ckpt, f"{ckpt_path.parent}/{ckpt_path.stem}_no_opt.ckpt")
print(f"Finished saving checkpoint to {ckpt_path.parent}/{ckpt_path.stem}_no_opt.ckpt")