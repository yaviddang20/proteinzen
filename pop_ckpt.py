import torch
from pathlib import Path

ckpt = torch.load(f"{Path(__file__).parent}/proteinzen_weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch=3278-step=780000.ckpt", map_location="cpu", weights_only=False)
ckpt.pop("optimizer_states", None)
ckpt.pop("lr_schedulers", None)                                                                                                                                      
ckpt["optimizer_states"] = []                                                                                                                                        
ckpt["lr_schedulers"] = []     
torch.save(ckpt, f"{Path(__file__).parent}/proteinzen_weights/motif_scaffolding_model/lightning_logs/version_0/checkpoints/epoch=3278-step=780000_no_opt.ckpt")
print("Finished saving checkpoint")