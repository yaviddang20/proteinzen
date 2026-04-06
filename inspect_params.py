import torch                                                                                                
from proteinzen.runtime.lmod import BiomoleculeModule                                                                                                                
                                                                                                                                                                    
ckpt = torch.load("/datastor1/dy4652/proteinzen/outputs/geom_identityRot_256_conformer_3std_stereo/train/lightning_logs/version_3/checkpoints/last.ckpt", map_location="cpu", weights_only=False)                                                                                                                         
# just look at the keys                                                                                                                                              
param_names = [k for k in ckpt['state_dict'].keys() if k.startswith('model.')]                                                                                       
for n in param_names:                                                                                                                                                
    print(n)