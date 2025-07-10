import torch
import matplotlib.pyplot as plt
from pnpflow.methods.pnp_dataloaders import get_pnp_loader
from pnpflow.methods.pnp_flow import PNP_FLOW
from pnpflow.degradations import Denoising 
from ddpm_unet.denoising_diffusion_pytorch import Unet as DDPMUnet
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # Paramètres du modèle DDPM U-Net
    parser.add_argument('--dim',           type=int,   default=256)
    parser.add_argument('--dim_mults',     nargs='+',  type=int,   default=[1,1,2,2,4,4])
    parser.add_argument('--num_channels',  type=int,   default=3)
    parser.add_argument('--dropout',       type=float, default=0.0)
    # Paramètres PnP
    parser.add_argument('--model',         type=str,   default='indep_couplage')
    parser.add_argument('--method',        type=str,   default='pnp_flow',
                        help="Nom de la méthode PnP (pour save_path, logs, ...)")
    parser.add_argument('--noise_type',    type=str,   default='none')
    parser.add_argument('--num_samples',   type=int,   default=1)
    parser.add_argument('--steps_pnp',     type=int,   default=1000)
    parser.add_argument('--lr_pnp',        type=float, default=1e-3)
    # Nouveau : stratégie de décroissance du LR
    parser.add_argument('--gamma_style',
                        type=str,
                        default='constant',
                        choices=['1_minus_t','sqrt_1_minus_t','constant','alpha_1_minus_t'],
                        help="Forme de décroissance du pas lr_pnp en fonction de t")
    # Puissance α pour 'alpha_1_minus_t'
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help="Exponent α utilisé si gamma_style='alpha_1_minus_t'")
    # Chargement des données
    parser.add_argument('--data_root',     type=str,   required=True,
                        help="Chemin vers le dossier contenant HES/HE")
    parser.add_argument('--dim_image',     type=int,   default=256)
    parser.add_argument('--batch_size',    type=int,   default=1)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--eval_split',    type=str,   default='test')
    return parser.parse_args()

def imshow(tensor, ax, title):
    # tensor : 1×3×H×W ou 3×H×W
    t = tensor.detach().cpu()
    # si batch dimension :
    if t.ndim == 4:
        t = t[0]
    # 1) denormalize
    t = t * 0.5 + 0.5
    # 2) clamp
    t = t.clamp(0,1)
    # 3) to numpy H×W×3
    img = t.permute(1,2,0).numpy()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

def generate_one(pnp, he_img, degradation, args):
    H, H_adj = degradation.H, degradation.H_adj
    x = H_adj(he_img).to(he_img.device)
    delta = 1 / args.steps_pnp
    # On convertit lr_pnp en Tensor pour que learning_rate_strat renvoie toujours un Tensor
    lr_pnp_tensor = torch.tensor(args.lr_pnp, device=x.device)
    with torch.no_grad():
        for i in range(args.steps_pnp):
            t    = torch.full((x.shape[0],), delta * i, device=x.device)
            lr_t = pnp.learning_rate_strat(lr_pnp_tensor, t)
            grad = pnp.grad_datafit(x, he_img, H, H_adj)
            x    = x - lr_t.view(-1,1,1,1) * grad
            z_tilde = pnp.interpolation_step(x, t.view(-1,1,1,1))
            x       = pnp.denoiser(z_tilde, t)
    return x.clamp(0,1)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Instanciation du modèle Flow-Matching (U-Net DDPM)
    model = DDPMUnet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        channels=args.num_channels,
        dropout=args.dropout
    ).to(device)
    ckpt = torch.load('/home/salim/Desktop/PFE/implimentation/PNP_FM/PnP-Flow (UNET DDPM)/model/hes/flow_matching_indep/model_final.pt', map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    pnp = PNP_FLOW(model, device, args)

    # 2) Préparer le loader paires (HES, HE)
    loader = get_pnp_loader(
        data_root   = args.data_root,
        split       = args.eval_split,
        batch_size  = args.batch_size,
        dim_image   = args.dim_image,
        num_workers = args.num_workers,
        shuffle     = False
    )
    degradation = Denoising()  # identité H, H_adj = x ↦ x

    # 3) Récupérer une paire
    hes_real, he_input = next(iter(loader))
    he_input = he_input.to(device)

    # 4) Générer la HES à partir de la HE
    hes_gen = generate_one(pnp, he_input, degradation, args)

    # 5) Affichage
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    imshow(he_input,  axes[0], 'HE input')
    imshow(hes_gen,   axes[1], 'HES générée')
    imshow(hes_real,  axes[2], 'HES réelle')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
