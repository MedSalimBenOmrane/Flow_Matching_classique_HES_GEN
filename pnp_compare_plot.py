# pnp_compare_plot.py

import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pnpflow.methods.pnp_dataloaders import get_pnp_loader
from pnpflow.degradations import HE_HES_Degradation
from generate_pnp_fm import PNP_FLOW
from ddpm_unet.denoising_diffusion_pytorch import Unet as DDPMUnet

def parse_args():
    p = ArgumentParser("Comparaison HE réel, HES généré, HES réel")
    # données / checkpoint
    p.add_argument('--data_root',  type=str, required=True,
                  help="Répertoire Data_HES_HE")
    p.add_argument('--checkpoint', type=str, required=True,
                  help="Chemin vers model_final.pt")
    p.add_argument('--device',     type=str, default='cuda')

    # hyper-paramètres PnP
    p.add_argument('--steps_pnp',   type=int,   default=100,
                  help="Nombre d'itérations PnP")
    p.add_argument('--lr_pnp',      type=float, default=0.1,
                  help="Pas de gradient initial")
    p.add_argument('--noise_type',  type=str,   default='none',
                  choices=['none','gaussian','laplace'])
    p.add_argument('--sigma_noise', type=float, default=0.0,
                  help="σ du bruit si gaussian/laplace")
    p.add_argument('--num_samples', type=int,   default=1,
                  help="Échantillons par itération")

    # architecture DDPMUnet
    p.add_argument('--dim',        type=int,   required=True,
                  help="dim (latent) du UNet")
    p.add_argument('--dim_mults',  nargs='+', type=int, required=True,
                  help="dim_mults, ex: 1 1 2 2 4 4")
    p.add_argument('--num_channels', type=int, default=3,
                  help="Canaux d'entrée/sortie")
    p.add_argument('--dropout',      type=float, default=0.0,
                  help="Dropout du UNet")

    # flow-matching / lr-schedule
    p.add_argument('--model',       type=str, default='indep_couplage',
                  choices=['ot','indep_couplage','rectified'],
                  help="Type de flow-matching pré-entraîné")
    p.add_argument('--method',      type=str, default='pnp_flow',
                   choices=['pnp_flow','pnp_gs','pnp_diff'],
                   help="Algorithme PnP à utiliser")
    p.add_argument('--gamma_style', type=str, default='1_minus_t',
                  choices=['1_minus_t','sqrt_1_minus_t','constant','alpha_1_minus_t'],
                  help="Schedule du pas γ(t)")
    p.add_argument('--alpha',       type=float, default=1.0,
                  help="Exposant si gamma_style='alpha_1_minus_t'")
    return p.parse_args()

def main():
    args = parse_args()
    # ajustement lr selon type de bruit
    if args.noise_type == 'gaussian':
        args.lr_pnp *= args.sigma_noise**2
    elif args.noise_type == 'laplace':
        args.lr_pnp *= args.sigma_noise

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) instanciation du UNet DDPM
    model = DDPMUnet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        channels=args.num_channels,
        dropout=args.dropout
    ).to(device)

    # 2) chargement du state_dict
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd)
    model.eval()
    

    # 3) wrap Plug-and-Play
    pnp = PNP_FLOW(model, device, args)

    # 4) un seul batch pour l'affichage
    loader = get_pnp_loader(args.data_root, split='test', batch_size=1, dim_image=128, shuffle=False)
    hes, he = next(iter(loader))
    hes, he = hes.to(device), he.to(device)
    # On garde une copie 3-canaux pour l'affichage final
    he_display = he.clone()

   # On ne prend que les 2 premiers canaux pour l'observation
    he2 = he[:, :2, :, :]

    # 5) ajout de bruit sur HE si demandé
    if args.noise_type == 'gaussian':
        he_noisy = he2 + torch.randn_like(he2) * args.sigma_noise
    elif args.noise_type == 'laplace':
       noise = torch.distributions.laplace.Laplace(
           torch.zeros_like(he2), args.sigma_noise).sample().to(device)
       he_noisy = he2 + noise
    else:
        he_noisy = he2

    # 6) initialization x0 = Hᵀ y
    H = HE_HES_Degradation()
    x = H.H_adj(he_noisy).to(device)

    # 7) boucle PnP-Flow
    delta = 1.0 / args.steps_pnp
    for i in range(args.steps_pnp):
        t = torch.full((x.size(0),), delta*i, device=device)
        lr_t = pnp.learning_rate_strat(args.lr_pnp, t)
        z = x - lr_t.view(-1,1,1,1) * pnp.grad_datafit(
           x, he_noisy, H.H, H.H_adj
       )
        x_new = torch.zeros_like(x)
        for _ in range(args.num_samples):
            zt = pnp.interpolation_step(z, t.view(-1,1,1,1))
            x_new += pnp.denoiser(zt, t)
        x = x_new / args.num_samples

    # 8) affichage
    to_np = lambda img: img.cpu().squeeze().permute(1,2,0).numpy()
    for img, title in [(to_np(he_display), "HE réel"),
                       (to_np(x),    "HES généré"),
                       (to_np(hes),  "HES réel")]:
        plt.figure(); plt.imshow((img*0.5+0.5).clip(0,1))
        plt.title(title); plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
