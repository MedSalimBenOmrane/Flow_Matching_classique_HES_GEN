import os
import torch
import numpy as np
import skimage.io as io
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
import ot
from matplotlib import pyplot as plt
import torch.cuda.amp as amp
from pnpflow.models import InceptionV3
import pnpflow.fid_score as fs
from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils


class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d             = args.dim_image
        self.num_channels  = args.num_channels
        self.device        = device
        self.args          = args
        self.lr            = args.lr
        self.model         = model.to(device)
        # ─── SANITY CHECK ───
        # on vérifie que model(x, t) fonctionne et renvoie la bonne shape
        with torch.no_grad():
            x = torch.randn(
                2,
                self.num_channels,
                self.d,
                self.d,
                device=self.device
            )
            t = torch.rand(2, device=self.device)           # vecteur (B,)
            out = self.model(x, t)
            print(f"[Sanity] Forward OK → output shape: {out.shape}")
        # ─────────────────────

    def train_FM_model(self, train_loader, opt, num_epoch):
        # Outer loop on epochs, on affiche juste le compte de l'époque
        for ep in range(num_epoch):
            print(f"\n→ Epoch {ep+1}/{num_epoch}")

            # Barre de progression batch-par-batch avec ETA et affichage du loss
            batch_bar = tqdm(
                train_loader,
                desc="   Batches",
                unit="batch",
                leave=False
            )

            for iteration, (x, labels) in enumerate(batch_bar):
                if x.size(0) == 0:
                    continue
                x = x.to(self.device)

                # Échantillonnage latent
                # Échantillonnage latent : même shape que x
                # → on évite tout décalage si x a été recadré/autre
                z = torch.randn_like(x, device=self.device, requires_grad=True)
                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                # compute coupling par transport optimal
                # on garde les deux nuages x0, x1 identiques en taille
                x0 = z.clone()
                x1 = x.clone()
                # flatten sur la même dimension “features”
                x0_flat = x0.view(x0.size(0), -1)
                x1_flat = x1.view(x1.size(0), -1)
                # (a) version GPU + torch.cdist
                M_t = torch.cdist(x0_flat, x1_flat, p=2) ** 2   # pairwise squared‐Euclidiennes
                M = M_t.detach().cpu().numpy()
                # uniform weights
                a = b = np.ones(x0_flat.size(0)) / x0_flat.size(0)
                plan = ot.emd(a, b, M)
                p = plan.flatten()
                p = p / p.sum()
                choices = np.random.choice(
                    plan.shape[0] * plan.shape[1],
                    p=p, size=len(x0), replace=True
                )
                i, j = np.divmod(choices, plan.shape[1])
                x0 = x0[i]
                x1 = x1[j]
                xt = t1 * x1 + (1 - t1) * x0

                # calcul du loss et update
                # ----- forward en mixed precision -----
                with amp.autocast():
                    pred = self.model(xt, t1.squeeze())
                    loss = torch.sum((pred - (x1 - x0))**2) / x.shape[0]

                # ----- backward + step -----
                opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()

                # Save loss in txt file
                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

                # On affiche le loss dans la barre
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            # en fin d'époque : échantillonnage, sauvegardes, FID tous les 5 epochs
            self.sample_plot(x, ep)
            if (ep + 1) % 5 == 0:
                torch.save(
                    self.model.state_dict(),
                    self.model_path + f'model_{ep+1}.pt'
                )
                fid_value = self.compute_fast_fid(2048)
                with open(self.save_path + 'fid.txt', 'a') as file:
                    file.write(f'Epoch: {ep+1}, FID: {fid_value}\n')

    def apply_flow_matching(self, NO_samples):
        self.model.eval()
        with torch.no_grad():
            model_class = cnf(self.model)
            latent = torch.randn(
                NO_samples,
                self.num_channels,
                self.d,
                self.d,
                device=self.device
            )
            z_t = odeint(
                model_class,
                latent,
                torch.tensor([0.0, 1.0]).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )
            x = z_t[-1].detach()
        self.model.train()
        return x

    def sample_plot(self, x, ep=None):
        os.makedirs(self.save_path + 'results_samplings/', exist_ok=True)
        reco = utils.postprocess(self.apply_flow_matching(16), self.args)
        utils.save_samples(
            reco.detach().cpu(), x[:16].cpu(),
            self.save_path + f'results_samplings/samplings_ep_{ep}.pdf',
            self.args
        )
        # save training samples once
        if ep == 0:
            gt = utils.postprocess(x[:16], self.args)
            utils.save_samples(
                gt.cpu(), gt.cpu(),
                self.save_path + f'results_samplings/train_samples_ep_{ep}.pdf',
                self.args
            )

    def compute_fast_fid(self, num_samples):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(self.device)
        # samples réels
        data_v = next(iter(self.full_train_set))
        gt = data_v[0].to(self.device)[:num_samples]
        gt = gt.permute(0, 2, 3, 1).cpu().numpy()
        if gt.shape[-1] == 1:
            gt = np.concatenate([gt, gt, gt], axis=-1)
        gt = np.transpose(gt, (0, 3, 1, 2))
        m1, s1 = fs.calculate_activation_statistics(gt, model, 50, 2048, self.device)

        # samples générés
        samples = []
        n_iter = 50
        for _ in range(n_iter):
            gen_batch = self.apply_flow_matching(num_samples // n_iter).cpu()
            samples.append(gen_batch)
        gen = torch.cat(samples, dim=0)
        gen = torch.clip(gen.permute(0, 2, 3, 1), 0, 1).numpy()
        if gen.shape[-1] == 1:
            gen = np.concatenate([gen, gen, gen], axis=-1)
        gen = np.transpose(gen, (0, 3, 1, 2))
        m2, s2 = fs.calculate_activation_statistics(gen, model, 50, 2048, self.device)

        return fs.calculate_frechet_distance(m1, s1, m2, s2)

    def train(self, data_loaders):
        self.save_path   = os.path.join(self.args.root,   f"results/{self.args.dataset}/ot/")
        self.model_path  = os.path.join(self.args.root,   f"model/{self.args.dataset}/ot/")
        os.makedirs(self.save_path,  exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        train_loader = data_loaders['train']

        # loader complet pour le FID
        # loader complet pour le FID (on précise bien dim_image !)
        full_data = DataLoaders(
            dataset_name    = self.args.dataset,
            data_root       = self.args.data_root,
            batch_size_train= 2048,
            batch_size_test = 2048,
            dim_image       = self.args.dim_image,
            num_workers     = self.args.num_workers
        ).load_data()
        self.full_train_set = full_data['train']

        # fichier info modèle
        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write("PARAMETERS\n")
            file.write(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n")
            file.write(f"Number of epochs: {self.args.num_epoch}\n")
            file.write(f"Batch size: {self.args.batch_size_train}\n")
            file.write(f"Learning rate: {self.lr}\n")

        # optimisateur
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # AMP gradient scaler
        self.scaler = amp.GradScaler()

        # lancement de l’entraînement avec tqdm
        self.train_FM_model(train_loader, opt, num_epoch=self.args.num_epoch)

        # sauvegarde finale
        torch.save(self.model.state_dict(), self.model_path + 'model_final.pt')


class cnf(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
            return self.model(x, t.repeat(x.shape[0]))
