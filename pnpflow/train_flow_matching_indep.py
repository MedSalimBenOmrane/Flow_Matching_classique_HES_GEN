import os
import torch
import numpy as np
import skimage.io as io
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
from matplotlib import pyplot as plt
import torch.cuda.amp as amp
from pnpflow.models import InceptionV3
import pnpflow.fid_score as fs
from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import glob
import re
class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)
    def sample_grid(self, epoch):
        out_dir = os.path.join(self.save_path, 'grid_samples')
        os.makedirs(out_dir, exist_ok=True)
        samples = self.apply_flow_matching(16)
        samples = utils.postprocess(samples, self.args)   # en théorie [0,1] mais parfois un peu hors bornes
        # assembler la grille
        grid = vutils.make_grid(samples, nrow=4, normalize=False)
        npgrid = grid.permute(1,2,0).cpu().numpy()
        # CLAMP
        npgrid = np.clip(npgrid, 0.0, 1.0)
        # sauve en PNG
        fname = os.path.join(out_dir, f'samples_epoch_{epoch}.png')
        plt.imsave(fname, npgrid)

    def train_FM_model(self, train_loader, opt, num_epoch, start_ep=0):
        # Outer loop on epochs, on affiche juste le compte de l'époque
        for ep in range(start_ep, num_epoch):
            print(f"\n→ Epoch {ep+1}/{num_epoch}")

            # Barre de progression batch-par-batch avec ETA et affichage du loss
            batch_bar = tqdm(
                train_loader,
                desc="   Batches",
                unit="batch",
                leave=False
            )

            for iteration, (x1, labels) in enumerate(batch_bar):
                if x1.size(0) == 0:
                    continue
                x1 = x1.to(self.device)

                # Échantillonnage latent
                x0 = torch.randn(
                    x1.shape[0],
                    self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=True
                )
                t1 = torch.rand(x1.shape[0], 1, 1, 1, device=self.device)

                # compute coupling par coupling independent
                xt = t1 * x1 + (1 - t1) * x0

                # calcul du loss et update
                # ----- forward en mixed precision -----
                with amp.autocast():
                    pred = self.model(xt, t1.squeeze())
                    loss = torch.sum((pred - (x1 - x0))**2) / x1.shape[0]

                # ----- backward + step -----
                opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()

                # Save loss in txt file
                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(
                        f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

                # On affiche le loss dans la barre
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            # en fin d'époque : échantillonnage, sauvegardes, FID tous les 5 epochs
            # tous les 5 epochs : checkpoint + grille 16 samples (pas de FID)
            if (ep + 1) % 5 == 0:
                # 1) sauvegarde du checkpoint
                torch.save({
                    'epoch': ep+1,
                    'model':    self.model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scaler':   self.scaler.state_dict(),
                }, self.model_path + f'model_{ep+1}.pt')

                # 2) génération d'une grille 4×4
                self.sample_grid(ep+1)

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
                self.save_path +
                f'results_samplings/train_samples_ep_{ep}.pdf',
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
        m1, s1 = fs.calculate_activation_statistics(
            gt, model, 50, 2048, self.device)

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
        m2, s2 = fs.calculate_activation_statistics(
            gen, model, 50, 2048, self.device)

        return fs.calculate_frechet_distance(m1, s1, m2, s2)

    def train(self, data_loaders):
        # dossiers dédiés au couplage indépendant
        self.save_path = os.path.join(
            self.args.root, f"results/{self.args.dataset}/flow_matching_indep/")
        self.model_path = os.path.join(
            self.args.root, f"model/{self.args.dataset}/flow_matching_indep/")
        os.makedirs(self.save_path,  exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        train_loader = data_loaders['train']

        # loader complet pour le FID
        full_data = DataLoaders(
            dataset_name     = self.args.dataset,
            data_root        = self.args.data_root,
            batch_size_train = 2048,
            batch_size_test  = 2048,
            dim_image        = self.args.dim_image,
            num_workers      = self.args.num_workers
        ).load_data()
        self.full_train_set = full_data['train']

        # fichier info modèle
        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write("PARAMETERS\n")
            file.write(
                f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n")
            file.write(f"Number of epochs: {self.args.num_epoch}\n")
            file.write(f"Batch size: {self.args.batch_size_train}\n")
            file.write(f"Learning rate: {self.lr}\n")

        # optimisateur + AMP scaler
        opt    = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scaler = amp.GradScaler()

        # 1) Récupère tous les fichiers model_*.pt dans le dossier
        all_ckpts = glob.glob(os.path.join(self.model_path, 'model_*.pt'))

        # 2) Filtre les noms VALIDES de la forme model_<NUM>.pt et extrait l'entier
        numeric_ckpts = []
        for path in all_ckpts:
            m = re.search(r'model_(\d+)\.pt$', path)
            if m:
                epoch = int(m.group(1))
                numeric_ckpts.append((epoch, path))

        # 3) Choisit le checkpoint au plus grand epoch
        if numeric_ckpts:
            epoch, last = max(numeric_ckpts, key=lambda x: x[0])
            # 4) Vérifie qu'il existe bien et n'est pas vide
            if os.path.isfile(last) and os.path.getsize(last) > 0:
                print(f"→ Recharge checkpoint d'époque {epoch} : {last}")
                ck = torch.load(last, map_location=self.device)
                self.model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['optimizer'])
                self.scaler.load_state_dict(ck['scaler'])
                start_ep = ck['epoch']
                self.sample_grid(start_ep)
            else:
                print(f"⚠️ Le fichier {last} est manquant ou vide. Nouveau départ.")
                start_ep = 0
        else:
            print("→ Aucun checkpoint numérique trouvé, entraînement depuis le début.")
            start_ep = 0

        # lancement de l’entraînement avec tqdm
        # lancement de l’entraînement à partir de start_ep
        self.train_FM_model(
            train_loader,
            opt,
            num_epoch=self.args.num_epoch,
            start_ep=start_ep
        )

        # sauvegarde finale
        torch.save(self.model.state_dict(), self.model_path + 'model_final.pt')



class cnf(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
            return self.model(x, t.repeat(x.shape[0]))
