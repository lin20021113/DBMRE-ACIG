import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
import torch.nn.functional as F

logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class CosineTripletMarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cos_ap = F.cosine_similarity(anchor, positive, dim=1)
        cos_an = F.cosine_similarity(anchor, negative, dim=1)
        loss = torch.relu(self.margin - cos_ap + cos_an).mean()
        return loss

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.ctr_loss = CosineTripletMarginLoss(margin=getattr(args, 'margin', 0.05))

    def do_train(self, model, dataloader, return_epoch_results=False):

        params = list(model[0].parameters()) + \
                 list(model[1].parameters()) + \
                 list(model[2].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net

        max_epochs = 50

        while epochs < max_epochs:
            epochs += 1
            warmup_epochs = 10

            dm_weight = 0.05 * (epochs / warmup_epochs) if epochs <= warmup_epochs else 0.05
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    output = model[0](text, audio, vision, is_distill=True)

                    # logits for homo GD
                    logits_homo.append(output['logits_l_homo'])
                    logits_homo.append(output['logits_v_homo'])
                    logits_homo.append(output['logits_a_homo'])

                    # reprs for homo GD
                    reprs_homo.append(output['repr_l_homo'])
                    reprs_homo.append(output['repr_v_homo'])
                    reprs_homo.append(output['repr_a_homo'])

                    # logits for hetero GD
                    logits_hetero.append(output['logits_l_hetero'])
                    logits_hetero.append(output['logits_v_hetero'])
                    logits_hetero.append(output['logits_a_hetero'])

                    # reprs for hetero GD
                    reprs_hetero.append(output['repr_l_hetero'])
                    reprs_hetero.append(output['repr_v_hetero'])
                    reprs_hetero.append(output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)

                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)

                    # edges for homo distill
                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # reconstruction loss
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    batch_size = output['s_l'].shape[0]
                    seq_len = output['s_l'].shape[1]

                    s_l_flat = output['s_l'].reshape(-1, output['s_l'].shape[-1])
                    c_l_flat = output['c_l'].reshape(-1, output['c_l'].shape[-1])
                    s_v_flat = output['s_v'].reshape(-1, output['s_v'].shape[-1])
                    c_v_flat = output['c_v'].reshape(-1, output['c_v'].shape[-1])
                    s_a_flat = output['s_a'].reshape(-1, output['s_a'].shape[-1])
                    c_a_flat = output['c_a'].reshape(-1, output['c_a'].shape[-1])

                    target = torch.full((s_l_flat.shape[0],), -1, device=self.args.device)


                    cosine_similarity_s_c_l = self.cosine(s_l_flat, c_l_flat, target)
                    cosine_similarity_s_c_v = self.cosine(s_v_flat, c_v_flat, target)
                    cosine_similarity_s_c_a = self.cosine(s_a_flat, c_a_flat, target)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    rec_homo_l = output['rec_homo_l']
                    rec_homo_v = output['rec_homo_v']
                    rec_homo_a = output['rec_homo_a']
                    rec_hetero_l = output['rec_hetero_l']
                    rec_hetero_v = output['rec_hetero_v']
                    rec_hetero_a = output['rec_hetero_a']

                    def compute_dm_contrastive_loss(rec_list, labels, loss_fn):
                        N = labels.size(0)
                        feats = torch.cat(rec_list, dim=0)          # (3N, D)
                        labels_flat = labels.squeeze().repeat(3)    # (3N,)
                        modality_ids = torch.cat([torch.full((N,), i, device=feats.device)
                                                  for i in range(3)], dim=0)

                        sim = F.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=2)

                        pos_mask = (labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)) & \
                                   (modality_ids.unsqueeze(0) != modality_ids.unsqueeze(1))
                        neg_mask = (labels_flat.unsqueeze(0) != labels_flat.unsqueeze(1)) & \
                                   (modality_ids.unsqueeze(0) == modality_ids.unsqueeze(1))

                        hardest_pos_sim, hardest_pos_idx = sim.masked_fill(~pos_mask, -1e9).max(dim=1)  # (3N,)

                        semi_hard_mask = neg_mask & \
                                         (sim > hardest_pos_sim.unsqueeze(1)) & \
                                         (sim < hardest_pos_sim.unsqueeze(1) + loss_fn.margin)

                        any_semi_hard = semi_hard_mask.sum(dim=1) > 0
                        chosen_neg_mask = torch.where(any_semi_hard.unsqueeze(1), semi_hard_mask, neg_mask)

                        neg_probs = chosen_neg_mask.float() + 1e-9
                        chosen_neg_idx = torch.multinomial(neg_probs, 1).squeeze()  # (3N,)

                        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
                        if valid.sum() == 0:
                            return torch.tensor(0.0, device=feats.device)

                        anchor_feats = feats[valid]
                        pos_feats = feats[hardest_pos_idx[valid]]
                        neg_feats = feats[chosen_neg_idx[valid]]

                        loss = loss_fn(anchor_feats, pos_feats, neg_feats)
                        return loss

                    loss_ctr_ho = compute_dm_contrastive_loss(
                        [rec_homo_l, rec_homo_v, rec_homo_a], labels, self.ctr_loss)
                    loss_ctr_he = compute_dm_contrastive_loss(
                        [rec_hetero_l, rec_hetero_v, rec_hetero_a], labels, self.ctr_loss)
                    
                    rec_homo_l_enh = output['rec_homo_l_enh']
                    rec_homo_v_enh = output['rec_homo_v_enh']
                    rec_homo_a_enh = output['rec_homo_a_enh']
                    loss_consist = (F.mse_loss(rec_homo_l, rec_homo_l_enh) +
                                    F.mse_loss(rec_homo_v, rec_homo_v_enh) +
                                    F.mse_loss(rec_homo_a, rec_homo_a_enh)) / 3.0

                    loss_ctr_dm = dm_weight * (loss_ctr_ho + loss_ctr_he) + 0.01 * loss_consist

                    # homo GD loss
                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # hetero GD loss
                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                        model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    intra_reg = 1e-4 * (
                        output['s_intra_l'].norm(p=2) +
                        output['s_intra_v'].norm(p=2) +
                        output['s_intra_a'].norm(p=2)
                    )
                    inter_reg = 1e-4 * (
                        output['c_inter_l'].norm(p=2) +
                        output['c_inter_v'].norm(p=2) +
                        output['c_inter_a'].norm(p=2)
                    )
                    module_reg = intra_reg + inter_reg

                    combined_loss = loss_task + \
                                    graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1 + \
                                    loss_ctr_dm + module_reg

                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters()) + \
                                 list(model[1].parameters()) + \
                                 list(model[2].parameters())

                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs

                model_save_path = './pt/dmd.pth'

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

        if return_epoch_results:
            return epoch_results
        else:
            return None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision, is_distill=True)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results