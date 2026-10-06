"""One epoch of the fixed IEMOCAP4 Teacher/Student training protocol."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .masking import generate_inputs, random_mask, continuous_random_mask, select_training_mask_rate
from .losses import MaskedCELoss, MaskedHiddenDistillationLoss


def make_masked_batch(data, batch_rate: float, random_training: bool = False):
    """Apply independent host/guest masks over padded T*B cells, in source order."""
    audio_host, text_host, visual_host = data[0], data[1], data[2]
    audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
    qmask, umask, labels = data[6], data[7], data[8]
    seqlen, batch = audio_host.size(0), audio_host.size(1)

    if random_training:
        matrix = continuous_random_mask(3, seqlen * batch, batch_rate)
    else:
        matrix = random_mask(3, seqlen * batch, batch_rate)
    audio_host_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
    text_host_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
    visual_host_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))

    if random_training:
        matrix = continuous_random_mask(3, seqlen * batch, batch_rate)
    else:
        matrix = random_mask(3, seqlen * batch, batch_rate)
    audio_guest_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
    text_guest_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
    visual_guest_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))
    assert batch_rate <= 0.700001, "at least one modality must remain"

    masked_audio_host = audio_host * audio_host_mask
    masked_audio_guest = audio_guest * audio_guest_mask
    masked_text_host = text_host * text_host_mask
    masked_text_guest = text_guest * text_guest_mask
    masked_visual_host = visual_host * visual_host_mask
    masked_visual_guest = visual_guest * visual_guest_mask

    complete = generate_inputs(audio_host, text_host, visual_host,
                                audio_guest, text_guest, visual_guest, qmask)[0]
    incomplete = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host,
                                 masked_audio_guest, masked_text_guest,
                                 masked_visual_guest, qmask)[0]
    input_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask,
                                 audio_guest_mask, text_guest_mask,
                                 visual_guest_mask, qmask)[0]
    return complete, incomplete, input_mask, qmask, umask, labels


def run_epoch(model, dataloader, device: torch.device, cls_loss: MaskedCELoss,
              hidden_loss: MaskedHiddenDistillationLoss, *, train: bool,
              optimizer=None, strategy: str = "constant", fixed_rate: float = 0.7,
              epoch_index: int = 0):
    if train and optimizer is None:
        raise ValueError("training requires an optimizer")
    model.train() if train else model.eval()

    predictions, labels_all, valid_all = [], [], []
    losses, cls_losses, hidden_losses = [], [], []
    train_rates = []
    missing_count = valid_feature_count = 0
    video_ids = []

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        batch_rate = (select_training_mask_rate(strategy, epoch_index, fixed_rate)
                      if train else fixed_rate)
        if train:
            train_rates.append(float(batch_rate))
        complete, incomplete, input_mask, qmask, umask, labels = make_masked_batch(
            data, batch_rate, random_training=(train and strategy == "random"))
        video_ids.extend(data[-1])

        valid = umask.to(device=device)
        complete = complete.to(device=device)
        incomplete = incomplete.to(device=device)
        labels = labels.to(device=device)
        qmask = qmask.to(device=device)
        input_mask = input_mask.to(device=device)
        output = model(complete, incomplete, valid, data[0].size(-1),
                       data[1].size(-1), data[2].size(-1), train)

        selected_missing = input_mask.sum(dim=-1).transpose(0, 1) < 3
        selected_missing = torch.logical_and(selected_missing, valid.bool())
        teacher_flat = output["teacher_logits"].transpose(0, 1).contiguous().view(
            -1, output["teacher_logits"].size(2))
        student_flat = output["student_logits"].transpose(0, 1).contiguous().view(
            -1, output["student_logits"].size(2))
        labels_flat = labels.view(-1)
        classification = (cls_loss(teacher_flat, labels_flat, valid) +
                          cls_loss(student_flat, labels_flat, valid)) / 2
        distillation = hidden_loss(output["student_hidden"],
                                   output["teacher_hidden"], selected_missing)
        loss = classification + distillation

        flat_valid = valid.view(-1).detach().cpu().numpy()
        predictions.append(student_flat.detach().cpu().numpy())
        labels_all.append(labels_flat.detach().cpu().numpy())
        valid_all.append(flat_valid)
        valid_utterances = int(valid.bool().sum().item())
        losses.append(float(loss.detach().item()) * valid_utterances)
        cls_losses.append(float(classification.detach().item()) * valid_utterances)
        hidden_losses.append(float(distillation.detach().item()) * valid_utterances)
        valid_feature_count += valid_utterances * 3
        missing_count += int(((3 - input_mask.sum(dim=-1).transpose(0, 1)) * valid.bool()).sum().item())

        if train:
            loss.backward()
            optimizer.step()

    if not predictions:
        raise RuntimeError("empty data loader")
    pred_scores = np.concatenate(predictions)
    y_true = np.concatenate(labels_all)
    sample_mask = np.concatenate(valid_all)
    y_pred = np.argmax(pred_scores, axis=1)
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_mask)
    waf = f1_score(y_true, y_pred, sample_weight=sample_mask, average="weighted")
    n_valid = np.sum(sample_mask)
    metric = {
        "accuracy": float(accuracy),
        "waf": float(waf),
        "loss": [round(float(np.sum(losses) / n_valid), 4),
                 round(float(np.sum(cls_losses) / n_valid), 4),
                 round(float(np.sum(hidden_losses) / n_valid), 4)],
        "samples": int(n_valid),
        "actual_missing_feature_fraction": float(missing_count / valid_feature_count),
        "batch_mask_rates": train_rates,
    }
    return metric
