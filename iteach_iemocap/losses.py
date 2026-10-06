"""Masked IEMOCAP classification and Teacher-to-Student hidden losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss(reduction="sum")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                utterance_mask: torch.Tensor) -> torch.Tensor:
        mask = utterance_mask.view(-1, 1)
        target = target.view(-1, 1)
        log_prob = F.log_softmax(pred, 1)
        return self.loss(log_prob * mask, (target * mask).squeeze(-1).long()) / mask.sum()


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        mask = mask.view(-1, 1)
        return self.loss(pred * mask, target * mask) / mask.sum()


class MaskedHiddenDistillationLoss(nn.Module):
    """MSE over valid missing-modality utterances; Teacher targets are detached."""

    def __init__(self):
        super().__init__()
        self.mse = MaskedMSELoss()
        self.weights = (0.5, 0.1, 0.05)  # nearest classifier layer first

    def forward(self, student_hidden: list[torch.Tensor],
                teacher_hidden: list[torch.Tensor],
                selected_mask: torch.Tensor) -> torch.Tensor:
        if len(student_hidden) < 3 or len(teacher_hidden) < 3:
            raise ValueError("distillation requires the last three layers")
        selected = selected_mask.to(dtype=torch.bool)
        if selected.ndim != 2:
            raise ValueError("selected_mask must be [batch,time]")
        if not bool(selected.any()):
            return sum(x.sum() * 0.0 for x in student_hidden[-3:])
        total = student_hidden[-1].sum() * 0.0
        for weight, student, teacher in zip(
                self.weights, reversed(student_hidden[-3:]), reversed(teacher_hidden[-3:])):
            if student.shape != teacher.shape or student.shape[:2] != (selected.shape[1], selected.shape[0]):
                raise ValueError("hidden tensors must be matching [time,batch,hidden]")
            student_batch = student.transpose(0, 1).contiguous()
            teacher_batch = teacher.detach().transpose(0, 1).contiguous()
            expanded = selected.to(device=student.device).unsqueeze(-1).expand_as(student_batch)
            expanded = expanded.to(dtype=student.dtype).contiguous()
            total = total + weight * self.mse(
                student_batch.reshape(-1), teacher_batch.reshape(-1), expanded.reshape(-1))
        return total
