import torch
import torch.nn as nn

def locloss(self, y_true, y_pred, weights, loss_type="mse"):
    if self.point_size == 4:
        if loss_type == "mse":
            mse = torch.mean(torch.square(y_pred - y_true) * weights, dim=1)
            return mse
        if loss_type == "l1":
            l1 = torch.mean(torch.abs(y_pred - y_true) * weights, dim=1)
            return l1
        if loss_type == "log":
            log_loss = torch.mean(torch.log(1 + torch.abs(y_pred - y_true)) * weights, dim=1)
            return log_loss

def lineloss(line):
        line_x = line[:, 0::2]
        line_y = line[:, 1::2]
        x_diff = line_x[:, 1:] - line_x[:, 0:-1]
        y_diff = line_y[:, 1:] - line_y[:, 0:-1]
        x_diff_start = x_diff[:, 1:]
        x_diff_end = x_diff[:, 0:-1]
        y_diff_start = y_diff[:, 1:]
        y_diff_end = y_diff[:, 0:-1]
        similarity = (x_diff_start * x_diff_end + y_diff_start * y_diff_end) / (
                torch.sqrt(torch.square(x_diff_start) + torch.square(y_diff_start)) * torch.sqrt(
                torch.square(x_diff_end) + torch.square(y_diff_end)) + 1e-10)
        slop_loss = torch.mean(1 - similarity, dim=1)
        x_diff_loss = torch.mean(torch.square(x_diff[:, 1:] - x_diff[:, 0:-1]), dim=1)
        y_diff_loss = torch.mean(torch.square(y_diff[:, 1:] - y_diff[:, 0:-1]), dim=1)
        return slop_loss, x_diff_loss + y_diff_loss
