import torch.nn as nn

class MultiTaskMLP(nn.Module):
    def __init__(self, dim, task):
        super(MultiTaskMLP, self).__init__()
        self.task = task
        if task == 'multitask':
            self.head_dysarthria = nn.Linear(dim, 2)
            self.head_severity = nn.Linear(dim, 4)
        elif task == 'dysarthria':
            self.head_dysarthria = nn.Linear(dim, 2)
        else:
            self.head_severity = nn.Linear(dim, 4)
       
    def forward(self, x):
        # 如果输入是3维 [batch, seq_len, dim]，取平均
        if x.ndim == 3:
            x = x.mean(dim=1)
        
        if self.task == 'multitask':
            out1 = self.head_severity(x)
            out2 = self.head_dysarthria(x)
            return out1, out2
        elif self.task == 'dysarthria':
            return self.head_dysarthria(x)
        else:
            return self.head_severity(x)