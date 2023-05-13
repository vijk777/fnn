import torch
import torch.distributed as dist


class ParameterGroup:
    def __init__(self, parameters, group=None):
        """
        Parameters
        ----------
        parameters : Mapping[str, torch.nn.Parameter]
            parameters to sync
        group : torch.distributed.ProcessGroup
            process group
        """
        assert dist.get_rank(group=group) >= 0

        self.parameters = dict(parameters)
        self.group = group
        self.ranks = dist.get_process_group_ranks(group=group)

    @torch.no_grad()
    def sync_params(self):
        obj_list = [None for _ in self.ranks]
        dist.all_gather_object(obj_list, self.parameters, group=self.group)

        for k, v in self.parameters.items():
            o = [_[k].to(v.device) for _ in obj_list]
            o = torch.stack(o, dim=0).mean(dim=0)
            v.copy_(o)

    @torch.no_grad()
    def sync_grads(self):
        obj = {k: v.grad for k, v in self.parameters.items()}
        obj_list = [None for _ in self.ranks]
        dist.all_gather_object(obj_list, obj, group=self.group)

        for k, v in self.parameters.items():
            o = (_[k] for _ in obj_list)
            o = [_.to(v.device) for _ in o if _ is not None]
            if o:
                v.grad = torch.stack(o, dim=0).mean(dim=0)
