import torch
import torch.distributed as dist


class Parameters:
    def __init__(self, parameters, rank, master_rank=0, group=None):
        """
        Parameters
        ----------
        parameters : Mapping[str, torch.nn.Parameter]
            parameters to sync
        rank : int
            process rank
        group : torch.distributed.ProcessGroup
            process group
        """
        self.parameters = dict(parameters)
        self.rank = int(rank)
        self.master_rank = int(master_rank)
        self.group = group
        self.world_size = dist.get_world_size(group)

    @torch.no_grad()
    def sync_values(self):
        if self.rank == self.master_rank:
            gather_list = [None] * self.world_size
        else:
            gather_list = None

        obj = {k: v.data for k, v in self.parameters.items()}
        dist.gather_object(obj, gather_list, dst=self.master_rank, group=self.group)

        if self.rank == self.master_rank:
            params = dict()
            for k, v in self.parameters.items():
                p = [_[k].to(v.device) for _ in gather_list]
                p = torch.stack(p, dim=0).mean(dim=0)
                params[k] = p
            broadcast_list = [params]
        else:
            broadcast_list = [None]

        dist.broadcast_object_list(broadcast_list, src=self.master_rank, group=self.group)

        for k, v in broadcast_list[0].items():
            self.parameters[k].data.copy_(v)

    @torch.no_grad()
    def sync_grads(self):
        if self.rank == self.master_rank:
            gather_list = [None] * self.world_size
        else:
            gather_list = None

        obj = {k: v.grad for k, v in self.parameters.items()}
        dist.gather_object(obj, gather_list, dst=self.master_rank, group=self.group)

        if self.rank == self.master_rank:
            grads = dict()
            for k, v in self.parameters.items():
                g = (_[k] for _ in gather_list)
                g = [_.to(v.device) for _ in g if _ is not None]
                if g:
                    grads[k] = torch.stack(g, dim=0).mean(dim=0)
            broadcast_list = [grads]
        else:
            broadcast_list = [None]

        dist.broadcast_object_list(broadcast_list, src=self.master_rank, group=self.group)

        for k, v in broadcast_list[0].items():
            p = self.parameters[k]
            p.grad = v.to(p.device)
