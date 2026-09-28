import torch
from rl.buffer import Rollout

def make(rewards,values,terminal,active,bootstrap):
    return Rollout(None,None,None,None,torch.tensor(values,dtype=torch.float32),
                   torch.tensor(rewards,dtype=torch.float32),torch.tensor(active,dtype=torch.bool),
                   torch.tensor(terminal,dtype=torch.bool),torch.tensor(bootstrap,dtype=torch.float32),[])

def test_terminal_no_bootstrap_and_padding_excluded():
    b = make([[1],[2],[0]],[[.5],[.7],[99]],[[False],[True],[True]],[[True],[True],[False]],[100])
    adv, ret = b.advantages(gamma=1,lam=1)
    torch.testing.assert_close(ret[:2,0],torch.tensor([3.,2.]))
    assert adv[-1,0] == 0

def test_timeout_bootstraps():
    b = make([[1],[2]],[[.5],[.7]],[[False],[False]],[[True],[True]],[3])
    _,ret = b.advantages(gamma=1,lam=1)
    torch.testing.assert_close(ret[:,0],torch.tensor([6.,5.]))
