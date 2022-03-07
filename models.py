from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models import ModelCatalog
import torch
from torch import nn

class RNNModel(RecurrentNetwork,torch.nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        RecurrentNetwork.__init__(self,obs_space, act_space, num_outputs, *args, **kwargs)
        torch.nn.Module.__init__(self)

        self.hidden_size = 256
        self.fc_head = nn.Sequential(
            nn.Linear(obs_space.shape[0], self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )

        self.rnn = nn.RNN(self.hidden_size,self.hidden_size,batch_first=True)

        self.fc_head2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )

        self.policy_fn = nn.Linear(self.hidden_size, num_outputs)
        self.value_fn = nn.Linear(self.hidden_size, 1)

        #torch.nn.init.xavier_uniform_(self.policy_fn.weight)
        #torch.nn.init.xavier_uniform_(self.value_fn.weight)

        #torch.nn.init.constant_(self.policy_fn.bias,0.0)
        #torch.nn.init.constant_(self.value_fn.bias,0.0)


    def forward_rnn(self,input_dict, state, seq_lens, *args,**kwargs):
        x = input_dict.float()

        #logging.warn(str(x.shape)+'  '+str(state[0].shape))

        x = self.fc_head(x)
        state = torch.permute(state[0],(1,0,2))
        x,new_state = self.rnn(x,state)
        new_state = [torch.permute(new_state,(1,0,2))]
        x = x#[:,-1]

        x = self.fc_head2(x)

        self.value_out = self.value_fn(x)

        x = self.policy_fn(x)
        return x,new_state

    def get_initial_state(self):
        return [torch.zeros((1,self.hidden_size),dtype=torch.float32)]

    def value_function(self):
        return torch.reshape(self.value_out,[-1])

#Registers the env for RLLIB
ModelCatalog.register_custom_model("rnn_model", RNNModel)