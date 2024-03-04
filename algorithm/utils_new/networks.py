import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)


# self.state_processor = StateProcessor(state_dim=(N, M, d), hidden_dim=128) # output = batch x M x N

class ValueNetwork(nn.Module): 
    """
    Value network: f(state, action) -> goodness
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        state_dim = N x M x d
            where 
                N is the number of client
                M is the number of classes + 2
                d is the dimension of the representation (dim of output of the feature extractor)
        
        action_dim = K
            where 
                K is the number of participants per communication round
        """
        super(ValueNetwork, self).__init__()
        N, M, d = state_dim   # 10, 10, 128
        
        # Input: batch x N x M x d
        self.linear_s1 = nn.Linear(d, hidden_dim)
        # Output: batch x N x M x hidden

        self.linear_s2 = nn.Linear(hidden_dim, 1)
        #Output: batch x N x M x 1

        self.linear1 = nn.Linear(N, hidden_dim)
        # Output: batch x M x hidden
        
        self.linear2 = nn.Linear(hidden_dim, 1)
        # Output: batch x M x 1 ---squeeze(-1)---> batch x M 
        
        self.linear3 = nn.Linear(M + 2 + action_dim, hidden_dim)
        # Output: batch x hidden
        
        self.linear4 = nn.Linear(hidden_dim, 1)
        # Output: batch x 1

        self.apply(init_weights)
        return
    
    def forward(self, raw_state, action):
        grad, loss, num_vol = raw_state
        x = F.relu(self.linear_s1(grad))
        x = F.relu(self.linear_s2(x)).squeeze(-1)

        x = torch.cat([x, loss.unsqueeze(-1), num_vol.unsqueeze(-1)], dim=2)    
        x = x.transpose(1,2)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x)).squeeze(-1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class PolicyNetwork(nn.Module):
    """
    Policy network: f(state) -> action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        N, M, d = state_dim   # 100, 10, 128
        
        # Input: batch x N x M x d
        self.linear_s1 = nn.Linear(d, hidden_dim)
        # Output: batch x N x M x hidden

        self.linear_s2 = nn.Linear(hidden_dim, 1)
        #Output: batch x N x M x 1

        # Input: batch x M x N
        self.linear1 = nn.Linear(M + 2, hidden_dim)
        # Output: batch x M x hidden
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # Output: batch x M x 1 ---squeeze(-1)---> batch x M 
        
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Output: batch x action_dim
        
        self.apply(init_weights)
        return
    
    def forward(self, raw_state, list_clients = None):
        grad, loss, num_vol = raw_state
        x = F.relu(self.linear_s1(grad))
        x = F.relu(self.linear_s2(x)).squeeze(-1)
        x = torch.cat([x, loss.unsqueeze(-1), num_vol.unsqueeze(-1)], dim=2)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x).squeeze(-1))
        # for i in range(state.shape[0]):
        #     for j in range(state.shape[1]):
        #         if torch.all(state[i, j, :, :] == 0):
        #             x[i, j] = -99999
        if (x.shape[0] == 1):
            print(x)
        x = F.softmax(x, dim=1)
        return x
    
# class PolicyNetwork(nn.Module):
#     """
#     Policy network: f(state) -> action
#     """
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super(PolicyNetwork, self).__init__()
#         N, M, d = state_dim   # 100, 10, 128
        
#         # # Input: batch x M x N
#         # self.linear1 = nn.Linear(N, hidden_dim)
#         # # Output: batch x M x hidden
        
#         # self.linear2 = nn.Linear(hidden_dim, 1)
#         # # Output: batch x M x 1 ---squeeze(-1)---> batch x M 

#         # Input: batch x N x M x d
#         self.linear_s1 = nn.Linear(d, hidden_dim)
#         # Output: batch x N x M x hidden

#         self.linear_s2 = nn.Linear(hidden_dim, 1)
#         #Output: batch x N x M x 1

#         self.linear = nn.Linear(M, 256)
#         # # Output: batch x action_dim
#         self.encoder = EncoderRNN(input_size=M, hidden_size=256)
#         self.decoder = AttnDecoderRNN(output_size=action_dim, hidden_size=256)
#         self.apply(init_weights)
#         return

#     def forward(self, state, list_clients = None):
#         x = F.relu(self.linear_s1(state))
#         x = F.relu(self.linear_s2(x)).squeeze(-1)    

#         output, hidden = self.encoder(x)

#         input_decoder = self.linear(x)
#         x = self.decoder(output, hidden, input_decoder)

#         # mask = torch.ones(state.shape[0], state.shape[1]).cuda()
#         for i in range(state.shape[0]):
#             for j in range(state.shape[1]):
#                 if torch.all(state[i, j, :, :] == 0):
#                     x[i, j] = -99999
#                     # print(i, j)

#         # if list_clients != None:
#         #     mask = torch.ones_like(x, dtype=torch.bool)
#         #     mask[:, list_clients] = False
#         #     x[mask] = -9999999
#         x = F.softmax(x, dim=1)
#         return x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # Input: batch x N x M
        output, hidden = self.gru(input)
        # Output: batch x N x hidden
        
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, encoder_hidden, input_vector=None):
        batch_size = encoder_outputs.size(0)
        decoder_hidden = encoder_hidden
        decoder_input = torch.empty(batch_size, 1, self.hidden_size).fill_(0.01).cuda()
        decoder_outputs = []
        for i in range(self.output_size):
            # decoder_input = input_vector[: , i:i+1 , :]

            decoder_output, decoder_hidden, decoder_input, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            # attentions.append(attn_weights)

            # if target_tensor is not None:
            #     # Teacher forcing: Feed the target as the next input
            #     decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            # else:
            #     # Without teacher forcing: use its own predictions as the next input
            #     _, topi = decoder_output.topk(1)
            #     decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1).squeeze(-1)

        return decoder_outputs


    def forward_step(self, input, hidden, encoder_outputs):
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((input, context), dim=2)

        x, hidden = self.gru(input_gru, hidden)
        output2 = self.out(x)

        return output2, hidden, x, attn_weights

    
class StateProcessor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        """
        For example: state_dim = (100, 10, 128)
        """
        super().__init__()
        _, _, d = state_dim   # 100, 10, 128
        self.linear1 = nn.Linear(d, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
        return
    
    def forward(self, state):
        """
        The state is expected to have the dimension of (batch x N x M x d)
        """
        out = self.linear1(state)               # batch x N x M x hidden_dim
        out = F.relu(out)
        
        out = self.linear2(out).squeeze(-1)     # batch x N x M 
        out = F.relu(out)
        
        return out.transpose(1,2)               # batch x M x N