import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# Character-level name generation experiment.
# The goal is to compare different recurrent architectures
# in their ability to model phonetic patterns in Indian names.

# DATASET 
# Dataset class builds a character vocabulary and provides
# simple encoding/decoding utilities. Start and end tokens (^,$)
# help the model learn name boundaries.

class NameDataset:

    def __init__(self, path):

        names = open(path).read().lower().splitlines()
        # Lowercasing ensures consistent character statistics across the corpus.

        self.names = ["^" + n + "$" for n in names]
        # Explicit start/end tokens make sequence generation more stable.

        chars = sorted(list(set("".join(self.names))))

        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}

        self.vocab_size = len(chars)

    def encode(self,name):
        return torch.tensor([self.stoi[c] for c in name])

    def decode(self,idxs):
        return "".join([self.itos[i] for i in idxs])


# MODELS

# Three recurrent variants are implemented:
# Vanilla RNN for baseline behaviour,
# BLSTM to incorporate bidirectional context during training,
# and a lightweight attention-based RNN to reweight hidden states.

class VanillaRNN(nn.Module):

    def __init__(self,vocab,hidden):
        super().__init__()

        self.emb = nn.Embedding(vocab,hidden)
        # Character embedding layer maps discrete symbols into dense vectors.
        self.rnn = nn.RNN(hidden,hidden,batch_first=True)
        # Simple recurrent transition used as baseline sequence model.
        self.fc = nn.Linear(hidden,vocab)

    def forward(self,x,h=None):

        x = self.emb(x)
        out,h = self.rnn(x,h)
        out = self.fc(out)

        return out,h


class BLSTM(nn.Module):

    # Bidirectional recurrence provides both past and future context
    # during training, though generation still proceeds left-to-right.

    def __init__(self,vocab,hidden):
        super().__init__()

        self.emb = nn.Embedding(vocab,hidden)

        self.lstm = nn.LSTM(
            hidden,
            hidden,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden*2,vocab)

    def forward(self,x,h=None):

        x = self.emb(x)
        out,h = self.lstm(x,h)
        out = self.fc(out)

        return out,h


class AttentionRNN(nn.Module):

    # Linear attention layer produces timestep-wise importance weights.

    def __init__(self,vocab,hidden):
        super().__init__()

        self.emb = nn.Embedding(vocab,hidden)
        self.rnn = nn.RNN(hidden,hidden,batch_first=True)

        self.attn = nn.Linear(hidden,hidden)
       

        self.fc = nn.Linear(hidden,vocab)

    def init_hidden(self, batch):
       return torch.zeros(1, batch, self.emb.embedding_dim)

    def forward(self,x,h=None):

        x = self.emb(x)
        out,h = self.rnn(x,h)

       
        attn_weights = torch.softmax(self.attn(out), dim=1)
        
        # Hidden states are scaled locally instead of forming a global context vector.
        out = attn_weights * out

        out = self.fc(out)

        return out,h


#TRAIN
# Training is done using teacher forcing at the character level.
# Gradient clipping is used to avoid exploding gradients.

def train(model,dataset,epochs=20,lr=0.003):

    opt = torch.optim.Adam(model.parameters(),lr=lr)

    loss_fn = nn.CrossEntropyLoss()
   
    # Sequence loss measures next-character prediction accuracy.
    losses=[]

    for ep in range(epochs):

        total=0

        for name in dataset.names:

            x = dataset.encode(name[:-1]).unsqueeze(0)
            y = dataset.encode(name[1:]).unsqueeze(0)

            opt.zero_grad()

           
            if hasattr(model, "init_hidden"):
               h = model.init_hidden(1)
               out,_ = model(x, h)
            else:
               out,_ = model(x)

            loss = loss_fn(out.view(-1,dataset.vocab_size),y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()

            total+=loss.item()

        print(ep,total)

        losses.append(total)

    return losses


#  SAMPLING 
# Name generation is performed autoregressively by sampling
# from the softmax distribution at each timestep.

# Temperature controls randomness vs determinism in generated names.
def sample(model,dataset,maxlen=20,temp=1.0):

    model.eval()

    x = torch.tensor([[dataset.stoi["^"]]])

    h=None
    name=""

    for _ in range(maxlen):

        out,h = model(x,h)

        logits = out[0,-1] / temp

        prob = torch.softmax(logits,dim=0)

        idx = torch.multinomial(prob,1).item()

        ch = dataset.itos[idx]

        if ch=="$":
            break

        name += ch

        x = torch.tensor([[idx]])

    return name


def generate_names(model,dataset,n=100):

    return [sample(model,dataset) for _ in range(n)]


#  EVALUATION 

# Novelty checks memorization behaviour,
# while diversity reflects variability in generated samples.
def novelty(gen,train):

    new = [g for g in gen if g not in train]

    return len(new)/len(gen)

def diversity(gen):

    return len(set(gen))/len(gen)


#  MAIN 

# Models are trained sequentially for controlled comparison
# under identical dataset and optimisation settings.
data = NameDataset("TrainingNames.txt")

models = {
    "RNN":VanillaRNN(data.vocab_size,64),
    "BLSTM":BLSTM(data.vocab_size,32),
    "ATTN":AttentionRNN(data.vocab_size,128)
}

results={}

# Loss curves help visualise convergence differences
# between recurrent architectures.
for name,model in models.items():

    print("Training",name)

    losses = train(model,data)

    plt.plot(losses,label=name)

    gen = generate_names(model,data,200)

    nov = novelty(gen,data.names)
    div = diversity(gen)

    results[name]=(nov,div)

    print("Samples:",gen[:10])
    print("Novelty",nov,"Diversity",div)

plt.legend()
plt.title("Training Loss Comparison")
plt.show()

print(results)

# Empirically, the vanilla RNN produced the most realistic short names,
# while BLSTM sometimes over-fit character patterns and attention models
# showed higher variability in generated sequences.