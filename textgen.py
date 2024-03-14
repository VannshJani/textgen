import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import data
from utils import fetch_text_file_from_github

st.title('Text Generation with MLPs and LSTM')

st.write('This is a simple text generation app using a simple Multi-Layer Perceptron (MLP) and a Long Short-Term Memory (LSTM) model. The models are trained on the text of Shakespeare. The models are trained to predict the next character in the text given a sequence of characters. The models are then used to generate new text by sampling from the predicted distribution of the next character.')

input_text = st.text_input('Enter a sentence to generate text from:')

num_chars = st.slider('Number of characters to generate:', 1, 200, 20)

st.sidebar.title('Model Details')
model_name = st.sidebar.selectbox('Select model:', ['MLP', 'LSTM'])

# data = open('https://github.com/VannshJani/textgen/blob/main/input.txt', 'r').read()
repo_url = 'https://github.com/VannshJani/textgen'  
file_path = 'https://github.com/VannshJani/textgen/blob/main/input.txt' 

data = fetch_text_file_from_github(repo_url, file_path)
st.write(data)
unique_chars = list(set(''.join(data)))
unique_chars.sort()
vocab_dict = {i:ch for i, ch in enumerate(unique_chars)}
vocab_dict_inv = {ch:i for i, ch in enumerate(unique_chars)}

class NextChar(nn.Module):
  def __init__(self, block_size=8, vocab_size=65, emb_dim=8, hidden_dims = [64, 64]):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_dims[0])
    self.lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.lin3 = nn.Linear(hidden_dims[1], vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = torch.sin(self.lin2(x))
    x = self.lin3(x)
    return x
    
# using LSTM
class LSTM(nn.Module):
    def __init__(self, block_size=8, vocab_size=65, emb_dim=8, hidden_dims = [1024, 1024]):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dims[0], num_layers=2, batch_first=True,bias = True)
        self.lin = nn.Linear(hidden_dims[0], vocab_size)
    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.lin(x)
        return x



if model_name == 'MLP':
    path = 'Downloads/model.pth'
    model = NextChar()
    model.load_state_dict(torch.load(path))
elif model_name == 'LSTM':
    path = 'lstm_model.pth'
    model = LSTM()
    model.load_state_dict(torch.load(path))

g = torch.Generator()
g.manual_seed(4000002)
def generate_name(model,sentence, itos=vocab_dict, stoi=vocab_dict_inv, block_size=8, max_len=10):
    original_sentence = sentence
    if len(sentence) < block_size:
        sentence = " " * (block_size - len(sentence)) + sentence
    using_for_predicction = sentence[-block_size:].lower()
    context = [stoi[word] for word in using_for_predicction]
    prediction = ""
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        prediction += ch
        context = context[1:] + [ix]

    return original_sentence + prediction

if st.button('Generate Text'):
    generated_text = generate_name(model, input_text, vocab_dict, vocab_dict_inv, block_size=8, max_len=num_chars)
    st.write('Generated text:')
    st.write(generated_text)
    
    # visualizing embeddings using t-SNE
    emb = model.emb.weight.data.cpu().numpy()
    emb = emb[:100]
    tsne = TSNE(n_components=2, random_state=0, perplexity=15, n_iter=3000)
    emb_tsne = tsne.fit_transform(emb)
    plt.figure(figsize=(10, 10))
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1])
    for i, txt in enumerate(list(vocab_dict.values())[:100]):
        plt.annotate(txt, (emb_tsne[i, 0], emb_tsne[i, 1]))
    plt.title('t-SNE visualization of embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    st.pyplot(plt)

