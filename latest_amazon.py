# %%
import os
import re
import nltk
import torch
import gensim
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Hyperparameters
NUM_EPOCHS = 20  # Reduced epochs
LEARNING_RATE = 5e-4  # Increased learning rate
BATCH_SIZE = 32
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
STYLE_DIM = 32
CONTENT_DIM = 128
KL_WEIGHT = 0.05  # Fixed weight instead of annealing
MAX_SEQ_LENGTH = 50
GRADIENT_CLIP = 5.0
TEACHER_FORCING_RATIO = 0.5  # Reduced
LAMBDA_STYLE = 0.1
LAMBDA_CONTENT = 0.1
LAMBDA_ADV = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

# %%

 
# Path to the data in Google Drive
data_dir = './data/reduced_amazon_dataset'

def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'":]', '', text) 
    return text

def build_vocab(data_dir, min_freq=2):
    word_counter = Counter()
    special_tokens = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
    }

    for filename in ["sentiment.train.0", "sentiment.train.1"]:
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = preprocess_text(line)
                    words = text.strip().split()
                    word_counter.update(words)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            continue

    # Filter words based on min_freq
    filtered_words = {word: count for word, count in word_counter.items() if count >= min_freq}
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    vocab = dict(special_tokens)
    for idx, (word, _) in enumerate(sorted_words, start=len(special_tokens)):
        vocab[word] = idx

    return vocab

def build_noun_vocab(data_dir, vocab):
    noun_counter = Counter()
    for filename in ["sentiment.train.0", "sentiment.train.1"]:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = preprocess_text(line)
                tokens = text.strip().split()
                # Simple noun heuristic instead of NLTK
                nouns = [word for word in tokens if word.isalpha() and not word in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}]
                noun_counter.update(nouns)

    noun_vocab = {noun: idx for idx, (noun, _) in enumerate(noun_counter.items())}
    return noun_vocab
 

# %%
def extract_nouns(token_ids, id_to_word):
    tokens = [id_to_word.get(idx.item(), '<UNK>') for idx in token_ids]
    pos_tags = nltk.pos_tag(tokens)
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    return nouns

def create_bow_vector(nouns, noun_vocab):
    bow_vector = torch.zeros(len(noun_vocab), dtype=torch.float32)
    for noun in nouns:
        idx = noun_vocab.get(noun, None)
        if idx is not None:
            bow_vector[idx] += 1.0
    if len(nouns) > 0:
        bow_vector /= len(nouns)
    return bow_vector

def initialize_embeddings(vocab, embedding_dim=EMBEDDING_DIM, glove_path=None):
    """Initialize embeddings with GloVe if available, else random."""
    embedding_matrix = np.random.uniform(
        -0.1, 0.1, (len(vocab), embedding_dim)
    ).astype(np.float32)

    if glove_path is not None:
        print("Loading pre-trained GloVe embeddings...")
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        found = 0
        for word, idx in vocab.items():
            if word in embeddings_index:
                embedding_matrix[idx] = embeddings_index[word]
                found += 1
        print(f"Found {found} out of {len(vocab)} words in GloVe.")
    else:
        print("No GloVe path provided. Using random embeddings.")

    # Set padding token embedding to zeros
    embedding_matrix[0] = 0
    return torch.FloatTensor(embedding_matrix)

# %%
class TextDataset(Dataset):
    def __init__(self, data_dir, vocab, noun_vocab, max_length=MAX_SEQ_LENGTH, split='train'):
        self.data = []
        self.vocab = vocab
        self.noun_vocab = noun_vocab
        self.max_length = max_length

        files = ["sentiment.train.0", "sentiment.train.1"] if split == 'train' else ["sentiment.test.0", "sentiment.test.1"]

        # Common words that usually aren't nouns
        self.non_nouns = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}

        for filename in files:
            file_path = os.path.join(data_dir, filename)
            label = 1 if filename.endswith('.1') else 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Preprocess and tokenize
                        tokens = preprocess_text(line).strip().split()
                        tokens = ['<BOS>'] + tokens + ['<EOS>']  # Add BOS and EOS tokens
                        tokens = tokens[:max_length]  # Truncate if too long
                        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

                        if len(token_ids) == 0:
                            continue  # Skip empty samples

                        # Create BoW vector without NLTK
                        bow_vector = torch.zeros(len(self.noun_vocab), dtype=torch.float32)
                        potential_nouns = [word for word in tokens if word.isalpha() and word not in self.non_nouns]

                        for noun in potential_nouns:
                            idx = self.noun_vocab.get(noun, None)
                            if idx is not None:
                                bow_vector[idx] += 1.0

                        if potential_nouns:  # Normalize if we found any nouns
                            bow_vector /= len(potential_nouns)

                        self.data.append((token_ids, label, bow_vector))
            except FileNotFoundError:
                print(f"Warning: {file_path} not found")
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids, label, bow_vector = self.data[idx]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            bow_vector
        )

def collate_fn(batch):
    inputs, labels, bow_vectors = zip(*batch)

    # Get sequence lengths
    lengths = torch.tensor([len(seq) for seq in inputs])
    max_len = max(lengths)

    # Pad sequences
    padded_inputs = []
    attention_mask = []

    for seq in inputs:
        padding_length = max_len - len(seq)
        padded_seq = torch.cat([seq, torch.zeros(padding_length, dtype=torch.long)])
        mask = torch.cat([torch.ones(len(seq)), torch.zeros(padding_length)]).bool()

        padded_inputs.append(padded_seq)
        attention_mask.append(mask)

    return {
        'input_ids': torch.stack(padded_inputs),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.tensor(labels),
        'lengths': lengths,
        'bow_vectors': torch.stack(bow_vectors)
    }

# %%
class DisentangledVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, style_dim, content_dim, vocab, embedding_matrix=None):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<PAD>'])

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)

        # Encoder
        self.encoder_rnn = nn.GRU(
            embedding_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True
        )

        # Projections for style and content
        encoder_dim = hidden_dim * 2
        self.style_mu = nn.Linear(encoder_dim, style_dim)
        self.style_logvar = nn.Linear(encoder_dim, style_dim)
        self.content_mu = nn.Linear(encoder_dim, content_dim)
        self.content_logvar = nn.Linear(encoder_dim, content_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(style_dim + content_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(
            embedding_dim + style_dim + content_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder_rnn(packed)
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(hidden.size(0), -1)

        style_mu = self.style_mu(hidden)
        style_logvar = self.style_logvar(hidden)
        content_mu = self.content_mu(hidden)
        content_logvar = self.content_logvar(hidden)

        return style_mu, style_logvar, content_mu, content_logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, style, content, x=None, lengths=None, temperature=0.8):  # Add temperature parameter with default 0.8
        batch_size = style.size(0)
        z = torch.cat([style, content], dim=1)
        hidden = self.latent_to_hidden(z).unsqueeze(0)

        if self.training and x is not None:
            embedded = self.embedding(x)
            z_expanded = z.unsqueeze(1).expand(-1, embedded.size(1), -1)
            decoder_input = torch.cat([embedded, z_expanded], dim=-1)
            output, _ = self.decoder_rnn(decoder_input, hidden)
            logits = self.output_fc(output) / temperature  # Apply temperature
            return F.log_softmax(logits, dim=-1)
        else:
            current_token = torch.full((batch_size, 1),
                                self.vocab['<BOS>'],
                                dtype=torch.long,
                                device=style.device)
            outputs = []

            for _ in range(MAX_SEQ_LENGTH):
                embedded = self.embedding(current_token)
                z_expanded = z.unsqueeze(1)
                decoder_input = torch.cat([embedded, z_expanded], dim=-1)
                output, hidden = self.decoder_rnn(decoder_input, hidden)
                logits = self.output_fc(output) / temperature  # Apply temperature
                probs = F.softmax(logits[:, -1], dim=-1)
                current_token = torch.multinomial(probs, 1)
                outputs.append(current_token)

                if (current_token == self.vocab['<EOS>']).all():
                    break

            return torch.cat(outputs, dim=1)

    def forward(self, x, lengths):
        style_mu, style_logvar, content_mu, content_logvar = self.encode(x, lengths)
        style = self.reparameterize(style_mu, style_logvar)
        content = self.reparameterize(content_mu, content_logvar)
        recon_x = self.decode(style, content, x, lengths)
        return recon_x, style_mu, style_logvar, content_mu, content_logvar, style, content

# %%
class StyleClassifier(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, style_dim * 2),
            nn.BatchNorm1d(style_dim * 2),  # Consider BatchNorm instead of LayerNorm
            nn.ReLU(),  # ReLU might help gradient flow
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(style_dim * 2, style_dim),
            nn.BatchNorm1d(style_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(style_dim, 1)
        )

    def forward(self, style_embedding):
        return self.net(style_embedding)  # Remove sigmoid here, use BCE with logits

# %%
class ContentClassifier(nn.Module):
    def __init__(self, content_dim, noun_vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim, content_dim * 2),
            nn.LayerNorm(content_dim * 2),
            nn.Tanh(), 
            nn.Linear(content_dim * 2, content_dim),
            nn.LayerNorm(content_dim),
            nn.Tanh(), 
            nn.Linear(content_dim, noun_vocab_size)
        )

    def forward(self, content_embedding):
        return self.net(content_embedding)

# %%
class AdversarialClassifier(nn.Module):
    """Base class for adversarial classifiers with gradient reversal"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x, alpha=1.0):
        # Gradient reversal during backprop
        if self.training:
            x = x + alpha * x.detach() - x
        return self.net(x)

class AdversarialStyleClassifier(AdversarialClassifier):
    def __init__(self, content_dim):
        super().__init__(content_dim, 1)

    def forward(self, content_embedding, alpha=1.0):
        return torch.sigmoid(super().forward(content_embedding, alpha))

class AdversarialContentClassifier(AdversarialClassifier):
    def __init__(self, style_dim, noun_vocab_size):
        super().__init__(style_dim, noun_vocab_size)

    def forward(self, style_embedding, alpha=1.0):
        return torch.sigmoid(super().forward(style_embedding, alpha))

# %%
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def adversarial_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits.squeeze(), targets)

def style_classification_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

def content_classification_loss(logits, bow_vectors):
    return F.binary_cross_entropy_with_logits(logits, bow_vectors)

class KLAnnealer:
    """KL annealing scheduler"""
    def __init__(self, total_steps, start=0.0, stop=1.0):
        self.total_steps = total_steps
        self.start = start
        self.stop = stop
        self.current_step = 0

    def step(self):
        self.current_step = min(self.current_step + 1, self.total_steps)

    def get_weight(self):
        # Sigmoid schedule
        x = 10 * (self.current_step/self.total_steps - 0.5)
        weight = self.stop / (1 + np.exp(-x))
        return max(self.start, min(weight, self.stop))

class LanguageModelLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.criterion = nn.NLLLoss(ignore_index=pad_idx, reduction='none')

    def forward(self, logits, targets, mask=None):
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.float)
        mask_flat = mask.reshape(-1)

        loss = self.criterion(logits_flat, targets_flat)
        loss = loss * mask_flat

        return loss.sum() / (mask_flat.sum() + 1e-8)

def sample_sequence(logits, temperature=1.0):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)

# %%
def train_model(vae, style_classifier, content_classifier, adv_style_classifier, adv_content_classifier,
                data_loader, num_epochs, learning_rate):
    vae_optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=0.01)
    style_clf_optimizer = optim.RMSprop(list(style_classifier.parameters()), lr=learning_rate*0.1)
    content_clf_optimizer = optim.RMSprop(list(content_classifier.parameters()), lr=learning_rate*0.1) 
    adv_style_optimizer = optim.AdamW(adv_style_classifier.parameters(), lr=learning_rate*0.1)
    adv_content_optimizer = optim.AdamW(adv_content_classifier.parameters(), lr=learning_rate*0.1)

    recon_criterion = LanguageModelLoss(vocab['<PAD>'])

    for epoch in range(num_epochs):
        total_loss = defaultdict(float)
        batch_count = 0

        for batch_idx, batch in enumerate(data_loader, start=1):
            batch_count += 1

            # Move batch to device
            inputs = batch['input_ids'].to(DEVICE)
            masks = batch['attention_mask'].to(DEVICE)
            style_labels = batch['labels'].float().to(DEVICE)
            bow_vectors = batch['bow_vectors'].to(DEVICE)
            lengths = batch['lengths'].to(DEVICE)

            # Forward pass through VAE
            outputs = vae(inputs, lengths)
            recon_x, style_mu, style_logvar, content_mu, content_logvar, style_z, content_z = outputs

            # Shift targets for next-token prediction
            targets = inputs[:, 1:]  # Remove first token (BOS)
            decoder_mask = masks[:, 1:]  # Shift mask accordingly

            # Calculate reconstruction loss on shifted sequences
            recon_loss = recon_criterion(recon_x[:, :-1], targets, decoder_mask)

            # KL divergence losses
            kl_style = kl_divergence(style_mu, style_logvar)
            kl_content = kl_divergence(content_mu, content_logvar)
            kl_loss = (kl_style + kl_content) * KL_WEIGHT # KL annealing

            # Train discriminators
            adv_style_loss = adversarial_loss(adv_style_classifier(content_z.detach()), style_labels)
            adv_content_loss = adversarial_loss(adv_content_classifier(style_z.detach()), bow_vectors)

            # Backward pass for discriminators
            adv_style_optimizer.zero_grad()
            adv_content_optimizer.zero_grad()
            adv_style_loss.backward()
            adv_content_loss.backward()
            adv_style_optimizer.step()
            adv_content_optimizer.step()

            # Train VAE and classifiers
            style_logits = style_classifier(style_z)
            content_logits = content_classifier(content_z)
            style_loss = style_classification_loss(style_logits, style_labels)
            content_loss = content_classification_loss(content_logits, bow_vectors)
            
            style_logits_cls = style_classifier(style_z.detach())
            content_logits_cls = content_classifier(content_z.detach())
            style_loss_cls = style_classification_loss(style_logits_cls, style_labels)
            content_loss_cls = content_classification_loss(content_logits_cls, bow_vectors)
            # Adversarial losses for generator
            adv_style_loss_g = -adversarial_loss(adv_style_classifier(content_z), style_labels)
            adv_content_loss_g = -adversarial_loss(adv_content_classifier(style_z), bow_vectors)

            # Total loss
            loss = (recon_loss +
                    kl_loss +
                    LAMBDA_STYLE * style_loss +
                    LAMBDA_CONTENT * content_loss +
                    LAMBDA_ADV * (adv_style_loss_g + adv_content_loss_g))

            # Backward pass
            vae_optimizer.zero_grad() 
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(vae.parameters(), GRADIENT_CLIP)

            # Optimizer step
            vae_optimizer.step() 
            
            style_clf_optimizer.zero_grad()
            style_loss_cls.backward()
            style_clf_optimizer.step()

            content_clf_optimizer.zero_grad()
            content_loss_cls.backward()
            content_clf_optimizer.step()

            # Update running loss
            total_loss['total'] += loss.item()
            total_loss['recon'] += recon_loss.item()
            total_loss['kl'] += kl_loss.item()
            total_loss['style'] += style_loss.item()
            total_loss['content'] += content_loss.item()
            total_loss['adv'] += (adv_style_loss_g.item() + adv_content_loss_g.item())

            # Print statistics every 100 batches
            if batch_idx % 100 == 0:
                avg_loss = {k: v / batch_count for k, v in total_loss.items()}
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}] - "
                    f"Total Loss: {avg_loss['total']:.4f}, Recon Loss: {avg_loss['recon']:.4f}, "
                    f"KL Loss: {avg_loss['kl']:.4f}, Style Loss: {avg_loss['style']:.4f}, "
                    f"Content Loss: {avg_loss['content']:.4f}, Adv Loss: {avg_loss['adv']:.4f}")

        # Log epoch metrics after each epoch
        avg_loss = {k: v / len(data_loader) for k, v in total_loss.items()}
        print(f"Epoch {epoch + 1} Completed. Average Losses:")
        for k, v in avg_loss.items():
            print(f"{k.capitalize()} Loss: {v:.4f}")

# %%
class CounterfactualGenerator:
    def __init__(self, style_classifier, lambda_cf=0.1):
        self.classifier = style_classifier
        self.lambda_cf = lambda_cf

    def generate(self, style_z, target_style, confidence=0.9, steps=100, lr=0.01):
        s_prime = style_z.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([s_prime], lr=lr)

        target = torch.full_like(target_style, confidence)

        for step in range(steps):
            optimizer.zero_grad()

            # Get classifier prediction
            logits = self.classifier(s_prime)
            probs = torch.sigmoid(logits)

            # Counterfactual loss
            cf_loss = (probs - target).pow(2).mean() + \
                     self.lambda_cf * torch.norm(s_prime - style_z, p=1, dim=1).mean()

            cf_loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f'Step {step}: loss = {cf_loss.item():.4f}, '
                      f'prob = {probs.mean().item():.4f}')

        return s_prime.detach()

    def interpolate(self, style_z, target_style, strengths=[0.2, 0.5, 0.8]):
        cf_style = self.generate(style_z, target_style)
        results = []

        for alpha in strengths:
            interpolated = style_z + alpha * (cf_style - style_z)
            results.append(interpolated)

        return results

# %%


# %%


# %%
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Set up data paths
glove_path = "glove.6B.300d.txt"  # Set to None if you don't have GloVe embeddings

print("Building vocabularies...")
vocab = build_vocab(data_dir)
noun_vocab = build_noun_vocab(data_dir, vocab)
print(f"Vocab size: {len(vocab)}")
print(f"Noun vocab size: {len(noun_vocab)}")

# Initialize embeddings
print("Initializing embeddings...")
embedding_matrix = initialize_embeddings(vocab, embedding_dim=EMBEDDING_DIM, glove_path=glove_path)

# Create dataset and dataloader
print("Creating datasets...")
train_dataset = TextDataset(data_dir, vocab, noun_vocab, split='train')
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
print(f"Created dataloader with {len(train_loader)} batches")

# %%
from collections import defaultdict

# Initialize models
print("Initializing models...")
vocab_size = len(vocab)
noun_vocab_size = len(noun_vocab)

# Initialize VAE
vae = DisentangledVAE(
    vocab_size, EMBEDDING_DIM, HIDDEN_DIM, STYLE_DIM, CONTENT_DIM,
    vocab, embedding_matrix
).to(DEVICE)

# Initialize classifiers
style_classifier = StyleClassifier(STYLE_DIM).to(DEVICE)
content_classifier = ContentClassifier(CONTENT_DIM, noun_vocab_size).to(DEVICE)
adv_style_classifier = AdversarialStyleClassifier(CONTENT_DIM).to(DEVICE)
adv_content_classifier = AdversarialContentClassifier(STYLE_DIM, noun_vocab_size).to(DEVICE)



# %%
# Initialize style transfer pipeline
#style_transfer = StyleTransferPipeline(vae, style_classifier, vocab)

print("Starting training...")
try:
    train_model(
        vae, style_classifier, content_classifier,
        adv_style_classifier, adv_content_classifier,
        train_loader, 50, LEARNING_RATE
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {str(e)}")
    raise

# Save models
print("Saving models...")
torch.save({
    'vae_state_dict': vae.state_dict(),
    'style_classifier_state_dict': style_classifier.state_dict(),
    'content_classifier_state_dict': content_classifier.state_dict(),
    'vocab': vocab,
    'noun_vocab': noun_vocab
}, 'style_transfer_model_amazon.pt')
print("Models saved successfully!")
 

