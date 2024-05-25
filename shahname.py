import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

params = {
    "epochs": 10,
    "learning_rate": 3e-4,
    "batch_size": 8,
    "embedding_dim": 8,
    "nhead": 2,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dropout": 0.1,
    "block_size": 13,
    "dim_feedforward": 4,
}
load_model = False
save_model = True
model_filename = "models/shahname.pt"

mesra_delimiter = "\t"
beyt_delimiter = "\n"


class Head(nn.Module):
    """
    Self-attention head layer.
    """

    def __init__(self, embedding_dim, head_size, dropout=0.0):
        super().__init__()

        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        _, _, C = q.shape
        value, key, query = self.value(v), self.key(k), self.query(q)
        weights = query @ key.transpose(-2, -1) * C**-0.5

        if mask is not None:
            weights = weights.masked_fill(mask, float("-inf"))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ value
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), f"{embedding_dim=} must be divisible by {num_heads=}"
        head_size = embedding_dim // num_heads

        self.ln = nn.LayerNorm(embedding_dim)
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        v, k, q = self.ln(v), self.ln(k), self.ln(q)
        out = torch.cat([head(v, k, q, mask) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dim_feedforward, dropout=0.0):
        super().__init__()

        # feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, dim_feedforward * embedding_dim),
            nn.ReLU(),
            nn.Linear(dim_feedforward * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ffn(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        # multi-head self attention with no mask. All nodes may communicate.
        self.attn = MultiheadAttention(embedding_dim, num_heads, dropout)
        self.ffn = FeedForward(embedding_dim, dim_feedforward)

    def forward(self, v, k, q):
        out = q + self.attn(v, k, q)
        out = out + self.ffn(out)
        return out


def generate_square_subsequent_mask(sz, device):
    return torch.tril(torch.ones(sz, sz).to(device)) == 0


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        # multi-head self attention with triangular mask. Nodes communicate only
        # with previous nodes.
        self.attn = MultiheadAttention(embedding_dim, num_heads, dropout)
        # Reusing Encoder as the top part of the decoder with a multi-head
        # cross-attention and a feed-forward network on top of it.
        self.attn_ffn = EncoderBlock(embedding_dim, num_heads, dim_feedforward, dropout)

    def forward(self, enc_out, dec_in, mask):
        out = dec_in
        out = out + self.attn(out, out, out, mask)
        out = out + self.attn_ffn(enc_out, enc_out, out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_encoder_layers,
        num_decoder_layers,
        block_size,
        device,
        embedding_dim,
        nhead,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.device = device

        self.src_emb = nn.Embedding(src_vocab_size, embedding_dim)
        self.src_pos = nn.Embedding(block_size, embedding_dim)

        self.tgt_emb = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.tgt_pos = nn.Embedding(block_size, embedding_dim)

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim,
                    nhead,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim,
                    nhead,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.proj = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        _, srcT = src.shape
        src_positions = torch.arange(srcT).unsqueeze(0).to(self.device)
        src_out = self.src_emb(src) + self.src_pos(src_positions)
        src_out = self.dropout(src_out)

        for encoder in self.encoders:
            src_out = encoder(src_out, src_out, src_out)

        _, tgtT = tgt.shape
        tgt_positions = torch.arange(tgtT).unsqueeze(0).to(self.device)
        tgt_out = self.tgt_emb(tgt) + self.tgt_pos(tgt_positions)
        tgt_out = self.dropout(tgt_out)

        mask = generate_square_subsequent_mask(tgtT, self.device)

        for decoder in self.decoders:
            tgt_out = decoder(src_out, tgt_out, mask)

        tgt_out = self.proj(tgt_out)
        tgt_out = self.dropout(tgt_out)

        return tgt_out


def split_beyts(filename, split_files):
    with open(filename) as f:
        lines = f.read().splitlines(keepends=True)

    lines = [line.split(mesra_delimiter) for line in lines]
    src = [line[0] for line in lines if len(line) == 2]
    tgt = [mesra_delimiter + line[1] for line in lines if len(line) == 2]

    with open(split_files[0], "w") as f:
        f.write(beyt_delimiter.join(src))

    with open(split_files[1], "w") as f:
        f.write("".join(tgt))


def get_tokenizer(filenames):
    tk = Tokenizer(BPE(unk_token="[UNK]"))
    tk.enable_padding(pad_id=3)
    # tk.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tk.train(filenames, trainer)

    return tk


def split(data, train_ratio=0.8, val_ratio=0.1):
    src, tgt = data
    assert len(src) == len(tgt), "expeted the same source and target sizes."

    ntrain = int(train_ratio * len(src))
    nval = int(val_ratio * len(src))
    indices = torch.randperm(len(src))

    train = (src[indices][:ntrain], tgt[indices][:ntrain])
    val = (src[indices][ntrain : ntrain + nval], tgt[indices][ntrain : ntrain + nval])
    test = (src[indices][ntrain + nval :], tgt[indices][ntrain + nval :])

    return train, val, test


def get_batch(data, batch_size, device):
    """
    Generates a batch of examples.
    """
    src, tgt = data
    assert len(src) == len(tgt), "expeted the same source and target sizes."

    indices = torch.randint(len(src), (batch_size,))

    x = src[indices].to(device)
    y = tgt[indices].to(device)

    return x, y


def get_loss(logits, y, ignore_index):
    """
    Computes cross-entropy loss, given logits and labels.
    """
    B, T, C = logits.shape
    # F.cross_entropy expects size C, (B, C), or (B, C, ...)
    # logits shape is (B, T, C), so we flatten the first two dimensions.
    return F.cross_entropy(
        logits.view(B * T, C), y.reshape(B * T), ignore_index=ignore_index
    )


def generate(first_mesra, tk, model, device):
    """
    Generates second mesra.
    """
    token_ids = tk.encode(first_mesra).ids
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    y = torch.zeros((1, 1), dtype=torch.long, device=device)

    while True:
        logits = model(x, y)
        # only consider the last logit
        logits = logits[:, -1, :]
        score = F.softmax(logits, dim=-1)
        next_token_id = score.multinomial(1)
        if "\n" in tk.id_to_token(next_token_id):
            break
        y = torch.cat((y, next_token_id), dim=1)

    y = y.view(-1)
    return " ".join([tk.id_to_token(t) for t in y[1:]])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    text_file = "data/shahname.txt"
    split_files = ["data/shahname_src.txt", "data/shahname_tgt.txt"]
    split_beyts(text_file, split_files)

    tk = get_tokenizer(split_files)
    vocab_size = tk.get_vocab_size()
    print(f"vocab size: {vocab_size}")

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_encoder_layers=params["num_encoder_layers"],
        num_decoder_layers=params["num_decoder_layers"],
        block_size=params["block_size"],
        device=device,
        embedding_dim=params["embedding_dim"],
        nhead=params["nhead"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    ).to(device)

    num_params = sum([p.nelement() for p in model.parameters()])
    print(f"model parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10
    )

    if load_model:
        state = torch.load(model_filename)

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    num_params = sum([p.nelement() for p in model.parameters()])
    print(f"\nmodel parameters: {num_params}")

    first_mesra = "چو ترکان بدیدند کارجاسپ رفت"
    print(f"\nfirst mesra: {first_mesra}")

    with open("data/shahname_src.txt") as f:
        src_lines = f.read().splitlines()

    src_token_ids = torch.tensor(
        [x.ids for x in tk.encode_batch(src_lines)], dtype=torch.long
    )

    with open("data/shahname_tgt.txt") as f:
        tgt_lines = f.read().splitlines(keepends=True)

    tgt_token_ids = torch.tensor(
        [x.ids for x in tk.encode_batch(tgt_lines)], dtype=torch.long
    )

    train, val, _ = split((src_token_ids, tgt_token_ids), 0.9, 0.1)

    train_losses, val_losses = [], []

    for epoch in range(params["epochs"]):
        print(f"epoch {epoch} / {params['epochs']}")

        model.eval()
        with torch.no_grad():
            second_mesra = generate(first_mesra, tk, model, device)
            print(f"second mesra:\n{second_mesra}")

            src, tgt = get_batch(val, params["batch_size"], device)

            logits = model(src, tgt[:, :-1])
            vloss = get_loss(logits, tgt[:, 1:], ignore_index=tk.token_to_id("[PAD]"))
            val_losses.append(vloss.item())

        model.train()
        src, tgt = get_batch(train, params["batch_size"], device)

        logits = model(src, tgt[:, :-1])
        loss = get_loss(logits, tgt[:, 1:], ignore_index=tk.token_to_id("[PAD]"))
        train_losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if save_model:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, model_filename)

        scheduler.step(train_losses[-1])
