import os
import torch
import torch.optim as optim
import itertools
import time
from .transformer import Model
from .dataloader import get_data_loaders


def model_runner(
    # --- model setup ---
    batch_size: int = 4,
    embed_size: int = 32,
    ff_hidden_size: int = 64,
    num_layers: int = 2,
    seq_len: int = 64,
    num_heads: int = 4,
    
    # --- training setup ---
    global_steps: int = 1000,
    minibatch: int = 4,
    eval_freq: int = 100,
    lr: float = 0.001,
    lr_min: float = 0.0001,
    checkpoint_dir: str = "./checkpoints",
    dataset_dir: str = "./datasets"
):
    """
    Run training of a small transformer model on WikiText.
    """

    steps = global_steps // minibatch
    val_batch_count = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data loaders, model, optimizer, loss --- #
    train_data, val_data, meta = get_data_loaders(batch_size=batch_size, 
                                                      seq_len=seq_len, 
                                                      dataset_dir=dataset_dir)
    vocab_size = meta["vocab_size"]
    crossentropy_loss = torch.nn.CrossEntropyLoss()

    model = Model(vocab_size=vocab_size,
                seq_len=seq_len,
                embed_size=embed_size,
                ff_hidden_size=ff_hidden_size,
                num_layers=num_layers,
                num_heads=num_heads).to(device)
    
    model = model.to(torch.bfloat16)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr_min)

    train_itr, val_itr = itertools.cycle(train_data), itertools.cycle(val_data)

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params / 1e6:.2f}M parameters.")
    print(f"Training for {steps} steps with minibatch size {minibatch}...")
    for its in range(steps+1):
        acc_loss = 0.0
        for _ in range(minibatch):
            x, y = next(train_itr)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = crossentropy_loss(y_hat.view(-1, vocab_size), y.view(-1)) / minibatch
            acc_loss += loss.item()
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if its % eval_freq == 0 and its > 0:
            acc_val_loss = 0.0
            acc_time = 0.0
            for _ in range(val_batch_count):
                mili_start = time.time()
                x, y = next(val_itr)
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                acc_val_loss += crossentropy_loss(y_hat.view(-1, vocab_size), y.view(-1)).item()
                mili_end = time.time()
                acc_time += mili_end - mili_start
            acc_val_loss /= val_batch_count
            time_in_mili = (acc_time / val_batch_count) * 1000.0
            perplexity = 2.71828 ** acc_val_loss
            print(f"Step {its}/{steps}: Train Loss: {acc_loss:.4f}, Val Loss: {acc_val_loss:.4f}, Perplexity: {perplexity:.4f}, Time/Iter: {time_in_mili:.4f}ms")
            
            acc_val_loss = 0.0
            acc_time = 0.0
        acc_loss = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)
    time_id = int(time.time())
    # torch.save(model.state_dict(), f"{checkpoint_dir}/tinyLM_{time_id}.ckpt")

    hyperparams = {
    'batch_size': batch_size,
    'embed_size': embed_size,
    'ff_hidden_size': ff_hidden_size,
    'num_layers': num_layers,
    'seq_len': seq_len,
    'num_heads': num_heads,

    # training setup
    'global_steps': global_steps,
    'minibatch': minibatch,
    'eval_freq': eval_freq,
    'lr': lr,
    'lr_min': lr_min,
}
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': its if 'step' in locals() else 0,
        'args': hyperparams,
    }, f"{checkpoint_dir}/tinyLM_{time_id}.ckpt")
    print("Training complete.", f"Model saved to .../tinyLM_{time_id}.ckpt")