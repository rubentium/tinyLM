import argparse
from src.optimizer import model_runner

def parse_args():
    parser = argparse.ArgumentParser(description="Run small Transformer training.")

    # --- model setup ---
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=2048)
    parser.add_argument("--ff_hidden_size", type=int, default=16384)
    parser.add_argument("--num_layers", type=int, default=36)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=64)

    # --- training setup ---
    parser.add_argument("--global_steps", type=int, default=10000)
    parser.add_argument("--minibatch", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)

    # --- directories--- #
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--datasets_dir", type=str, default="/mloscratch/homes/navasard/datasets")

    return parser.parse_args()

def main():
    args = parse_args()

    # Call model_runner with all arguments
    model_runner(
        batch_size=args.batch_size,
        embed_size=args.embed_size,
        ff_hidden_size=args.ff_hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        global_steps=args.global_steps,
        minibatch=args.minibatch,
        eval_freq=args.eval_freq,
        lr=args.lr,
        lr_min=args.lr_min,
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.datasets_dir
    )

if __name__ == "__main__":
    main()
