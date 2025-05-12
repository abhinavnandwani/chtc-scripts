import argparse
import tqdm
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from utils import *

def train(train_ids, train_label_map, val_ids, val_label_map, args, fold_idx=None, logger=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = None #StandardScaler()

    if scaler is not None:
        for donor in tqdm.tqdm(train_ids, desc="Fitting scaler"):
            path = os.path.join(args.data_path, donor + f".{args.file_format}")
            df = pd.read_csv(path, index_col=0) if args.file_format == 'csv' else pd.read_parquet(path)
            scaler.partial_fit(df.values)
    
    train_ds = EmbDataset(train_ids, args.data_path, train_label_map, scaler, True, args.file_format)
    val_ds   = EmbDataset(val_ids,   args.data_path, val_label_map,   scaler, True, args.file_format)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    
    input_dim   = args.input_dim or train_ds[0][0].shape[0]
    
    num_classes = args.num_classes or len(set(train_label_map.values()))
    if num_classes==2:
        num_classes=1
    
    hidden_dims = [int(h) for h in args.hidden_dims.split(",")] if args.hidden_dims else []
    
    model = (LogisticRegressionModel(input_dim, num_classes) if args.model_type == 'logistic'
             else MLP(input_dim, hidden_dims, num_classes)).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    # train epochs
    print ("Training init...")
    for epoch in range(args.epochs):
        
        model.train(); 
        total_loss=0.0; 
        preds, truths=[],[]
        
        for i, (bx, by, batch_weights) in enumerate(
            tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        ):
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(bx); loss = criterion(out, by)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * bx.size(0)
            preds.extend((out>0).long().cpu().tolist()); truths.extend(by.cpu().long().tolist())
        train_loss = total_loss / len(train_loader.dataset)
        train_bal  = balanced_accuracy_score(truths, preds)
    
    # eval
    model.eval(); 
    total_val_loss=0.0; 
    vpreds,vtruths=[],[]; 
    count=0
    
    with torch.no_grad():
        for bx, by, wt in val_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            out = model(bx); loss = criterion(out,by).mean().item()
            bs = by.size(0); total_val_loss += loss*bs; count+=bs
            vpreds.extend((out>0).long().cpu().tolist()); vtruths.extend(by.cpu().long().tolist())
    avg_val_loss = total_val_loss/count if count else 0
    avg_bal      = balanced_accuracy_score(vtruths, vpreds)
    logger.info(f"Complete training{' fold '+str(fold_idx) if fold_idx else ''}...")
    return train_loss, train_bal, avg_val_loss, avg_bal


def cross_validate(args):
    logger = logging.getLogger()
    sample_ids, labels, label_map = get_metadata_cv(
        args.meta, args.sample_column, args.phenotype_column
    )
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    train_losses, train_bals, val_losses, val_bals = [], [], [], []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(sample_ids, labels), 1):
        trains = [sample_ids[i] for i in tr_idx]
        vals   = [sample_ids[i] for i in va_idx]
        tr_map = {sid: label_map[sid] for sid in trains}
        va_map = {sid: label_map[sid] for sid in vals}
        logger.info(f"===== Fold {fold_idx}/{args.n_splits} =====")
        t_loss, t_bal, v_loss, v_bal = train(
            trains, tr_map,
            vals,   va_map,
            args, fold_idx, logger
        )
        logger.info(f"Fold {fold_idx}: train_loss={t_loss:.4f}, train_bal={t_bal:.4f}, val_loss={v_loss:.4f}, val_bal={v_bal:.4f}")
        train_losses.append(t_loss); train_bals.append(t_bal)
        val_losses.append(v_loss);     val_bals.append(v_bal)
    avg_train_loss = sum(train_losses)/len(train_losses)
    avg_train_bal  = sum(train_bals)/len(train_bals)
    avg_val_loss   = sum(val_losses)/len(val_losses)
    avg_val_bal    = sum(val_bals)/len(val_bals)
    logger.info(f"AVG train_loss={avg_train_loss:.4f}, AVG train_bal={avg_train_bal:.4f}, AVG val_loss={avg_val_loss:.4f}, AVG val_bal={avg_val_bal:.4f}")
    hp = {
        'epochs': args.epochs,
        'hidden_dims': args.hidden_dims,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'num_workers': args.num_workers,
        'model_type': args.model_type,
        'n_splits': args.n_splits
    }
    summary = {**hp,
               'avg_train_loss': avg_train_loss,
               'avg_train_bal':  avg_train_bal,
               'avg_val_loss':   avg_val_loss,
               'avg_val_bal':    avg_val_bal}
    summary_df = pd.DataFrame([summary])
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = f"{args.out_path}/cv_summary_{ts}.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved CV summary to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Train a classification model on cell embeddings.")
    parser.add_argument(
        '--meta', dest='meta', type=str,
        default="/media/sayalialatkar/T9/Sayali/FoundationModels/scBrainLLM/phenotype_classification_files/c15x_split_seed42.pkl",
        help="Path to the train/test split metadata pickle file."
    )
    parser.add_argument(
        '--fm_model_name', default="scGPT",
        help="Model name to load inferred cell embeddings (default: scGPT)"
    )
    parser.add_argument(
        '--model_type', type=str, choices=['mlp','logistic'], default='mlp',
        help="Model to use: 'mlp' or 'logistic'"
    )
    parser.add_argument(
        '--data_path', type=str,
        default='/media/sayalialatkar/T9/Sayali/FoundationModels/scGPT-main/results/zero-shot/extracted_cell_embeddings_full_body',
        help="Directory containing data files (default is scGPT path)."
    )
    parser.add_argument(
        '--out_path', type=str,
        default='/media/sayalialatkar/T9/Sayali/FoundationModels/scBrainLLM/results',
        help="Directory to store results."
    )
    parser.add_argument(
        '--file_format', type=str, default='csv', choices=['csv', 'parquet'],
        help="File format of input data files."
    )
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of "train" split to hold out for validation.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for splitting.')
    parser.add_argument('--sample_column', type=str, default='SubID',
                        help="Column name for sample IDs in metadata.")
    parser.add_argument('--phenotype_column', type=str, default='c15x',
                        help="Column name for phenotype labels in metadata.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--input_dim', type=int, default=None,
                        help='Input feature dimension (auto if unset).')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes.')
    parser.add_argument('--hidden_dims', type=str, default="128,64",
                        help='Comma-separated hidden layer sizes, e.g. "128,64".')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--n_splits', type=int, default=5,
                        help="Validation splits.")
    parser.add_argument('--accumulation_steps', type=int, default=0,
                        help="Number of steps to accumulate gradients before updating.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of DataLoader worker threads.")
    parser.add_argument('--do_cv', action='store_true',
                        help='Run 5-fold cross validation')
    parser.add_argument('--log_file', type=str,
                        default='/media/sayalialatkar/T9/Sayali/FoundationModels/scBrainLLM/results/train.log',
                        help="File to write training logs to (default: train.log)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(args.log_file)
    log_filename = f"{base}_{timestamp}{ext}"
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format='%(asctime)s %(levelname)s: %(message)s')

    args.data_path = get_model_data_path(args.fm_model_name)
    logger = logging.getLogger()
    logger.info(f"Hyperparameters: epochs={args.epochs}, hidden_dims={args.hidden_dims}, lr={args.lr}, "
                f"batch_size={args.batch_size}, accumulation_steps={args.accumulation_steps}, "
                f"num_workers={args.num_workers}, model_type={args.model_type}, n_splits={args.n_splits}")
    if args.do_cv:
        cross_validate(args)
    else:
        tr_ids, va_ids, tr_map, va_map = get_metadata(
            args.meta,
            sample_column=args.sample_column,
            phen_column=args.phenotype_column,
            val_split=args.val_split,
            random_state=args.random_state
        )
        logger.info("Starting single-split training...")
        train(tr_ids, tr_map, va_ids, va_map, args, None, logger)

if __name__ == '__main__':
    main()
