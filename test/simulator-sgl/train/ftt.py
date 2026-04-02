# ===== ft_transformer_regressor.py (numeric+categorical, residual, regularized) =====
import math, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- scalers ----------
class StandardScaler1D:
    def __init__(self): self.mean_=None; self.std_=None
    def fit(self, x: np.ndarray):
        self.mean_=x.mean(axis=0); self.std_=x.std(axis=0); self.std_[self.std_<1e-8]=1.0
    def transform(self, x: np.ndarray)->np.ndarray: return (x - self.mean_) / self.std_
    def inverse_transform(self, x: np.ndarray)->np.ndarray: return x * self.std_ + self.mean_

# ---------- utils ----------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-9
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + eps)
    r2 = float(1.0 - ss_res / ss_tot)
    mape = float(np.mean(np.abs(resid) / (np.abs(y_true) + eps)) * 100.0)
    med_ape = float(np.median(np.abs(resid) / (np.abs(y_true) + eps)) * 100.0)
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "median_ape": med_ape}

class NpDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: Optional[np.ndarray], y: np.ndarray):
        self.Xn = X_num.astype(np.float32)
        self.Xc = None if X_cat is None else X_cat.astype(np.int64)
        self.y  = y.astype(np.float32).reshape(-1,1)
    def __len__(self): return self.Xn.shape[0]
    def __getitem__(self, i):
        if self.Xc is None: return self.Xn[i], self.y[i]
        return (self.Xn[i], self.Xc[i]), self.y[i]

# ---------- modules ----------
class FeatureDropout(nn.Module):
    def __init__(self, p: float = 0.1): super().__init__(); self.p = p
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p<=0: return tokens
        B,S,D = tokens.shape
        mask = (torch.rand(B,S, device=tokens.device) > self.p).float().unsqueeze(-1)
        mask[:,0,:] = 1.0  # keep CLS
        return tokens * mask

class NumericTokenizer(nn.Module):
    def __init__(self, n_num: int, d_tok: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d_tok) / math.sqrt(d_tok))
        self.bias   = nn.Parameter(torch.zeros(n_num, d_tok))
    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: [B, Fn] -> [B, Fn, d]
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class CategoricalTokenizer(nn.Module):
    def __init__(self, cardinalities: List[int], d_tok: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card+1, d_tok) for card in cardinalities])  # +1 for OOV
    def forward(self, x_cat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_cat is None: return None
        toks = [emb(x_cat[:,i].clamp_min(0)) for i,emb in enumerate(self.embs)]  # [B,d] each
        return torch.stack(toks, dim=1)  # [B, Fc, d]

class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cards: List[int], d_tok=96, n_heads=4, n_layers=3, ffn_mult=2.0, dropout=0.15):
        super().__init__()
        self.use_cls = True
        self.cls = nn.Parameter(torch.zeros(1,1,d_tok))
        self.num_tok = NumericTokenizer(n_num, d_tok) if n_num>0 else None
        self.cat_tok = CategoricalTokenizer(cat_cards, d_tok) if len(cat_cards)>0 else None
        self.feat_drop = FeatureDropout(p=0.10)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=n_heads, dim_feedforward=int(d_tok*ffn_mult),
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok,d_tok), nn.GELU(), nn.Linear(d_tok,1))

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        toks = []
        if self.use_cls:
            toks.append(self.cls.expand(x_num.size(0) if x_num is not None else x_cat.size(0), 1, -1))
        if self.num_tok is not None and x_num is not None:
            toks.append(self.num_tok(x_num))
        if self.cat_tok is not None and x_cat is not None:
            toks.append(self.cat_tok(x_cat))
        seq = torch.cat(toks, dim=1)    # [B, 1+Fn+Fc, d]
        seq = self.feat_drop(seq)
        enc = self.encoder(seq)
        z = enc[:,0]
        return self.head(z)

class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)

@dataclass
class FTTConfig:
    d_tok:int=96; n_heads:int=4; n_layers:int=3; ffn_mult:float=2.0; dropout:float=0.15
    batch_size:int=4096; lr:float=5e-4; weight_decay:float=2e-4
    max_epochs:int=600; warmup_epochs:int=5; early_stop_patience:int=50
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    loss:str="huber"; huber_delta:float=1.0; verbose:int=1
    ema_decay:float=0.999
    use_plateau:bool=True  # ReduceLROnPlateau

class _FTTWrapper:
    def __init__(self, model: FTTransformer, num_cols: List[str], cat_cols: List[str],
                 x_num_scaler: Optional[StandardScaler1D], y_scaler: StandardScaler1D,
                 cat_maps: Dict[str, Dict], device: str, baseline_tr_mean: Optional[float]=None):
        self.model=model; self.num_cols=num_cols; self.cat_cols=cat_cols
        self.xs=x_num_scaler; self.ys=y_scaler; self.cat_maps=cat_maps; self.device=device
        self.baseline_tr_mean = baseline_tr_mean  # 用于预测时缺 baseline 的兜底（可选）

    def _cats_to_ids(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.cat_cols: return None
        arr = []
        for c in self.cat_cols:
            m = self.cat_maps[c]
            # 未见类别 → 0（OOV）
            ids = np.array([m.get(v, 0) for v in df[c].tolist()], dtype=np.int64)
            arr.append(ids)
        return np.stack(arr, axis=1)

    @torch.no_grad()
    def predict(self, X: Union[np.ndarray, pd.DataFrame], baseline_values: Optional[np.ndarray]=None) -> np.ndarray:
        # baseline_values: 与 X 同长度（若用残差训练，预测时加回；否则忽略）
        if isinstance(X, pd.DataFrame):
            X_num = X[self.num_cols].values.astype(np.float32) if self.num_cols else np.zeros((len(X),0),np.float32)
            X_cat = self._cats_to_ids(X) if self.cat_cols else None
        else:
            X_num = X.astype(np.float32); X_cat = None  # 纯数值 numpy
        if self.xs is not None and X_num.shape[1]>0: Xn = self.xs.transform(X_num)
        else: Xn = X_num
        Xn_t = torch.from_numpy(Xn).to(self.device)
        Xc_t = None if X_cat is None else torch.from_numpy(X_cat).to(self.device)

        self.model.eval()
        preds_std=[]; bsz=8192
        for i in range(0, Xn_t.size(0), bsz):
            yhat_std = self.model(Xn_t[i:i+bsz], None if Xc_t is None else Xc_t[i:i+bsz]).cpu().numpy().reshape(-1)
            preds_std.append(yhat_std)
        delta_std = np.concatenate(preds_std, axis=0)
        delta_raw = self.ys.inverse_transform(delta_std)

        if baseline_values is not None:
            return baseline_values.reshape(-1) + delta_raw
        elif self.baseline_tr_mean is not None:
            return self.baseline_tr_mean + delta_raw
        else:
            return delta_raw

# -------- main API --------
def perform_ft_transformer_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    val_data: Tuple[Optional[Union[np.ndarray,pd.DataFrame]], Optional[np.ndarray]],
    y_values: np.ndarray,
    cfg: Optional[FTTConfig] = None,
    cat_cols: Optional[List[str]] = None,
    baseline_values: Optional[np.ndarray] = None,   # 如果做“残差训练”，传入 baseline（train split）
    val_baseline_values: Optional[np.ndarray] = None,  # val 对应 baseline
) -> Dict:
    """
    feature_cols: 数值特征列（建议把 ID 类移出去，放到 cat_cols）
    cat_cols:     类别特征列名（如 model_id/config_id/version 等）
    baseline_values: 若提供，则训练目标为 y - baseline（残差）；预测时自动加回 baseline
    """
    cfg = cfg or FTTConfig()
    device = cfg.device
    num_cols = feature_cols[:]
    cat_cols = cat_cols or []

    # ---- slice splits ----
    if isinstance(df, pd.DataFrame): Xn_tr = df[num_cols].values.astype(np.float32) if num_cols else np.zeros((len(df),0), np.float32)
    else: raise ValueError("df must be DataFrame with numeric columns")

    y_tr_raw = y_values.astype(np.float32).reshape(-1)

    # handle cat
    cat_maps = {}
    Xc_tr = None
    if cat_cols:
        for c in cat_cols:
            uniq = pd.Series(df[c].astype(str)).unique().tolist()
            # 预留 0 for OOV，从 1 开始编号
            cat_maps[c] = {v:i+1 for i,v in enumerate(uniq)}
        Xc_tr = np.stack([np.array([cat_maps[c].get(str(v),0) for v in df[c].tolist()], dtype=np.int64) for c in cat_cols], axis=1)

    # ---- residual target ----
    if baseline_values is not None:
        baseline_tr = baseline_values.reshape(-1).astype(np.float32)
        y_tr_resid_raw = y_tr_raw - baseline_tr
    else:
        baseline_tr = None
        y_tr_resid_raw = y_tr_raw

    # ---- scalers ----
    x_scaler = None
    if Xn_tr.shape[1] > 0:
        x_scaler = StandardScaler1D(); x_scaler.fit(Xn_tr)
        Xn_tr = x_scaler.transform(Xn_tr)
    y_scaler = StandardScaler1D(); y_scaler.fit(y_tr_resid_raw.reshape(-1,1))
    y_tr_std = y_scaler.transform(y_tr_resid_raw.reshape(-1,1)).reshape(-1)

    # ---- val ----
    X_va, y_va_raw = val_data if val_data is not None else (None, None)
    if X_va is not None:
        if isinstance(X_va, pd.DataFrame):
            Xn_va = X_va[num_cols].values.astype(np.float32) if num_cols else np.zeros((len(X_va),0),np.float32)
            Xc_va = None
            if cat_cols:
                Xc_va = np.stack([np.array([cat_maps[c].get(str(v),0) for v in X_va[c].tolist()], dtype=np.int64) for c in cat_cols], axis=1)
        else:
            Xn_va = X_va.astype(np.float32); Xc_va = None
        if x_scaler is not None and Xn_va.shape[1]>0: Xn_va = x_scaler.transform(Xn_va)
        y_va_raw = y_va_raw.astype(np.float32).reshape(-1)
        if val_baseline_values is not None:
            baseline_va = val_baseline_values.reshape(-1).astype(np.float32)
            y_va_resid_raw = y_va_raw - baseline_va
        else:
            baseline_va = None
            y_va_resid_raw = y_va_raw
        y_va_std = y_scaler.transform(y_va_resid_raw.reshape(-1,1)).reshape(-1)
    else:
        Xn_va = Xc_va = baseline_va = y_va_std = y_va_raw = None

    # ---- model/loss/opt/sched ----
    cat_cards = [len(cat_maps[c]) for c in cat_cols]
    model = FTTransformer(n_num=Xn_tr.shape[1], cat_cards=cat_cards,
                          d_tok=cfg.d_tok, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                          ffn_mult=cfg.ffn_mult, dropout=cfg.dropout).to(device)

    def huber(pred, target, delta=cfg.huber_delta):
        diff = torch.abs(pred - target)
        return torch.where(diff<delta, 0.5*diff**2, delta*(diff-0.5*delta)).mean()

    criterion = huber if cfg.loss=="huber" else (nn.L1Loss() if cfg.loss=="mae" else nn.MSELoss())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.use_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, threshold=1e-3, min_lr=1e-5)
    else:
        def lr_lambda(ep):
            if ep < cfg.warmup_epochs: return (ep+1)/max(1,cfg.warmup_epochs)
            t = (ep - cfg.warmup_epochs)/max(1,(cfg.max_epochs-cfg.warmup_epochs)); return 0.5*(1+math.cos(math.pi*min(1.0,t)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ema = EMA(model, decay=cfg.ema_decay)

    # ---- data loaders ----
    train_ds = NpDataset(Xn_tr, Xc_tr, y_tr_std)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0)

    def eval_on(Xn, Xc, y_raw, baseline_raw):
        model.eval(); ema.copy_to(model)
        with torch.no_grad():
            preds_std=[]; bsz=8192
            Xn_t = torch.from_numpy(Xn).to(device) if Xn is not None else torch.zeros((len(y_raw),0), dtype=torch.float32, device=device)
            Xc_t = None if Xc is None else torch.from_numpy(Xc).to(device)
            for i in range(0, len(y_raw), bsz):
                xb_num = Xn_t[i:i+bsz]; xb_cat = None if Xc_t is None else Xc_t[i:i+bsz]
                out_std = model(xb_num, xb_cat).cpu().numpy().reshape(-1)
                preds_std.append(out_std)
            delta_std = np.concatenate(preds_std, axis=0)
            delta_raw = y_scaler.inverse_transform(delta_std)
            if baseline_raw is not None:
                yhat = baseline_raw.reshape(-1) + delta_raw
            else:
                yhat = delta_raw
        return _metrics(y_raw, yhat), yhat

    # ---- train ----
    best_val = float("inf"); best_state=None; no_improve=0; start=time.time()
    for ep in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            (xb_num, xb_cat), yb = batch if isinstance(batch[0], tuple) else ((batch[0], None), batch[1])
            xb_num = xb_num.to(device); xb_cat = None if xb_cat is None else xb_cat.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb_num, xb_cat)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
        # val
        if Xn_va is not None:
            val_metrics, _ = eval_on(Xn_va, Xc_va, y_va_raw, baseline_va)
            score = val_metrics["mae"]
            if cfg.use_plateau: scheduler.step(score)
            else: scheduler.step()
            if cfg.verbose and (ep%10==0 or ep==cfg.max_epochs-1):
                print(f"[FTT] ep {ep:03d}  val_mae={score:.3f}")
            if score + 1e-6 < best_val:
                best_val = score; best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                    if cfg.verbose: print(f"[FTT] Early stop at ep {ep}")
                    break
        else:
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k,v in best_state.items()})

    wrapper = _FTTWrapper(model, num_cols, cat_cols, x_scaler, y_scaler, cat_maps, device,
                          baseline_tr_mean=float(baseline_tr.mean()) if baseline_tr is not None else None)

    # train metrics
    train_pred = wrapper.predict(df[num_cols + cat_cols] if cat_cols else df[num_cols], baseline_values)
    results = {"model": wrapper, "train_metrics": _metrics(y_tr_raw, train_pred), "fit_time_sec": time.time()-start}

    if Xn_va is not None:
        val_pred = wrapper.predict(X_va if isinstance(X_va, pd.DataFrame) else pd.DataFrame(X_va, columns=num_cols), val_baseline_values)
        results["val_metrics"] = _metrics(y_va_raw, val_pred)
        results["val_baseline_metrics"] = _metrics(y_va_raw, np.full_like(y_va_raw, fill_value=float(y_tr_raw.mean())))
    return results
