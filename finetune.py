import os
import shutil
import torch
from tokenizers import Tokenizer
from model import GPTLanguageModel

# ==========================================
# 1. バックアップ処理
# ==========================================
original_ckpt = "baby_gpt_best.pth"
backup_ckpt = "ckpt_base_literature.pth"

if not os.path.exists(original_ckpt):
    raise FileNotFoundError(f"'{original_ckpt}' が見つかりません。")

if not os.path.exists(backup_ckpt):
    shutil.copy2(original_ckpt, backup_ckpt)
    print(f"🔒 バックアップ成功: '{original_ckpt}' を '{backup_ckpt}' にコピーしました。")
else:
    print(f"✅ バックアップは既に存在します: '{backup_ckpt}'")

# デバイスの自動検出
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"使用中のデバイス: {device}")

# ==========================================
# 2. 事前学習済みモデルのロード
# ==========================================
checkpoint = torch.load(original_ckpt, map_location=device)
hparams = checkpoint['hyperparameters']
vocab_size = checkpoint['vocab_size']

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=hparams['n_embd'],
    block_size=hparams['block_size'],
    n_head=hparams['n_head'],
    n_layer=hparams['n_layer'],
    dropout=0.2
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
print("事前学習済みモデルをロードしました。")

# ==========================================
# 3. データの準備
# ==========================================
tokenizer = Tokenizer.from_file("tokenizer.json")
encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids

with open('chat_data.txt', 'r', encoding='utf-8') as f:
    chat_text = f.read()

data = torch.tensor(encode(chat_text), dtype=torch.long)
print(f"ファインチューニング用データトークン数: {len(data)}")

# ==========================================
# 4. ハイパーパラメータ (事後学習用)
# ==========================================
learning_rate = 3e-4  # 破滅的忘却を防ぐため低め
max_iters = 300       # 短期学習ステップ
batch_size = 8        # データが少ないのでバッチサイズも小さく
block_size = hparams['block_size'] # 元のコンテキスト長を維持

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 小規模データのための安全なバッチ取得関数
def get_batch():
    max_idx = len(data) - block_size - 1
    if max_idx <= 0:
        raise ValueError("データが短すぎます。データ量を増やすか block_size を下げてください。")
    ix = torch.randint(max_idx, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 5. ファインチューニングループ
# ==========================================
print("\n--- 事後学習（ファインチューニング）を開始します ---")
model.train()

for iter in range(max_iters + 1):
    xb, yb = get_batch()
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 50 == 0:
        print(f"[ステップ {iter:3d}] ロス: {loss.item():.4f}")

# ==========================================
# 6. ファインチューニング済みモデルの保存
# ==========================================
save_path = "baby_gpt_chat.pth"
chat_checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'hyperparameters': hparams
}
torch.save(chat_checkpoint, save_path)
print(f"\n✅ 事後学習が完了しました！モデルを '{save_path}' に保存しました。")
