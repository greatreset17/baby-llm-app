import os
import torch
from tqdm import tqdm
from model import GPTLanguageModel

# ==========================================
# ハイパーパラメータの設定 (超軽量で高速に学習できるように調整)
# ==========================================
batch_size = 64        # 1回の学習ステップで同時に処理するサンプル数
block_size = 256       # モデルが一度に見るコンテキスト（サブワード数）の長さ（視野を2倍に拡張）
max_iters = 8000       # 最大学習ステップ数（早期終了前提で広めに設定）
eval_interval = 500    # ロスの計算と、成長過程のテキスト生成を行う間隔
learning_rate = 5e-4   # 学習率（33Mモデルに最適なスケーリング）
eval_iters = 50        # 評価時のサンプルステップ数
n_embd = 384           # 埋め込み次元（ベクトルの太さ）
n_head = 6             # マルチヘッド・アテンションのヘッド数
n_layer = 12           # Transformerブロックの積み重ね段数（8段から12段へスケールアップ➔33Mパラメータ脳へ）
dropout = 0.3          # ドロップアウト率（0.2から0.3へ強化し丸暗記による過学習を強力に防止）

# デバイスの自動検出（MacのApple Siliconがある場合は高速な mps を使用）
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"使用中のデバイス: {device}")

# 乱数シードの固定（再現性のため）
torch.manual_seed(1337)

# ==========================================
# 1. データの読み込みとトークナイザーの準備
# ==========================================
corpus_path = "corpus.txt"
if not os.path.exists(corpus_path):
    raise FileNotFoundError(f"'{corpus_path}' が見つかりません。")

with open(corpus_path, 'r', encoding='utf-8') as f:
    text = f.read()

# BPEトークナイザーのロード
from tokenizers import Tokenizer
tokenizer_path = "tokenizer.json"
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"'{tokenizer_path}' が見つかりません。先に tokenizer_train.py を実行してください。")

tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
print(f"ボキャブラリ数（BPE語彙数）: {vocab_size}")

encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids  # 文字列 -> IDリスト
decode = lambda l: tokenizer.decode(l)                                # IDリスト -> 文字列

# テキスト全体を数値IDのテンソルに変換
data = torch.tensor(encode(text), dtype=torch.long)

# ==========================================
# 2. データのチャンク化とシャッフル (検証ロスの過学習・乖離を防ぐためのシャッフルデータローダ)
# ==========================================
# コンテキスト長 block_size (128) の入力 x と、1トークン未来にずらしたターゲット y (128) を
# 完璧に切り出すため、あらかじめ (block_size + 1 = 129) トークンずつのチャンクに切り分けます。
chunk_len = block_size + 1
num_chunks = len(data) // chunk_len

# 余剰トークンをカットし、(チャンク数, 129) の2次元テンソルに変形します
chunks = data[:num_chunks * chunk_len].view(num_chunks, chunk_len)

# 各作品（桃太郎、走れメロス、こころ等）のデータが連続した塊のまま学習・検証に分割されるのを防ぎ、
# 全作品からランダムかつ均等に学習・検証データをサンプリングできるよう、チャンク単位でシャッフルします。
# ※再現性のために手動で乱数ジェネレータのシードを固定します。
g = torch.Generator().manual_seed(1337)
shuffled_indices = torch.randperm(num_chunks, generator=g)
chunks = chunks[shuffled_indices]

# シャッフルされたチャンクを訓練データ (90%) と検証データ (10%) に分割し、GPUへ配置します。
n = int(0.9 * num_chunks)
train_chunks = chunks[:n].to(device)
val_chunks = chunks[n:].to(device)

print(f"データ分割情報 ➔ 総チャンク数: {num_chunks} | 訓練用チャンク: {len(train_chunks)} | 検証用チャンク: {len(val_chunks)}")

def get_batch(split):
    chunks_split = train_chunks if split == 'train' else val_chunks
    # GPU上で直接ランダムに batch_size 個のチャンクを選択します
    ix = torch.randint(len(chunks_split), (batch_size,), device=device)
    batch_chunks = chunks_split[ix] # Shape: (batch_size, chunk_len)
    
    # x (入力) はチャンクの末尾1文字を除く 128文字
    # y (教師ラベル) はチャンクの先頭1文字を除く 128文字 (xから1文字未来にシフトした形)
    # ※後続のモデル内での .view() 操作（非連続テンソル不可）のバグを防ぐため、.contiguous() を呼び出します
    x = batch_chunks[:, :-1].contiguous()
    y = batch_chunks[:, 1:].contiguous()
    return x, y

# ==========================================
# 3. 損失（ロス）の推定関数
# ==========================================
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() # モデルを評価モードに切り替え
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # モデルを訓練モードに戻す
    return out

# ==========================================
# 4. モデルと最適化アルゴリズム（Optimizer）の初期化
# ==========================================
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)
model = model.to(device)

# モデルの総パラメータ数を表示
num_params = sum(p.numel() for p in model.parameters())
print(f"モデルの総パラメータ数: {num_params:,} (約 {num_params/1000000:.2f}M)")

# AdamW オプティマイザを設定
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 5. 訓練（トレーニング）ループ
# ==========================================
print("\n--- 訓練を開始します ---")
print("定期的に表示される『赤ちゃんの独り言』の成長にご注目ください！")

best_val_loss = float('inf')  # 過去最高の検証ロスを記憶する変数

for iter in range(max_iters + 1):

    # 定期的な評価と生成サンプルの出力
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        val_loss = losses['val'].item()
        print(f"\n[ステップ {iter:4d}] 訓練ロス: {losses['train']:.4f} | 検証ロス: {val_loss:.4f}")
        
        # 検証ロスが過去最低（＝一番賢い瞬間）を更新したら、その脳みそ（完全な互換オブジェクト）を保存！
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'hyperparameters': {
                    'n_embd': n_embd,
                    'block_size': block_size,
                    'n_head': n_head,
                    'n_layer': n_layer
                }
            }
            torch.save(best_checkpoint, 'baby_gpt_best.pth')
            print(f"🔥 最高知性を更新！ベストモデルを保存しました (Val Loss: {val_loss:.4f})")
        
        # モデルに「むかしむかし、」という最初の文字を与えて、続きを喋らせてみます（100文字）
        context = torch.tensor([encode("むかしむかし、")], dtype=torch.long, device=device)
        generated_ids = model.generate(context, max_new_tokens=100)[0].tolist()
        print(f"ーーー 赤ちゃんの独り言（ステップ {iter}）ーーー")
        print(decode(generated_ids))
        print("ーーーーーーーーーーーーーーーーーーーーーー")

    # 訓練用データをバッチで取得
    xb, yb = get_batch('train')

    # フォワードパスとロスの計算
    logits, loss = model(xb, yb)
    
    # バックプロパゲーション（誤差逆伝播）と重みの更新
    optimizer.zero_grad(set_to_none=True) # 勾配の初期化
    loss.backward()                       # 勾配の計算
    optimizer.step()                      # パラメータの更新

# ==========================================
# 6. 学習済みモデルの保存
# ==========================================
print("\n--- 訓練が完了しました！ ---")
# BPEトークナイザーの設定ファイルは tokenizer.json として外出しで存在するため、
# チェックポイントにはモデル構造と重み、ハイパーパラメータのみを保存します。
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'hyperparameters': {
        'n_embd': n_embd,
        'block_size': block_size,
        'n_head': n_head,
        'n_layer': n_layer
    }
}
torch.save(checkpoint, 'baby_gpt.pth')
print("学習済みモデルを 'baby_gpt.pth' に保存しました。")
