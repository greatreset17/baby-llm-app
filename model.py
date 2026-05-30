import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 超小型・赤ちゃんGPT (Transformer) モデルの定義
# ==========================================

class Head(nn.Module):
    """
    シングル・セルフアテンション・ヘッド (Single Head of Self-Attention)
    各文字が「他のどの文字に注目すべきか」を計算する脳の一部です。
    """
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        # Query, Key, Value を計算するための線形変換
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # 自己回帰のためのマスク（下三角行列）。未来のトークンを見えなくする壁です。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 入力 x のサイズ: (バッチサイズ B, コンテキスト長 T, 埋め込み次元 C)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # アテンション・スコア（関連度）の計算: Q と K の内積
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # 最後にスケーリング（平方根で割る）を行うことで学習を安定させます。
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        
        # 因果関係マスク（未来の文字へのアテンションを -無限大 にしてシャットアウト）
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # ソフトマックス関数で「注目度の割合（合計1.0）」に変換
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # 注目度に基づいて、Value（価値・情報）を重み付け合計
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    マルチヘッド・アテンション (Multi-Head Attention)
    複数の異なる視点（アテンション・ヘッド）で同時に並列思考します。
    """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        # 複数のヘッドを並列に並べます
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        # 各ヘッドの出力を統合して、元の埋め込み次元に戻すための全結合層
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 各ヘッドの計算結果をチャネル(C)方向に結合 (concatenate)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 結合した出力を統合
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    フィードフォワード・ネットワーク (MLP)
    アテンションで集めた情報を整理し、じっくり「思考」する線形層です。
    """
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 次元を4倍に広げて複雑な表現を学習可能にする
            nn.GELU(),                    # 活性化関数（非線形性）
            nn.Linear(4 * n_embd, n_embd), # 元の次元に戻す
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformerブロック (Transformer Block)
    「アテンション（通信）」と「フィードフォワード（個別思考）」を1つにまとめた基本単位。
    これを何段も積み重ねてモデルを構成します。
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        # アテンション層（文字同士の関係性を探る）
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        # フィードフォワード層（個々の文字の理解を深める）
        self.ffwd = FeedForward(n_embd, dropout)
        # レイヤー正規化 (Layer Normalization)。勾配消失を防ぎ、学習をスムーズにします。
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差接続 (Residual Connections) + Pre-LN（LayerNormを先に適用）を採用
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    赤ちゃんGPTモデル本体 (Decoder-only Transformer)
    文字を受け取り、自己回帰的に「次の文字」の確率分布を出力します。
    """
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        
        # 1. トークン埋め込み層 (Token Embedding): 文字IDをベクトル表現に変換
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 2. 位置埋め込み層 (Position Embedding): 文字が「何文字目にあるか」の位置情報を追加
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 3. Transformerブロックの積み重ね
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        
        # 4. 最終レイヤー正規化
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 5. 言語モデル・ヘッド: ベクトル表現から次の「文字IDの確率（ロジット）」へ変換
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 重みの初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx と targets のサイズはどちらも (B, T) のテンソル

        # トークン埋め込み
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # 位置埋め込みの取得
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        
        # 埋め込み表現の合計
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Transformerブロックの適用
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        
        # 最終ロジットの計算（ボキャブラリ内の各文字の出現スコア）
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # 損失関数（クロスエントロピー損失）の計算
            # PyTorchのF.cross_entropyは (B*T, C) の形状を期待するため変形します
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        自己回帰的に新しい文字を生成します。
        """
        for _ in range(max_new_tokens):
            # コンテキストサイズ（block_size）を超えないように入力を切り取ります
            idx_cond = idx[:, -self.block_size:]
            # 予測の実行
            logits, loss = self(idx_cond)
            # 最後の文字位置の予測（ロジット）のみを取得
            logits = logits[:, -1, :] # (B, vocab_size)
            # ソフトマックスを適用して次の文字の確率分布に変換
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # 確率分布に基づいて次の文字をサンプリング（多項分布サンプリング）
            # これにより、常に一番確率の高い文字だけでなく、多様な文章が生成されます
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 生成した文字をこれまでの入力の末尾に結合して、次の予測へ進みます（自己回帰）
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
