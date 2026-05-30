import os
import torch
from model import GPTLanguageModel

# デバイスの自動検出
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==========================================
# 1. 学習済みモデルのロード
# ==========================================
# 過去最低検証ロスを記録したベストモデルを優先ロードし、無ければ最終モデルをロードします
model_path = "baby_gpt_best.pth"
if not os.path.exists(model_path):
    model_path = "baby_gpt.pth"
    if not os.path.exists(model_path):
        print("エラー: 学習済みモデル（'baby_gpt_best.pth' または 'baby_gpt.pth'）が見つかりません。")
        print("先に 'python3 train.py' を実行してモデルを訓練してください。")
        exit(1)

print(f"モデル '{model_path}'（最高知性ベストモデル）をロード中...")
checkpoint = torch.load(model_path, map_location=device)

# BPEトークナイザーのロード
from tokenizers import Tokenizer
tokenizer_path = "tokenizer.json"
if not os.path.exists(tokenizer_path):
    print(f"エラー: トークナイザー '{tokenizer_path}' が見つかりません。先に tokenizer_train.py を実行してください。")
    exit(1)

tokenizer = Tokenizer.from_file(tokenizer_path)

# 保存されたパラメータの復元
vocab_size = checkpoint['vocab_size']
hparams = checkpoint['hyperparameters']

encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids  # 文字列 -> IDリスト
decode = lambda l: tokenizer.decode(l)                                # IDリスト -> 文字列

# モデル構造の再構築
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=hparams['n_embd'],
    block_size=hparams['block_size'],
    n_head=hparams['n_head'],
    n_layer=hparams['n_layer']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("モデルのロードが完了しました！")
print("--------------------------------------------------")
print("★ 赤ちゃんLLM テキスト生成インターフェース ★")
print("お好きな書き出し（プロンプト）を入力すると、赤ちゃんLLMが続きを喋ります。")
print("（※ひらがなや、昔話に出てきそうな漢字を使うと上手く動きやすいです）")
print("終了するには 'exit' または 'quit' と入力してください。")
print("--------------------------------------------------\n")

while True:
    try:
        # プロンプトの入力
        prompt = input("プロンプトを入力してください (例: むかしむかし、): ")
        if prompt.strip().lower() in ['exit', 'quit']:
            print("生成インターフェースを終了します。お疲れ様でした！")
            break
            
        if not prompt:
            prompt = "むかしむかし、" # デフォルト値
            print(f"入力が空だったため、デフォルトの「{prompt}」を使用します。")

        # ※バイトレベルBPEを採用しているため、未知文字エラーは発生せず、安全にサブワード/バイト単位にエンコードされます。

        # 生成する文字数を設定
        length_input = input("生成する文字数を入力してください (デフォルト: 200): ")
        try:
            max_new_tokens = int(length_input) if length_input.strip() else 200
        except ValueError:
            max_new_tokens = 200
            print("数字が検出されなかったため、デフォルトの 200 文字を生成します。")

        # テンソルに変換して生成
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        print(f"\n--- 生成開始 (プロンプト: 「{prompt}」) ---")
        
        # モデルに入力して自己回帰的にテキストを生成
        generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
        
        print(decode(generated_ids))
        print("--- 生成完了 ---\n")

    except KeyboardInterrupt:
        print("\n生成インターフェースを終了します。")
        break
