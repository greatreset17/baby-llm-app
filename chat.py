import torch
from tokenizers import Tokenizer
from model import GPTLanguageModel
import os

# デバイスの自動検出
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"使用中のデバイス: {device}")

# トークナイザーのロード
tokenizer_path = "tokenizer.json"
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError("tokenizer.json が見つかりません。")
tokenizer = Tokenizer.from_file(tokenizer_path)
encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
decode = lambda l: tokenizer.decode(l)

# ファインチューニング済みモデルのロード
model_path = "baby_gpt_chat.pth"
if not os.path.exists(model_path):
    print(f"'{model_path}' が見つかりません。まずは finetune.py を実行してください。")
    exit()

checkpoint = torch.load(model_path, map_location=device)
hparams = checkpoint['hyperparameters']
vocab_size = checkpoint['vocab_size']

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=hparams['n_embd'],
    block_size=hparams['block_size'],
    n_head=hparams['n_head'],
    n_layer=hparams['n_layer']
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"対話モデル '{model_path}' をロードしました。")
print("------------------------------------------")
print("チャットを開始します。（終了するには 'quit' または 'exit' と入力）")

while True:
    user_input = input("\nあなた: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    if not user_input.strip():
        continue
    
    # プロンプトの構築
    prompt = f"問：{user_input} 答："
    encoded_prompt = encode(prompt)
    
    # テンソル化してデバイスへ
    context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    
    # 生成するトークン数（block_size を超えないように）
    max_new_tokens = hparams['block_size'] - len(encoded_prompt)
    if max_new_tokens <= 0:
        print("AI: (入力が長すぎます。もっと短くしてください)")
        continue

    # 生成処理
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    generated_text = decode(generated_ids)
    
    # プロンプト以降の部分（AIの返答）を抽出
    if "答：" in generated_text:
        # "答："より後ろの部分を取得
        response = generated_text.split("答：", 1)[1]
        
        # 次の「問：」などが生成されたらそこでカット
        if "問：" in response:
            response = response.split("問：")[0]
        # 改行があればそこでカット（一問一答の1行分だけ出力）
        if "\n" in response:
            response = response.split("\n")[0]
            
        print(f"AI: {response.strip()}")
    else:
        print(f"AI: {generated_text}")
