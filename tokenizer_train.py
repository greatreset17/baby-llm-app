import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing

def main():
    print("=== BPEサブワード・トークナイザーの訓練を開始します ===")
    
    corpus_path = "corpus.txt"
    if not os.path.exists(corpus_path):
        print(f"エラー: '{corpus_path}' が見つかりません。先に corpus.txt を作成してください。")
        return

    # ボキャブラリサイズの設定 (3MBのデータに対して、4000がノイズと単語抽出のベストバランスです)
    vocab_size = 4000
    
    # バイトレベルBPEトークナイザーの初期化
    # ※バイトレベルにすることで、未知の漢字が来ても絶対にクラッシュせず安全に分解処理できます。
    tokenizer = ByteLevelBPETokenizer(lowercase=False)
    
    print("トークナイザーをデータに基づいて学習中...")
    tokenizer.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=2,  # 2回以上出現する単語・文字のペアを統合
        special_tokens=[
            "<s>",     # 文頭 (Start of String)
            "<pad>",   # パディング (Padding)
            "</s>",    # 文末 (End of String)
            "<unk>",   # 未知語 (Unknown)
            "<mask>"   # マスク (Mask)
        ]
    )
    
    # テンプレートプロセッサの設定（必要に応じて特殊トークンを自動付与するため）
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    # トークナイザー設定を保存
    output_path = "tokenizer.json"
    tokenizer.save(output_path)
    print(f"➔ トークナイザーの訓練が完了し、'{output_path}' に保存されました！")
    print(f"実際の語彙数 (Vocab Size): {tokenizer.get_vocab_size()}")
    print("--------------------------------------------------")
    
    # === トークン分割のテスト検証 ===
    test_text = "むかしむかし、ジョバンニはおじいさんと川へ行った。メロスは激怒した。"
    print(f"■ テストテキスト: 「{test_text}」")
    
    # エンコード（ID列への変換）
    encoded = tokenizer.encode(test_text)
    print(f"➔ トークン分割結果:")
    print(encoded.tokens)
    print(f"➔ トークンID列 (長さ {len(encoded.ids)}):")
    print(encoded.ids)
    
    # デコード（文字列への復元）
    decoded = tokenizer.decode(encoded.ids)
    print(f"➔ デコード（復元結果）:")
    print(f"「{decoded}」")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
