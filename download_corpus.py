import urllib.request
import re
import os

# 青空文庫の作品URLリスト (HTML版)
# ※青空文庫のHTMLは Shift_JIS (cp932) でエンコードされています
WORKS = {
    "ごん狐": "https://www.aozora.gr.jp/cards/000121/files/628_14895.html",
    "蜘蛛の糸": "https://www.aozora.gr.jp/cards/000879/files/92_14545.html",
    "注文の多い料理店": "https://www.aozora.gr.jp/cards/000081/files/43754_17659.html",
    "走れメロス": "https://www.aozora.gr.jp/cards/000035/files/1567_14913.html",
    "山月記": "https://www.aozora.gr.jp/cards/000119/files/624_14544.html",
    "羅生門": "https://www.aozora.gr.jp/cards/000879/files/127_15260.html",
    "銀河鉄道の夜": "https://www.aozora.gr.jp/cards/000081/files/456_15050.html",
    "檸檬": "https://www.aozora.gr.jp/cards/000074/files/424_19826.html",
    "人間失格": "https://www.aozora.gr.jp/cards/000035/files/301_14912.html",
    "こころ": "https://www.aozora.gr.jp/cards/000148/files/773_14560.html",
    "吾輩は猫である": "https://www.aozora.gr.jp/cards/000148/files/789_14547.html",
    "舞姫": "https://www.aozora.gr.jp/cards/000129/files/2058_19628.html",
    # === 新規追加分 (10MB突破用の検証済み長編傑作群) ===
    "三四郎": "https://www.aozora.gr.jp/cards/000148/files/58842_76759.html",
    "それから": "https://www.aozora.gr.jp/cards/000148/files/56143_50921.html",
    "坊っちゃん": "https://www.aozora.gr.jp/cards/000148/files/752_14964.html",
    "門": "https://www.aozora.gr.jp/cards/000148/files/785_14941.html",
    "道草": "https://www.aozora.gr.jp/cards/000148/files/783_14833.html",
    "彼岸過迄": "https://www.aozora.gr.jp/cards/000148/files/1077_14950.html",
    "斜陽": "https://www.aozora.gr.jp/cards/000035/files/1565_8559.html",
    "李陵": "https://www.aozora.gr.jp/cards/000119/files/1737_14534.html",
    "草枕": "https://www.aozora.gr.jp/cards/000148/files/1048_14896.html",
    "硝子戸の中": "https://www.aozora.gr.jp/cards/000148/files/1066_14834.html",
    "坑夫": "https://www.aozora.gr.jp/cards/000148/files/772_14954.html",
    "津軽": "https://www.aozora.gr.jp/cards/000035/files/2275_15264.html",
    "黄金仮面": "https://www.aozora.gr.jp/cards/001779/files/57241_73705.html",
    "孤島の鬼": "https://www.aozora.gr.jp/cards/001779/files/57849_71930.html"
}

def clean_aozora_html(html_content):
    """
    青空文庫のHTMLからルビやメタデータを除去し、きれいなプレーンテキストを抽出します。
    """
    # 1. main_text クラスの div の開始位置を特定
    start_tag = '<div class="main_text">'
    start_idx = html_content.find(start_tag)
    if start_idx != -1:
        # main_text より後ろをスライス
        text = html_content[start_idx + len(start_tag):]
    else:
        text = html_content

    # 2. フッター（bibliographical_information / 図書カードなど）の開始位置を特定してカット
    # ネストされた div の影響を受けずに、本文が終わった部分で切り出します
    end_tags = [
        '<div class="bibliographical_information">',
        '<div class="bibliographical_information"',
        '<div class="card">',
        '<div class="annotation">'
    ]
    for end_tag in end_tags:
        end_idx = text.find(end_tag)
        if end_idx != -1:
            text = text[:end_idx]
            break

    # 3. ルビのフリガナ部分 (<rt>...</rt>) と補助文字 (<rp>...</rp>) を完全に消去
    text = re.sub(r'<rt>.*?</rt>', '', text, flags=re.DOTALL)
    text = re.sub(r'<rp>.*?</rp>', '', text, flags=re.DOTALL)

    # 4. その他のすべてのHTMLタグを除去
    text = re.sub(r'<[^>]+>', '', text)

    # 5. 青空文庫特有の注記 ［＃...］ を消去
    text = re.sub(r'［＃[^］]+］', '', text)

    # 6. 空白行の整理とトリミング
    lines = [line.strip() for line in text.splitlines()]
    # 空行は除き、綺麗な行だけで再構成
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)

def main():
    print("=== 青空文庫から学習データのダウンロードを開始します ===")
    
    # 既存の昔話コーパス（元々のコーパス）の読み込み
    original_corpus = ""
    original_path = "corpus.txt"
    if os.path.exists(original_path):
        # 一旦バックアップをとっておく、または元のデータをロード
        with open(original_path, 'r', encoding='utf-8') as f:
            original_corpus = f.read()
        print("既存の昔話コーパスを読み込みました。")
    else:
        print("警告: 既存の corpus.txt が見つかりませんでした。新規作成します。")

    merged_text = original_corpus.strip() + "\n\n"

    for title, url in WORKS.items():
        print(f"・『{title}』をダウンロード中... ({url})")
        try:
            # リクエストの送信 (User-Agentを設定してクローリング)
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            )
            with urllib.request.urlopen(req) as response:
                raw_data = response.read()
                # Aozora Bunko HTML is encoded in Shift_JIS (cp932)
                html_content = raw_data.decode('cp932', errors='ignore')
            
            # テキストのクレンジング
            cleaned_text = clean_aozora_html(html_content)
            print(f"  ➔ クレンジング完了! (文字数: {len(cleaned_text)}文字)")
            
            # マージ
            merged_text += f"\n\n# --- {title} ---\n" + cleaned_text
            
        except Exception as e:
            print(f"  ❌ エラーが発生しました: {e}")

    # 重複改行の調整と保存
    merged_text = re.sub(r'\n{3,}', '\n\n', merged_text).strip() + "\n"

    with open(original_path, 'w', encoding='utf-8') as f:
        f.write(merged_text)

    print("\n=== コーパスの拡張が完了しました！ ===")
    print(f"保存先: {os.path.abspath(original_path)}")
    print(f"最終ファイルサイズ: {os.path.getsize(original_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
