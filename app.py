import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from tokenizers import Tokenizer
from model import GPTLanguageModel

# ==========================================
# 1. ページ設定とデザイン（CSS）の注入
# ==========================================
st.set_page_config(
    page_title="赤ちゃん文豪AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Fonts の読み込みとプレミアムデザインの CSS インジェクション
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+JP:wght@300;400;600;800&display=swap');
    
    /* フォントファミリーの適用 */
    html, body, [class*="css"], .stMarkdown, p, span, li, button, input {
        font-family: 'Outfit', 'Noto Sans JP', sans-serif !important;
    }

    /* メインヘッダーの美しいグラデーションタイトル */
    .main-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        text-align: center;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: #88888b;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* スペックバッジのコンテナ */
    .spec-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .spec-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    
    .spec-label {
        color: #999;
    }
    
    .spec-value {
        font-weight: 600;
        color: #FF8E53;
    }

    /* メッセージ要素の微調整と微小なフェードインアニメーション */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.4s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. モデルとトークナイザーのロード（キャッシュ）
# ==========================================
@st.cache_resource
def load_model_and_tokenizer():
    # デバイス自動検出
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model_path = "baby_gpt_best.pth"
    if not os.path.exists(model_path):
        model_path = "baby_gpt.pth"
        if not os.path.exists(model_path):
            return None, None, device
            
    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    hparams = checkpoint['hyperparameters']
    
    # トークナイザー読み込み
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        return None, None, device
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # モデル構築
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
    
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# ==========================================
# 3. Token Streamingジェネレータ (差分デコード搭載)
# ==========================================
@torch.no_grad()
def generate_stream(model, tokenizer, prompt, max_new_tokens, temperature=0.8, top_k=40, device="cpu"):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    prompt_ids = encode(prompt)
    
    # 入力テンソル化 (Batch=1)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # 差分デコードのための状態保持 (UTF-8の部分マルチバイト境界化を防ぐ)
    accumulated_ids = []
    current_text = ""
    
    for _ in range(max_new_tokens):
        # 視野 (block_size) を超えないように入力をクロップ
        idx_cond = idx[:, -model.block_size:]
        
        # 予測
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Shape: (1, vocab_size)
        
        # 温度 (Temperature) の適用
        logits = logits / max(temperature, 1e-5)
        
        # Top-K フィルタリングの適用
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 確率分布化
        probs = F.softmax(logits, dim=-1)
        
        # サンプリング
        idx_next = torch.multinomial(probs, num_samples=1)
        token_id = idx_next.item()
        
        # 文脈と累積の更新
        accumulated_ids.append(token_id)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 差分デコードの実行
        full_decoded_text = tokenizer.decode(accumulated_ids)
        new_chars = full_decoded_text[len(current_text):]
        current_text = full_decoded_text
        
        # リアルタイムにストリーミング出力
        yield new_chars

# ==========================================
# 4. サイドバーの実装（コントロール ＆ テンプレート）
# ==========================================
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 10px;"><h2 style="color: #FF8E53; font-weight:800; font-size: 1.5rem; margin:0;">✍️ 赤ちゃん文豪AI</h2></div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #888; text-align:center; font-size:0.85rem; margin-bottom:20px;">あなたのMacが育てた、世界に一つの脳細胞</div>', unsafe_allow_html=True)
    
    # A. モデルスペック情報
    st.markdown('<h3 style="font-size:1.0rem; font-weight:600; margin-bottom:10px; color:#ddd;">🧠 モデル構成スペック</h3>', unsafe_allow_html=True)
    
    if model is not None:
        st.markdown(f"""
        <div class="spec-card">
            <div class="spec-item">
                <span class="spec-label">総パラメータ数</span>
                <span class="spec-value">24.45 M (24,454,816)</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">コンテキスト視野</span>
                <span class="spec-value">{model.block_size} トークン</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Transformer層数</span>
                <span class="spec-value">12 層</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">アテンションヘッド</span>
                <span class="spec-value">6 ヘッド</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">BPE語彙数 (Vocab)</span>
                <span class="spec-value">{tokenizer.get_vocab_size()} 語</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">稼働デバイス</span>
                <span class="spec-value" style="color: #5eff5e;">{device.type.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("モデルまたはトークナイザーがロードできません。'baby_gpt_best.pth' または 'baby_gpt.pth'、および 'tokenizer.json' がカレントディレクトリに存在することを確認してください。")

    # B. 推論パラメータ制御
    st.markdown('<h3 style="font-size:1.0rem; font-weight:600; margin-top:20px; margin-bottom:10px; color:#ddd;">⚙️ 生成オプション</h3>', unsafe_allow_html=True)
    
    temperature = st.slider(
        "創造性 (Temperature)",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.05,
        help="値が大きいほど創造的で予測不可能な表現になり、小さいほど安定した(手堅い)文章になります。"
    )
    
    top_k = st.slider(
        "Top-K フィルタリング",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        help="確率上位の指定語彙のみからサンプリングを行うことで、不自然な文字や文法崩れをシャットアウトします。"
    )
    
    max_new_tokens = st.slider(
        "最大生成文字数 (Max Tokens)",
        min_value=50,
        max_value=400,
        value=200,
        step=10,
        help="一回のプロンプト入力でAIが追加生成する長さの上限です。"
    )
    
    # C. ワンクリックテンプレート
    st.markdown('<h3 style="font-size:1.0rem; font-weight:600; margin-top:25px; margin-bottom:10px; color:#ddd;">📜 文豪テンプレート</h3>', unsafe_allow_html=True)
    st.markdown('<div style="color: #888; font-size:0.8rem; margin-bottom:10px;">クリックするとAIが即座にその続きをストリーミング生成します：</div>', unsafe_allow_html=True)
    
    presets = {
        "🐱 漱石風（吾輩は猫である）": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所で",
        "🏃 太宰風（走れメロス）": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、",
        "🌌 賢治風（銀河鉄道の夜）": "ジョバンニは、学校の授業が終わると、街を通って大きな活版所に上がっていきました。入るとすぐにインクの匂いが",
        "🦊 南吉風（ごん狐）": "むかしむかし、あるところに、おじいさんとおばあさんが住んでいました。おじいさんは山へ芝刈りに、おばあさんは川へ",
        "🕵️ 乱歩風（明智小五郎）": "「おい、明智君、いったいこの事件をどう思うかね？」波越警部は焦ったように、枕元でじっと考え込んでいる探偵の"
    }
    
    # テンプレートボタンの処理
    for label, text in presets.items():
        if st.button(label, use_container_width=True):
            st.session_state.temp_prompt = text
            st.session_state.trigger_preset = True

    # D. チャット履歴の消去
    st.markdown("<hr style='margin: 25px 0 15px 0; opacity: 0.15;'/>", unsafe_allow_html=True)
    if st.button("💬 会話をリセット", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 5. メインチャットインターフェース
# ==========================================
st.markdown('<h1 class="main-title">✍️ 赤ちゃん文豪 AI</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BPEサブワードと24.4Mパラメータで蘇る、ブラウザ上の近代日本文学クロスオーバー</div>', unsafe_allow_html=True)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初回起動時のウェルカムメッセージ
if not st.session_state.messages:
    welcome_text = """はじめまして！私はあなたに育てていただいた **「赤ちゃん文豪AI」** です。
    
夏目漱石や太宰治、江戸川乱歩などの名作（計12.5MB）から日本語を学んだ、2,445万パラメータのトランスフォーマー脳細胞（最高知性ベストモデル）を持っています。
    
お好きな書き出し（プロンプト）を入力していただくか、サイドバーの **「文豪テンプレート」** をクリックしてみてください。私の中に眠る文豪たちのエッセンスが交差し、続きの文章を紡ぎ出します！"""
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome_text
    })

# 過去の会話履歴をレンダリング
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ワンクリックテンプレートのトリガー判定
preset_triggered = False
if "trigger_preset" in st.session_state and st.session_state.trigger_preset:
    user_prompt = st.session_state.temp_prompt
    st.session_state.trigger_preset = False  # フラグリセット
    preset_triggered = True

# チャット入力フォーム
if not preset_triggered:
    user_prompt = st.chat_input("赤ちゃん文豪に続きを書かせたいプロンプトを入力してください...")

# プロンプトが送信されたときの処理
if user_prompt:
    # ユーザーの入力を表示
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # アシスタント（AI）の生成処理
    with st.chat_message("assistant"):
        if model is None:
            st.error("モデルが正常にロードされていません。カレントディレクトリにモデルファイルが配置されているかご確認ください。")
        else:
            # リアルタイムにストリーミング生成して表示
            # ※ st.write_stream はジェネレータが yield した文字を即座にUIに反映させます
            full_response = st.write_stream(
                generate_stream(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device
                )
            )
            
            # 生成結果を会話履歴に保存
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # テンプレート選択直後の場合は再読み込みして表示を更新
            if preset_triggered:
                st.rerun()
