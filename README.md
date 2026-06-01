# openbmb-MiniCPM5-1B-container

[openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) をローカルで動かし、ネットワーク遮断・最小権限の Docker コンテナ内で CLI 対話するアプリケーションです。ホストに NVIDIA GPU がある場合はビルド時に自動検出し、GPU 推論用イメージを生成します。

## 機能

- ビルド時にモデルをイメージへ焼き込み、実行時は完全オフライン
- モデル推論プロセス（`model` ユーザー）と対話 CLI（`chat` ユーザー）を分離
- Unix ソケット経由の JSON プロトコル（HTTP/API サーバーなし）
- パスワード認証後のみ CLI 対話を開始
- オプションで [MiniCPM5 ツール呼び出し](https://huggingface.co/openbmb/MiniCPM5-1B)（ホワイトリストのみ、デフォルト全有効、`--tools` で調整）
- ビルド時に `nvidia-smi` が使えるホストでは CUDA 入りイメージを自動生成し、実行時は GPU で推論（応答が速い）

## 前提

- Docker / Docker Compose
- ディスク空き 約 5GB 以上（モデル + イメージ）
- **CPU**: GPU 不要（応答は遅め）。macOS の Docker では通常こちら
- **GPU（任意）**: NVIDIA ドライバ + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。ビルドホストで `nvidia-smi` が成功すると GPU イメージが選ばれる

## クイックスタート

```bash
# 1. ビルド（初回はモデル DL のため 10〜20 分程度）
CHAT_PASSWORD='your-secret' ./scripts/build.sh

# 2. 起動してログイン → 対話
./scripts/run.sh

# 3. 停止
./scripts/stop.sh
```

ビルド時、ローカルにモデルがあれば自動で `.models/MiniCPM5-1B/` へ配置し Hugging Face からの再 DL をスキップします。配置元の優先順位は次のとおりです。

1. 既に完全な `.models/MiniCPM5-1B/`（または `MODEL_DIR` で指定したパス）
2. Hugging Face キャッシュ（`~/.cache/huggingface/hub/...`）
3. 既存の Docker イメージ `minicpm5-1b-chat:latest`

初回ビルドで Docker 内 DL した場合、成功後にモデルを `.models/MiniCPM5-1B/` へ保存するため、2 回目以降のビルドは高速になります。

```bash
# 配置のみ確認・実行
./scripts/prepare_model.sh

# 任意の保存先
MODEL_DIR=/data/MiniCPM5-1B CHAT_PASSWORD='your-secret' ./scripts/build.sh
```

### GPU 推論（NVIDIA）

ホストで `nvidia-smi` が成功する場合、`./scripts/build.sh` は `runtime-gpu` ターゲットで CUDA 12.4 + PyTorch (cu124) 入りイメージをビルドし、ラベル `minicpm.inference=gpu` を付与します。`./scripts/run.sh` はそのラベルを見て `docker-compose.gpu.yml` を自動マージし、コンテナに GPU を割り当てます。

```bash
# GPU なしマシンから GPU サーバー向けイメージを明示ビルド
MINICPM_GPU=1 CHAT_PASSWORD='your-secret' ./scripts/build.sh

# 実行時に GPU compose を明示（イメージラベルと併用可）
MINICPM_GPU=1 ./scripts/run.sh

# CPU イメージで誤って GPU compose を付けない
MINICPM_GPU=0 ./scripts/run.sh
```

起動後、`docker logs minicpm5-1b-chat` に `CUDA available` が出ていれば GPU 推論です。GPU 時もオフライン・プロセス分離・`network_mode: none`（デフォルト）は CPU 版と同じです。

コード変更後にイメージを作り直さず反映する場合:

```bash
./scripts/sync.sh
```

### 対話コマンド

ログイン成功後、生成設定（トークン量・推論モード・サンプリング・応答言語）を対話形式で指定できます。各項目の説明と有効範囲（閾値）が表示されたあと入力します。Enter のみでデフォルト値が使われ、セッション中はその設定が維持されます。

| 設定 | 説明 |
|------|------|
| `response_language` | 応答言語。`auto`（デフォルト）でユーザー入力言語に追従。`ja` / `en` 等で固定も可 |
| 環境変数 `CHAT_RESPONSE_LANGUAGE` | ログイン時プロンプトのデフォルト値（例: `ja`） |
| `enabled_tools` | 有効ツール ID（カンマ区切り）。省略時はホワイトリスト全ツール有効 |
| 環境変数 `CHAT_TOOLS` | ログイン引数 `--tools` 未指定時の指定（例: `-http_get,-web_search` または `calculate,count_text`） |

### ツール（`--tools`）

デフォルトではホワイトリスト登録済みの全ツールが有効です。ログイン時に禁止・限定できます。

```bash
# 全ツール ON（デフォルト）
./scripts/run.sh

# ネットワークツールのみ禁止（オフラインのまま）
./scripts/run.sh -- --tools=-http_get,-web_search

# 許可リストで限定
./scripts/run.sh -- --tools calculate,current_datetime

# 全無効
./scripts/run.sh -- --tools none

# HTTP ツール（GET のみ）— ネットワーク override が必要
./scripts/run.sh --network -- --tools -calculate
```

| ID | 種別 | 説明 |
|----|------|------|
| `calculate` | ローカル | 安全な四則演算式の評価 |
| `current_datetime` | ローカル | UTC 現在時刻（ISO 8601） |
| `count_text` | ローカル | 文字数・単語数・行数 |
| `convert_units` | ローカル | 長さ・質量・温度・バイト換算 |
| `http_get` | ネットワーク | URL を **GET のみ**で取得（SSRF 対策あり） |
| `web_search` | ネットワーク | Wikipedia + DuckDuckGo Instant Answer（**GET JSON**）。任意で `CHAT_SEARX_BASE_URL` に SearXNG |

`web_search` は HTML スクレイピングを使わず、Docker から CAPTCHA なしで使える JSON API のみ利用します。一般 Web 全文検索が必要な場合は、信頼できる自前 SearXNG を `CHAT_SEARX_BASE_URL`（例: `https://searx.example.com`）で指定してください。

思考モードはログイン時に **Hybrid（auto）固定** です。`apply_chat_template` へ `enable_thinking` を渡さず、公式 MiniCPM5-1B と同様にモデルが思考の要否を判断します。運用者向けに `CHAT_ENABLE_THINKING=0|1` で上書きできます。`web_search` では `max_new_tokens` を 256 程度にすると要約しやすくなります。パーサーは XML に加え `{"name":"...","arguments":{...}}` 形式の JSON も受け付けます。

| コマンド | 説明 |
|---------|------|
| `/help` | ヘルプ表示 |
| `/clear` | 会話履歴クリア |
| `/exit` | 終了 |

## 開発

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not integration"
ruff check src tests
pre-commit run --all-files
```

## セキュリティ設計

### コンテナ hardening

| 設定 | 目的 |
|------|------|
| `network_mode: none` | 外部通信の完全遮断 |
| `read_only: true` | root FS 書き込み禁止 |
| `cap_drop: [ALL]` | Linux capability 最小化 |
| `no-new-privileges:true` | 権限昇格防止 |
| `tmpfs: [/tmp, /run]` | 揮発性領域のみ書き込み可 |
| ポート公開なし | ネットワークサービス不提供 |

### プロセス分離

| ユーザー | 権限 |
|---------|------|
| `model` (UID 1001) | `/models` 読み取り専用、Unix ソケット待受 |
| `chat` (UID 1002) | ソケット接続のみ（モデル weights へ直接アクセス不可） |

### 推論の制限

- `trust_remote_code=False`（任意コード実行を禁止）
- ツールは **ホワイトリストのみ**（デフォルト全有効、`-id` で禁止または許可リストで限定）
- MiniCPM5 の `<function>` / `<param>` はトークナイザ上の special token のため、推論デコードでは `skip_special_tokens=False` が必須
- ツール実行は **GET のみ**・SSRF ブロック・レスポンスサイズ上限（HTTP ツール）
- メッセージ数・文字数に上限。`max_new_tokens` はモデルコンテキスト長（131,072）まで指定可能（ログイン時デフォルト 1000。必要なら明示的に下げる）。プロンプトが長い場合は残りコンテキストまで推論時に自動調整。CPU 推論では大きな値は時間・メモリコストが大きい
- 平文パスワードはイメージに含めず、bcrypt ハッシュのみ埋め込み

### セキュリティチェックリスト

- [x] デフォルトは実行時ネットワーク遮断（`network_mode: none`）
- [x] HTTP ツールは `docker-compose.network.yml` / `--network` で明示的に egress を許可
- [x] `chat` ユーザーは `/models` を読めない（`chmod 750`, `model` 所有）
- [x] root FS read-only
- [x] HTTP リスナーなし（外向きクライアントのみ、任意）
- [x] ツール名ホワイトリスト・ラウンド上限・`trust_remote_code=False`
- [x] 平文パスワードを git / イメージに含めない

## アーキテクチャ

```
Host: scripts/build.sh → prepare_model → docker build (CPU or GPU target from nvidia-smi)
Host: scripts/run.sh   → docker compose [+ gpu.yml] up → docker exec chat-login

Container (offline):
  entrypoint → model_server (user: model, device_map auto on GPU) → /run/model.sock
  chat-login → chat_cli (user: chat) → Unix socket
```

## ライセンス

Apache License 2.0（本リポジトリ）。モデルは [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) のライセンスに従います。
