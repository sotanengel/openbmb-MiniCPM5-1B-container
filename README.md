# openbmb-MiniCPM5-1B-container

[openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) をローカル CPU 環境で動かし、ネットワーク遮断・最小権限の Docker コンテナ内で CLI 対話するアプリケーションです。

## 機能

- ビルド時にモデルをイメージへ焼き込み、実行時は完全オフライン
- モデル推論プロセス（`model` ユーザー）と対話 CLI（`chat` ユーザー）を分離
- Unix ソケット経由の JSON プロトコル（HTTP/API サーバーなし）
- パスワード認証後のみ CLI 対話を開始
- オプションで [MiniCPM5 ツール呼び出し](https://huggingface.co/openbmb/MiniCPM5-1B)（ホワイトリストのみ、`--tools` で指定）

## 前提

- Docker / Docker Compose
- ディスク空き 約 5GB 以上（モデル + イメージ）
- CPU 推論（GPU 不要、応答は遅め）

## クイックスタート

```bash
# 1. ビルド（初回はモデル DL のため 10〜20 分程度）
CHAT_PASSWORD='your-secret' ./scripts/build.sh

# 2. 起動してログイン → 対話
./scripts/run.sh

# 3. 停止
./scripts/stop.sh
```

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
| `enabled_tools` | 有効ツール ID（カンマ区切り）。省略時は `none`（ツール無効） |
| 環境変数 `CHAT_TOOLS` | ログイン引数 `--tools` 未指定時のデフォルト（例: `calculate,count_text`） |

### ツール（`--tools`）

デフォルトではツールは無効です。ログイン時に有効化します。

```bash
# ローカル安全ツールのみ（オフラインのまま）
./scripts/run.sh -- --tools calculate,current_datetime

# HTTP ツール（GET のみ）— ネットワーク override が必要
./scripts/run.sh --network -- --tools http_get,web_search
```

| ID | 種別 | 説明 |
|----|------|------|
| `calculate` | ローカル | 安全な四則演算式の評価 |
| `current_datetime` | ローカル | UTC 現在時刻（ISO 8601） |
| `count_text` | ローカル | 文字数・単語数・行数 |
| `convert_units` | ローカル | 長さ・質量・温度・バイト換算 |
| `http_get` | ネットワーク | URL を **GET のみ**で取得（SSRF 対策あり） |
| `web_search` | ネットワーク | DuckDuckGo HTML へ **GET のみ**で検索 |

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
- ツールは **ホワイトリストのみ**（デフォルト無効、`--tools` で明示有効化）
- ツール実行は **GET のみ**・SSRF ブロック・レスポンスサイズ上限（HTTP ツール）
- メッセージ数・文字数・ `max_new_tokens` に上限
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
Host: scripts/build.sh → docker build (HF download)
Host: scripts/run.sh   → docker compose up → docker exec chat-login

Container (offline):
  entrypoint → model_server (user: model) → /run/model.sock
  chat-login → chat_cli (user: chat) → Unix socket
```

## ライセンス

Apache License 2.0（本リポジトリ）。モデルは [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) のライセンスに従います。
