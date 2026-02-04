# Local-TTS App Proxy

启动脚本，用于运行 App Proxy 和 Ngrok。

## 快速开始

```bash
# 启动 App Proxy + Ngrok
sh start.sh

# 仅本地运行（不启动 Ngrok）
sh start.sh --local
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--local` | 跳过 Ngrok，仅在本地运行 |

App Proxy 运行在 **8000** 端口。

## 停止服务

按 `Ctrl+C` 即可停止所有服务。
