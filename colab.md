```
import os
from google.colab import userdata

# 从 Colab 隐私设置中获取并注入
try:
    os.environ["NGROK_TOKEN"] = userdata.get('NGROK_TOKEN')
    print("✅ Token 已从 Secrets 安全注入")
    !git clone https://github.com/Maxwin-z/colab-app-proxy
    os.chdir('/content')
    os.chdir('colab-app-proxy')
    !bash ./start.sh
except userdata.SecretNotFoundError:
    print("❌ 未在 Secrets 中找到 NGROK_TOKEN")
```
