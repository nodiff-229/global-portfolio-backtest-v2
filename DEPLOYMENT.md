# Streamlit Cloud 部署指南

## 一键部署步骤

### 1. 访问 Streamlit Cloud
打开：https://docs.streamlit.io/deploy

### 2. 登录 GitHub
- 点击 **"Sign in with GitHub"**
- 授权 Streamlit 访问你的 GitHub

### 3. 添加应用
- 点击 **"New app"**
- 选择仓库：`nodiff-229/global-portfolio-backtest-v2`
- 分支：`master`
- 主文件：`app.py`

### 4. 点击部署
- 点击 **"Deploy!"**
- 等待 1-2 分钟

### 5. 获取 URL
部署完成后，你会得到类似这样的 URL：
```
https://global-portfolio-backtest-v2-nodiff.streamlit.app
```

---

## 部署后检查

- ✅ 配置面板显示正常
- ✅ ETF 数据能加载（yfinance 免费数据）
- ✅ 回测结果图表正常显示
- ✅ 下载功能可用

---

## 注意事项

### 免费额度
- 每月 700 小时运行时间
- 24 小时无访问会进入休眠
- 首次访问需等待 30-60 秒唤醒

### 数据源
- 使用 yfinance（雅虎财经）- 免费，无需 API key
- 数据覆盖全球主要 ETF

### 性能
- 回测 1998-2026 约需 10-20 秒
- 云端服务器性能可能不如本地

---

## 部署完成后

把新的 Streamlit URL 发给 nodiff，他会保存到记忆里！
