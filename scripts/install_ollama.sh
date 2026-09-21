#!/bin/bash
# 安装 Ollama: 多镜像轮询下载
set -e
cd ~
REL="v0.32.14"
FILE="ollama-linux-amd64.tgz"
MIRRORS=(
  "https://mirror.ghproxy.com/https://github.com/ollama/ollama/releases/download/${REL}/${FILE}"
  "https://ghfast.top/https://github.com/ollama/ollama/releases/download/${REL}/${FILE}"
  "https://gh-proxy.com/https://github.com/ollama/ollama/releases/download/${REL}/${FILE}"
  "https://ghproxy.net/https://github.com/ollama/ollama/releases/download/${REL}/${FILE}"
  "https://github.com/ollama/ollama/releases/download/${REL}/${FILE}"
)

if [ -x ~/ollama/bin/ollama ]; then
  echo "ollama 已存在: $(~/ollama/bin/ollama --version)"
  exit 0
fi

mkdir -p ollama
OK=0
for url in "${MIRRORS[@]}"; do
  echo "=== 尝试: $url ==="
  if curl -L --max-time 240 --connect-timeout 10 -o ollama.tgz "$url" && [ -s ollama.tgz ] && [ $(stat -c%s ollama.tgz) -gt 1000000 ]; then
    echo "下载成功: $(stat -c%s ollama.tgz) 字节"
    OK=1
    break
  else
    echo "失败, 换下一个镜像"
    rm -f ollama.tgz
  fi
done

if [ $OK -ne 1 ]; then
  echo "所有镜像均失败"
  exit 1
fi

tar -C ollama -xzf ollama.tgz
rm -f ollama.tgz
echo "=== 安装完成 ==="
~/ollama/bin/ollama --version
