# GitHub 发布说明

本地稳定发布分支和 `v2` 标签准备完成后，在 GitHub 创建一个空仓库。不要勾选自动生成
README、`.gitignore` 或许可证，以免首次推送产生无关冲突。

## 绑定远程仓库

```bash
git remote add origin https://github.com/<username>/<repository>.git
git remote -v
```

如果已经存在 `origin`，使用下面的命令修改地址：

```bash
git remote set-url origin https://github.com/<username>/<repository>.git
```

## 发布代码和标签

```bash
git push -u origin main
git push origin v2
```

随后在 GitHub 的 Releases 页面选择 `v2` 创建 Release，标题建议使用
`STUDY RAG V1-A v2`，发布说明使用 `docs/v2-release-notes.md`。

## 发布前检查

```bash
git status
git show --stat v2
python -m pytest -q
```

确认仓库中没有 `.env`、私人 PDF、`vector_db/`、`agent_state/`、模型权重或运行数据库。
首次公开发布前还应自行选择许可证；未添加许可证时，默认保留全部著作权。
