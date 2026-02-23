# Smart Quiz MVP（个性化智能出题最小可行版本）

这个仓库提供一个可直接起步的 MVP 骨架：

- 使用 SQLite 存 metadata 题库与学生作答记录。
- 使用 sentence-transformers 生成题目摘要 embedding。
- 使用 FAISS 建立向量索引。
- 使用 FastAPI 暴露 `/select` 接口，按目标标签和难度筛题，再按学生画像打分随机抽题。

## 目录结构

```text
.
├── backend/
│   ├── api.py
│   └── selection.py
├── data/
│   └── questions.json
├── scripts/
│   ├── import_questions.py
│   ├── build_embeddings.py
│   └── build_faiss.py
└── requirements.txt
```

## Step 0：安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1：导入题库到 SQLite

```bash
python scripts/import_questions.py
```

会在 `data/questions.db` 中创建：

- `questions`（题目元数据 + 完整题面）
- `student_records`（学生作答日志）

## Step 2：生成 embedding

```bash
python scripts/build_embeddings.py
```

输出：

- `data/embeddings.npy`
- `data/ids.npy`

## Step 3：构建 FAISS 索引

```bash
python scripts/build_faiss.py
```

输出：

- `data/faiss.index`
- `data/faiss_ids.npy`

## Step 4：启动 API

```bash
uvicorn backend.api:app --reload --port 8000
```

检查健康接口：

```bash
curl http://127.0.0.1:8000/health
```

调用选题接口：

```bash
curl -X POST http://127.0.0.1:8000/select \
  -H 'Content-Type: application/json' \
  -d '{
    "student_id": "stu_001",
    "target_tags": ["代数"],
    "n": 2,
    "difficulty": 3
  }'
```

## 现在能做什么（MVP能力）

- 题目导入、查询和筛选。
- 按标签 + 难度窗口（±1）检索候选题。
- 基于学生能力与弱项加权打分，并保留随机性抽题。

## 下一步建议

1. 在 `student_records` 上实现画像计算（ability + tag_stats）并接入 `/select`。
2. 真正接入向量召回（API 里先 SQL 过滤，再 FAISS top-k 排序）。
3. 增加 `/submit` 接口，写入作答记录并更新学生画像。
4. 对错题触发 LLM 评分/解析（按需调用，控制 token）。
