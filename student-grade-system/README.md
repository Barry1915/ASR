# 学生成绩关系系统（Java CLI）

无需 Maven/Gradle，直接用 `javac` 编译运行。

## 结构
- `src/main/java/com/example/gradesystem` 源码
- `data/` 持久化 TSV 文件（程序运行时自动创建）
- `compile.sh`、`run.sh` 编译与运行脚本

## 使用
```bash
./compile.sh
./run.sh
```

数据保存在 `data/students.tsv`、`data/courses.tsv`、`data/enrollments.tsv`。
