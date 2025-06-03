import os
import json

root_dir = r"D:\work\paperRelated\并行代码生成\开源仓库\results\HPCL results"

# 遍历所有文件夹和文件
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".jsonl"):
            file_path = os.path.join(dirpath, filename)
            print(f"处理文件: {file_path}")
            processed_lines = []
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        try:
                            record = json.loads(line)
                            predict = record.get("predict", "").strip()
                            label = record.get("label", "").strip()
                            processed_lines.append({"predict": predict, "label": label})
                        except json.JSONDecodeError:
                            print(f"  ⛔ 跳过无效 JSON 行 in {filename}")

                # 覆盖原文件
                with open(file_path, "w", encoding="utf-8") as outfile:
                    for record in processed_lines:
                        json.dump(record, outfile, ensure_ascii=False)
                        outfile.write("\n")
                print(f"  ✅ 完成处理: {filename}")
            except Exception as e:
                print(f"  ❌ 处理失败: {filename}, 错误信息: {e}")
print(f"  ✅ 完成处理")