# 传统工序影像的分类误差与说明改版

本地数据与术语材料供业务人员整理分析资料使用，所有记录均为合成样例。

## 目录

- `fixtures/`：仓库自带的合成输入，只用于说明字段结构和实体关联，样本量不足以得出统计结论。
- `docs/`：字段说明、标注手册（业务规程）与工具实现说明。
- `craftreview/`：可运行的分类评估、细类资格、迁移与复核排期工具。
- `examples/`：`build_demo.py` 生成的可复现演示数据（`demo/` 完整流程，`leaky/` 家族泄漏反例）；生成物不入库，运行脚本即可重建。
- `tests/`：输入结构检查与业务规则回归测试。

## 环境和测试

Python 3.11 标准库，无需第三方依赖。

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q .
```

## 运行

```bash
python3 examples/build_demo.py
python3 -m craftreview analyze --data examples/demo
python3 -m craftreview analyze --data examples/leaky    # 隔离无效，不报总体准确率
python3 -m craftreview analyze --data fixtures          # 样例不足，全部退回父类
python3 -m craftreview catalog-diff --data examples/demo --from-version v1 --to-version v2
```

`analyze` 支持 `--json`、`--models`、`--from-version` / `--to-version`、`--period`、`--no-model`。

## 输入文件

`primary.csv` 为必需；其余文件存在时才加载。

| 文件 | 用途 | 关键字段 |
| --- | --- | --- |
| `primary.csv` | 片段与标签、数据划分 | `clip,family,tradition,task,context,split,coarse,fine,version` |
| `observations.csv` | 结构化观测（非视觉） | `clip,task,context,duration_ms,landmark_count`，其余数值列自动视为信号 |
| `reviews.csv` | 片段级复核 | `review,clip,reviewer,tradition,coarse,fine,verdict,reviewed_at` |
| `findings.csv` | 双评核对结论 | `finding,clips,verdict,reviewer,decided_at`；`verdict=same_performance` 表示同一表演 |
| `catalog.csv` | 目录版本 | `code,version,level,parent,label,scope_notes,approved,max_granularity` |
| `courses.csv` | 课程章节与用途 | `course,chapter,clip,catalog_code,explanation,severity,status,valid_from` |
| `support-1.csv` | 复核人名册 | `reviewer,tradition,skill,minutes,period` |
| `support-2.csv` | 策略覆盖 | `policy,item,rule`，只能收紧手册下限 |

`scope_notes` 支持 `task:join|carve; context:oak|elm` 形式，用于限定细类在哪些工序和器具上下文中成立。

## 判断口径

1. **家族隔离**：训练、校准、检验按 `family` 分组。同一家族跨组即拒绝评估，不输出总体准确率。双评核对判定同源的片段会合并家族，合并后可能使依赖“两个独立家族”的资格失效。
2. **细类上线**：需要两个不同证据家族的独立细类复核、目录批准且在适用范围内、精度下界达到手册阈值（默认单侧 95% Wilson，0.80）。只有父类一致的双评、细类分歧、被同源核对撤销、家族不足、片源只有单一非训练家族，都会退回父类。
3. **上下文**：细类只在 `(工序, 器具)` 上下文内区分类别，同名细类在不同器具下不共享证据。
4. **输出**：每段要么给出细类、要么给出父类、要么拒判；报告按地域（`tradition`）、工序、可标注粒度分别给出误分、拒判与覆盖。
5. **选型**：先剔除隔离无效的模型，再优先比较高代价用途上的细类错误次数，其次比较课程用途加权成本与覆盖率；总体准确率不作为首要指标。
6. **迁移**：已定版章节只有在细类资格成立且新版目录批准时才换说明；新版仍允许的父类说明继续引用；草案章节和细类分歧不迁移。
7. **复核排期**：优先补“差一个独立家族 / 精度不足 / 高成本小类”的缺口；只安排具备该传承支系资质的人，同一人不能充当第二评；复核不会增加独立家族的情形不占用时间；报告下一期的能力缺口。

## 样例数据的结论

仓库自带的 `fixtures/` 只有 5 个片段、无独立双评、无结构化观测，且 `F1` 同时出现在 train 与 test。工具据此输出：无细类可上线，全部继续沿用父类；评估与迁移均不产生结论。这是预期行为，不是运行错误。
