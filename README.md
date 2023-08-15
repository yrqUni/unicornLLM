# 🦄️ unicornLLM
unicornLLM 一个中文增强大模型研究项目，本项目由[严瑞卿](https://github.com/yrqUni)和[郑凌瀚](https://github.com/zlh1992)创建。
* [unicorn](https://github.com/yrqUni/unicorn)是一个研究项目集成体，本项目是其一部分。

## 代码
代码请进入各分支查看，main分支仅作入口展示页面使用。
### [分支1：dschat版本](https://github.com/yrqUni/unicornLLM/tree/dschat)
+ 基于dschat，支持LoRA、fp16和ZeRO-1/2/3。
+ 在极端情况下，支持极低显存训练（几乎无最低显存要求，需要Offload梯度、优化器状态和模型参数至CPU和RAM）。
### [分支2：PEFT版本](https://github.com/yrqUni/unicornLLM/tree/peft)
+ 基于peft，支持LoRA、8Bit、fp16和ZeRO-1/2。
+ 在极端情况下，支持极低显存训练（约10G，需要Offload梯度和优化器状至CPU和RAM）。
+ 注意！8Bit训练效果不稳定，请注意观察训练情况，慎重使用。
