# 🦄️ unicorn
unicorn 一个中文增强大模型研究项目，本项目由[严瑞卿](https://github.com/yrqUni)和[郑凌瀚](https://github.com/zlh1992)创建。

## 代码
代码请进入各分支查看，main分支仅作入口展示页面使用。
### 1) https://github.com/yrqUni/unicorn/tree/dschat
+ 基于dschat，支持LoRA、fp16和ZeRO-1/2/3。
+ 在极端情况下，支持极低显存训练（几乎无最低显存要求，需要Offload梯度、优化器状态和模型参数至CPU和RAM）。
### 2) https://github.com/yrqUni/unicorn/tree/peft
+ 基于peft，支持LoRA、8Bit、fp16和ZeRO-1/2。
+ 在极端情况下，支持极低显存训练（约10G，需要Offload梯度和优化器状至CPU和RAM）。
+ 注意！8Bit训练效果不稳定，请注意观察训练情况，慎重使用。

## 试用链接
http://116.63.188.130:1017/
