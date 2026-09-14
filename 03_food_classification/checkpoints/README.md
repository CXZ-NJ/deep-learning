# Checkpoints

训练生成的 `.pth` 模型保存在这里，但不会提交到 GitHub。

修复后的 checkpoint 只保存模型参数、优化器状态、epoch 和最佳验证准确率，不再直接保存完整 Python 模型对象。
