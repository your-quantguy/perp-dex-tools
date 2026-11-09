# 飞书通知配置指南

## 🎉 新版本：群自定义机器人 Webhook（推荐）

### ✨ 特点

✅ **无需后台配置** - 直接在群里添加机器人即可  
✅ **富文本卡片** - 支持颜色、图标、多栏布局  
✅ **代码控制样式** - 所有格式在代码中定义  
✅ **交互式通知** - 更直观、更美观的通知效果

### 📋 配置步骤

#### 1. 在飞书群中添加自定义机器人

1. 打开你想接收通知的飞书群
2. 点击右上角 "..." -> "设置"
3. 选择 "群机器人" -> "添加机器人"
4. 选择 "自定义机器人"

#### 2. 配置机器人

1. **机器人名称**：自定义，如 "交易机器人通知"
2. **关键词**（可选）：
   - 如果需要安全验证，可以设置关键词（如 "交易"）
   - 不需要可以跳过
3. **IP白名单**（可选）：一般不需要设置

#### 3. 获取 Webhook URL

配置完成后，你会得到一个 Webhook URL，格式类似：
```
https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

#### 4. 配置环境变量

在你的 `.env` 文件中添加：

```bash
# 方式1: 无关键词（推荐）
LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/你的webhook地址

# 方式2: 有关键词
LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/你的webhook地址
LARK_WEBHOOK_KEYWORD=交易  # 你设置的关键词
```

### 🎨 通知效果

#### 启动通知
- 🚀 绿色卡片
- 显示实例ID、交易所、交易对
- 显示完整策略参数

#### 错误通知
- ⚠️ 红色卡片
- @所有人提醒
- 显示错误类型和详情
- 显示堆栈预览

#### 每小时统计
- 📊 蓝色卡片
- 显示各项操作统计
- 显示预估手续费
- 显示统计周期

### 🆚 对比旧版本

| 特性 | 新版本（Webhook） | 旧版本（流程） |
|------|------------------|---------------|
| 配置难度 | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| 后台配置 | ❌ 不需要 | ✅ 需要 |
| 样式自定义 | ✅ 支持 | ❌ 固定 |
| 颜色图标 | ✅ 支持 | ❌ 不支持 |
| 卡片布局 | ✅ 多栏 | ❌ 单栏 |
| 更新维护 | ✅ 代码中 | ❌ 后台 |

---

## 🔙 旧版本：流程 Webhook（不推荐）

### ⚠️ 缺点

- ❌ 需要在飞书后台配置流程
- ❌ 需要手动创建字段解析规则
- ❌ 样式固定，无法自定义
- ❌ 维护麻烦，修改需要同时改代码和后台

### 📋 配置步骤（如果你仍要使用）

1. 登录飞书管理后台
2. 创建自动化流程
3. 配置 Webhook 触发器
4. 添加字段解析规则
5. 配置消息发送动作
6. 获取 Token 并配置到 `.env`:
   ```bash
   LARK_TOKEN=你的流程token
   ```

---

## 🚀 快速测试

配置完成后，运行交易机器人测试通知：

```bash
# 启动机器人，会发送启动通知
python runbot.py --exchange backpack --ticker SOL --quantity 0.1
```

如果配置正确，你会在飞书群中收到一条漂亮的卡片通知！

---

## 🐛 故障排除

### 问题1: 没有收到通知

**检查清单：**
- ✅ 确认 `.env` 中配置了 `LARK_WEBHOOK_URL`
- ✅ 确认 Webhook URL 格式正确
- ✅ 确认机器人已添加到群中
- ✅ 检查终端是否有错误日志

### 问题2: 提示关键词验证失败

**解决方法：**
- 在 `.env` 中配置 `LARK_WEBHOOK_KEYWORD=你的关键词`
- 或者在飞书后台取消关键词验证

### 问题3: 通知内容显示不正常

**可能原因：**
- Webhook URL 错误
- 关键词配置错误
- 网络连接问题

---

## 📚 更多信息

- [飞书自定义机器人文档](https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot)
- [飞书消息卡片设计指南](https://open.feishu.cn/document/common-capabilities/message-card/message-cards-content/using-card-template)

---

## 🎯 推荐配置

```bash
# .env 文件配置示例
LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx
# LARK_WEBHOOK_KEYWORD=  # 建议不设置关键词，简化配置
```

简单、高效、美观！🎉
