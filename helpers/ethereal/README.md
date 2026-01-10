# Ethereal Subaccount 绑定工具

## 使用方法

本目录包含两个辅助工具，用于完成 Ethereal “linked signer” 绑定所需的双重签名流程：

1. `subaccount_eip712.html`

   - 作用：用浏览器 + MetaMask 生成主钱包的 EIP-712 `signature`（授权把 linked signer 绑定到子账户），并生成需要的请求体/命令。
   - 用法：
     - 用浏览器打开该文件，连接主钱包。
     - 填入 `signer`（待绑定的 linked signer 地址），保持 `subaccount label` 默认 `primary`。
     - 点击 “Sign Typed Data” 生成 `signature`，页面会自动生成 `signerSignature` 的 Python 命令。
     - 将 linked signer 私钥放在 `.env` 的 `ETHEREAL_PRIVATE_KEY` 后，运行生成的命令，得到 `signerSignature`，粘贴回页面。
     - 点击 “Link Subaccount” 发送 `/v1/linked-signer/link` 请求，状态区会显示成功/失败。
   - 安全性：主钱包签名在浏览器/钱包内完成，避免以明文暴露主钱包私钥；官方 UI 暂未提供该绑定流程，因此用本地页面完成交互。

2. `sign_linked_signer.py`
   - 作用：用 linked signer 私钥生成 `signerSignature`（证明你控制待绑定的 signer 地址）。
   - 前置：在 `.env` 配置 `ETHEREAL_PRIVATE_KEY` 为 linked signer 私钥。
   - 用法示例（参数来自页面提示）：

    ```bash
    python helpers/ethereal/sign_linked_signer.py \
       --sender <主钱包地址> \
       --subaccount <bytes32 子账户名> \
       --nonce <页面给出的 nonce> \
       --signed-at <页面给出的 signedAt>
    ```

    仅输出签名 hex，将其填入 HTML 的 `signerSignature` 输入框即可。

## 为什么需要用工具绑定子账户？

- Ethereal 官方界面暂未提供子账户绑定 linked signer 的前端；需要用户自己完成 EIP-712 授权。
- 主钱包私钥不应离开钱包/浏览器环境，HTML + MetaMask 可避免在本地脚本中明文处理主钱包私钥。
- linked signer 私钥只用于生成 `signerSignature`，可放在 `.env` 后由 Python 脚本离线签名。

## 可能遇到的问题

1. subaccount label 目前只支持 'primary' 的 bytes32 编码，使用其他编码会报错（暂时不知道原因）
2. subaccount address 如果之前添加过，然后 revoke 了，不能再次添加，需要换一个新的钱包来作为 subaccount 绑定
3. subaccount 不需要 gas ，临时生成一个即可（例如 `cast wallet new`）

---

请在安全环境中使用以上工具，并确认网络指向官方 API（默认 `https://api.ethereal.trade`）。\*\*\*
