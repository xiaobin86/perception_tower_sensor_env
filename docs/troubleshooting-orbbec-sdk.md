# Orbbec Gemini 336L 常见问题

## USB PAL 错误

### 症状

启动相机时报错：

```
Component constructor threw an exception: Usb pal is not exist, please check the build config that you have enabled BUILD_USB_PAL
```

### 根因

Orbbec SDK 预编译库在构建时没有启用 `BUILD_USB_PAL` 选项，导致运行时找不到 USB 平台抽象层模块。

关键证据：

```bash
# 有问题的 SDK (v2.9.3) - 没有 libob_usb.so 依赖
$ ldd libOrbbecSDK.so.2.9.3
# 无 libob_usb.so

# 正常的 SDK (v2.8.6) - 有完整的 USB PAL 支持
$ strings libOrbbecSDK.so.2.8.6 | grep "LinuxUsbPal"
# 有输出，说明 USB PAL 模块存在
```

### 解决方案

确保使用 SDK v2.8.6（已验证可用），不要使用 v2.9.3。

当前 `docker/orbbec-v2-main.tar.gz` 中的 SDK 已经是 v2.8.6，构建时会自动使用正确版本。

Dockerfile 中已添加构建时验证：

```bash
SDK_FILE=$(find /opt/orbbec_ws/install/orbbec_camera/lib -name "libOrbbecSDK.so.*.*.*" -not -type l | head -1)
strings "$SDK_FILE" | grep -q "LinuxUsbPal" && echo "✓ USB PAL module present" || echo "✗ USB PAL missing"
```

### 验证方法

```bash
# 检查 SDK 是否包含 USB PAL
strings /opt/orbbec_ws/install/orbbec_camera/lib/libOrbbecSDK.so.* | grep -i "usb pal"

# 正常输出应包含：
# LinuxUsbPal
# ObLibuvcDevicePort
```

---

## 参考信息

| SDK 版本 | USB PAL 支持 | 状态 |
|---------|-------------|------|
| v2.8.6 | ✓ | 可用 |
| v2.9.3 | ✗ | 不可用 |
| v1.10.27 | ✓ | API 不兼容 v2 |

---

## 环境要求

- ROS2 Humble
- Docker with USB device passthrough (`--device /dev:/dev`)
- Orbbec Gemini 336L 固件版本 >= 1.2.20
