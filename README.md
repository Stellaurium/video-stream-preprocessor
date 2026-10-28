# Video Stream Preprocessor / ScreenPrep

将侧拍屏幕的实时画面进行平面透视校正和裁剪，通过本机HTTP输出无损最新帧。相机位置稳定时只需标定一次。

## 首次运行（Windows x64）

1. 安装 **Python 3.11 x64**，建议加入PATH。安装入口会实际校验版本与64位架构；即使系统的py启动器登记错误，也不会使用Python 2.x继续安装。
2. 解压/clone本仓库，运行 **tools/安装依赖.cmd**，创建独立 `.venv` 并安装固定版本依赖。
3. 将 `ffmpeg.exe` 放入 `bin/`，或安装到系统PATH，详见 [FFmpeg准备](bin/README.md)。
4. 双击 **启动软件.cmd**。以后直接使用这个入口，首次setup不必重复执行。

本项目独立运行，不要求另一个项目的源码放在旁边。两个应用通过HTTP协作，详见[模块协作接口](docs/模块协作接口.md)。 可独立使用GUI或无界面入口。

## 安装参数与重试

双击 `tools/安装依赖.cmd` 可完成通常安装。命令行/自动化使用：

```powershell
# 指定已安装的解释器；路径仅为示例，替换为自己的Python位置
.\tools\安装依赖.cmd -Python "C:\Python311\python.exe" -NoPause
# 仅检查解释器，不创建环境、不下载依赖
.\tools\安装依赖.cmd -CheckOnly -NoPause
# 开发验收，预处理项目同时安装pytest
.\tools\安装依赖.cmd -Dev -NoPause
```

也支持 `APP_PYTHON` 环境变量指定解释器。setup失败会返回非零退出码；修正网络或路径后可重复执行，已安装依赖和校验正确的模型会跳过。OCR可用 `-SkipModels`先安装依赖，之后必须运行 `tools/prepare_models.py`准备模型才能识别。

FFmpeg是独立前置依赖：可以使用系统PATH中的已安装版本，或放到bin；setup会明确显示找到的位置，未找到会提示准备方法。它不会从另一个项目复制程序。

移动源码到新位置后，应在新位置重新创建.venv并运行setup；Windows虚拟环境内的命令行入口可能记录旧路径，不应只靠剪切环境来迁移。

## 文档

- [用户操作手册](使用说明.md)
- [实现与维护](docs/实现与维护.md)
- [维护工具](tools/README.md)
- [源码发布、隐私和迁移](docs/源码发布.md)
- [标定配置与代码接口](docs/接口与配置.md)

## 仓库与本机文件

`screenprep/`和 `app.py` 是运行代码；`tests/`是独立可重复的回归测试。`.venv/`、`data/`、本地二进制、模型权重和题库均由忽略规则排除。setup从依赖源/模型源下载运行材料；使用已有模型时本地OCR不需要上传题图。LLM API是可选功能，仅明确发送时联网调用。

临时实验统一放在仓库外的测试目录。本仓库不包含历史验收截图、私人题目、旧EXE或代码备份。
