# FFmpeg

本目录不提交可执行文件。请自行从FFmpeg官网列出的Windows构建来源获取FFmpeg，将 `ffmpeg.exe` 放在这里，或将FFmpeg加入系统PATH。软件先查本目录，再查PATH。

上游入口：<https://ffmpeg.org/download.html>。摄像头设备枚举需要FFmpeg；PyAV已随Python依赖安装，但不能替代这里的独立命令行程序。现有本机ffmpeg.exe可以继续使用。

若另行分发第三方二进制，请同时遵守其原始许可证和源码提供要求；下载时保留所选构建的上游许可证与源码来源；本仓库不携带本机二进制或其安装副本。
