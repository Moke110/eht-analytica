# EHT Analytica

## 项目概述

EHT Analytica 利用深度学习模型（U-Net Ensemble ×5）自动识别、追踪 EHT (Engineered Heart Tissue) pillars 并计算下游收缩-舒张指标。

**EHT 背景**: PDMS 支柱 + 心肌细胞 + 纤维蛋白组装成的体外心肌组织，具有自主收缩-舒张功能。通过追踪支柱位移计算收缩力、频率、T80 等关键功能指标。

**技术栈**: Python (FastAPI) 后端 + Vue 3 (Vite) 前端。

## 快速启动

```bash
# 双击 dev.bat，或手动在两个终端中分别运行：
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9876 --reload
cd frontend && npm run dev
# 浏览器打开 http://localhost:5173
```

## 项目结构

```
EHT_Analytica/
├── backend/                    # FastAPI 后端
│   ├── main.py                 # 应用入口 + 静态文件托管
│   ├── api/                    # API 路由
│   │   ├── track.py            # /api/track/* — 视频、模型、追踪
│   │   ├── analyze.py          # /api/analyze/* — CSV 分析与保存
│   │   └── system.py           # /api/system/* — 健康检查、文件对话框、设备信息
│   ├── models/schemas.py       # Pydantic 请求/响应模型
│   ├── services/               # 业务逻辑层
│   │   ├── video_service.py    # 视频会话管理
│   │   ├── tracking_service.py # 模型加载 (GPU/CPU) + 追踪执行
│   │   ├── analysis_service.py # 分析流水线
│   │   └── task_manager.py     # 任务生命周期 + SSE 进度推送
│   └── utils/config.py         # 持久化配置 (模型路径、文件历史)
├── frontend/                   # Vue 3 前端 (Vite)
│   ├── src/
│   │   ├── App.vue             # 根组件，Tab 切换 Track/Analyze，自动加载模型
│   │   ├── components/
│   │   │   ├── track/          # TrackPanel, Sidebar (DEVICE), VideoCanvas, Progress
│   │   │   ├── analyze/        # AnalyzePanel, CsvList, Progress
│   │   │   └── shared/         # FilePicker (路径历史), StatusIndicator
│   │   ├── composables/        # useApi, useSse, useTaskProgress
│   │   └── assets/main.css
│   └── vite.config.js          # 代理 /api → 127.0.0.1:9876
├── functions/                  # 纯功能函数 (无 GUI/Web 依赖)
│   ├── video_processor.py      # 视频元数据提取 + 帧扫描
│   ├── tracker.py              # ROI 追踪引擎 (TorchScript 推理)
│   ├── length_data_analyzer.py # 长度→力转换 + 峰值检测 + 周期分割 + T80
│   └── roi.py                  # 纯数据 ROI 类
├── model/track_unet_v2/        # U-Net 追踪模型 (.pt + .pth)
├── src/test/                   # 测试视频 + CSV 数据
├── dev.bat                     # 开发模式一键启动
└── requirements.txt
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/system/health | 健康检查 |
| GET | /api/system/config | 获取配置 |
| POST | /api/system/config/save-path | 保存路径历史 |
| GET | /api/system/device | 设备信息 (GPU/CPU + 显存/内存) |
| POST | /api/system/native-file-dialog | 原生文件对话框 |
| POST | /api/system/open-folder | 在文件管理器中打开 |
| POST | /api/system/list-csv | 列出文件夹中 CSV |
| POST | /api/track/video/open | 打开视频，返回首帧 |
| POST | /api/track/video/process | 后台扫描所有帧 |
| POST | /api/track/model/load | 加载 TorchScript 模型 (GPU优先) |
| GET | /api/track/model/status | 模型状态 (设备、路径、显存) |
| POST | /api/track/start | 开始追踪 |
| DELETE | /api/track/task/{id} | 取消任务 |
| GET | /api/track/task/{id}/stream | SSE 进度流 |
| POST | /api/analyze/validate | 验证 CSV |
| POST | /api/analyze/start | 开始分析 |
| POST | /api/analyze/save | 保存结果 |
| GET | /api/analyze/task/{id}/stream | SSE 进度流 |

## 功能完成情况

### 已实现
- FastAPI + Vue 3 Web 应用，替代原 wxPython GUI
- Track 页签: 视频打开、ROI 绘制、实时预览缩放、模型管理
- Analyze 页签: CSV 批量分析、结果保存
- 模型自动加载（启动时从持久化路径加载）+ 失败错误提示
- GPU 推理 (CUDA)，RTX 4070 Ti SUPER 上 4 ROI × 10003 帧 ≈ 18 分钟（vs CPU 16 小时）
- DEVICE 模块：侧边栏顶部显示设备型号和显存/内存
- 文件路径历史分离存储（Recording + Model 各自记忆上次目录）
- SSE 任务进度实时推送，支持取消
- 路径历史持久化 (`config/app_config.json`)

### 待完成
- 桌面打包: PyInstaller + pywebview
- 单元测试补充
- 部署到 NAS/服务器

## 开发环境

- **Python**: conda `eht` 环境, PyTorch 2.12.0+cu126
- **Node.js**: 系统 `npm`
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER (16376 MB)
- **端口**: 后端 9876, 前端开发 5173
