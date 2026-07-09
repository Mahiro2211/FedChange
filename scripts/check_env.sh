#!/bin/bash
# =============================================================================
# 共享环境依赖检查 —— 在运行实验脚本前确认必要的 Python 库已安装。
#
# 用法（在脚本开头 source 本文件，再调用检查函数）:
#     source "$(dirname "$0")/check_env.sh"
#     check_env_core          # 检查核心依赖（所有实验都需要）
#     check_env_torchange     # 额外检查 torchange 基线依赖（torchange_fed / changen2 需要）
#
# 行为：
#   - 缺核心依赖 → 打印中文提示 + 安装命令，以非 0 退出（阻止后续训练）。
#   - 缺 torchange 依赖 → 同样提示并退出；torchange 是可选库，BIT-CD 实验不会调用此函数。
#   - 全部就绪 → 静默通过（返回 0）。
#
# 依赖判定用 importlib.util.find_spec（与 import 行为一致），不实际导入，
# 因此不会触发重型库的初始化，速度快。
# =============================================================================

# 检查单个库是否可导入。成功返回 0，失败返回 1。
_lib_present() {
    python - "$1" <<'PY' 2>/dev/null
import importlib.util as u, sys
sys.exit(0 if u.find_spec(sys.argv[1]) else 1)
PY
}

# 通用检查：第一个参数是要检查的 "import名|pip名" 列表（空格分隔），
# 第二个参数是该档的 pip 安装命令（用于提示）。
_check_libs() {
    local label="$1"; shift
    local install_hint="$1"; shift
    local missing=()
    for entry in "$@"; do
        local import_name="${entry%%|*}"   # | 左侧 = import 时的名字
        if ! _lib_present "$import_name"; then
            # | 右侧 = pip 安装名（用于显示），缺省与 import 名相同
            local pip_name="${entry#*|}"
            missing+=("${pip_name}")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        echo "" >&2
        echo "❌ 缺少 ${label} 依赖：" >&2
        echo "   ${missing[*]}" >&2
        echo "" >&2
        echo "请先安装（在 FedChange/ 目录下）：" >&2
        echo "   ${install_hint}" >&2
        echo "" >&2
        echo "完整安装说明见 README.md「环境配置」章节。" >&2
        return 1
    fi
    return 0
}

# 核心依赖：所有实验（BIT-CD 集中式 / 联邦、划分生成）都需要。
# 格式："import名|pip名"，无 | 时两者同名。
check_env_core() {
    _check_libs "核心" \
        "pip install -r requirements.txt" \
        "torch|torch" "torchvision|torchvision" "einops|einops" \
        "numpy|numpy" "PIL|pillow" "loguru|loguru"
}

# torchange 基线依赖：仅 torchange_fed / changen2 步骤需要。
# 注意 import 名与 pip 包名不同（scikit-image→skimage, segmentation-models-pytorch→segmentation_models_pytorch）。
check_env_torchange() {
    _check_libs "torchange 基线" \
        "pip install torchange \"albumentations>=2.0.0\" tifffile scikit-image datasets ever-beta segmentation-models-pytorch timm" \
        "torchange|torchange" \
        "albumentations|albumentations" \
        "tifffile|tifffile" \
        "skimage|scikit-image" \
        "datasets|datasets" \
        "ever|ever-beta" \
        "segmentation_models_pytorch|segmentation-models-pytorch" \
        "timm|timm"
}
