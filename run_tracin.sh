#!/bin/bash

# TracIn 批量执行脚本

echo "=========================================="
echo "TracIn Batch Execution Script"
echo "=========================================="
echo ""

# 配置文件列表
CONFIG_FILES=(
    "./configs/config_datacleaning.yaml"
    "./configs/config_datacleaning_all.yaml"
)

# 记录开始时间
START_TIME=$(date +%s)

# 循环执行
for CONFIG in "${CONFIG_FILES[@]}"
do
    echo ""
    echo "=========================================="
    echo "Starting: $CONFIG"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    # 执行
    python tracin_main.py --config "$CONFIG"
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo ""
        echo "SUCCESS: $CONFIG completed"
    else
        echo ""
        echo "ERROR: $CONFIG failed"
        echo "Stopping execution"
        exit 1
    fi
    
    echo ""
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "All tasks completed"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="