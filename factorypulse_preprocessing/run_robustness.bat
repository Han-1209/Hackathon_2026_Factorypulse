@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  FactoryPulse 強健性測試（實驗 G + 實驗 H）
echo  G 感測器缺失 約 1-3 分鐘
echo  H 雜訊注入   約 15-40 分鐘（要重讀 15 個原始 .mat）
echo ============================================
echo.
python train_robustness.py %*
echo.
echo ============================================
echo  完成。結果在 results\robustness_summary.csv
echo ============================================
pause
