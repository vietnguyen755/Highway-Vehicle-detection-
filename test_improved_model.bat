@echo off
echo ============================================================
echo TESTING IMPROVED MODEL WITH BETTER TRUCK/BUS CLASSIFICATION
echo ============================================================
echo.
echo Model: yolov8m_stage2_improved
echo Video: stage2_final_test.mp4
echo Output: detection_output_improved.mp4
echo.

call venv\Scripts\activate.bat
python main.py --video stage2_final_test.mp4 --output detection_output_improved.mp4

echo.
echo ============================================================
echo TESTING COMPLETED!
echo ============================================================
echo Check detection_output_improved.mp4 for results
pause

