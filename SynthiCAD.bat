@echo off
REM num_runs - number of times to run main.py script
set num_runs=1
REM Kindly provide the path of SynthiCAD to use this batch file, example is given below
cd /d D:\SynthiCAD
for /l %%x in (1,1,%num_runs%) do (
    echo %%x
    blenderproc run code/main.py code/resources/Dataset/*.ply code/output
)