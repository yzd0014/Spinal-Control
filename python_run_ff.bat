@echo off
setlocal

rem Set the number of times to execute the Python script
set iterations=8

rem Loop through the iterations
for /l %%i in (1, 1, %iterations%) do (
    echo Running iteration %%i
    rem Call the Python script
    python .\train_invertpendulum.py 2
)

endlocal