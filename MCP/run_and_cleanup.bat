@echo off

echo.
echo --- Activating Conda environment 'learn_llm'...
call conda activate learn_llm

echo.
echo --- Starting the MCP Agent Client (agent.py)...
echo.

python agent.py

echo.
echo --- Script finished. ---