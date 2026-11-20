@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat

for %%s in (
    ingest\01_games.py
    ingest\02_player_box.py
    ingest\03_shots.py
    ingest\04_schedule.py
    ingest\05_pbp.py
    ingest\06_league.py
    ingest\07_tracking.py
    ingest\08_draft.py
    ingest\09_league_tracking.py
    ingest\10_contracts.py
    ingest\11_league_schedule.py
    process\build_stints.py
    process\rapm.py
    process\on_off.py
    process\lineups.py
    process\team_lineups.py
    process\spacing.py
    process\game_shots.py
    process\league_overview.py
    process\cap_sheet.py
    process\draft_prep.py
    process\draft_model.py
    process\team_views.py
    ingest\validate.py
) do (
    echo.
    echo === %%s ===
    python "%%s"
    if errorlevel 1 (
        echo.
        echo FAILED on %%s - see error above.
        pause
        exit /b 1
    )
)

echo.
echo Refresh complete %date% %time%
echo Open Power BI and click Home ^> Refresh.
pause