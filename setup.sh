#!/bin/bash

# venv を作成するパス（このスクリプトのディレクトリ内に作成）
VENVPATH="`dirname $0`/venv"
ABS_VENVPATH=`cd $(dirname ${0}) && pwd`

echo "<<<  Start Setup.  <<<"

# venv が作成されていれば venv は作成しない
if [ ! -e $VENVPATH ]; then
    if [ "$(uname)" == "Darwin" ]; then
        python3 -m venv $VENVPATH
        echo "Create venv (at $ABS_VENVPATH)"
    elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]; then
        python -m venv $VENVPATH
        echo "Create venv (at $ABS_VENVPATH)"
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        python3 -m venv $VENVPATH
        echo "Create venv (at $ABS_VENVPATH)"
    else
        "Unknown OS"
    fi
else
    echo "[Warning]: Already exist venv! Please remove $VENVPATH if you want to setup again."
fi

# venv を有効化
if [ "$(uname)" == "Darwin" ]; then
    source "$VENVPATH/bin/activate"
elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]; then
    . venv/Scripts/activate
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    source "$VENVPATH/bin/activate"
else
    "Unknown OS"
fi

# black: Python コードフォーマッター
# pylint: コード解析
# opencv-python: Python用OpenCV
pip install black pylint opencv-python

# venv を無効化
deactivate

echo ">>> Finished Setup. >>>"
exit 0