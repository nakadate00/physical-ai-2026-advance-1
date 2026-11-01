# AgriPioneer の ROS 2 統合を検証するためのイメージ.
# MuJoCo のオフスクリーン描画は GPU 無しのコンテナでも動くよう OSMesa を使う.
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ca-certificates \
        libosmesa6 libosmesa6-dev libgl1 libglfw3 \
        ros-humble-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

# TLS 傍受型プロキシ環境向け: ca/ に組織のルート CA (*.crt) を置くと取り込む.
# 証明書が無ければ何もしない (ca/README.md 参照).
COPY ca/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates \
 && pip3 config set global.cert /etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN pip3 install --no-cache-dir \
        "numpy>=1.24,<2" \
        "mujoco==3.2.7" \
        "scipy>=1.10" \
        "scikit-learn>=1.3" \
        "joblib>=1.3" \
        "pillow>=10.0" \
        "torch==2.4.1" --extra-index-url https://download.pytorch.org/whl/cpu

# MuJoCo を GPU 無しのヘッドレス環境で描画させる
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
# スタンドアロン実装のコアモジュールを ROS ノードから import できるようにする
ENV PYTHONPATH=/work/src:${PYTHONPATH}
ENV RCUTILS_COLORIZED_OUTPUT=1

WORKDIR /work/ros2_ws
CMD ["bash"]
