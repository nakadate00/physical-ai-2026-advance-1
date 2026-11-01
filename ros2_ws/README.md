# ROS 2 統合（オプション）

スタンドアロン実装（`src/agripioneer`）のコアモジュールをそのまま再利用し、
知覚・計画・制御をトピックで疎結合にした ROS 2 Humble パッケージ。

```
simulator_node   MuJoCo の物理と草地。センサ配信と刈刃の駆動
perception_node  /camera/image_raw → U-Net → /grass/map (OccupancyGrid)
planner_node     /grass/map → クラスタ分解 + TSP → /plan (Path)
controller_node  /plan + /odom → Pure Pursuit → /cmd_vel (Twist)
```

| トピック | 型 | 発行元 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | controller_node |
| `/odom` | `nav_msgs/Odometry` | simulator_node |
| `/camera/image_raw` | `sensor_msgs/Image` (rgb8, 96×96) | simulator_node |
| `/camera/pose` | `geometry_msgs/PoseStamped` | simulator_node |
| `/mow/progress` | `std_msgs/Int32MultiArray` `[刈取済, 対象]` | simulator_node |
| `/grass/map` | `nav_msgs/OccupancyGrid` (61×61 @3 cm) | perception_node |
| `/plan` | `nav_msgs/Path` | planner_node |
| `/mission/phase` | `std_msgs/String` | planner_node |
| `/controller/finished` | `std_msgs/Bool` | controller_node |
| `~/replan` | `std_srvs/Trigger` (service) | planner_node |

## 実行

GPU なしのコンテナで完結する（MuJoCo のオフスクリーン描画は OSMesa を使う）。

```bash
# リポジトリのルートで
docker build -t agripioneer-ros -f ros2_ws/Dockerfile ros2_ws
docker run --rm -v "$PWD":/work agripioneer-ros bash -lc '/work/ros2_ws/run_demo.sh 150'
```

TLS 傍受型プロキシ環境では `ros2_ws/ca/` に組織のルート CA を置いてからビルドする
（`ros2_ws/ca/README.md` 参照）。

## 動作確認の結果

`docs/ros2_demo.log` に実行ログを残している。1 坪・草 206 本の条件で

```
SURVEY（探査, 6 点の経路）→ 地図被覆率 95.5 % → 第 1 巡 草むら 1 箇所 / 13.73 m
→ 地図被覆率 100 % → 第 2 巡 草むら 2 箇所 / 3.11 m → 刈り残しなし
刈り取り 205/205 (100.0 %)、走行 21.84 m、所要 約 81 s
```

スタンドアロン実装（17.03 m）より走行距離が長いのは、ROS 版が
`VERIFY` フェーズのしきい値漸減を簡略化しているため。
