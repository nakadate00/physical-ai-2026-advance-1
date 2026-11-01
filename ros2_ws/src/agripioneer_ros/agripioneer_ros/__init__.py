"""AgriPioneer の ROS 2 ノード群.

スタンドアロン実装 (`src/agripioneer`) のコアモジュールをそのまま再利用し,
知覚・計画・制御をトピックで疎結合にした構成. 各ノードの責務は

    simulator_node   MuJoCo の物理と草地, センサ配信, 刈刃の駆動
    perception_node  画像 -> U-Net -> 占有格子地図 (OccupancyGrid)
    planner_node     地図 -> クラスタ分解 + TSP -> 経路 (Path)
    controller_node  経路 + 自己位置 -> 速度指令 (Twist)
"""
