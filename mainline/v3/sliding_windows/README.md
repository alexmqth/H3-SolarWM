# V3 Sliding Window（SW，滑动窗口）

目的：用最近5个祖先KV（Key/Value，注意力键和值缓存）限制历史video KV容量，并独立比较Global / Local旋转位置编码（RoPE）。

[SW-G：Global](global/README.md)158帧有限可行；[SW-L：Local](local/README.md)当前无训练方向无联合收益。正式Baseline不变。

[协议差异、结果和视频](../../../report/v3/sliding_windows/README.md)。本层是定义导航，视频位于报告的global/videos和local/videos。
