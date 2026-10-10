# V2：两条并行修复路线

V2a RGB Anchor通过视觉条件一致性和adaptation改善124帧基本结构，动作控制仍失败。V2b Same-σ保留历史/当前联合双向去噪，持续A/D124帧动作与基本结构可用，但计算昂贵、没有persistent KV。两者没有权重继承关系。

- [V2a RGB Anchor](v2a_rgb_anchor/README.md)
- [V2b Same-σ](v2b_same_sigma_local_bidir/README.md)

严格因果与persistent KV的已验收结果恢复为[V3 Original Feasibility Baseline](../v3/v3_baseline/README.md)。V3-SW-G/SW-L是其独立Sliding Window候选。短暂的V2c命名已被用户最新决定取代。
