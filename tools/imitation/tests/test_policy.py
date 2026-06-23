from tools.imitation.policy import score, pick

# tiny hand-built tree: if x[0] <= 0.5 -> leaf value 0.1 else leaf value 0.9
TREE = {"feature": [0, -1, -1], "threshold": [0.5, 0.0, 0.0],
        "left": [1, -1, -1], "right": [2, -1, -1], "value": [0.0, 0.1, 0.9]}

def test_score_walks_tree():
    assert score(TREE, [0.0]) == 0.1
    assert score(TREE, [1.0]) == 0.9

def test_pick_argmax_respects_single_select():
    # option 1 has higher feature -> higher score -> chosen
    state = []
    opts = [[0.0], [1.0]]
    out = pick(TREE, opts, state, min_count=1, max_count=1)
    assert out == [1]

def test_pick_multi_select_returns_top_k():
    opts = [[1.0], [0.0], [1.0]]
    out = pick(TREE, opts, [], min_count=2, max_count=2)
    assert len(out) == 2 and set(out) <= {0, 2}
