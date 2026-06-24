from sklearn.tree import DecisionTreeClassifier
from tools.imitation.train import export_tree
from tools.imitation.policy import score

def test_export_matches_sklearn():
    X = [[0.0], [0.1], [0.9], [1.0]]
    y = [0, 0, 1, 1]
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    tree = export_tree(clf)
    for x in X:
        # exported P(class=1) must equal sklearn's predict_proba[:,1]
        p = clf.predict_proba([x])[0][1]
        assert abs(score(tree, x) - p) < 1e-9
