# レポート課題１
1910094 植木 駿介


```python
import numpy as np

#いずれの式も閾値は0とした
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(w*x) + b
    if y <= 0:
    	return 0
    return 1

def NAND(x1, x2):
    #NANDはANDの否定なのでANDの式のバイアス、重みを(-1)倍した値に設定した
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(w*x) + b
    if y <= 0:
    	return 0
    return 1

def OR(x1, x2):
    #ORは(0,0)のみ計算結果がマイナスになるようにバイアスをマイナスの値に設定した
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.1
    y = np.sum(w*x) + b
    if y <= 0:
    	return 0
    return 1

def XOR(x1, x2): 
	#XORはNANDとORの結果のANDで表せる
	x = OR(x1, x2)
	y = NAND(x1, x2)
	z = AND(x, y)
	if z == 0:
		return 0
	return 1

#import時など自分が実行された時でないときに実行されないようにする(if __name__ == '__main__':)
if __name__ == '__main__':
	a = [['AND '],['NAND '],['OR '],['XOR ']]
    #list a にそれぞれの結果を保存する
	for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
		a[0].append(AND(xs[0], xs[1]))
		a[1].append(NAND(xs[0], xs[1]))
		a[2].append(OR(xs[0], xs[1]))
		a[3].append(XOR(xs[0], xs[1]))
    #結果の表示
	for i in a:
		print(i)

```

    ['AND ', 0, 0, 0, 1]
    ['NAND ', 1, 1, 1, 0]
    ['OR ', 0, 1, 1, 1]
    ['XOR ', 0, 1, 1, 0]
    

## 感想
今回は手動でバイアスや重みを決めたのでグラフの直線をイメージして値を決めた。いつもはjupyter notebookを使っていなかったので今回使ってみて結果など様々なことを残すことができ便利だと感じた。
