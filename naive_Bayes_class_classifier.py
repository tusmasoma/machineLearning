import math
import sys
import json

from janome.tokenizer import Tokenizer

def split(doc, word_class=["形容詞", "形容動詞", "感動詞", "副詞", "連体詞", "名詞", "動詞"]):
    t = Tokenizer()
    tokens = t.tokenize(doc)

    word_list = []
    for token in tokens:
        #if '名詞' in token.part_of_speech or '形容詞' in token.part_of_speech:
        word_list.append(token.surface)
    return [word for word in word_list]

def getwords(doc):
    words = [s.lower() for s in split(doc)]  #splitメソッドで分割した文字列を全ての小文字に変換
    return tuple(w for w in words)




class NaiveBayes:
    def __init__(self) -> None:
        self.quantity_vocabularies = set() # 特微量の集合(重複なし)
        self.quantity_count_for_each_class = {} #クラスごとの特微量の出現回数を保存 {クラスA : {特微量1:出現回数,特微量2:出現回数,...},クラスB : ....}
        self.class_count = {} #{クラスA:回数} 訓練データの中でそれぞれのクラスが出た回数

    def train(self,sampleData):
        """
        sampleData = (特微量,クラス)
        """
        quantityList = getwords(sampleData[0])

        for quantity in quantityList:
            self.quantity_count_for_each_class.setdefault(sampleData[1], {}) #self.quantity_count_for_each_class = {クラス:{}} 最初のループ以外は無視していると思える
            self.quantity_count_for_each_class[sampleData[1]].setdefault(quantity,0) #self.quantity_count_for_each_class = {クラス:{特微量1:0},...}
            self.quantity_count_for_each_class[sampleData[1]][quantity] += 1   #self.quantity_count_for_each_class = {クラス:{特微量1:1},...}
            self.quantity_vocabularies.add(quantity)  #self.quantity_vocabularies = {特微量1,特微量2,...}

        self.class_count.setdefault(sampleData[1],0)
        self.class_count[sampleData[1]] += 1  #訓練データの中でクラスがでた回数を記録
    
    #クラスの出現確率 P(A)
    def __class_probability(self,A):
        """
        P(A) = 訓練データのうち、クラスAである数 / 訓練データの数
        """
        return float(self.class_count[A] / sum(self.class_count.values()))
    
    # 推定フェーズ：あるカテゴリの中に単語が登場した回数
    def __incategory(self,A,Hi):
        #指定した特微量が訓練データのクラスにない場合がある
        if Hi in self.quantity_count_for_each_class[A]:
            return float(self.quantity_count_for_each_class[A][Hi])
        return 0.0
    
    #P(Hi|A) ゼロ頻度問題の対策あり
    def __quantity_conditional_probability(self,A,Hi):
        """
        P(Hi|A) = (「クラスAの中で特微量Hiが出現する回数」+ 1) / (「クラスAの特微量の総数」+ 「訓練データの特微量の総数(重複なし)」)
        """
        return (self.__incategory(A,Hi) + 1.0) / (sum(self.quantity_count_for_each_class[A].values()) + len(self.quantity_vocabularies))
    
    #アンダーフロー対策をしたP(A|H1,...Hn)を計算
    def __score(self,A,test_quantityList):
        score = math.log(self.__class_probability(A))
        for Hi in test_quantityList:
            score+=math.log(self.__quantity_conditional_probability(A,Hi))
        return score
    
    #分類
    def classifier(self,test_quantity):
        """
        渡された特微量(test_quantity)をHとして、それぞれ独立と仮定して、H = (H1)U....U(Hn)とおく。
        このHを用いて、全ての訓練データのクラスAに対して、P(A|H)を求める。
        つまり、
        for classA in train_class_list:
            P(classA|H)
        そして、P(A|H)が最大となるAを特微量Hの推定されるクラスとする
        """
        best = None
        max_score = -sys.maxsize
        test_quantityList = getwords(test_quantity)
        for classA in self.quantity_count_for_each_class.keys():
            score = self.__score(classA,test_quantityList)
            if score > max_score:
                max_score = score
                best = classA
        return best

